import argparse
import io
import os
from typing import Any, Dict, List

import pandas as pd
import tqdm
import yaml
from PIL import Image

from common import ensure_dir, expand_input_paths, filter_paths_for_shard

try:
    from vllm import LLM, SamplingParams
except Exception as e:  # pragma: no cover
    LLM = None
    SamplingParams = None
    _VLLM_IMPORT_ERROR = e


def _load_image_from_row(sample: Dict[str, Any], image_cols: List[str], path_cols: List[str]) -> Image.Image:
    for c in image_cols:
        if c in sample and sample[c] is not None:
            v = sample[c]
            if isinstance(v, (bytes, bytearray)):
                return Image.open(io.BytesIO(v)).convert("RGB")
            if isinstance(v, Image.Image):
                return v.convert("RGB")
    for c in path_cols:
        if c in sample and sample[c]:
            return Image.open(sample[c]).convert("RGB")
    raise KeyError(f"no image found in columns={image_cols} or path_columns={path_cols}")


def process_batch(model, batch, args, *, image_cols: List[str], path_cols: List[str]):
    def get_cropped_images_and_indices(batch):
        all_image = []
        valid_indices = []
        for _, row in batch.iterrows():
            try:
                image = _load_image_from_row(row.to_dict(), image_cols, path_cols)
                all_image.append(image)
                valid_indices.append(1)
            except Exception:
                # Keep length alignment; mark invalid.
                all_image.append(Image.new("RGB", (32, 32), color=(0, 0, 0)))
                valid_indices.append(-1)
        return all_image, valid_indices

    
    all_image, valid_indices = get_cropped_images_and_indices(batch)
    # from IPython import embed; embed()
    this_batch_size = 4
    all_simple_caption = []
    all_detail_caption = []

    spotter_prompt = ' Please describe all the visual concepts in the image in detail, but use concise words withy no more than 120 words.'

    for offset in range(0, len(all_image), this_batch_size):
        batch_images = all_image[offset: offset + this_batch_size]
        prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{spotter_prompt} ASSISTANT:"

        sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     max_tokens=256, n=4, best_of=8)

        batched_data = []
        for im in batch_images:
            batched_data.append({
                'prompt': prompt,
                "multi_modal_data": {
                    "image": im
                }
            })

        outputs = model.generate(batched_data, sampling_params=sampling_params)

        res = []
        for output in outputs:
            cur_res = []
            for o in output.outputs:
                cur_res.append(o.text)
            res.append(cur_res)
        
        all_detail_caption.extend(res)
    
    spotter_prompt = ' Please provide a simple description for this image.'
    for offset in range(0, len(all_image), this_batch_size):
        batch_images = all_image[offset: offset + this_batch_size]
        prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{spotter_prompt} ASSISTANT:"
        sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     max_tokens=256, n=4, best_of=8)

        batched_data = []
        for im in batch_images:
            batched_data.append({
                'prompt': prompt,
                "multi_modal_data": {
                    "image": im
                }
            })
        outputs = model.generate(batched_data, sampling_params=sampling_params)


        res = []
        for output in outputs:
            cur_res = []
            for o in output.outputs:
                cur_res.append(o.text)
            res.append(cur_res)
        
        all_simple_caption.extend(res)

    return all_image, all_simple_caption, all_detail_caption, valid_indices


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main(chunk_index, chunk_num, node_index, node_num, args):
    if LLM is None:  # pragma: no cover
        raise RuntimeError(
            "vLLM is required for captioning stage. "
            f"Import error: {_VLLM_IMPORT_ERROR}"
        )

    cfg = _load_config(args.config)
    data_paths = cfg.get("data_paths") or cfg.get("hdfs_data_paths") or []
    out_dir = cfg.get("output_dir") or ""
    image_cols = cfg.get("image_columns") or ["image", "frame", "binary"]
    path_cols = cfg.get("image_path_columns") or ["image_path", "path"]

    ensure_dir("reservoir/processed_data")
    ensure_dir("reservoir/temp")

    model = LLM(model=args.model_path, max_model_len=4096)

    all_inputs = expand_input_paths(data_paths)
    shard_inputs = filter_paths_for_shard(
        all_inputs,
        node_index=node_index,
        node_num=node_num,
        chunk_index=chunk_index,
        chunk_num=chunk_num,
    )

    for parquet_path in shard_inputs:
        base_name = os.path.basename(parquet_path)
        if out_dir and os.path.exists(os.path.join(out_dir, base_name)):
            print(f"file {parquet_path} already merged, skipping")
            continue

        print(f"processing {parquet_path}")
        processed_data = []
        df = pd.read_parquet(parquet_path)
        batch_size = 4

        for offset in tqdm.trange(0, len(df), batch_size):
            batch = df.loc[offset : offset + batch_size - 1].reset_index(drop=True)
            all_image, all_simple_caption, all_detail_caption, valid_indices = process_batch(
                model, batch, args, image_cols=image_cols, path_cols=path_cols
            )
            for index, (simp_cap, detail_cap, validness) in enumerate(
                zip(all_simple_caption, all_detail_caption, valid_indices)
            ):
                sample = batch.loc[index].to_dict()
                sample["general_caption"] = simp_cap
                sample["detail_caption"] = detail_cap
                sample["validness"] = bool(validness and sample.get("validness", 1))
                processed_data.append(sample)

        processed_df = pd.DataFrame(processed_data).reset_index(drop=True)
        output_path = os.path.join("reservoir/processed_data", base_name)
        processed_df.to_parquet(output_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/captions.yaml")
    parser.add_argument('--chunk_index', type=int, default=0)
    parser.add_argument('--chunk_num', type=int, default=4)
    parser.add_argument('--node_index', type=int, default=0)
    parser.add_argument('--model_path', type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument('--node_num', type=int, default=10)
    parser.add_argument('--lvlm', type=str, default="llava")
    parser.add_argument('--llm', type=str, default="llama2")
    args = parser.parse_args()

    main(
        chunk_index=args.chunk_index, 
        chunk_num=args.chunk_num,
        node_index=args.node_index,
        node_num=args.node_num,
        args=args
    )
