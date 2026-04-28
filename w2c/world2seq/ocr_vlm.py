import argparse
import io
import os
import pickle
import sys
from collections import defaultdict
from typing import Any, Dict, List

import pandas as pd
import tqdm
import yaml
from PIL import Image

from common import ensure_dir, expand_input_paths, filter_paths_for_shard

# Ensure the repository root is importable when run as a script.
_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common import group_boxes_by_tag

try:
    from vllm import LLM, SamplingParams
except Exception as e:  # pragma: no cover
    LLM = None
    SamplingParams = None
    _VLLM_IMPORT_ERROR = e

llava_ori_prompt = 'A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human\'s questions. USER: <image>\n'

def shape_filterer(shape_attribute):
    # return only valid color. If No Color, Return None
    valid_shapes = [
        "round", "square", "rectangular", "circular", "oval", "triangular", 
        "hexagonal", "cylindrical", "spherical", "cubical", "conical", "flat", 
        "curved", "angular", "geometric", "asymmetrical", "symmetrical", 
        "elliptical", "tubular", "polygonal"
    ]

    id, q, sp = shape_attribute

    sp = sp.replace(".", " ").replace("_", " ").replace(',', " ")
    sp = sp.lower().split()
    
    found_shape = []
    for sub_shape in valid_shapes:
        if sub_shape in sp:
            found_shape.append(sub_shape)

    if len(found_shape) != 0:
        new_sp = " and ".join(found_shape)
    else:
        new_sp = None

    return [id, q, new_sp]


def color_filterer(color_attribute):
    # color_set1 = [
    #     "bright", "vibrant", "dull", "pastel", "dark", "light", 
    #     "vivid", "pale", "neon", "muted", "rich", "deep", 
    #     "faded", "saturated", "bold", "subtle", "soft", 
    #     "intense", "warm", "cool", "metallic", "glossy", 
    #     "matte", "shiny", "fluorescent"
    # ]

    color_set = [
        "red", "blue", "green", "yellow", "orange", "purple", 
        "pink", "black", "white", "gray", "brown", "beige", 
        "maroon", "navy", "teal", "turquoise", "silver", 
        "gold", "bronze", "ivory", "charcoal", "indigo", 
        "violet", "cyan", "magenta", "lime", "olive", "coral", 
        "aubergine", "mustard", "jade", "fuchsia", "burgundy", 
        "lavender", "peach", "salmon", "amber", "emerald", "sapphire", 
        "ruby", "cerulean", "crimson", "scarlet", "mauve", "periwinkle", 
        "chocolate", "tangerine", "raspberry", "mint", "rose", "taupe"
    ]

    id, q, ans = color_attribute

    ans = ans.replace(".", " ").replace("_", " ").replace(',', " ")
    ans = ans.lower().split()

    found_color = []
    for sub_color in color_set:
        if sub_color in ans:
            found_color.append(sub_color)

    if len(found_color) == 0:
        new_sp = None
    else:
        new_sp = " and ".join(found_color)

    return [id, q, new_sp]
 


def grouping(tag_list, box_list):
    
    # 如果一些tags有一样的名字，且数量很多，我们就根据他们的boxes联通性做grouping
    
    tag_list = [item.split('(')[0] for item in tag_list]
    grouped_boxes = group_boxes_by_tag(box_list, tag_list)
    
    grouped_names = []
    grouped_box_ids = []
    group_area = []

    # 存3个以上，算是grouping
    for tag, conjunct_list in grouped_boxes.items():
        for conj in conjunct_list:
            if len(conj) >= 3:
                grouped_names.append(tag)
                grouped_box_ids.append(conj)
                a, b, c, d = 10000, 10000, 0, 0
                for box_id in conj:
                    box = box_list[box_id]
                    a = min(a, box[0])
                    b = min(b, box[1])
                    c = max(c, box[2])
                    d = max(d, box[3])
                group_area.append([a, b, c, d])

    return grouped_names, grouped_box_ids, group_area


def merge_two_box(box1, box2):
    a = min(box1[0], box2[0])
    b = min(box1[1], box2[1])
    c = max(box1[2], box2[2])
    d = max(box1[3], box2[3])

    return [a, b, c, d]



def ocr_reformat_box(boxes, input_text):
    # result = eval(input_item)
    # boxes = result["boxes"]
    # scores = result["scores"]

    scores = [1. for i in range(len(boxes))]

    def reform(box):
        min_x = min([a[0] for a in box])
        max_x = max([a[0] for a in box])
        min_y = min([a[1] for a in box])
        max_y = max([a[1] for a in box])

        return [min_x, min_y, max_x, max_y]

    new_boxes = [reform(box) for box,score in zip(boxes, scores) if score > 0.5]
    new_text = [text for text,score in zip(input_text, scores) if score > 0.5]

    return new_boxes, new_text


def find_smallest_covering_box(box_a, box_b):
    # from IPython import embed; embed()
    def is_covering(box, target):
        return box[0] <= target[0] and box[1] <= target[1] and box[2] >= target[2] and box[3] >= target[3]

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    results = []

    for a in box_a:
        covering_boxes = [(i, b) for i, b in enumerate(box_b) if is_covering(b, a)]
        if not covering_boxes:
            results.append(-1)
        else:
            smallest_box = min(covering_boxes, key=lambda x: box_area(x[1]))
            results.append(smallest_box[0])

    return results


def load_image(image_pil):
    transform = T.Compose(
        [
            # T.Resize([640], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image


def check_valid_relation(box_list, id1, id2, w, h):
    def get_area(box):
        a, b, c, d = box
        area = (c-a) * (d-b)
        return area
    def overlap(box1, box2):
        minx1, miny1, maxx1, maxy1 = box1
        minx2, miny2, maxx2, maxy2 = box2
        minx = max(minx1, minx2)
        miny = max(miny1, miny2)
        maxx = min(maxx1, maxx2)
        maxy = min(maxy1, maxy2)
        if minx > maxx or miny > maxy:
            return False
        else:
            return True
    box1, box2 = box_list[id1], box_list[id2]
    
    box1_area = get_area(box1)
    box2_area = get_area(box2)

    if box1_area / (w*h) > 0.05 and box2_area / (w*h) > 0.05:
        return True

    if box1_area / (w*h) < 0.05 and box2_area / (w*h) < 0.05:
        return False
    
    else:
        overlap_box = overlap(box1, box2)
        if overlap_box:
            return True
        else:
            return False


def process_batch(model, batch, args):
    def get_area(coordinates):
        return (coordinates[3] - coordinates[1]) * (coordinates[2] - coordinates[0])

    # need images and corresponding captions

    # I need to get boxes, tags and etc.
    def get_cropped_images_and_indices(batch):
        all_image = []
        valid_indices = []
        
        all_element_boxes = []
        all_element = []
        all_ocr_boxes = []
        all_ocr_text = []

        def clean_element(el):
            if not 'a' <= el[-1] <= 'z':
                el = el[:-1]
            return el

        for index, sample in batch.iterrows():
            try:
                if 'frame' in sample:
                    image = Image.open(io.BytesIO(sample['frame'])).convert("RGB")
                elif 'binary' in sample:
                    image = Image.open(io.BytesIO(sample['binary'])).convert("RGB")
                else:
                    image = Image.open(io.BytesIO(sample['image'])).convert("RGB")
                # boxes = [[round(coordinate) for coordinate in coordinates] for coordinates in sample['SAM_merged_cluster_centers']]
                all_image.append(image)
                valid_indices.append(1)
                element = pickle.loads(sample['dino_phrases'])
                element = [clean_element(item.split("(")[0]) for item in element]
                boxes = pickle.loads(sample['dino_box'])
                ocr_text = [] # sample['ocr_texts_text']
                ocr_box = [] # eval(sample['ocr_info'])

                all_element.append(element)
                all_element_boxes.append(boxes)
                all_ocr_boxes.append(ocr_box)
                all_ocr_text.append(ocr_text)

            except:
                tmp_image = Image.open("test_ocr.png").convert("RGB")
                all_image.append(tmp_image)
                valid_indices.append(-1)

                element = pickle.loads(sample['dino_phrases'])
                element = [item.split("(")[0] for item in element]
                boxes = pickle.loads(sample['dino_box'])
                ocr_text = [] # sample['ocr_texts_text']
                ocr_box = [] # eval(sample['ocr_info'])

                all_element.append(element)
                all_element_boxes.append(boxes)
                all_ocr_boxes.append(ocr_box)
                all_ocr_text.append(ocr_text)

 
        return all_image, valid_indices, all_element, all_element_boxes, all_ocr_boxes, all_ocr_text

    
    all_image, valid_indices, all_element, all_element_boxes, all_ocr_boxes, all_ocr_text = get_cropped_images_and_indices(batch)

    # from IPython import embed; embed()
    this_batch_size = 4
    # pad to 256
    origin_len = len(all_image)
    if origin_len == 0:
        return [], [], [], []

    add_len = this_batch_size - origin_len
    
    all_image = all_image + [all_image[-1]] * add_len
    valid_indices = valid_indices + [valid_indices[-1]] * add_len
    
    all_element = all_element + [all_element[-1]] * add_len
    all_element_boxes = all_element_boxes + [all_element_boxes[-1]] * add_len
    all_ocr_boxes = all_ocr_boxes + [all_ocr_boxes[-1]] * add_len
    all_ocr_text = all_ocr_text + [all_ocr_text[-1]] * add_len

    all_groups = []
    all_group_son_ids = []
    all_group_boxes = []

    all_element_father_ids = []
    for e_list, e_box_list in zip(all_element, all_element_boxes):
        cur_group_names, cur_group_box_ids, cur_group_area = grouping(e_list, e_box_list)

        all_groups.append(cur_group_names)
        all_group_son_ids.append(cur_group_box_ids)
        all_group_boxes.append(cur_group_area)

        element_father = [-1 for i in range(len(e_list))]
        for g_id, g_box_ids in enumerate(cur_group_box_ids):
            for box_id in g_box_ids:
                element_father[box_id] = g_id
        all_element_father_ids.append(element_father)

    # repack all images, all questions, and corresponding indices
    def repack_attribute(total_image, total_element, total_element_boxes):
        packed_total_im = []
        packed_total_element = []
        packed_total_e_boxes = []
        packed_father_indices = []
        packed_describe_question = []

        for image_id, (im, e_list, e_box_list) in enumerate(zip(total_image, total_element, total_element_boxes)):
            for e, e_box in zip(e_list, e_box_list):
                # Special Crop Image!
                bounding_box = e_box
                x, y = (bounding_box[0] + bounding_box[2]) / 2, (bounding_box[1] + bounding_box[3]) / 2
                w, h = (bounding_box[2] - bounding_box[0]), bounding_box[3] - bounding_box[1]
                new_box = [max(0, x - 0.5 * w), max(0, y - 0.5 * h), min(x + 0.5 * w, im.width), min(y + 0.5 * h, im.height)]

                try:
                    new_im = im.crop(new_box)
                except:
                    new_im = im

                packed_total_im.append(new_im)
                packed_total_element.append(e)
                packed_total_e_boxes.append(e_box)
                packed_father_indices.append(image_id)
                
                llava_ori_prompt = 'A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human\'s questions. USER: <image>\n'
                # describe_question = f" From the image, provide only one sentence that describes {e} (you should try your best to include attributes like shape, color or material), especially, using {e} as the begining of your answer. ASSISTANT:"
                describe_question = f" List all the ocr text in. ASSISTANT:"
                final_describe_prompt = llava_ori_prompt + describe_question

                packed_describe_question.append(final_describe_prompt)


        return packed_total_im, packed_total_element, packed_total_e_boxes, packed_describe_question, packed_father_indices


    for offset in range(0, len(all_image), this_batch_size):
        batch_images = all_image[offset: offset + this_batch_size]
        raw_images = batch_images.copy()
        # batch_images = [load_image(item) for item in batch_images]

        batch_element = all_element[offset: offset + this_batch_size]
        batch_element_box = all_element_boxes[offset: offset + this_batch_size]
        batch_element_fathers = all_element_father_ids[offset: offset + this_batch_size]

        # from IPython import embed; embed()
        
        packed_total_im, packed_total_element, packed_total_e_boxes, \
            packed_describe_question, packed_father_indices = repack_attribute(batch_images, batch_element, batch_element_box)

        mini_batch_size = 4
        # tokenizer_model_max_length = getattr(args, 'tokenizer_model_max_length', 2048)
        args.tokenizer_padding_side = "left"
        # args.tokenizer_model_max_length = 2048
        args.image_aspect_ratio = "pad"

        describe_res = []

        ocr_prompt = " List all the ocr text in the image, answer with only the ocr text  or 'no' if there isn't any ocr text."

        for start in range(0, len(packed_total_im), mini_batch_size):
            # from IPython import embed; embed()
            mini_packed_total_im = packed_total_im[start: min(start + mini_batch_size, len(packed_total_im))]
            mini_packed_describe_question = packed_describe_question[start: min(start + mini_batch_size, len(packed_total_im))]
            # need image_sizes

            prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{ocr_prompt} </s> ASSISTANT: "
            sampling_params = SamplingParams(temperature=0.8,
                                        top_p=0.95,
                                        max_tokens=256, n=1, best_of=4)

            batched_data = []
            for im in mini_packed_total_im:
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
            
            describe_res.extend(res)

        # process_relation_batch_packed_element
        final_describe_res = [[] for i in range(this_batch_size)]
        
        for describe_sub, father_id in zip(describe_res, packed_father_indices):
            final_describe_res[father_id].append(describe_sub)

    return all_image[:origin_len], final_describe_res[:origin_len], valid_indices[:origin_len]


def prepare_model_for_inference(model, dtype):
    model.cuda()
    model.eval()
    if dtype is not None:
        model.to(dtype)


def visualize(image, boxes, semantic_tags, out_file='reservoir/temp.jpg', linewidth=6):
    import matplotlib.pyplot as plt

    image = np.array(image) 

    image_h, image_w = image.shape[:2]
    fig, ax = plt.subplots(figsize=(image_w/100, image_h/100), dpi=100)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(image)

    draw_label_setting = {'facecolor': 'black', 'alpha': 0.8, 'pad': 0.7, 'edgecolor': 'none'}
    color = np.random.rand(len(semantic_tags), 3)
    n_terms = 10
    delta_y = math.floor(image_h/25)
    for i, (box, label) in enumerate(zip(boxes, semantic_tags)):
        box = [round(i, 2) for i in box]
        ax.add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], edgecolor=color[i], facecolor=(0,0,0,0), lw=linewidth))
        label = label.strip()
        label_ = label.split()
        if len(label_) < n_terms:
            ax.text(box[0], box[1], label, fontsize=6, bbox=draw_label_setting, verticalalignment='top', color="gray")
        else:
            n_labels = (len(label_)-1)//n_terms+1
            for label_idx in range(n_labels):
                start, end = label_idx * n_terms, min((n_terms * (label_idx+1), len(label_)))
                this_label = ' '.join(label_[start:end])
                this_y = box[1] + delta_y * label_idx
                ax.text(box[0], this_y, this_label, fontsize=6, bbox=draw_label_setting, verticalalignment='top', color="gray")

    plt.savefig(out_file, format='jpeg')
    plt.close()



def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main(chunk_index, chunk_num, node_index, node_num, args):
    if LLM is None:  # pragma: no cover
        raise RuntimeError(
            "vLLM is required for VLM-OCR stage. "
            f"Import error: {_VLLM_IMPORT_ERROR}"
        )

    cfg = _load_config(args.config)
    data_paths = cfg.get("data_paths") or cfg.get("hdfs_data_paths") or []
    out_dir = cfg.get("output_dir") or ""

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
            all_image, all_ocr_information, valid_indices = process_batch(model, batch, args)
            for index, (valid_ind, ocr_info) in enumerate(zip(valid_indices, all_ocr_information)):
                sample = batch.loc[index].to_dict()
                sample["validness"] = bool(valid_ind and sample.get("validness", 1))
                sample["ocr_info_vlm_pred"] = ocr_info
                processed_data.append(sample)

        processed_df = pd.DataFrame(processed_data).reset_index(drop=True)
        output_path = os.path.join("reservoir/processed_data", base_name)
        processed_df.to_parquet(output_path)
    
    # for file in this_processed_files:
    #     os.system(f"hdfs dfs -put {file} {yml['hdfs_data_processed_path']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/ocr.yaml")
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
