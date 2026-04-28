import argparse
import io
import math
import os
import pickle
import sys
from typing import Any, Dict, List

import pandas as pd
import torch
import tqdm
import yaml
from PIL import Image

# Ensure the repository root is importable when run as a script.
_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# GroundingDINO repo is expected to be available either as a sibling of `w2c/`
# (default for this repo layout) or provided via env var.
_GD_REPO = os.environ.get("GROUNDINGDINO_REPO") or os.path.abspath(os.path.join(_PROJECT_ROOT, "..", "GroundingDINO"))
if _GD_REPO not in sys.path:
    sys.path.insert(0, _GD_REPO)

import torchvision
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from common import ensure_dir, expand_input_paths, filter_paths_for_shard
from common import (
    ADE20K_CLASS_NAMES,
    COCO_CLASS_NAMES,
    LVIS_CLASS_NAMES,
    OBJECTS365_CLASS_NAMES,
    OPEN_IMAGES_V4_BOXABLE_CLASS_NAMES,
    VISUAL_GENOME_CLASS_NAMES,
)

total_class_names = COCO_CLASS_NAMES + LVIS_CLASS_NAMES + ADE20K_CLASS_NAMES + OBJECTS365_CLASS_NAMES + \
    OPEN_IMAGES_V4_BOXABLE_CLASS_NAMES + VISUAL_GENOME_CLASS_NAMES

box_thres = 0.23
text_thres = 0.2
iou_thres = 0.5


def get_grounding_output(model, raw_image, image, caption, box_threshold, text_threshold,device="cpu"):
    caption = [", ".join(list(set(item))[:40]) for item in caption]
    caption = [item.lower() for item in caption]
    caption = [item.strip() for item in caption]
    caption = [item + '.' if not item.endswith(".") else item for item in caption]
    # if not caption.endswith("."):
    #     caption = caption + "."

    model = model.to(device)

    image = torch.stack(image)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image, captions=caption)
    logits_batch = outputs["pred_logits"].cpu().sigmoid()  # (nq, 128)
    boxes_batch = outputs["pred_boxes"].cpu()  # (nq, 4)

    batch_boxes, batch_scores, batch_phrases = [], [], []

    for sample_id, (logits, boxes) in enumerate(zip(logits_batch, boxes_batch)):
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 128
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        image_pil = raw_image[sample_id]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption[sample_id])
        # build pred
        pred_phrases = []
        scores = []

        final_boxes = []
        for index_id, (logit, box) in enumerate(zip(logits_filt, boxes_filt)):
            try:
                pred_phrase_list, phrase_score = get_phrases_from_posmap(logit > text_threshold, logit, tokenized, tokenlizer)
            except:
                from IPython import embed; embed()

            pred_phrase_list = [item.strip() for item in pred_phrase_list]
            pred_phrase_list = [item.lstrip(", ") for item in pred_phrase_list]

            if len(phrase_score) == 0:
                continue

            assert len(pred_phrase_list) == len(phrase_score)
            
            # 可以多个phrase? 2-3个phrase
            final_phrase = ""
            best_s = 0.
            for p, p_s in zip(pred_phrase_list, phrase_score):
                if p_s > best_s:
                    final_phrase = p
                    best_s = p_s
            
            if best_s >= text_thres:
                pred_phrase = final_phrase
                final_boxes.append(box.unsqueeze(0))
                pred_phrases.append(pred_phrase + f"({best_s})")
                scores.append(logit.max().item())
        
        if len(final_boxes) > 0:
            final_boxes = torch.cat(final_boxes, dim=0)
            scores = torch.Tensor(scores)
            boxes_filt = final_boxes

            size = image_pil.size
            H, W = size[1], size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()

            nms_idx = torchvision.ops.nms(final_boxes, scores, iou_thres).numpy().tolist()
            boxes_filt = boxes_filt[nms_idx].numpy().tolist()
            scores = scores[nms_idx]
            pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        else:
            boxes_filt = []
            scores = []
            pred_phrases = []

        batch_boxes.append(boxes_filt)
        batch_scores.append(scores)
        batch_phrases.append(pred_phrases)

    return batch_boxes, batch_scores, batch_phrases


import spacy
spacy_nlp = spacy.load("en_core_web_sm")


import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# 确保下载了所需的数据包
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

from common import class_to_objects

def get_tag_phrases(o_cap, c_cap, skip_phrase_list):

    if isinstance(o_cap, bytes):
        o_cap = pickle.loads(o_cap)
    if isinstance(c_cap, bytes):
        c_cap = pickle.loads(c_cap)
    
    if isinstance(o_cap, np.ndarray):
        o_cap = list(o_cap)
    if isinstance(c_cap, np.ndarray):
        c_cap = list(c_cap)

    if isinstance(o_cap, list):
        o_cap = " ".join(o_cap)
    if isinstance(c_cap, list):
        c_cap = " ".join(c_cap)
    def left_remove(noun_p):
        np_list = noun_p.split()
        if np_list[0].lower() in ['a', 'an', 'the', 'his', 'her', 'my', 'your', "their", "its"]:
            return " ".join(np_list[1:])
        else:
            return noun_p

    words = word_tokenize(o_cap)
    tagged_words = pos_tag(words)

    o_noun_phrases = [left_remove(word.lower()) if tag.startswith('NN') else "_=_" for word, tag in tagged_words]
    o_noun_phrases = " ".join(o_noun_phrases).split("_=_")
    o_noun_phrases = [item.strip() for item in o_noun_phrases if item != " " ]

    words = word_tokenize(c_cap)
    tagged_words = pos_tag(words)
    c_noun_phrases = [left_remove(word.lower()) if tag.startswith('NN') else "_=_" for word, tag in tagged_words]
    c_noun_phrases = " ".join(c_noun_phrases).split("_=_")
    c_noun_phrases = [item.strip() for item in c_noun_phrases if item != " "]

    # # use spacy for words
    # doc = spacy_nlp(o_cap)
    # o_noun_phrases = [left_remove(nouns.text.lower()) for nouns in doc.noun_chunks]

    # doc = spacy_nlp(c_cap)
    # c_noun_phrases = [left_remove(nouns.text.lower()) for nouns in doc.noun_chunks]

    entity_o_1_gram = o_cap.split()
    entity_o_1_gram = [item.lower() for item in entity_o_1_gram]
    entity_o_1_gram = [item if (item[-1] <= 'z' and item[-1] >= 'a') else item[:-1] for item in entity_o_1_gram]

    entity_c_1_gram = c_cap.split()
    entity_c_1_gram = [item.lower() for item in entity_c_1_gram]
    entity_c_1_gram = [item if (item[-1] <= 'z' and item[-1] >= 'a') else item[:-1] for item in entity_c_1_gram]

    valid_tags = []

    valid_tags.extend(o_noun_phrases)
    valid_tags.extend(c_noun_phrases)
    valid_tags = [wnl.lemmatize(tag, "n") for tag in valid_tags]
    valid_tags = list(set(valid_tags))

    # remove contain tags
    for k in valid_tags:
        if k.lower() in class_to_objects.keys():
            for v in class_to_objects[k]:
                if v.lower() in valid_tags:
                    if k in valid_tags:
                        valid_tags.remove(k)

    # 如果phrase本身有包含关系，取最长的?
    valid_tags = [item.lower() for item in valid_tags]
    skip_phrase_list = [item.lower().rstrip(":") for item in skip_phrase_list]

    valid_tags = list(set(valid_tags))

    for v in valid_tags:
        if v.lower() in skip_phrase_list:
            valid_tags.remove(v)

    if "" in valid_tags:
        valid_tags.remove("")

    return valid_tags


def load_image(image_pil):
    # load image
    # image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.Resize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image


def process_batch(model, batch):
    from common import skip_phrase_list
    def get_area(coordinates):
        return (coordinates[3] - coordinates[1]) * (coordinates[2] - coordinates[0])

    # need images and corresponding captions
    def get_cropped_images_and_indices(batch):
        all_image = []
        valid_indices = []
        all_valid_tags = []

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
                simple_caption = sample['general_caption']
                detail_caption = sample['detail_caption']

                selected_phrases = get_tag_phrases(simple_caption, detail_caption, skip_phrase_list)
                all_valid_tags.append(selected_phrases)
            except:
                tmp_image = Image.open("test_ocr.png").convert("RGB")
                all_image.append(tmp_image)
                valid_indices.append(-1)
                simple_caption = sample['general_caption']
                detail_caption = sample['detail_caption']

                selected_phrases = get_tag_phrases(simple_caption, detail_caption, skip_phrase_list)
                all_valid_tags.append(selected_phrases)
 
        return all_image, valid_indices, all_valid_tags

    
    all_image, valid_indices, all_valid_tags = get_cropped_images_and_indices(batch)

    # from IPython import embed; embed()
    this_batch_size = 32
    # pad to 128
    origin_len = len(all_image)

    if origin_len == 0:
        return [], [], [], [], []
    
    all_image = all_image + [all_image[-1]] * (128 - len(all_image))
    valid_indices = valid_indices + [valid_indices[-1]] * (128 - len(valid_indices))
    all_valid_tags = all_valid_tags + [all_valid_tags[-1]] * (128 - len(all_valid_tags))

    all_boxes, all_scores, all_phrases = [], [], []
    
    for offset in range(0, len(all_image), this_batch_size):
        batch_images = all_image[offset: offset + this_batch_size]
        raw_images = batch_images.copy()
        batch_images = [load_image(item) for item in batch_images]

        batch_tags = all_valid_tags[offset: offset + this_batch_size]

        boxes_filt, scores, pred_phrases = get_grounding_output(model, raw_images, batch_images, batch_tags, box_thres, text_thres, device="cuda")    
        all_boxes.extend(boxes_filt)
        all_scores.extend(scores)
        all_phrases.extend(pred_phrases)

    all_image = all_image[:origin_len]
    valid_indices = valid_indices[:origin_len]
    all_boxes = all_boxes[:origin_len]
    all_scores = all_scores[:origin_len]
    all_phrases = all_phrases[:origin_len]

    return all_image, valid_indices, all_boxes, all_scores, all_phrases


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
    cfg = _load_config(args.config)
    data_paths = cfg.get("data_paths") or cfg.get("hdfs_data_paths") or []
    out_dir = cfg.get("output_dir") or ""

    model_config_path = cfg.get("groundingdino_config") or "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    model_checkpoint_path = cfg.get("groundingdino_ckpt") or "./checkpoints/groundingdino_swinb_cogcoor.pth"

    def load_model(model_config_path: str, model_checkpoint_path: str, device: str):
        gd_args = SLConfig.fromfile(model_config_path)
        gd_args.device = device
        model = build_model(gd_args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model

    ground_model = load_model(model_config_path, model_checkpoint_path, device="cuda")

    ensure_dir("reservoir/processed_data")
    ensure_dir("reservoir/temp")

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
        batch_size = 32

        for offset in tqdm.trange(0, len(df), batch_size):
            batch = df.loc[offset : offset + batch_size - 1].reset_index(drop=True)
            all_image, valid_indices, all_boxes, all_scores, all_phrases = process_batch(ground_model, batch)
            for index, (valid_ind, boxes, scores, phrases) in enumerate(
                zip(valid_indices, all_boxes, all_scores, all_phrases)
            ):
                sample = batch.loc[index].to_dict()
                sample["dino_box"] = pickle.dumps(boxes)
                sample["dino_scores"] = pickle.dumps(scores)
                sample["dino_phrases"] = pickle.dumps(phrases)
                sample["validness"] = bool(valid_ind and sample.get("validness", 1))
                processed_data.append(sample)

        processed_df = pd.DataFrame(processed_data).reset_index(drop=True)
        output_path = os.path.join("reservoir/processed_data", base_name)
        processed_df.to_parquet(output_path)
    
    # for file in this_processed_files:
    #     os.system(f"hdfs dfs -put {file} {yml['hdfs_data_processed_path']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/boxes.yaml")
    parser.add_argument('--chunk_index', type=int, default=0)
    parser.add_argument('--chunk_num', type=int, default=4)
    parser.add_argument('--node_index', type=int, default=0)
    parser.add_argument('--node_num', type=int, default=10)
    args = parser.parse_args()

    main(
        chunk_index=args.chunk_index, 
        chunk_num=args.chunk_num,
        node_index=args.node_index,
        node_num=args.node_num,
        args=args
    )
