import argparse
import io
import os
import pickle
import sys
from collections import defaultdict
from typing import Any, Dict

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

llava_ori_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n"

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
 

import spacy
spacy_nlp = spacy.load("en_core_web_sm")

from common import COCO_CLASS_NAMES, LVIS_CLASS_NAMES, ADE20K_CLASS_NAMES, OBJECTS365_CLASS_NAMES, OPEN_IMAGES_V4_BOXABLE_CLASS_NAMES, VISUAL_GENOME_CLASS_NAMES

total_class_names = COCO_CLASS_NAMES + LVIS_CLASS_NAMES + ADE20K_CLASS_NAMES + OBJECTS365_CLASS_NAMES + \
    OPEN_IMAGES_V4_BOXABLE_CLASS_NAMES + VISUAL_GENOME_CLASS_NAMES

def get_tag_phrases(cap, skip_phrase_list):
    def left_remove(noun_p):
        np_list = noun_p.split()
        if np_list[0].lower() in ['a', 'an', 'the', 'his', 'her', 'my', 'your', "their", "its"]:
            return " ".join(np_list[1:])
        else:
            return noun_p

    doc = spacy_nlp(cap)
    valid_tags = [left_remove(nouns.text.lower()) for nouns in doc.noun_chunks]

    valid_tags = [item.strip() for item in valid_tags if item != " " ]
    valid_tags = [item.lower() for item in valid_tags]
    skip_phrase_list = [item.lower().rstrip(":") for item in skip_phrase_list]
    valid_tags = list(set(valid_tags))

    for v in valid_tags:
        if v.lower() in skip_phrase_list:
            valid_tags.remove(v)
    if '' in valid_tags:
        valid_tags.remove('')

    # 分词完后，每个term必须有一个属于visual concept的东西~
    for v in valid_tags:
        all_words = v.split()
        is_concept = False
        for w in all_words:
            if w in total_class_names:
                is_concept = True
        if not is_concept:
            valid_tags.remove(v)

    return valid_tags


def process_batch(model, batch, args):
    # A bunch of yes or no questions. At the same time, given scorer for the generated cases.
    # This time, we use samples with no-ocr input, to ablate the gains brought by ocr-tools

    def clean_element(el):
        if not 'a' <= el[-1] <= 'z':
            el = el[:-1]
        return el

    all_image = []
    image_validness = []

    all_element = []
    all_element_description = []
    all_element_boxes = []
    
    all_area_ids = []
    all_relation_description = []

    all_global_cap = []
    
    for index, sample in batch.iterrows():
        try:
            if 'frame' in sample:
                image = Image.open(io.BytesIO(sample['image'])).convert("RGB")
            elif 'binary' in sample:
                image = Image.open(io.BytesIO(sample['binary'])).convert("RGB")
            else:
                image = Image.open(io.BytesIO(sample['frame'])).convert("RGB")
            # boxes = [[round(coordinate) for coordinate in coordinates] for coordinates in sample['SAM_merged_cluster_centers']]
            all_image.append(image)
            image_validness.append(1)
            element = pickle.loads(sample['dino_phrases'])
            element = [clean_element(item.split("(")[0]) for item in element]
            element_describe = pickle.loads(sample['describe_label'])

            element_boxes = pickle.loads(sample['dino_box'])

            all_element.append(element)
            all_element_description.append(element_describe)
            all_element_boxes.append(element_boxes)

            element_son_id = pickle.loads(sample['area_element_id_pair'])
            all_area_ids.append(element_son_id)

            area_caption = pickle.loads(sample['area_captions'])
            all_relation_description.append(area_caption)

            global_cap = sample['general_caption']
            all_global_cap.append(global_cap)

        except:
            tmp_image = Image.open("test_ocr.png").convert("RGB")
            all_image.append(tmp_image)
            image_validness.append(-1)

            element = pickle.loads(sample['dino_phrases'])
            element = [clean_element(item.split("(")[0]) for item in element]
            element_describe = pickle.loads(sample['describe_label'])

            element_boxes = pickle.loads(sample['dino_box'])

            all_element.append(element)
            all_element_description.append(element_describe)
            all_element_boxes.append(element_boxes)

            element_son_id = pickle.loads(sample['area_element_id_pair'])
            all_area_ids.append(element_son_id)

            area_caption = pickle.loads(sample['area_captions'])
            all_relation_description.append(area_caption)

            global_cap = sample['global_caption']
            all_global_cap.append(global_cap)
    

    def repack_element_checker(all_image, all_element, element_boxes):   
        batch_im = []
        batch_question = []
        batch_map_id = []

        item_id = 0
        for im, e_list, e_box in zip(all_image, all_element, element_boxes):
            for e, cur_box in zip(e_list, e_box):
                bounding_box = cur_box
                x, y = (bounding_box[0] + bounding_box[2]) / 2, (bounding_box[1] + bounding_box[3]) / 2
                w, h = (bounding_box[2] - bounding_box[0]), bounding_box[3] - bounding_box[1]
                w, h = max(w, 200), max(h, 200)
                new_box = [max(0, x - 0.65 * w), max(0, y - 0.65 * h), min(x + 0.65 * w, im.width), min(y + 0.65 * h, im.height)]

                try:
                    new_im = im.crop(new_box)
                except:
                    new_im = im

                llava_ori_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n"
                valid_question = f" Is '{e}' a valid and visible visual concept in the image? Answer yes or no with only one single word. ASSISTANT:"

                final_prompt = llava_ori_prompt + valid_question

                batch_im.append(new_im)
                batch_question.append(final_prompt)
                batch_map_id.append(item_id)
            item_id += 1
        return batch_im, batch_question, batch_map_id


    def repack_element_caption_checker(all_image, all_element, all_element_desc, element_boxes):
        batch_im = []
        batch_question = []
        batch_map_id = []
        batch_e_id = []
        batch_phrase = []
        from common import skip_phrase_list

        item_id = 0
        for im, e_list, e_cap, cur_box in zip(all_image, all_element, all_element_desc, element_boxes):
            cur_item_phrase = []
            for e_id, (e, sub_box) in enumerate(zip(e_cap, cur_box)):
                e_cap_phrases = []
                for item in e:
                    e_cap_phrases.extend(get_tag_phrases(item, skip_phrase_list)) 
                e_cap_phrases = list(set(e_cap_phrases))  
                
                bounding_box = sub_box
                x, y = (bounding_box[0] + bounding_box[2]) / 2, (bounding_box[1] + bounding_box[3]) / 2
                w, h = (bounding_box[2] - bounding_box[0]), bounding_box[3] - bounding_box[1]
                w, h = max(w, 200), max(h, 200)
                new_box = [max(0, x - 0.65 * w), max(0, y - 0.65 * h), min(x + 0.65 * w, im.width), min(y + 0.65 * h, im.height)]

                try:
                    new_im = im.crop(new_box)
                except:
                    new_im = im

                for sub_phrase in e_cap_phrases:
                    llava_ori_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n"
                    valid_question = f" Is '{sub_phrase}' a valid and visible visual concept in the image? Answer yes or no with only one single word. ASSISTANT:"

                    final_prompt = llava_ori_prompt + valid_question

                    batch_im.append(new_im)
                    batch_question.append(final_prompt)
                    batch_map_id.append(item_id)
                    batch_e_id.append(e_id)
                cur_item_phrase.append(e_cap_phrases)
            batch_phrase.append(cur_item_phrase)
            item_id += 1

        
        return batch_im, batch_question, batch_map_id, batch_e_id, batch_phrase

    def repack_group_checker(all_image, all_element):
        batch_im = []
        batch_question = []
        batch_map_id = []

        batch_groups = []

        item_id = 0
        for im, e_list in zip(all_image, all_element):
            occur = defaultdict(int)
            for e in e_list:
                occur[e] += 1
            
            for group_key in occur.keys():
                parse_times = occur[group_key]
                llava_ori_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n"
                valid_question = f" Is there {parse_times} or more {group_key} in the image? Answer yes or no with a single word. ASSISTANT:"

                final_prompt = llava_ori_prompt + valid_question

                batch_im.append(im)
                batch_question.append(final_prompt)
                batch_map_id.append(item_id)
            batch_groups.append(list(occur.keys()))
            item_id += 1
        return batch_im, batch_question, batch_map_id, batch_groups
    
    def repack_relation_checker(all_image, all_relation, element_boxes):
        batch_im = []
        batch_question = []
        batch_map_id = []
        batch_e_id = []
        batch_phrase = []
        from common import skip_phrase_list

        item_id = 0
        for im, rel_cap, cur_box in zip(all_image, all_relation, element_boxes):
            cur_item_phrase = []
            for e_id, (e, sub_box) in enumerate(zip(rel_cap, cur_box)):
                e_cap_phrases = []
                for item in e:
                    e_cap_phrases.extend(get_tag_phrases(item, skip_phrase_list)) 
                e_cap_phrases = list(set(e_cap_phrases))  
                
                bounding_box = sub_box
                x, y = (bounding_box[0] + bounding_box[2]) / 2, (bounding_box[1] + bounding_box[3]) / 2
                w, h = (bounding_box[2] - bounding_box[0]), bounding_box[3] - bounding_box[1]
                w, h = max(w, 200), max(h, 200)
                new_box = [max(0, x - 1.0 * w), max(0, y - 1.0 * h), min(x + 1.0 * w, im.width), min(y + 1.0 * h, im.height)]

                # try:
                #     new_im = im.crop(new_box)
                # except:
                #     new_im = im

                new_im = im

                for sub_phrase in e_cap_phrases:
                    llava_ori_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n"
                    valid_question = f" Is '{sub_phrase}' a valid and visible visual concept in the image? Answer yes or no with only one single word. ASSISTANT:"

                    final_prompt = llava_ori_prompt + valid_question

                    batch_im.append(new_im)
                    batch_question.append(final_prompt)
                    batch_map_id.append(item_id)
                    batch_e_id.append(e_id)
                cur_item_phrase.append(e_cap_phrases)
            batch_phrase.append(cur_item_phrase)
            item_id += 1
        
        return batch_im, batch_question, batch_map_id, batch_e_id, batch_phrase
    


    this_batch_size = 1
    # pad to 256
    origin_len = len(all_image)
    if origin_len == 0:
        return [], [], [], [], [], [], [], []

    add_len = this_batch_size - origin_len

    all_image = all_image + [all_image[-1]] * add_len
    image_validness = image_validness + [image_validness[-1]] * add_len

    all_element = all_element + [all_element[-1]] * add_len
    all_element_description = all_element_description + [all_element_description[-1]] * add_len
    
    all_relation_description = all_relation_description + [all_relation_description[-1]] * add_len

    all_element_validness = []
    all_element_desc_validness = []

    all_group_validness = []

    all_group_phrases = []
    all_caption_phrases = []

    all_relation_validness = []
    all_relation_phrases = []
    

    for offset in range(0, len(all_image), this_batch_size):
        batch_images = all_image[offset: offset + this_batch_size]
        
        batch_element = all_element[offset: offset + this_batch_size]
        batch_element_description = all_element_description[offset: offset + this_batch_size]

        batch_relation_description = all_relation_description[offset: offset + this_batch_size]

        packed_total_im, packed_ques, packed_map_id = repack_element_checker(batch_images, batch_element, all_element_boxes)
    
        mini_batch_size = 1
        # tokenizer_model_max_length = getattr(args, 'tokenizer_model_max_length', 2048)
        args.tokenizer_padding_side = "left"
        # args.tokenizer_model_max_length = 2048
        args.image_aspect_ratio = "pad"

        element_checker = []

        for start in tqdm.trange(0, len(packed_total_im), mini_batch_size):
            # from IPython import embed; embed()
            mini_packed_total_im = packed_total_im[start: min(start + mini_batch_size, len(packed_total_im))]
            mini_packed_describe_question = packed_ques[start: min(start + mini_batch_size, len(packed_total_im))]
            
            sampling_params = SamplingParams(temperature=0.8,
                                        top_p=0.95,
                                        max_tokens=256, n=1, best_of=8)

            batched_data = []
            for cur_im, cur_prompt in zip(mini_packed_total_im, mini_packed_describe_question):
                batched_data.append({
                    'prompt': cur_prompt,
                    "multi_modal_data": {
                        "image": cur_im
                    }
                })
            outputs = model.generate(batched_data, sampling_params=sampling_params)

            res = []
            for output in outputs:
                cur_res = []
                for o in output.outputs:
                    cur_res.append(o.text)
                res.append(cur_res)

            element_checker.extend(res)
        

        packed_cap_total_im, packed_cap_ques, caption_map_id, caption_element_id, caption_phrase = repack_element_caption_checker(batch_images, batch_element, batch_element_description, all_element_boxes)
        all_caption_phrases.extend(caption_phrase)

        element_caption_checker = []
        for start in tqdm.trange(0, len(packed_cap_total_im), mini_batch_size):
            # from IPython import embed; embed()
            mini_packed_total_im = packed_cap_total_im[start: min(start + mini_batch_size, len(packed_cap_total_im))]
            mini_packed_describe_question = packed_cap_ques[start: min(start + mini_batch_size, len(packed_cap_total_im))]
            sampling_params = SamplingParams(temperature=0.8,
                                        top_p=0.95,
                                        max_tokens=256, n=1, best_of=8)

            batched_data = []
            for cur_im, cur_prompt in zip(mini_packed_total_im, mini_packed_describe_question):
                batched_data.append({
                    'prompt': cur_prompt,
                    "multi_modal_data": {
                        "image": cur_im
                    }
                })
            outputs = model.generate(batched_data, sampling_params=sampling_params)

            res = []
            for output in outputs:
                cur_res = []
                for o in output.outputs:
                    cur_res.append(o.text)
                res.append(cur_res)
            
            element_caption_checker.extend(res)

        
        # ----------------- for relation checker -------------------
        packed_rel_total_im, packed_rel_ques, relation_map_id, relation_element_id, relation_phrase = repack_relation_checker(batch_images, batch_relation_description, all_element_boxes)
        all_relation_phrases.extend(relation_phrase)

        element_relation_checker = []
        for start in tqdm.trange(0, len(packed_rel_total_im), mini_batch_size):
            # from IPython import embed; embed()
            mini_packed_total_im = packed_rel_total_im[start: min(start + mini_batch_size, len(packed_rel_total_im))]
            mini_packed_describe_question = packed_rel_ques[start: min(start + mini_batch_size, len(packed_rel_total_im))]
            sampling_params = SamplingParams(temperature=0.8,
                                        top_p=0.95,
                                        max_tokens=256, n=1, best_of=8)

            batched_data = []
            for cur_im, cur_prompt in zip(mini_packed_total_im, mini_packed_describe_question):
                batched_data.append({
                    'prompt': cur_prompt,
                    "multi_modal_data": {
                        "image": cur_im
                    }
                })
            outputs = model.generate(batched_data, sampling_params=sampling_params)

            res = []
            for output in outputs:
                cur_res = []
                for o in output.outputs:
                    cur_res.append(o.text)
                res.append(cur_res)
            element_relation_checker.extend(res)
    

        packed_group_total_im, packed_group_ques, group_map_id, b_groups = repack_group_checker(batch_images, batch_element)
        all_group_phrases.extend(b_groups)

        group_checker = []
        for start in tqdm.trange(0, len(packed_group_total_im), mini_batch_size):
            # from IPython import embed; embed()
            mini_packed_total_im = packed_group_total_im[start: min(start + mini_batch_size, len(packed_group_total_im))]
            mini_packed_describe_question = packed_group_ques[start: min(start + mini_batch_size, len(packed_group_total_im))]
            sampling_params = SamplingParams(temperature=0.8,
                                        top_p=0.95,
                                        max_tokens=256, n=1, best_of=8)

            batched_data = []
            for cur_im, cur_prompt in zip(mini_packed_total_im, mini_packed_describe_question):
                batched_data.append({
                    'prompt': cur_prompt,
                    "multi_modal_data": {
                        "image": cur_im
                    }
                })
            outputs = model.generate(batched_data, sampling_params=sampling_params)

            res = []
            for output in outputs:
                cur_res = []
                for o in output.outputs:
                    cur_res.append(o.text)
                res.append(cur_res)
            group_checker.extend(res)

        # Rearrange these checker information to group infomations
        final_element_check = [[] for i in range(this_batch_size)]
        for v, map_id in zip(element_checker, packed_map_id):
            final_element_check[map_id].append(v)
        
        final_element_caption_check = [[] for i in range(this_batch_size)]
        for v, map_id, e_id in zip(element_caption_checker, caption_map_id, caption_element_id):
            while e_id >= len(final_element_caption_check[map_id]):
                final_element_caption_check[map_id].append([])
            final_element_caption_check[map_id][e_id].append(v)
    
        final_relation_check = [[] for i in range(this_batch_size)]
        for v, map_id, e_id in zip(element_relation_checker, relation_map_id, relation_element_id):
            while e_id >= len(final_relation_check[map_id]):
                final_relation_check[map_id].append([])
            final_relation_check[map_id][e_id].append(v)
        
        final_group_check = [[] for i in range(this_batch_size)]
        for v, map_id in zip(group_checker, group_map_id):
            final_group_check[map_id].append(v)

        all_element_validness.extend(final_element_check)
        all_element_desc_validness.extend(final_element_caption_check)
        all_group_validness.extend(final_group_check)
        all_relation_validness.extend(final_relation_check)

    return all_element_validness[:origin_len], all_element_desc_validness[:origin_len], all_group_validness[:origin_len], all_group_phrases[:origin_len], all_caption_phrases[:origin_len], all_relation_validness[:origin_len], all_relation_phrases[:origin_len], image_validness[:origin_len]


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
            "vLLM is required for validator stage. "
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
        batch_size = 1

        for offset in tqdm.trange(0, len(df), batch_size):
            batch = df.loc[offset : offset + batch_size - 1].reset_index(drop=True)

            (
                element_validness,
                element_desc_validness,
                group_validness,
                group_phrases,
                caption_phrase,
                relation_validness,
                relation_phrases,
                image_validness,
            ) = process_batch(model, batch, args)

            for index, (
                valid_e,
                valid_ed,
                valid_g,
                g_phrases,
                c_phrases,
                rel_val,
                rel_phrase,
                im_validness,
            ) in enumerate(
                zip(
                    element_validness,
                    element_desc_validness,
                    group_validness,
                    group_phrases,
                    caption_phrase,
                    relation_validness,
                    relation_phrases,
                    image_validness,
                )
            ):
                sample = batch.loc[index].to_dict()
                sample["element_validness"] = pickle.dumps(valid_e)
                sample["element_desc_validness"] = pickle.dumps(valid_ed)
                sample["group_validness"] = pickle.dumps(valid_g)
                sample["group_phrase"] = pickle.dumps(g_phrases)
                sample["caption_phrase"] = pickle.dumps(c_phrases)
                sample["relation_validness"] = pickle.dumps(rel_val)
                sample["relation_phrases"] = pickle.dumps(rel_phrase)
                sample["validness"] = bool(im_validness and sample.get("validness", 1))
                processed_data.append(sample)

        processed_df = pd.DataFrame(processed_data).reset_index(drop=True)
        output_path = os.path.join("reservoir/processed_data", base_name)
        processed_df.to_parquet(output_path)
    
    # for file in this_processed_files:
    #     os.system(f"hdfs dfs -put {file} {yml['hdfs_data_processed_path']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/validate.yaml")
    parser.add_argument('--chunk_index', type=int, default=0)
    parser.add_argument('--chunk_num', type=int, default=4)
    parser.add_argument('--node_index', type=int, default=0)
    parser.add_argument('--node_num', type=int, default=10)
    parser.add_argument('--model_path', type=str, default="llava-hf/llava-1.5-7b-hf")
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
