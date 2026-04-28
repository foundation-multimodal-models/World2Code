#!/usr/bin/env bash
set -euo pipefail

# Stage 1 (local open-source version):
# 1) caption (general + detail)
# 2) boxes based on captions
# 3) OCR (VLM-based)
#
# Configure input/output via YAML in `configs/`.

node_index="${1:-0}"
node_num="${NODE_NUM:-1}"

if command -v nvidia-smi >/dev/null 2>&1; then
  chunk_num="${CHUNK_NUM:-$(nvidia-smi -L | wc -l | tr -d ' ')}"
else
  chunk_num="${CHUNK_NUM:-1}"
fi
if [ "${chunk_num}" -le 0 ]; then chunk_num=1; fi

model_path="${MODEL_PATH:-llava-hf/llava-v1.6-vicuna-7b-hf}"

mkdir -p reservoir/output reservoir/processed_data reservoir/temp

for (( chunk_index=0; chunk_index<chunk_num; chunk_index++ )); do
  CUDA_VISIBLE_DEVICES="${chunk_index}" \
    nohup python3 world2seq/captions.py \
      --config configs/captions.yaml \
      --model_path "${model_path}" \
      --chunk_index "${chunk_index}" --chunk_num "${chunk_num}" \
      --node_index "${node_index}" --node_num "${node_num}" \
      > "reservoir/output/caption_${chunk_index}_node_${node_index}.log" 2>&1 &
done
wait
python3 merge_results.py --config configs/captions.yaml

for (( chunk_index=0; chunk_index<chunk_num; chunk_index++ )); do
  CUDA_VISIBLE_DEVICES="${chunk_index}" \
    nohup python3 world2seq/boxes.py \
      --config configs/boxes.yaml \
      --chunk_index "${chunk_index}" --chunk_num "${chunk_num}" \
      --node_index "${node_index}" --node_num "${node_num}" \
      > "reservoir/output/bbox_${chunk_index}_node_${node_index}.log" 2>&1 &
done
wait
python3 merge_results.py --config configs/boxes.yaml

for (( chunk_index=0; chunk_index<chunk_num; chunk_index++ )); do
  CUDA_VISIBLE_DEVICES="${chunk_index}" \
    nohup python3 world2seq/ocr_vlm.py \
      --config configs/ocr.yaml \
      --model_path "${model_path}" \
      --chunk_index "${chunk_index}" --chunk_num "${chunk_num}" \
      --node_index "${node_index}" --node_num "${node_num}" \
      > "reservoir/output/ocr_${chunk_index}_node_${node_index}.log" 2>&1 &
done
wait
python3 merge_results.py --config configs/ocr.yaml
