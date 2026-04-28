#!/usr/bin/env bash
set -euo pipefail

# Stage 3 (local open-source version): validation / filtering.

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
    nohup python3 world2seq/validate.py \
      --config configs/validate.yaml \
      --model_path "${model_path}" \
      --chunk_index "${chunk_index}" --chunk_num "${chunk_num}" \
      --node_index "${node_index}" --node_num "${node_num}" \
      > "reservoir/output/validate_${chunk_index}_node_${node_index}.log" 2>&1 &
done
wait
python3 merge_results.py --config configs/validate.yaml
