#!/usr/bin/env bash
set -euo pipefail

# Open-source friendly entrypoint:
# - Pure local filesystem IO (no HDFS)
# - Assumes dependencies are already installed in the current Python env

node_index="${1:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

bash "${SCRIPT_DIR}/stage1.sh" "${node_index}"
bash "${SCRIPT_DIR}/stage2.sh" "${node_index}"
bash "${SCRIPT_DIR}/stage3.sh" "${node_index}"
