#!/usr/bin/env python3
import argparse
import os
import shutil
from typing import Any, Dict

import yaml


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_results(config_path: str, *, processed_dir: str = "reservoir/processed_data") -> None:
    cfg = _load_yaml(config_path)
    out_dir = cfg.get("output_dir")
    if not out_dir:
        raise ValueError(f"`output_dir` is required in config: {config_path}")

    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isdir(processed_dir):
        return

    for name in sorted(os.listdir(processed_dir)):
        if not name.endswith(".parquet"):
            continue
        src = os.path.join(processed_dir, name)
        dst = os.path.join(out_dir, name)
        # Move when possible to mimic the old "upload then remove" behavior.
        shutil.move(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config containing `output_dir`")
    ap.add_argument("--processed_dir", default="reservoir/processed_data")
    args = ap.parse_args()
    merge_results(args.config, processed_dir=args.processed_dir)


if __name__ == "__main__":
    main()

