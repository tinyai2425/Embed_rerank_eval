#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU wiki embedding inference (raw results only, no STSB metrics).

Usage:
  python Eval_GPU_embed_wiki.py <input_json_path> <ip> <port> <model_id>

Writes under the case file's sibling GPU/ directory:
  GPU-Results-1.parquet
  GPU-Results-1.jsonl
"""

import json
import os
import sys
from typing import Tuple

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Function"))

import embed_gpu_runner

VERSION_NUM = 1


def parse_args():
    if len(sys.argv) != 5:
        print("Usage: python Eval_GPU_embed_wiki.py <input_json_path> <ip> <port> <model_id>")
        sys.exit(1)
    input_json_path = sys.argv[1]
    ip = sys.argv[2]
    try:
        port = int(sys.argv[3])
    except ValueError:
        print("[ERR] <port> must be an integer")
        sys.exit(1)
    model_id = sys.argv[4]
    return input_json_path, ip, port, model_id


def parse_project_base_and_filename(input_json_path: str) -> Tuple[str, str]:
    abs_path = os.path.abspath(input_json_path)
    return os.path.dirname(abs_path), os.path.basename(abs_path)


if __name__ == "__main__":
    input_json_path, ip, port, model_id = parse_args()
    project_base, file_name = parse_project_base_and_filename(input_json_path)

    gpu_output_dir = os.path.join(project_base, "GPU")
    os.makedirs(gpu_output_dir, exist_ok=True)

    print("Notice: Proxy settings should be disabled before proceeding.")
    print(f"[INFO] Using server http://{ip}:{port}/v1  model={model_id}")
    print(f"[INFO] Processing version {VERSION_NUM}...")

    df_results = embed_gpu_runner.run_gpu_embeddings_from_tests(
        tests_json_path=os.path.join(project_base, file_name),
        server_ip=ip,
        server_port=port,
        model_id=model_id,
        output_dir=gpu_output_dir,
    )

    parquet_path = os.path.join(gpu_output_dir, f"GPU-Results-{VERSION_NUM}.parquet")
    jsonl_path = os.path.join(gpu_output_dir, f"GPU-Results-{VERSION_NUM}.jsonl")

    try:
        df_results.to_parquet(parquet_path, index=False)
        print(f"[SAVE] {parquet_path}")
    except Exception as e:
        print(f"[WARN] Failed to save Parquet: {e}\n       Install pyarrow or fastparquet")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df_results.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    print(f"[SAVE] {jsonl_path}")
    print(f"[DONE] Version {VERSION_NUM} finished (parquet & jsonl only).")
