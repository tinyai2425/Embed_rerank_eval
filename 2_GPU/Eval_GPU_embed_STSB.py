#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU STSB embedding eval.

Usage:
  python Eval_GPU_embed_STSB.py <input_json_path> <ip> <port> <model_id> <version_flag>

Writes under the case file's sibling GPU/ directory:
  GPU-Results-<version>.parquet / .jsonl
  GPU_eval_v<version>.txt
  GPU_eval_overall.txt
"""

import json
import os
import sys
from typing import Tuple

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Function"))

import embed_gpu_runner
import embed_metrics
import embed_report


def parse_args():
    if len(sys.argv) != 6:
        print("Usage: python Eval_GPU_embed_STSB.py <input_json_path> <ip> <port> <model_id> <version_flag>")
        sys.exit(1)
    return sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])


def parse_project_base_and_filename(input_json_path: str) -> Tuple[str, str]:
    abs_path = os.path.abspath(input_json_path)
    return os.path.dirname(abs_path), os.path.basename(abs_path)


def get_version_nums(version_flag: int):
    if version_flag < 1:
        raise ValueError("version_flag must be >= 1")
    return list(range(1, version_flag + 1))


if __name__ == "__main__":
    input_json_path, ip, port, model_id, version_flag = parse_args()
    project_base, file_name = parse_project_base_and_filename(input_json_path)
    version_nums = get_version_nums(version_flag)

    gpu_output_dir = os.path.join(project_base, "GPU")
    os.makedirs(gpu_output_dir, exist_ok=True)

    print("Notice: Proxy settings should be disabled before proceeding.")
    print(f"[INFO] Using server http://{ip}:{port}/v1  model={model_id}")

    tests_path = os.path.join(project_base, file_name)
    for version_num in version_nums:
        print(f"[INFO] Processing version {version_num}...")
        df_results = embed_gpu_runner.run_gpu_embeddings_from_tests(
            tests_json_path=tests_path,
            server_ip=ip,
            server_port=port,
            model_id=model_id,
            output_dir=gpu_output_dir,
        )

        parquet_path = os.path.join(gpu_output_dir, f"GPU-Results-{version_num}.parquet")
        jsonl_path = os.path.join(gpu_output_dir, f"GPU-Results-{version_num}.jsonl")
        df_results.to_parquet(parquet_path, index=False)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for _, row in df_results.iterrows():
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        print(f"[SAVE] {parquet_path}")
        print(f"[SAVE] {jsonl_path}")

        metrics = embed_metrics.evaluate_stsb_from_df(df_results, l2norm=True)
        embed_report.write_eval_txt(
            gpu_dir=gpu_output_dir,
            version_num=version_num,
            metrics=metrics,
            df=df_results,
            tests_json_path=tests_path,
            server_ip=ip,
            server_port=port,
            model_id=model_id,
        )
        print(f"[DONE] Evaluation for version {version_num}")
