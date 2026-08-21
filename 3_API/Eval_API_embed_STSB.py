#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API STSB embedding eval.

Usage:
  python Eval_API_embed_STSB.py <api_result_path>

Writes under the log file's sibling API/ directory:
  API-Results.parquet / API-Results.jsonl
  API_eval.txt
  API_eval_overall.txt
"""

import json
import os
import sys
from typing import Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "Function"))
sys.path.append(os.path.join(SCRIPT_DIR, "..", "2_GPU", "Function"))

import embed_api_parser
import embed_metrics
import embed_report_api


def parse_args():
    if len(sys.argv) != 2:
        print("Usage: python Eval_API_embed_STSB.py <api_result_path>")
        sys.exit(1)
    return sys.argv[1]


def parse_project_base_and_filename(path: str) -> Tuple[str, str]:
    abs_path = os.path.abspath(path)
    return os.path.dirname(abs_path), os.path.basename(abs_path)


if __name__ == "__main__":
    api_result_path = parse_args()
    project_base, file_name = parse_project_base_and_filename(api_result_path)

    api_output_dir = os.path.join(project_base, "API")
    os.makedirs(api_output_dir, exist_ok=True)

    try:
        df_api = embed_api_parser.parse_embedding_api_results(api_result_path)
    except Exception as e:
        print(f"[ERROR] Failed to parse API results: {e}")
        sys.exit(1)

    parquet_path = os.path.join(api_output_dir, "API-Results.parquet")
    jsonl_path = os.path.join(api_output_dir, "API-Results.jsonl")
    df_api.to_parquet(parquet_path, index=False)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df_api.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    print(f"[SAVE] {parquet_path}")
    print(f"[SAVE] {jsonl_path}")

    metrics = embed_metrics.evaluate_stsb_from_df(df_api, l2norm=True)
    embed_report_api.write_api_eval_txt(
        api_dir=api_output_dir,
        metrics=metrics,
        df=df_api,
        tests_log_path=os.path.join(project_base, file_name),
    )
    print("[DONE] API embedding evaluation complete.")
