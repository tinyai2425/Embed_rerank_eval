#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API rerank eval from a 3-line /v1/reranks log.

Usage:
  python Eval_API_rerank.py <api_result_path> [k_values]

Writes under the log file's sibling API/ directory:
  API-Results.jsonl / API-Results.parquet
  evaluation_results.md
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "Function"))
sys.path.append(os.path.join(SCRIPT_DIR, "..", "2_GPU", "Function"))

from rerank_api_parser import parse_rerank_api_results
from rerank_eval import (
    build_qrels_and_run_from_scored_items,
    evaluate_and_write_markdown,
    parse_k_values,
)


def main():
    if len(sys.argv) not in (2, 3):
        print("Usage: python Eval_API_rerank.py <api_result_path> [k_values]")
        sys.exit(1)

    api_result_path = os.path.abspath(sys.argv[1])
    k_values = parse_k_values(sys.argv[2] if len(sys.argv) == 3 else "[5,10]")
    if not os.path.isfile(api_result_path):
        sys.exit(f"[ERR] API result file not found: {api_result_path}")

    project_base = os.path.dirname(api_result_path)
    api_output_dir = os.path.join(project_base, "API")
    os.makedirs(api_output_dir, exist_ok=True)

    print(f"Using k values: {k_values}")
    try:
        df = parse_rerank_api_results(api_result_path)
    except Exception as e:
        print(f"[ERROR] Failed to parse API results: {e}")
        sys.exit(1)
    print(f"Loaded {len(df)} scored pairs")

    jsonl_path = os.path.join(api_output_dir, "API-Results.jsonl")
    parquet_path = os.path.join(api_output_dir, "API-Results.parquet")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"[SAVE] {parquet_path}")
    except Exception as e:
        print(f"[WARN] Failed to save Parquet: {e}")
    print(f"[SAVE] {jsonl_path}")

    items = df.to_dict(orient="records")
    qrels, run = build_qrels_and_run_from_scored_items(items)
    print(f"Built qrels with {len(qrels)} queries")
    print(f"Built run with {len(run)} queries")
    evaluate_and_write_markdown(qrels, run, k_values, api_output_dir)
    print("[DONE] API rerank evaluation complete.")


if __name__ == "__main__":
    main()
