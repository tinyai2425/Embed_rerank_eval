#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare GPU wiki embeddings vs API wiki embeddings.

Usage:
  python Eval_GPU_API_embed_wiki.py <gpu_jsonl_path> <api_txt_path>

Creates a GPU-API directory (next to the GPU result dir, or reuses it)
and writes compare details + excel there.
"""

import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Function"))

from embed_gpu_api_compare import (
    compare_and_collect,
    make_by_vertical,
    parse_api_txt,
    parse_gpu_jsonl,
    resolve_gpu_api_outdir,
)


def main():
    if len(sys.argv) != 3:
        print("Usage: python Eval_GPU_API_embed_wiki.py <gpu_jsonl_path> <api_txt_path>")
        sys.exit(1)

    gpu_path = os.path.abspath(sys.argv[1])
    api_path = os.path.abspath(sys.argv[2])
    if not os.path.isfile(gpu_path):
        sys.exit(f"[ERR] GPU file not found: {gpu_path}")
    if not os.path.isfile(api_path):
        sys.exit(f"[ERR] API file not found: {api_path}")

    out_dir = resolve_gpu_api_outdir(gpu_path, api_path)
    os.makedirs(out_dir, exist_ok=True)
    gpu_base = os.path.splitext(os.path.basename(gpu_path))[0]
    details_parquet = os.path.join(out_dir, f"{gpu_base}-compare-details.parquet")
    excel_path = os.path.join(out_dir, f"{gpu_base}-compare.xlsx")

    print(f"[READ] GPU: {gpu_path}")
    gpu_df = parse_gpu_jsonl(gpu_path)
    print(f"[READ] API: {api_path}")
    api_df = parse_api_txt(api_path)

    print("[CALC] cosine similarity ...")
    details = compare_and_collect(gpu_df, api_df)

    print(f"[SAVE] details parquet -> {details_parquet}")
    details.to_parquet(details_parquet, index=False)

    api_series = details["api_total_time_s"].dropna()
    summary = pd.DataFrame([{
        "count": int(details["cosine"].count()),
        "min": float(details["cosine"].min()),
        "max": float(details["cosine"].max()),
        "mean": float(details["cosine"].mean()),
        "API_min_s": float(api_series.min()) if len(api_series) else None,
        "API_max_s": float(api_series.max()) if len(api_series) else None,
        "API_mean_s": float(api_series.mean()) if len(api_series) else None,
    }])
    by_vertical = make_by_vertical(details)

    print(f"[SAVE] Excel -> {excel_path}")
    with pd.ExcelWriter(excel_path) as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        by_vertical.to_excel(w, sheet_name="by_vertical", index=False)

    print("\n===== SUMMARY =====")
    print(summary.to_string(index=False))
    print("\n===== BY VERTICAL (cosine_mean desc, top 20) =====")
    print(by_vertical.head(20).to_string(index=False))
    print(f"\n[DONE] reports written to {out_dir}")


if __name__ == "__main__":
    main()
