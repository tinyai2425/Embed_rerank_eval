#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate STSB embedding cases (GPU JSON + API JSONL).

Usage:
  python Gen_embed_STSB_cases.py <model_config.json> [MAX_CASES]

  - no MAX_CASES  -> full dataset
  - MAX_CASES > 0 -> head(MAX_CASES)
"""

import json
import os
import sys
from types import SimpleNamespace

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Function"))

from embed_utils import create_api_embedding_test, create_gpu_embedding_test, set_model_config
from paths import data_set_dir, make_project_dir

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.join(data_set_dir(), "C-MTEB", "test-C-METB-STSB.parquet")


def main():
    if len(sys.argv) not in (2, 3):
        print("Usage: python Gen_embed_STSB_cases.py <model_config.json> [MAX_CASES]")
        sys.exit(1)

    config_path = sys.argv[1]
    max_cases = None
    if len(sys.argv) == 3:
        try:
            max_cases = int(sys.argv[2])
            if max_cases <= 0:
                max_cases = None
        except ValueError:
            raise ValueError("MAX_CASES must be an integer")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(f"Dataset not found: {PARQUET_PATH}")

    with open(config_path, "r", encoding="utf-8") as f:
        mp = SimpleNamespace(**json.load(f))
    if not getattr(mp, "model_name", None):
        raise ValueError("Config must contain model_name")
    set_model_config(mp)

    project_name, project_path = make_project_dir(mp.model_name)
    print(f"PROJECT_NAME {project_name}")
    print(f"Full path to project dir: {project_path}")

    df = pd.read_parquet(PARQUET_PATH)
    need = {"sentence1", "sentence2", "score"}
    if not need.issubset(df.columns):
        raise ValueError(f"parquet must contain {need}, got {df.columns.tolist()}")
    df = df.dropna(subset=["sentence1", "sentence2", "score"])

    base = os.path.splitext(os.path.basename(PARQUET_PATH))[0]
    iter_df = df if max_cases is None else df.head(max_cases)

    gpu_cases, api_cases = [], []
    for i, row in iter_df.iterrows():
        s1, s2 = str(row["sentence1"]), str(row["sentence2"])
        gold = float(row["score"])
        tc_name = f"{project_name}-{base}-{i}"
        gpu_cases.append(create_gpu_embedding_test(tc_name, [s1, s2], gold))
        api_cases.append(create_api_embedding_test(tc_name, [s1, s2], gold))

    suffix = "ALL" if max_cases is None else str(len(iter_df))
    gpu_json_path = os.path.join(project_path, f"{project_name}-{base}-GPU-{suffix}.json")
    api_jsonl_path = os.path.join(project_path, f"{project_name}-{base}-API-{suffix}.jsonl")

    with open(gpu_json_path, "w", encoding="utf-8") as f:
        json.dump(gpu_cases, f, ensure_ascii=False, indent=2)
    print(f"[GPU] {len(gpu_cases)} cases saved -> {gpu_json_path}")

    with open(api_jsonl_path, "w", encoding="utf-8") as f:
        for case in api_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"[API] {len(api_cases)} cases saved -> {api_jsonl_path}")


if __name__ == "__main__":
    main()
