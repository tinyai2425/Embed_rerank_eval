#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate wiki embedding perf cases (GPU JSON + API JSONL).

Usage:
  python Gen_embed_wiki_cases.py <model_config.json> [MAX_CASES] [TARGET_LENGTHS]

  MAX_CASES: omit or 0 = full dataset
  TARGET_LENGTHS: comma-separated token lengths, default 1000
                  e.g. 1000   or  512,1024,2048

Config must include model_name and TOKENIZER_CONFIG_PATH.
"""

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Function"))

from embed_utils import set_model_config
from paths import data_set_dir, make_project_dir

PARQUET_PATH = os.path.join(data_set_dir(), "WIKI_perf", "wiki_CN_ENG_merged.parquet")
DEFAULT_TARGET_LENGTHS = [1000]


def parse_target_lengths(raw: str):
    parts = [p.strip() for p in str(raw).replace(";", ",").split(",") if p.strip()]
    if not parts:
        raise ValueError("TARGET_LENGTHS is empty")
    lengths = []
    for part in parts:
        try:
            n = int(part)
        except ValueError:
            raise ValueError(f"TARGET_LENGTHS must be integers, got {part!r}")
        if n <= 0:
            raise ValueError(f"Each target length must be > 0, got {n}")
        lengths.append(n)
    return lengths


def json_safe(o):
    if o is None or isinstance(o, (str, int, float, bool)):
        return o
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [json_safe(x) for x in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return [json_safe(x) for x in o.tolist()]
    return str(o)


def _to_source_list(x):
    if isinstance(x, np.ndarray):
        return [str(v) for v in x.tolist()]
    if isinstance(x, (list, tuple, set)):
        return [str(v) for v in x]
    if pd.isna(x):
        return []
    return [str(x)]


def _to_int_or_none(x):
    return None if pd.isna(x) else int(x)


def truncate_to_exact_tokens(tokenizer, text: str, target_tokens: int) -> str:
    ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        max_length=target_tokens,
        truncation=True,
    )
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def main():
    if len(sys.argv) not in (2, 3, 4):
        print("Usage: python Gen_embed_wiki_cases.py <model_config.json> [MAX_CASES] [TARGET_LENGTHS]")
        sys.exit(1)

    config_path = sys.argv[1]
    max_cases = None
    if len(sys.argv) >= 3:
        try:
            max_cases = int(sys.argv[2])
            if max_cases <= 0:
                max_cases = None
        except ValueError:
            raise ValueError("MAX_CASES must be an integer")
    target_lengths = DEFAULT_TARGET_LENGTHS
    if len(sys.argv) == 4:
        target_lengths = parse_target_lengths(sys.argv[3])

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(f"Dataset not found: {PARQUET_PATH}")

    with open(config_path, "r", encoding="utf-8") as f:
        mp = SimpleNamespace(**json.load(f))
    if not getattr(mp, "model_name", None):
        raise ValueError("Config must contain model_name")
    if not getattr(mp, "TOKENIZER_CONFIG_PATH", None):
        raise ValueError("Config must contain TOKENIZER_CONFIG_PATH")
    if not os.path.exists(mp.TOKENIZER_CONFIG_PATH):
        raise FileNotFoundError(f"TOKENIZER_CONFIG_PATH not found: {mp.TOKENIZER_CONFIG_PATH}")
    set_model_config(mp)

    project_name, project_path = make_project_dir(mp.model_name)
    print(f"PROJECT_NAME {project_name}")
    print(f"Full path to project dir: {project_path}")
    print(f"TARGET_LENGTHS {target_lengths}")

    df = pd.read_parquet(PARQUET_PATH)
    need = {"text", "word_count", "source", "num_sources"}
    if not need.issubset(df.columns):
        raise ValueError(f"parquet must contain {need}, got {df.columns.tolist()}")
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df["source"] = df["source"].apply(_to_source_list)
    df["word_count"] = df["word_count"].apply(_to_int_or_none)
    df["num_sources"] = df["num_sources"].apply(_to_int_or_none)

    iter_df = df if max_cases is None else df.head(max_cases)

    tokenizer = AutoTokenizer.from_pretrained(mp.TOKENIZER_CONFIG_PATH, use_fast=True)
    tokenizer.model_max_length = 10**12

    gpu_cases_all = []
    api_lines_all = []
    for i, row in iter_df.iterrows():
        raw_text = str(row["text"])
        for length in target_lengths:
            cut_text = truncate_to_exact_tokens(tokenizer, raw_text, length)
            tc_name = f"{project_name}-perf-test-prefill-{length}-embedding-wiki-{i}"
            gpu_cases_all.append({
                "testCaseName": tc_name,
                "input": cut_text,
            })
            api_lines_all.append({
                "model": mp.model_name,
                "testCaseName": tc_name,
                "input": [cut_text],
            })

    suffix = "ALL" if max_cases is None else str(len(iter_df))
    len_tag = "L" + "_".join(str(x) for x in target_lengths)
    gpu_json_path = os.path.join(
        project_path, f"{project_name}-perf-embedding-wiki-GPU-{len_tag}-{suffix}.json"
    )
    api_jsonl_path = os.path.join(
        project_path, f"{project_name}-perf-embedding-wiki-API-{len_tag}-{suffix}.jsonl"
    )

    with open(gpu_json_path, "w", encoding="utf-8") as f:
        json.dump(gpu_cases_all, f, ensure_ascii=False, indent=2, default=json_safe)
    with open(api_jsonl_path, "w", encoding="utf-8") as f:
        for obj in api_lines_all:
            f.write(json.dumps(obj, ensure_ascii=False, default=json_safe) + "\n")

    print(f"[GPU] {len(gpu_cases_all)} cases saved -> {gpu_json_path}")
    print(f"[API] {len(api_lines_all)} cases saved -> {api_jsonl_path}")
    print("Done.")


if __name__ == "__main__":
    main()
