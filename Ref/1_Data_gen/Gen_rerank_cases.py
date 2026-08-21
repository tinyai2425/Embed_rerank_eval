#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Gen_rerank_cases.py

import sys
import os
import json
from types import SimpleNamespace

sys.path.append(os.path.abspath("Function"))
import rerank as rrk  # noqa

CORPUS_DIR = "../data_set/Rerank/RerankerEval_Small_merged.pkl"


def get_unique_subdir(base_dir: str, prefix: str) -> str:
    counter = 1
    while os.path.exists(os.path.join(base_dir, f"{prefix}-{counter}")):
        counter += 1
    return f"{prefix}-{counter}"


def main():
    # 用法：python Gen_rerank_cases.py <model_config.json> <MAX_query>
    if len(sys.argv) != 3:
        print("用法: python Gen_rerank_cases.py <model_config.json> <MAX_query>")
        sys.exit(1)

    config_path = sys.argv[1]
    try:
        MAX_query = int(sys.argv[2])
    except ValueError:
        raise ValueError("MAX_query 必须是整数")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    if not os.path.exists(CORPUS_DIR):
        raise FileNotFoundError(f"找不到数据集文件: {CORPUS_DIR}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    mp = SimpleNamespace(**config_dict)

    PROJECT_BASE = f"../../model-eval-storage/{mp.model_name}"
    PROJECT_PREFIX = "project"
    PROJECT_NAME = get_unique_subdir(PROJECT_BASE, PROJECT_PREFIX)
    PROJECT_PATH = f"{PROJECT_BASE}/{PROJECT_NAME}"
    os.makedirs(PROJECT_PATH, exist_ok=True)

    print(f"PROJECT_NAME {PROJECT_NAME}")
    print(f"Full path to project dir: {PROJECT_PATH}")

    rrk.set_model_config(mp)

    instruction = "Given a web search query, retrieve relevant passages that answer the query"

    all_cases = rrk.generate_rerank_tests(
        project_name=PROJECT_NAME,
        dataset_path=CORPUS_DIR,
        instruction=instruction,
        max_query=MAX_query,
    )

    out_path = f"{PROJECT_PATH}/{PROJECT_NAME}-RERANK-GPU-MAXQ{MAX_query}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_cases, f, indent=4, ensure_ascii=False)

    print(f"{len(all_cases)} GPU rerank test cases saved to {out_path}")


if __name__ == "__main__":
    main()
