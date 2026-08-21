#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate rerank cases for GPU (model-family specific) and API (generic).

Usage:
  python Gen_rerank_cases.py <model_config.json> <MAX_query>

GPU cases depend on rerank_family in the config (default: qwen3).
API cases always use the generic {model, query, documents} interface;
system prompt / instruct are applied server-side and are not sent.

To add a new GPU family later, add Function/rerank_<family>.py and register
it in GPU_GENERATORS below.
"""

import json
import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Function"))

from paths import data_set_dir, make_project_dir
from rerank_common import generate_api_rerank_tests, set_model_config
from rerank_qwen3 import generate_gpu_rerank_tests as generate_gpu_rerank_tests_qwen3

CORPUS_PATH = os.path.join(data_set_dir(), "Rerank", "RerankerEval_Small_merged.pkl")

GPU_GENERATORS = {
    "qwen3": generate_gpu_rerank_tests_qwen3,
}


def main():
    if len(sys.argv) != 3:
        print("Usage: python Gen_rerank_cases.py <model_config.json> <MAX_query>")
        sys.exit(1)

    config_path = sys.argv[1]
    try:
        max_query = int(sys.argv[2])
    except ValueError:
        raise ValueError("MAX_query must be an integer")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(f"Dataset not found: {CORPUS_PATH}")

    with open(config_path, "r", encoding="utf-8") as f:
        mp = SimpleNamespace(**json.load(f))
    if not getattr(mp, "model_name", None):
        raise ValueError("Config must contain model_name")
    set_model_config(mp)

    family = getattr(mp, "rerank_family", "qwen3")
    if family not in GPU_GENERATORS:
        known = ", ".join(sorted(GPU_GENERATORS))
        raise ValueError(
            f"Unsupported rerank_family={family!r}. Known: {known}. "
            "Add Function/rerank_<family>.py and register it in GPU_GENERATORS."
        )

    project_name, project_path = make_project_dir(mp.model_name)
    print(f"PROJECT_NAME {project_name}")
    print(f"Full path to project dir: {project_path}")
    print(f"rerank_family: {family}")

    gpu_cases = GPU_GENERATORS[family](
        project_name=project_name,
        dataset_path=CORPUS_PATH,
        max_query=max_query,
    )
    api_cases = generate_api_rerank_tests(
        project_name=project_name,
        dataset_path=CORPUS_PATH,
        max_query=max_query,
    )

    gpu_out = os.path.join(project_path, f"{project_name}-RERANK-GPU-MAXQ{max_query}.json")
    api_out = os.path.join(project_path, f"{project_name}-RERANK-API-MAXQ{max_query}.jsonl")

    with open(gpu_out, "w", encoding="utf-8") as f:
        json.dump(gpu_cases, f, indent=4, ensure_ascii=False)
    print(f"{len(gpu_cases)} GPU rerank cases saved to {gpu_out}")

    with open(api_out, "w", encoding="utf-8") as f:
        for case in api_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"{len(api_cases)} API rerank cases saved to {api_out}")


if __name__ == "__main__":
    main()
