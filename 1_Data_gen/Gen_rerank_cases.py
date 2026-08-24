#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate rerank cases for GPU (model-family specific) and API (generic).

Usage:
  python Gen_rerank_cases.py <model_config.json> <MAX_query> [MAX_PROMPT_LEN]

MAX_PROMPT_LEN is optional. 0 or omitted = no length filter.
When set (e.g. 1024 or 8192), keep only pairs whose assembled GPU prompt
(system + instruct + query + document + chat template) is within that many
tokens. Requires TOKENIZER_CONFIG_PATH in the config.

To add a new GPU family later, add Function/rerank_<family>.py and register
it in GPU_GENERATORS / LENGTH_COUNTERS below.
"""

import json
import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Function"))

from paths import data_set_dir, make_project_dir
from rerank_common import generate_api_rerank_tests, iter_query_doc_pairs, set_model_config
from rerank_qwen3 import (
    DEFAULT_INSTRUCTION,
    PromptLengthCounter,
    filter_pairs_by_prompt_len,
)
from rerank_qwen3 import generate_gpu_rerank_tests as generate_gpu_rerank_tests_qwen3

CORPUS_PATH = os.path.join(data_set_dir(), "Rerank", "RerankerEval_Small_merged.pkl")

GPU_GENERATORS = {
    "qwen3": generate_gpu_rerank_tests_qwen3,
}

LENGTH_COUNTERS = {
    "qwen3": PromptLengthCounter,
}


def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python Gen_rerank_cases.py <model_config.json> <MAX_query> [MAX_PROMPT_LEN]")
        sys.exit(1)

    config_path = sys.argv[1]
    try:
        max_query = int(sys.argv[2])
    except ValueError:
        raise ValueError("MAX_query must be an integer")
    max_prompt_len = 0
    if len(sys.argv) == 4:
        try:
            max_prompt_len = int(sys.argv[3])
            if max_prompt_len < 0:
                raise ValueError
        except ValueError:
            raise ValueError("MAX_PROMPT_LEN must be a non-negative integer (0 = no limit)")

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
    print(f"MAX_query: {max_query}")

    pairs = list(iter_query_doc_pairs(CORPUS_PATH, max_query))
    n_raw = len(pairs)
    print(f"Raw pairs from first {max_query} queries: {n_raw}")

    if max_prompt_len > 0:
        tok_path = getattr(mp, "TOKENIZER_CONFIG_PATH", None)
        if not tok_path:
            raise ValueError("MAX_PROMPT_LEN > 0 requires TOKENIZER_CONFIG_PATH in the config")
        if not os.path.exists(tok_path):
            raise FileNotFoundError(f"TOKENIZER_CONFIG_PATH not found: {tok_path}")
        if family not in LENGTH_COUNTERS:
            raise ValueError(
                f"No prompt-length counter for rerank_family={family!r}. "
                "Add one in Function/rerank_<family>.py and register it in LENGTH_COUNTERS."
            )
        print(f"MAX_PROMPT_LEN: {max_prompt_len} (assembled chat prompt tokens)")
        print(f"Loading tokenizer from {tok_path} ...")
        counter = LENGTH_COUNTERS[family](tok_path, instruction=DEFAULT_INSTRUCTION)
        pairs, skipped = filter_pairs_by_prompt_len(pairs, counter, max_prompt_len)
        n_queries = len({p[1] for p in pairs})
        print(
            f"Kept {len(pairs)}/{n_raw} pairs across {n_queries} queries; "
            f"skipped {skipped} over {max_prompt_len} tokens"
        )
    else:
        print("MAX_PROMPT_LEN: 0 (no length filter)")

    gpu_cases = GPU_GENERATORS[family](
        project_name=project_name,
        dataset_path=CORPUS_PATH,
        max_query=max_query,
        pairs=pairs,
    )
    api_cases = generate_api_rerank_tests(
        project_name=project_name,
        dataset_path=CORPUS_PATH,
        max_query=max_query,
        pairs=pairs,
    )

    len_tag = f"-L{max_prompt_len}" if max_prompt_len > 0 else ""
    gpu_out = os.path.join(project_path, f"{project_name}-RERANK-GPU-MAXQ{max_query}{len_tag}.json")
    api_out = os.path.join(project_path, f"{project_name}-RERANK-API-MAXQ{max_query}{len_tag}.jsonl")

    with open(gpu_out, "w", encoding="utf-8") as f:
        json.dump(gpu_cases, f, indent=4, ensure_ascii=False)
    print(f"{len(gpu_cases)} GPU rerank cases saved to {gpu_out}")

    with open(api_out, "w", encoding="utf-8") as f:
        for case in api_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"{len(api_cases)} API rerank cases saved to {api_out}")


if __name__ == "__main__":
    main()
