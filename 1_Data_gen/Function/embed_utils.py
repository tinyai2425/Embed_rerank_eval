#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Embedding test-case builders (GPU JSON / API JSONL)."""

from typing import Any, Dict, List

mp = None


def set_model_config(model_params) -> None:
    global mp
    mp = model_params


def create_gpu_embedding_test(
    testCaseName: str,
    input_list: List[str],
    expect: Any = "",
) -> Dict[str, Any]:
    return {
        "testCaseName": testCaseName,
        "input": input_list,
        "expect": expect,
    }


def create_api_embedding_test(
    testCaseName: str,
    input_list: List[str],
    expect: Any = "",
) -> Dict[str, Any]:
    """OpenAI-compatible /v1/embeddings request, plus eval metadata."""
    if mp is None or not getattr(mp, "model_name", None):
        raise ValueError("Call set_model_config(mp) first; mp.model_name is required.")
    return {
        "model": mp.model_name,
        "testCaseName": testCaseName,
        "input": input_list,
        "expect": expect,
    }
