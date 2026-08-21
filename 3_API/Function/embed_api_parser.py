#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parse embedding API logs (3 lines per sample) into a DataFrame."""

import json
from typing import Any, Dict, List

import pandas as pd


def _err(prefix: str, i: int, line: str) -> ValueError:
    snippet = line[:200].replace("\n", "\\n")
    return ValueError(f"{prefix} at line {i + 1}: {snippet}")


def extract_embeddings(obj: Dict[str, Any]) -> List[List[float]]:
    """Support both {embeddings: [[...]]} and OpenAI {data: [{embedding: [...]}]}. """
    if "embeddings" in obj and isinstance(obj["embeddings"], list):
        embs = obj["embeddings"]
        if embs and not isinstance(embs[0], (list, tuple)):
            return [list(map(float, embs))]
        return [list(map(float, vec)) for vec in embs]

    if "data" in obj and isinstance(obj["data"], list):
        embeddings = []
        for item in obj["data"]:
            if isinstance(item, dict) and "embedding" in item:
                embeddings.append(list(map(float, item["embedding"])))
        if embeddings:
            return embeddings

    raise ValueError(f"Cannot find embeddings; keys={list(obj.keys())}")


def parse_embedding_api_results(log_path: str) -> pd.DataFrame:
    """
    Each sample is 3 non-empty lines:
      1) meta JSON: model / testCaseName / input / expect
      2) response JSON: embeddings or data[].embedding
      3) API_total_time: <float>
    """
    with open(log_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()
    lines: List[str] = [ln.strip() for ln in raw_lines if ln.strip() != ""]

    n = len(lines)
    if n == 0:
        raise ValueError("Empty API result file after removing blank lines.")
    if n % 3 != 0:
        raise ValueError(f"API result lines (excluding blanks) must be multiple of 3, got {n}.")

    results: List[Dict[str, Any]] = []
    for k in range(0, n, 3):
        i1, i2, i3 = k, k + 1, k + 2
        l1, l2, l3 = lines[i1], lines[i2], lines[i3]

        try:
            meta = json.loads(l1)
        except Exception:
            raise _err("Invalid JSON (meta)", i1, l1)
        if not isinstance(meta, dict):
            raise _err("Meta must be a JSON object", i1, l1)
        if "testCaseName" not in meta:
            raise _err('Missing "testCaseName" in meta', i1, l1)
        if "input" not in meta:
            raise _err('Missing "input" in meta', i1, l1)
        inp = meta["input"]
        if isinstance(inp, str):
            inp = [inp]
        if not isinstance(inp, list):
            raise _err('Invalid "input" (must be list or str) in meta', i1, l1)

        try:
            embs_obj = json.loads(l2)
        except Exception:
            raise _err("Invalid JSON (embeddings)", i2, l2)
        if not isinstance(embs_obj, dict):
            raise _err("Embeddings line must be a JSON object", i2, l2)
        try:
            embeddings = extract_embeddings(embs_obj)
        except Exception:
            raise _err("Missing or invalid embeddings", i2, l2)

        if not l3.startswith("API_total_time:"):
            raise _err('Third line must start with "API_total_time:"', i3, l3)
        try:
            elapsed_s = float(l3.split(":", 1)[1].strip())
        except Exception:
            raise _err("Failed to parse API_total_time as float", i3, l3)

        results.append({
            "testCaseName": meta["testCaseName"],
            "input": inp,
            "expect": meta.get("expect", ""),
            "embeddings": embeddings,
            "elapsed_s": elapsed_s,
        })

    df = pd.DataFrame(results)
    if not all(isinstance(x, list) for x in df["input"]):
        raise ValueError("Column 'input' contains non-list entries.")
    return df
