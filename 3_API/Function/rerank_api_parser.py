#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parse generic /v1/reranks API logs (3 lines per sample)."""

import json
from typing import Any, Dict, List

import pandas as pd


def _err(prefix: str, i: int, line: str) -> ValueError:
    snippet = line[:200].replace("\n", "\\n")
    return ValueError(f"{prefix} at line {i + 1}: {snippet}")


def extract_relevance_scores(resp: Dict[str, Any]) -> List[float]:
    data = resp.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError(f"Response missing data[]; keys={list(resp.keys())}")
    indexed = []
    for item in data:
        if not isinstance(item, dict) or "relevance_score" not in item:
            raise ValueError(f"Invalid reranking_result item: {item}")
        indexed.append((int(item.get("index", len(indexed))), float(item["relevance_score"])))
    indexed.sort(key=lambda x: x[0])
    return [score for _, score in indexed]


def parse_rerank_api_results(log_path: str) -> pd.DataFrame:
    """
    Each sample is 3 non-empty lines:
      1) request JSON (may include testCaseName / queryName / corpusName / expect)
      2) response JSON with data[].relevance_score
      3) API_total_time: <float>
    """
    with open(log_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()
    lines = [ln.strip() for ln in raw_lines if ln.strip() != ""]

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
            req = json.loads(l1)
        except Exception:
            raise _err("Invalid JSON (request)", i1, l1)
        try:
            resp = json.loads(l2)
        except Exception:
            raise _err("Invalid JSON (response)", i2, l2)
        if not l3.startswith("API_total_time:"):
            raise _err('Third line must start with "API_total_time:"', i3, l3)
        try:
            elapsed_s = float(l3.split(":", 1)[1].strip())
        except Exception:
            raise _err("Failed to parse API_total_time as float", i3, l3)

        try:
            scores = extract_relevance_scores(resp)
        except Exception:
            raise _err("Missing or invalid relevance_score", i2, l2)

        documents = req.get("documents") or []
        corpus_names = req.get("corpusNames")
        expects = req.get("expects")

        if len(scores) == 1 and "corpusName" in req:
            results.append({
                "testCaseName": req.get("testCaseName", ""),
                "queryName": str(req.get("queryName", "")),
                "corpusName": str(req.get("corpusName", "")),
                "query": req.get("query", ""),
                "document": documents[0] if documents else "",
                "expect": int(req.get("expect", 0)),
                "score": scores[0],
                "elapsed_s": elapsed_s,
            })
            continue

        if not corpus_names:
            corpus_names = [f"{req.get('queryName', 'q')}-doc{i}" for i in range(len(scores))]
        if expects is None:
            expects = [0] * len(scores)
        if len(scores) != len(corpus_names) or len(scores) != len(expects):
            raise _err(
                f"Score/corpus/expect length mismatch: "
                f"{len(scores)}/{len(corpus_names)}/{len(expects)}",
                i1,
                l1,
            )
        for i, score in enumerate(scores):
            results.append({
                "testCaseName": req.get("testCaseName", f"{req.get('queryName', '')}-{i}"),
                "queryName": str(req.get("queryName", "")),
                "corpusName": str(corpus_names[i]),
                "query": req.get("query", ""),
                "document": documents[i] if i < len(documents) else "",
                "expect": int(expects[i]),
                "score": score,
                "elapsed_s": elapsed_s,
            })

    return pd.DataFrame(results)
