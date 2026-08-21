#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare GPU vs API wiki embeddings; write reports into a GPU-API directory."""

import json
import os
import re
from typing import Any, List

import numpy as np
import pandas as pd

from embed_api_parser import extract_embeddings

TEST_CASE_NAME_PATTERN = re.compile(
    r"^(?P<project_name>[a-zA-Z0-9_]+(?:-[a-zA-Z0-9_]+)*-[0-9]+)-"
    r"(?P<flavor>[a-zA-Z0-9_]+)-test-"
    r"(?P<vertical>[a-zA-Z0-9_]+(?:-[a-zA-Z0-9_]+)*)-[0-9]+$"
)


def resolve_gpu_api_outdir(gpu_path: str, api_path: str) -> str:
    """Place GPU-API next to the GPU result directory (or reuse it if already there)."""
    gpu_dir = os.path.dirname(os.path.abspath(gpu_path))
    name = os.path.basename(gpu_dir)
    if name == "GPU-API":
        return gpu_dir
    if name == "GPU":
        return os.path.join(os.path.dirname(gpu_dir), "GPU-API")
    return os.path.join(gpu_dir, "GPU-API")


def _cosine(u, v) -> float:
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


def _as_list(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, tuple):
        return list(x)
    return x


def _normalize_input_to_list(inp: Any) -> List[str]:
    if isinstance(inp, list):
        return [str(i) for i in inp]
    if isinstance(inp, str):
        return [inp]
    return [str(inp)]


def _normalize_embeddings_to_list_of_lists(emb: Any) -> List[List[float]]:
    emb = _as_list(emb)
    if not isinstance(emb, list):
        raise TypeError("embeddings is not a list.")
    if len(emb) > 0 and not isinstance(emb[0], (list, tuple, np.ndarray)):
        return [list(map(float, emb))]
    return [list(map(float, vec)) for vec in emb]


def parse_gpu_jsonl(path: str) -> pd.DataFrame:
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tcn = obj.get("testCaseName") or obj.get("test_case_name")
            if not tcn:
                raise KeyError(f"[GPU] line {ln} missing testCaseName")
            inputs = _normalize_input_to_list(obj.get("input"))
            recs.append({
                "testCaseName": tcn,
                "input": inputs,
                "embeddings": _normalize_embeddings_to_list_of_lists(obj.get("embeddings")),
                "gpu_elapsed_s": float(obj.get("elapsed_s")) if obj.get("elapsed_s") is not None else None,
                "case_type": "pair" if len(inputs) == 2 else "single",
            })
    if not recs:
        raise ValueError("GPU jsonl is empty.")
    return pd.DataFrame.from_records(recs)


def parse_api_txt(path: str) -> pd.DataFrame:
    recs, buf = [], []

    def flush_triplet(trip: List[str], idx: int):
        if len(trip) != 3:
            raise ValueError(f"[API] group {idx} is not 3 lines: {trip}")
        j1 = json.loads(trip[0])
        j2 = json.loads(trip[1])
        if not trip[2].startswith("API_total_time:"):
            raise ValueError(f"[API] group {idx} line 3 must start with 'API_total_time:': {trip[2]}")
        tcn = j1.get("testCaseName") or j1.get("test_case_name")
        if not tcn:
            raise KeyError(f"[API] group {idx} missing testCaseName")
        api_total_time_s = float(trip[2].split(":", 1)[1].strip())
        inputs = _normalize_input_to_list(j1.get("input"))
        recs.append({
            "testCaseName": tcn,
            "input": inputs,
            "embeddings": _normalize_embeddings_to_list_of_lists(extract_embeddings(j2)),
            "api_total_time_s": api_total_time_s,
            "total_duration": j2.get("total_duration"),
            "load_duration": j2.get("load_duration"),
            "prompt_eval_count": j2.get("prompt_eval_count"),
            "case_type": "pair" if len(inputs) == 2 else "single",
        })

    with open(path, "r", encoding="utf-8") as f:
        idx = 0
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            buf.append(line)
            if len(buf) == 3:
                idx += 1
                flush_triplet(buf, idx)
                buf = []
    if buf:
        raise ValueError(f"[API] trailing incomplete group: {buf}")
    if not recs:
        raise ValueError("API txt is empty.")
    return pd.DataFrame.from_records(recs)


def extract_vertical(df: pd.DataFrame) -> pd.DataFrame:
    vertical = []
    for tcn in df["testCaseName"]:
        m = TEST_CASE_NAME_PATTERN.match(str(tcn))
        vertical.append(m.group("vertical") if m else "UNKNOWN")
    df = df.copy()
    df["vertical"] = vertical
    return df


def compare_and_collect(gpu_df: pd.DataFrame, api_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(
        gpu_df[["testCaseName", "input", "embeddings", "gpu_elapsed_s", "case_type"]],
        api_df[["testCaseName", "input", "embeddings", "api_total_time_s", "case_type"]],
        on=["testCaseName", "case_type"],
        suffixes=("_gpu", "_api"),
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No aligned cases; check testCaseName.")

    rows = []
    for _, r in merged.iterrows():
        tcn = r["testCaseName"]
        gpu_embs = _normalize_embeddings_to_list_of_lists(r["embeddings_gpu"])
        api_embs = _normalize_embeddings_to_list_of_lists(r["embeddings_api"])
        if len(gpu_embs) != len(api_embs):
            raise ValueError(f"{tcn} vector count mismatch: GPU={len(gpu_embs)} vs API={len(api_embs)}")
        for j in range(len(gpu_embs)):
            rows.append({
                "testCaseName": tcn,
                "pair_idx": j,
                "cosine": _cosine(gpu_embs[j], api_embs[j]),
                "api_total_time_s": r.get("api_total_time_s"),
                "gpu_elapsed_s": r.get("gpu_elapsed_s"),
                "case_type": r.get("case_type"),
            })
    details = pd.DataFrame.from_records(rows, columns=[
        "testCaseName", "pair_idx", "cosine", "api_total_time_s", "gpu_elapsed_s", "case_type",
    ])
    if details.empty:
        raise ValueError("No cosine similarities computed.")
    return details


def make_by_vertical(details: pd.DataFrame) -> pd.DataFrame:
    df = extract_vertical(details)

    def _agg(g: pd.DataFrame) -> pd.Series:
        api = g["api_total_time_s"].dropna()
        return pd.Series({
            "样本数(行)": int(len(g)),
            "pair样本数": int((g["case_type"] == "pair").sum()),
            "single样本数": int((g["case_type"] == "single").sum()),
            "cosine_min": round(float(g["cosine"].min()), 6),
            "cosine_max": round(float(g["cosine"].max()), 6),
            "cosine_mean": round(float(g["cosine"].mean()), 6),
            "API_min_s": round(float(api.min()), 6) if len(api) else None,
            "API_max_s": round(float(api.max()), 6) if len(api) else None,
            "API_mean_s": round(float(api.mean()), 6) if len(api) else None,
        })

    byv = df.groupby("vertical", dropna=False).apply(_agg).reset_index()
    byv = byv.sort_values(by="cosine_mean", ascending=False)
    return byv
