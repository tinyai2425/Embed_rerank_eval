#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_gpu_api_embeddings_by_vertical_v2.py

- 读取 GPU jsonl 与 API txt（3行一组）
- 逐对计算余弦相似度（pair/single 都支持）
- 仅按 vertical 分类统计（min/max/mean），并统计 API 时间的 min/max/mean
- 输出到 GPU 文件同目录：
  1) <gpu_basename>-compare-details.parquet   （逐条明细）
  2) <gpu_basename>-compare.xlsx              （summary / by_vertical 两个 sheet）
"""

import os, re, sys, json, argparse
from typing import Any, List
import numpy as np
import pandas as pd

# 从 testCaseName 里解析 vertical
TEST_CASE_NAME_PATTERN = re.compile(
    r"^(?P<project_name>[a-zA-Z0-9_]+(?:-[a-zA-Z0-9_]+)*-[0-9]+)-"
    r"(?P<flavor>[a-zA-Z0-9_]+)-test-"
    r"(?P<vertical>[a-zA-Z0-9_]+(?:-[a-zA-Z0-9_]+)*)-[0-9]+$"
)

def _cosine(u, v) -> float:
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))

def _as_list(x: Any) -> Any:
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, tuple): return list(x)
    return x

def _normalize_input_to_list(inp: Any) -> List[str]:
    if isinstance(inp, list): return [str(i) for i in inp]
    if isinstance(inp, str):  return [inp]
    return [str(inp)]

def _normalize_embeddings_to_list_of_lists(emb: Any) -> List[List[float]]:
    emb = _as_list(emb)
    if not isinstance(emb, list):
        raise TypeError("embeddings 不是列表结构。")
    if len(emb) > 0 and not isinstance(emb[0], (list, tuple, np.ndarray)):
        return [list(map(float, emb))]  # 一维向量包装为 [[...]]
    return [list(map(float, vec)) for vec in emb]

def parse_gpu_jsonl(path: str) -> pd.DataFrame:
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            tcn = obj.get("testCaseName") or obj.get("test_case_name")
            if not tcn: raise KeyError(f"[GPU] 第{ln}行缺少 testCaseName")
            inputs = _normalize_input_to_list(obj.get("input"))
            recs.append({
                "testCaseName": tcn,
                "input": inputs,
                "embeddings": _normalize_embeddings_to_list_of_lists(obj.get("embeddings")),
                "gpu_elapsed_s": float(obj.get("elapsed_s")) if obj.get("elapsed_s") is not None else None,
                "case_type": "pair" if len(inputs) == 2 else "single",
            })
    if not recs: raise ValueError("GPU jsonl 为空。")
    return pd.DataFrame.from_records(recs)

def _extract_embeddings(obj):
    # 旧格式:
    # {"embeddings": [[...], [...]]}
    if "embeddings" in obj:
        return obj["embeddings"]

    # OpenAI Compatible 格式:
    # {"data": [{"embedding": [...]}, ...]}
    if "data" in obj and isinstance(obj["data"], list):
        embeddings = []

        for item in obj["data"]:
            if isinstance(item, dict) and "embedding" in item:
                embeddings.append(item["embedding"])

        if embeddings:
            return embeddings

    raise TypeError(
        f"无法从 API 返回中找到 embeddings，"
        f"返回字段为: {list(obj.keys())}"
    )

def parse_api_txt(path: str) -> pd.DataFrame:
    recs, buf = [], []
    def flush_triplet(trip: List[str], idx: int):
        if len(trip) != 3:
            raise ValueError(f"[API] 第{idx}组不是3行：{trip}")
        j1 = json.loads(trip[0])
        j2 = json.loads(trip[1])
        if not trip[2].startswith("API_total_time:"):
            raise ValueError(f"[API] 第{idx}组三行目须以 'API_total_time:' 开头：{trip[2]}")
        tcn = j1.get("testCaseName") or j1.get("test_case_name")
        if not tcn: raise KeyError(f"[API] 第{idx}组缺少 testCaseName")
        api_total_time_s = float(trip[2].split(":",1)[1].strip())
        inputs = _normalize_input_to_list(j1.get("input"))
        recs.append({
            "testCaseName": tcn,
            "input": inputs,
            "embeddings": _normalize_embeddings_to_list_of_lists(_extract_embeddings(j2)),
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
            if not line: continue
            buf.append(line)
            if len(buf) == 3:
                idx += 1
                flush_triplet(buf, idx)
                buf = []
    if buf:
        raise ValueError(f"[API] 末尾有不完整分组：{buf}")
    if not recs: raise ValueError("API txt 为空。")
    return pd.DataFrame.from_records(recs)

def extract_vertical(df: pd.DataFrame) -> pd.DataFrame:
    """从 testCaseName 提取 vertical；匹配失败标记为 'UNKNOWN'。"""
    vertical = []
    for tcn in df["testCaseName"]:
        m = TEST_CASE_NAME_PATTERN.match(str(tcn))
        vertical.append(m.group("vertical") if m else "UNKNOWN")
    df = df.copy()
    df["vertical"] = vertical
    return df

def compare_and_collect(gpu_df: pd.DataFrame, api_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(
        gpu_df[["testCaseName","input","embeddings","gpu_elapsed_s","case_type"]],
        api_df[["testCaseName","input","embeddings","api_total_time_s","case_type"]],
        on=["testCaseName","case_type"],
        suffixes=("_gpu","_api"),
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("两侧没有可对齐用例，请检查 testCaseName。")

    rows = []
    for _, r in merged.iterrows():
        tcn = r["testCaseName"]
        gpu_embs = _normalize_embeddings_to_list_of_lists(r["embeddings_gpu"])
        api_embs = _normalize_embeddings_to_list_of_lists(r["embeddings_api"])
        if len(gpu_embs) != len(api_embs):
            raise ValueError(f"{tcn} 的向量数量不一致：GPU={len(gpu_embs)} vs API={len(api_embs)}")
        for j in range(len(gpu_embs)):
            cos = _cosine(gpu_embs[j], api_embs[j])
            rows.append({
                "testCaseName": tcn,
                "pair_idx": j,
                "cosine": cos,
                "api_total_time_s": r.get("api_total_time_s"),
                "gpu_elapsed_s": r.get("gpu_elapsed_s"),
                "case_type": r.get("case_type"),
            })
    details = pd.DataFrame.from_records(rows, columns=[
        "testCaseName","pair_idx","cosine","api_total_time_s","gpu_elapsed_s","case_type"
    ])
    if details.empty: raise ValueError("未计算到余弦相似度。")
    return details

def make_by_vertical(details: pd.DataFrame) -> pd.DataFrame:
    df = extract_vertical(details)

    def _agg(g: pd.DataFrame) -> pd.Series:
        api = g["api_total_time_s"].dropna()
        res = {
            "样本数(行)": int(len(g)),
            "pair样本数": int((g["case_type"]=="pair").sum()),
            "single样本数": int((g["case_type"]=="single").sum()),
            "cosine_min": round(float(g["cosine"].min()), 6),
            "cosine_max": round(float(g["cosine"].max()), 6),
            "cosine_mean": round(float(g["cosine"].mean()), 6),
            "API_min_s": round(float(api.min()), 6) if len(api) else None,
            "API_max_s": round(float(api.max()), 6) if len(api) else None,
            "API_mean_s": round(float(api.mean()), 6) if len(api) else None,
        }
        return pd.Series(res)

    byv = df.groupby("vertical", dropna=False).apply(_agg).reset_index()
    byv = byv.sort_values(by="cosine_mean", ascending=False)
    return byv

def main():
    ap = argparse.ArgumentParser(description="Compare GPU/API embeddings and export vertical-wise cosine stats (min/max/mean).")
    ap.add_argument("gpu_jsonl", help="GPU 结果 jsonl 路径")
    ap.add_argument("api_txt", help="API 结果 txt 路径（3行一组）")
    args = ap.parse_args()

    gpu_path = args.gpu_jsonl
    api_path = args.api_txt
    if not os.path.isfile(gpu_path): sys.exit(f"[错误] 找不到 GPU 文件：{gpu_path}")
    if not os.path.isfile(api_path): sys.exit(f"[错误] 找不到 API 文件：{api_path}")

    out_dir = os.path.dirname(os.path.abspath(gpu_path)) or "."
    gpu_base = os.path.splitext(os.path.basename(gpu_path))[0]
    details_parquet = os.path.join(out_dir, f"{gpu_base}-compare-details.parquet")
    excel_path      = os.path.join(out_dir, f"{gpu_base}-compare.xlsx")

    print(f"[读取] GPU: {gpu_path}")
    gpu_df = parse_gpu_jsonl(gpu_path)
    print(f"[读取] API: {api_path}")
    api_df = parse_api_txt(api_path)

    print("[计算] 余弦相似度 ...")
    details = compare_and_collect(gpu_df, api_df)

    print(f"[保存] 明细 Parquet -> {details_parquet}")
    details.to_parquet(details_parquet, index=False)

    # 总体 summary（含 API 时间的 min/max/mean）
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

    # 按 vertical 分类统计
    by_vertical = make_by_vertical(details)

    print(f"[保存] Excel -> {excel_path}")
    with pd.ExcelWriter(excel_path) as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        by_vertical.to_excel(w, sheet_name="by_vertical", index=False)

    print("\n===== SUMMARY =====")
    print(summary.to_string(index=False))
    print("\n===== BY VERTICAL（按 cosine_mean 降序，前 20）=====")
    print(by_vertical.head(20).to_string(index=False))
    print("\n[完成] 所有输出已写入 GPU 文件同目录。")

if __name__ == "__main__":
    main()
