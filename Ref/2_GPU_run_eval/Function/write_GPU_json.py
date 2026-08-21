# -*- coding: utf-8 -*-
"""
write_GPU_json.py
把 GPU-Results-*.parquet 精选列导出为 JSONL（每行一个用例）。
"""

import os
import json
import pandas as pd
from typing import List, Optional

DEFAULT_COLUMNS = ["testCaseName", "prompt", "response", "expect", "correct"]

def save_jsonl_selected_fields(
    parquet_path: str,
    out_path: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> str:
    """
    将 parquet 中的指定列导出为 JSONL 文件（UTF-8，无转义，逐行一条记录）。

    Args:
        parquet_path: 需要导出的 parquet 文件路径
        out_path: 输出 JSONL 文件路径；默认与 parquet 同目录同名（扩展名改为 .jsonl）
        columns: 需要导出的列名列表；默认使用 DEFAULT_COLUMNS

    Returns:
        实际写入的 JSONL 文件路径
    """
    if columns is None:
        columns = DEFAULT_COLUMNS

    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in parquet: {missing}")

    # 只保留需要的列，复制防止 SettingWithCopy 警告
    df = df[columns].copy()

    # 填空并做轻度类型处理：文本列转为 str，布尔列保留布尔
    text_cols = [c for c in columns if c != "correct"]
    for c in text_cols:
        # 将 NaN -> "" 再统一转成 str
        df[c] = df[c].fillna("").astype(str)

    # correct 若不是布尔，尽量转为布尔，否则填 False
    if "correct" in df.columns:
        if df["correct"].dtype == bool:
            pass
        else:
            df["correct"] = df["correct"].fillna(False).astype(bool)

    # 目标路径
    if out_path is None:
        base, _ = os.path.splitext(parquet_path)
        out_path = base + ".jsonl"

    # 顺序写入 JSONL；ensure_ascii=False 保留中文
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[SAVE] JSONL saved to {out_path}")
    return out_path
