import json
from typing import Any, Dict, List
import pandas as pd

def _err(prefix: str, i: int, line: str) -> ValueError:
    # 报错里带上 1-based 行号与内容片段，便于定位
    snippet = line[:200].replace("\n", "\\n")
    return ValueError(f"{prefix} at line {i+1}: {snippet}")

def parse_embedding_api_results(log_path: str) -> pd.DataFrame:
    """
    严格解析 API 日志（每 3 行一个样本）为 DataFrame：
      1) meta: {"model": "...", "testCaseName": "...", "input": [...], "expect": ...}
      2) embs: {"model": "...", "embeddings": [[...]]}
      3) time: "API_total_time: 5.132307"

    规则（严格）：
      - 允许跳过空行；去掉空行后，行数必须是 3 的倍数，否则报错。
      - 第 1 行必须是 JSON，且包含 "testCaseName" 和 "input"（list），"expect" 可选。
      - 第 2 行必须是 JSON，且包含 "embeddings"（list）。
      - 第 3 行必须以 "API_total_time:" 开头，且能解析为 float。
      - 任一组校验失败立即抛出 ValueError。

    输出列：
      - testCaseName: str
      - input: List[str]
      - expect: Any (float for STSB)
      - embeddings: List[List[float]]
      - elapsed_s: float
    """
    # 读取并剔除纯空行；严格解析
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
        i1, i2, i3 = k, k+1, k+2
        l1, l2, l3 = lines[i1], lines[i2], lines[i3]

        # 1) meta
        try:
            meta = json.loads(l1)
        except Exception:
            raise _err("Invalid JSON (meta)", i1, l1)
        if not isinstance(meta, dict):
            raise _err("Meta must be a JSON object", i1, l1)
        if "testCaseName" not in meta:
            raise _err('Missing "testCaseName" in meta', i1, l1)
        if "input" not in meta or not isinstance(meta["input"], list):
            raise _err('Missing or invalid "input" (must be list) in meta', i1, l1)

        # 2) embeddings
        try:
            embs = json.loads(l2)
        except Exception:
            raise _err("Invalid JSON (embeddings)", i2, l2)
        if not isinstance(embs, dict):
            raise _err("Embeddings line must be a JSON object", i2, l2)
        if "embeddings" not in embs or not isinstance(embs["embeddings"], list):
            raise _err('Missing or invalid "embeddings" (must be list)', i2, l2)

        # 3) time
        if not l3.startswith("API_total_time:"):
            raise _err('Third line must start with "API_total_time:"', i3, l3)
        try:
            elapsed_s = float(l3.split(":", 1)[1].strip())
        except Exception:
            raise _err("Failed to parse API_total_time as float", i3, l3)

        results.append({
            "testCaseName": meta["testCaseName"],
            "input": meta["input"],
            "expect": meta.get("expect", ""),
            "embeddings": embs["embeddings"],
            "elapsed_s": elapsed_s,
        })

    df = pd.DataFrame(results)
    # 最小一致性检查
    if not all(isinstance(x, list) for x in df["input"]):
        raise ValueError("Column 'input' contains non-list entries.")
    return df
