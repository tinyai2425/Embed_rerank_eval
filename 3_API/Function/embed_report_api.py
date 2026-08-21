import os
from typing import Dict

import numpy as np
import pandas as pd


def write_api_eval_txt(
    api_dir: str,
    metrics: Dict[str, float],
    df: pd.DataFrame,
    tests_log_path: str,
):
    valid_pairs = df["embeddings"].apply(lambda x: isinstance(x, list) and len(x) == 2).sum()

    lines = []
    lines.append("[API-EMB EVAL]")
    lines.append(f"Input file: {tests_log_path}")
    lines.append(f"Total cases: {len(df)}  (valid pairs: {valid_pairs})")
    lines.append(f"Spearman: {metrics.get('spearman', float('nan')):.6f}")
    lines.append(f"Pearson : {metrics.get('pearson', float('nan')):.6f}")

    valid_elapsed = df["elapsed_s"].dropna()
    if len(valid_elapsed) > 0:
        avg_s = float(np.mean(valid_elapsed))
        p95_s = float(np.percentile(valid_elapsed, 95))
        lines.append(f"Avg elapsed per case: {avg_s:.4f}s   P95: {p95_s:.4f}s")
    lines.append("")

    per_path = os.path.join(api_dir, "API_eval.txt")
    with open(per_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[SAVE] {per_path}")

    overall_path = os.path.join(api_dir, "API_eval_overall.txt")
    with open(overall_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[APPEND] {overall_path}")
