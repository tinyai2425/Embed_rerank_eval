from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def cosine(a: np.ndarray, b: np.ndarray, l2norm: bool = True) -> float:
    if l2norm:
        a = _l2_normalize(a[None, :])[0]
        b = _l2_normalize(b[None, :])[0]
    return float(np.dot(a, b))


def evaluate_stsb_from_df(df: pd.DataFrame, l2norm: bool = True) -> Dict[str, float]:
    """STSB: each row has 2 embeddings; expect is the gold score."""
    cosines, golds = [], []
    for _, row in df.iterrows():
        embs = row["embeddings"]
        if not isinstance(embs, list) or len(embs) != 2:
            continue
        e1 = np.asarray(embs[0], dtype="float32")
        e2 = np.asarray(embs[1], dtype="float32")
        if l2norm:
            e1 = e1 / (np.linalg.norm(e1) + 1e-12)
            e2 = e2 / (np.linalg.norm(e2) + 1e-12)
        cosines.append(float(np.dot(e1, e2)))
        try:
            golds.append(float(row.get("expect", float("nan"))))
        except Exception:
            golds.append(float("nan"))

    if not cosines or not golds:
        return {"spearman": float("nan"), "pearson": float("nan")}

    cosines = np.asarray(cosines, dtype="float32")
    golds = np.asarray(golds, dtype="float32")
    mask = ~np.isnan(golds)
    cosines = cosines[mask]
    golds = golds[mask]
    if len(cosines) == 0:
        return {"spearman": float("nan"), "pearson": float("nan")}

    sp, _ = spearmanr(cosines, golds)
    pe, _ = pearsonr(cosines, golds)
    return {"spearman": float(sp), "pearson": float(pe)}
