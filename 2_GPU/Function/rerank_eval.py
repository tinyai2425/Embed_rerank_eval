#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic rerank evaluation: qrels + run scores -> NDCG/MAP/Recall/P markdown.

Scoring itself is model-specific (GPU qwen3 logprobs vs API relevance_score).
This module only consumes already-computed scores.
"""

import os
from collections import defaultdict
from typing import Dict, List, Tuple

from EvaluationRetrieval import EvaluateRetrieval


def parse_k_values(k_values_str: str) -> List[int]:
    k_values_str = k_values_str.strip()
    if k_values_str.startswith("[") and k_values_str.endswith("]"):
        k_values_str = k_values_str[1:-1]
    k_values = []
    for k in k_values_str.split(","):
        k = k.strip()
        if k:
            k_values.append(int(k))
    return k_values


def build_qrels_and_run_from_scored_items(
    items: List[Dict],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, float]]]:
    """Each item: queryName, corpusName, expect, score."""
    qrels = defaultdict(dict)
    run = defaultdict(dict)
    for item in items:
        query_name = item.get("queryName", "")
        corpus_name = item.get("corpusName", "")
        if not query_name or not corpus_name:
            continue
        qrels[query_name][corpus_name] = int(item.get("expect", 0))
        if item.get("score") is not None:
            run[query_name][corpus_name] = float(item["score"])
    return dict(qrels), dict(run)


def save_results_to_markdown(
    output_path: str,
    k_values: List[int],
    ndcg: Dict,
    _map: Dict,
    recall: Dict,
    precision: Dict,
):
    md_content = ["# Rerank Model Evaluation Results", ""]
    for k in k_values:
        md_content.append(f"NDCG@{k}: {ndcg[f'NDCG@{k}']:.4f}")
    for k in k_values:
        md_content.append(f"MAP@{k}: {_map[f'MAP@{k}']:.4f}")
    for k in k_values:
        md_content.append(f"Recall@{k}: {recall[f'Recall@{k}']:.4f}")
    for k in k_values:
        md_content.append(f"P@{k}: {precision[f'P@{k}']:.4f}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))


def evaluate_and_write_markdown(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    k_values: List[int],
    output_dir: str,
    filename: str = "evaluation_results.md",
) -> str:
    missing_in_run = set(qrels.keys()) - set(run.keys())
    if missing_in_run:
        print(f"Warning: {len(missing_in_run)} queries in qrels but not in run")

    print("Evaluating...")
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, run, k_values)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print("  ".join([f"NDCG@{k}: {ndcg[f'NDCG@{k}']:.4f}" for k in k_values]))
    print("  ".join([f"MAP@{k}: {_map[f'MAP@{k}']:.4f}" for k in k_values]))
    print("  ".join([f"Recall@{k}: {recall[f'Recall@{k}']:.4f}" for k in k_values]))
    print("  ".join([f"P@{k}: {precision[f'P@{k}']:.4f}" for k in k_values]))

    os.makedirs(output_dir, exist_ok=True)
    output_markdown = os.path.join(output_dir, filename)
    save_results_to_markdown(output_markdown, k_values, ndcg, _map, recall, precision)
    print(f"\nEvaluation results saved to: {output_markdown}")
    return output_markdown
