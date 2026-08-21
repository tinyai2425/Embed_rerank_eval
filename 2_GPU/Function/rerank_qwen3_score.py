#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3-Reranker-0.6B score conversion: yes/no token logprobs -> relevance score."""

import math
from collections import defaultdict
from typing import Dict, List, Tuple


def calculate_score_from_response(response: List[Dict]) -> float:
    """Use exact 'yes' / 'no' tokens only; missing tokens default to -10.0."""
    logprobs_dict = {}
    for item in response:
        logprobs_dict[item["token"]] = item["logprob"]

    true_logit = logprobs_dict.get("yes", -10.0)
    false_logit = logprobs_dict.get("no", -10.0)
    true_score = math.exp(true_logit)
    false_score = math.exp(false_logit)
    return true_score / (true_score + false_score)


def build_qrels_and_run_from_qwen3_results(
    results: List[Dict],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, float]]]:
    qrels = defaultdict(dict)
    run = defaultdict(dict)
    for item in results:
        query_name = item.get("queryName", "")
        corpus_name = item.get("corpusName", "")
        expect = item.get("expect", 0)
        response = item.get("response", [])
        if not query_name or not corpus_name:
            continue
        qrels[query_name][corpus_name] = expect
        if response:
            run[query_name][corpus_name] = calculate_score_from_response(response)
    return dict(qrels), dict(run)
