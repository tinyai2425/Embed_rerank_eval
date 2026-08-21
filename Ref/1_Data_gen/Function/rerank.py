#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Function/rerank.py

import pickle
from typing import Any, Dict, List, Tuple

mp = None


def set_model_config(model_module: Any) -> None:
    global mp
    mp = model_module


def load_user_pkl(dataset_path: str) -> Tuple[Dict, Dict, Dict]:
    # ===== 你的原版：保持不动 =====
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
    if not all(k in data for k in ["corpus", "queries", "qrels"]):
        raise ValueError("User pkl must contain keys: corpus, queries, qrels")
    corpus, queries, qrels = data["corpus"], data["queries"], data["qrels"]
    cleaned_qrels: Dict[str, Dict[str, int]] = {}
    for qid, doc_scores in qrels.items():
        if not isinstance(doc_scores, dict):
            continue
        cleaned_qrels[qid] = {}
        for doc_id, score in doc_scores.items():
            if isinstance(score, dict):
                actual_score = score.get("score") or score.get("relevance") or score.get("rel")
                if actual_score is None:
                    values = [v for v in score.values() if isinstance(v, (int, float))]
                    actual_score = values[0] if values else 0
                score = actual_score
            if isinstance(score, (int, float)):
                cleaned_qrels[qid][doc_id] = int(score)
            elif isinstance(score, str):
                try:
                    cleaned_qrels[qid][doc_id] = int(float(score))
                except (ValueError, TypeError):
                    cleaned_qrels[qid][doc_id] = 0
            else:
                cleaned_qrels[qid][doc_id] = 0
    return corpus, queries, cleaned_qrels


def format_instruction(instruction: str, query: Any, doc: Any):
    return [
        {
            "role": "system",
            "content": 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".',
        },
        {
            "role": "user",
            "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}",
        },
    ]


def create_rerank_test(
    testCaseName: str,
    queryName: str,
    corpusName: str,
    messages,
    expect: int,
) -> Dict:
    # 严格按你定义的字段
    return {
        "testCaseName": testCaseName,
        "queryName": queryName,
        "corpusName": corpusName,
        "messages": messages,
        "expect": int(expect),
        "seed": mp.SEED,
        "temperature": mp.TEMPERATURE,
        "maxGenTokens": mp.maxGenTokens,
    }

def generate_rerank_tests(
    project_name: str,
    dataset_path: str,
    instruction: str,
    max_query: int = 0,
) -> List[Dict]:
    corpus, queries, qrels = load_user_pkl(dataset_path)

    qids = list(qrels.keys())
    if max_query > 0:
        qids = qids[:max_query]

    all_cases: List[Dict] = []

    for query_idx, qid in enumerate(qids):
        query_text = queries.get(qid)
        if query_text is None:
            continue

        for doc_idx, (doc_id, score) in enumerate(qrels[qid].items()):
            doc = corpus.get(doc_id)
            if not doc or "text" not in doc:
                continue

            messages = format_instruction(instruction, query_text, doc["text"])

            testCaseName = (
                f"{project_name}-Rerank-test-mixed-corpus-val-q{query_idx}-{doc_idx}"
            )

            all_cases.append(
                create_rerank_test(
                    testCaseName=testCaseName,
                    queryName=str(qid),
                    corpusName=str(doc_id),
                    messages=messages,
                    expect=score,
                )
            )

    return all_cases
