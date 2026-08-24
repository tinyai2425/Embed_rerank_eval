#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic rerank dataset loading and API case format.

API cases follow the public rerank interface (query + documents).
System prompt / instruct are applied server-side and are not sent.
"""

import pickle
from typing import Any, Dict, Iterable, List, Optional, Tuple

mp = None


def set_model_config(model_module: Any) -> None:
    global mp
    mp = model_module


def load_user_pkl(dataset_path: str) -> Tuple[Dict, Dict, Dict]:
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


def iter_query_doc_pairs(dataset_path: str, max_query: int = 0):
    """Yield (query_idx, qid, query_text, doc_idx, doc_id, doc_text, expect)."""
    corpus, queries, qrels = load_user_pkl(dataset_path)
    qids = list(qrels.keys())
    if max_query > 0:
        qids = qids[:max_query]

    for query_idx, qid in enumerate(qids):
        query_text = queries.get(qid)
        if query_text is None:
            continue
        for doc_idx, (doc_id, score) in enumerate(qrels[qid].items()):
            doc = corpus.get(doc_id)
            if not doc or "text" not in doc:
                continue
            yield query_idx, qid, query_text, doc_idx, doc_id, doc["text"], int(score)


def create_api_rerank_test(
    testCaseName: str,
    queryName: str,
    corpusName: str,
    query: str,
    document: str,
    expect: int,
) -> Dict[str, Any]:
    """Generic /v1/reranks request, plus eval metadata.

    Extra fields (testCaseName / queryName / corpusName / expect) are for
    logging and evaluation; the collector should only send model/query/documents.
    """
    if mp is None or not getattr(mp, "model_name", None):
        raise ValueError("Call set_model_config(mp) first; mp.model_name is required.")
    return {
        "model": mp.model_name,
        "query": query,
        "documents": [document],
        "testCaseName": testCaseName,
        "queryName": str(queryName),
        "corpusName": str(corpusName),
        "expect": int(expect),
    }


def generate_api_rerank_tests(
    project_name: str,
    dataset_path: str,
    max_query: int = 0,
    pairs: Optional[Iterable[Tuple]] = None,
) -> List[Dict]:
    if pairs is None:
        pairs = iter_query_doc_pairs(dataset_path, max_query)
    all_cases: List[Dict] = []
    for query_idx, qid, query_text, doc_idx, doc_id, doc_text, score in pairs:
        testCaseName = f"{project_name}-Rerank-test-mixed-corpus-val-q{query_idx}-{doc_idx}"
        all_cases.append(
            create_api_rerank_test(
                testCaseName=testCaseName,
                queryName=qid,
                corpusName=doc_id,
                query=query_text,
                document=doc_text,
                expect=score,
            )
        )
    return all_cases
