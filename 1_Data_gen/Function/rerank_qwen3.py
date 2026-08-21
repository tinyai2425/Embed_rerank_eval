#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3-Reranker-0.6B GPU case format.

GPU inference uses chat completions + yes/no logprobs.
The same system prompt and instruct are hardcoded on the API server
and are therefore not exposed in the /v1/reranks request body.
"""

from typing import Any, Dict, List

import rerank_common as rc
from rerank_common import iter_query_doc_pairs

DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

DEFAULT_SYSTEM_PROMPT = (
    'Judge whether the Document meets the requirements based on the Query and '
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
)


def format_instruction(instruction: str, query: Any, doc: Any) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}",
        },
    ]


def create_gpu_rerank_test(
    testCaseName: str,
    queryName: str,
    corpusName: str,
    messages,
    expect: int,
) -> Dict:
    if rc.mp is None:
        raise ValueError("Call set_model_config(mp) before creating GPU rerank cases.")
    return {
        "testCaseName": testCaseName,
        "queryName": queryName,
        "corpusName": corpusName,
        "messages": messages,
        "expect": int(expect),
        "seed": rc.mp.SEED,
        "temperature": rc.mp.TEMPERATURE,
        "maxGenTokens": rc.mp.maxGenTokens,
    }


def generate_gpu_rerank_tests(
    project_name: str,
    dataset_path: str,
    instruction: str = DEFAULT_INSTRUCTION,
    max_query: int = 0,
) -> List[Dict]:
    all_cases: List[Dict] = []
    for query_idx, qid, query_text, doc_idx, doc_id, doc_text, score in iter_query_doc_pairs(
        dataset_path, max_query
    ):
        messages = format_instruction(instruction, query_text, doc_text)
        testCaseName = f"{project_name}-Rerank-test-mixed-corpus-val-q{query_idx}-{doc_idx}"
        all_cases.append(
            create_gpu_rerank_test(
                testCaseName=testCaseName,
                queryName=str(qid),
                corpusName=str(doc_id),
                messages=messages,
                expect=score,
            )
        )
    return all_cases
