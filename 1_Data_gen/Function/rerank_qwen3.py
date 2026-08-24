#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3-Reranker-0.6B GPU case format and assembled-prompt length filter.

GPU inference uses chat completions + yes/no logprobs.
The same system prompt and instruct are hardcoded on the API server
and are therefore not exposed in the /v1/reranks request body.

Prompt length is the tokenized chat-templated string (system + instruct +
query + document + generation prompt), not the document body alone.
Common prefix/mid/suffix are tokenized once; query and document ids are cached.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from transformers import AutoTokenizer

import rerank_common as rc
from rerank_common import iter_query_doc_pairs

DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

DEFAULT_SYSTEM_PROMPT = (
    'Judge whether the Document meets the requirements based on the Query and '
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
)

# Re-tokenize the full assembled prompt only when the cached estimate is this
# close to the cutoff (BPE joins at part boundaries can shift length by a few).
_BOUNDARY_SLACK = 8

Pair = Tuple[int, Any, str, int, Any, str, int]


def format_instruction(instruction: str, query: Any, doc: Any) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}",
        },
    ]


def _apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


class PromptLengthCounter:
    """Count assembled GPU prompt tokens, reusing common and per-query/doc parts."""

    def __init__(self, tokenizer_path: str, instruction: str = DEFAULT_INSTRUCTION):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        self.tokenizer.model_max_length = 10**12
        self.instruction = instruction
        self.prefix, self.mid, self.suffix = self._split_template(instruction)
        self.prefix_n = self._n_tokens(self.prefix)
        self.mid_n = self._n_tokens(self.mid)
        self.suffix_n = self._n_tokens(self.suffix)
        self._query_n: Dict[Any, int] = {}
        self._doc_n: Dict[Any, int] = {}

    def _split_template(self, instruction: str) -> Tuple[str, str, str]:
        sentinel_q = "<<<Q>>>"
        sentinel_d = "<<<D>>>"
        templated = _apply_chat_template(
            self.tokenizer, format_instruction(instruction, sentinel_q, sentinel_d)
        )
        if sentinel_q not in templated or sentinel_d not in templated:
            raise ValueError("Chat template swallowed length sentinels; cannot split prompt.")
        prefix, rest = templated.split(sentinel_q, 1)
        mid, suffix = rest.split(sentinel_d, 1)
        return prefix, mid, suffix

    def _n_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def estimate(self, qid: Any, query: str, doc_id: Any, doc: str) -> int:
        if qid not in self._query_n:
            self._query_n[qid] = self._n_tokens(query)
        if doc_id not in self._doc_n:
            self._doc_n[doc_id] = self._n_tokens(doc)
        return self.prefix_n + self._query_n[qid] + self.mid_n + self._doc_n[doc_id] + self.suffix_n

    def exact(self, query: str, doc: str) -> int:
        assembled = self.prefix + query + self.mid + doc + self.suffix
        return self._n_tokens(assembled)

    def prompt_token_len(self, qid: Any, query: str, doc_id: Any, doc: str, max_len: Optional[int] = None) -> int:
        est = self.estimate(qid, query, doc_id, doc)
        if max_len is None:
            return est
        if est + _BOUNDARY_SLACK <= max_len or est > max_len + _BOUNDARY_SLACK:
            return est
        return self.exact(query, doc)


def filter_pairs_by_prompt_len(
    pairs: Iterable[Pair],
    counter: PromptLengthCounter,
    max_prompt_len: int,
) -> Tuple[List[Pair], int]:
    kept: List[Pair] = []
    skipped = 0
    for pair in pairs:
        _, qid, query_text, _, doc_id, doc_text, _ = pair
        n = counter.prompt_token_len(qid, query_text, doc_id, doc_text, max_prompt_len)
        if n > max_prompt_len:
            skipped += 1
            continue
        kept.append(pair)
    return kept, skipped


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
    pairs: Optional[Iterable[Pair]] = None,
) -> List[Dict]:
    if pairs is None:
        pairs = iter_query_doc_pairs(dataset_path, max_query)
    all_cases: List[Dict] = []
    for query_idx, qid, query_text, doc_idx, doc_id, doc_text, score in pairs:
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
