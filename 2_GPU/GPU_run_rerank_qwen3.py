#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-Reranker GPU run + evaluation in one pass.

Usage:
  python GPU_run_rerank_qwen3.py <input_json_path> <ip> <port> <model_id> [k_values]

Writes under the case file's sibling GPU/ directory:
  <basename>_Rerank_Results.jsonl
  evaluation_results.md

Add a new GPU_run_rerank_<family>.py for other rerank models; keep
Function/rerank_eval.py as the shared metric writer.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import openai

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Function"))

from rerank_eval import evaluate_and_write_markdown, parse_k_values
from rerank_qwen3_score import build_qrels_and_run_from_qwen3_results


def parse_args():
    if len(sys.argv) not in (5, 6):
        print(
            "Usage: python GPU_run_rerank_qwen3.py "
            "<input_json_path> <ip> <port> <model_id> [k_values]"
        )
        sys.exit(1)
    args = [a.strip() for a in sys.argv[1:]]
    k_values_str = args[4] if len(args) == 5 else "[5,10]"
    return args[0], args[1], int(args[2]), args[3], k_values_str


def convert_logprobs_to_serializable(logprobs):
    serializable_logprobs = []
    for logprob in logprobs:
        serializable_logprobs.append({
            "token": logprob.token,
            "logprob": logprob.logprob,
            "bytes": logprob.bytes,
        })
    return serializable_logprobs


def call_rerank_model(client, model_id, messages, test_case):
    try:
        temperature = test_case.get("temperature", 0)
        max_tokens = test_case.get("maxGenTokens", 1)
        resp = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=10,
            extra_body={
                "chat_template_kwargs": {
                    "add_generation_prompt": True,
                    "enable_thinking": False,
                },
            },
        )
        raw_logprobs = resp.choices[0].logprobs.content[0].top_logprobs
        return convert_logprobs_to_serializable(raw_logprobs)
    except Exception as e:
        print(f"Error in API call: {e}")
        return None


def parallel_fetch_rerank_results(test_cases, client, model_id):
    results = []
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=(os.cpu_count() or 4) * 5) as executor:
        futures = []
        for test in test_cases:
            future = executor.submit(
                call_rerank_model, client, model_id, test["messages"], test
            )
            futures.append((future, test))

        for i, (future, test) in enumerate(futures):
            lp = future.result()
            if lp is not None:
                results.append({
                    "testCaseName": test.get("testCaseName", ""),
                    "queryName": test.get("queryName", ""),
                    "corpusName": test.get("corpusName", ""),
                    "expect": test.get("expect", 0),
                    "response": lp,
                })
            print(f"\rProcessed {i + 1}/{len(test_cases)} test cases", end="", flush=True)

    print(f"\n[INFO] All completed in {time.time() - start_time:.2f}s")
    return results


def main():
    input_json_path, ip, port, model_id, k_values_str = parse_args()
    k_values = parse_k_values(k_values_str)
    print(f"Input file: {input_json_path}")
    print(f"Server: {ip}:{port}")
    print(f"Model ID: {model_id}")
    print(f"k_values: {k_values}")
    print("Notice: Proxy settings should be disabled before proceeding.")

    input_dir = os.path.dirname(os.path.abspath(input_json_path))
    base_name = os.path.splitext(os.path.basename(input_json_path))[0]
    gpu_dir = os.path.join(input_dir, "GPU")
    os.makedirs(gpu_dir, exist_ok=True)
    output_path = os.path.join(gpu_dir, f"{base_name}_Rerank_Results.jsonl")

    client = openai.OpenAI(
        base_url=f"http://{ip}:{port}/v1",
        api_key="no-api-key-needed",
    )

    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            test_cases = json.load(f)
        print(f"Loaded {len(test_cases)} test cases")
    except Exception as e:
        print(f"Error loading input file: {e}")
        sys.exit(1)

    results = parallel_fetch_rerank_results(test_cases, client, model_id)

    try:
        with open(output_path, "w", encoding="utf-8") as f_out:
            for item in results:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[DONE] Results saved to {output_path}")
        print(f"Total successful results: {len(results)}/{len(test_cases)}")
    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)

    qrels, run = build_qrels_and_run_from_qwen3_results(results)
    print(f"Built qrels with {len(qrels)} queries")
    print(f"Built run with {len(run)} queries")
    evaluate_and_write_markdown(qrels, run, k_values, gpu_dir)


if __name__ == "__main__":
    main()
