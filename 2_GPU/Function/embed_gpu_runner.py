import os
import json
import time
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI


def _make_client(server_ip: str, server_port: int) -> OpenAI:
    return OpenAI(
        base_url=f"http://{server_ip}:{server_port}/v1",
        api_key="no-api-key-needed",
    )


def _call_embeddings(client: OpenAI, model_id: str, inputs: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=model_id, input=inputs)
    return [item.embedding for item in resp.data]


def _normalize_input(inp: Any) -> List[str]:
    if isinstance(inp, list):
        return [str(x) for x in inp]
    return [str(inp)]


def _measure_case_embeddings(client: OpenAI, model_id: str, case: Dict[str, Any]) -> Dict[str, Any]:
    prompt_list = _normalize_input(case["input"])
    start = time.time()
    embs = _call_embeddings(client, model_id, prompt_list)
    elapsed = time.time() - start
    return {
        "testCaseName": case["testCaseName"],
        "input": prompt_list,
        "expect": case.get("expect", ""),
        "embeddings": embs,
        "elapsed_s": elapsed,
    }


def run_gpu_embeddings_from_tests(
    tests_json_path: str,
    server_ip: str,
    server_port: int,
    model_id: str,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Pull embeddings in parallel; output order matches input order."""
    with open(tests_json_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    client = _make_client(server_ip, server_port)

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(tests_json_path), "GPU")
    os.makedirs(output_dir, exist_ok=True)

    max_workers = (os.cpu_count() or 4) * 5
    results: List[Optional[Dict[str, Any]]] = [None] * len(test_cases)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_idx = {
            ex.submit(_measure_case_embeddings, client, model_id, case): idx
            for idx, case in enumerate(test_cases)
        }
        done = 0
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                rec = fut.result()
            except Exception as e:
                case = test_cases[idx]
                rec = {
                    "testCaseName": case["testCaseName"],
                    "input": case.get("input", []),
                    "expect": case.get("expect", ""),
                    "embeddings": None,
                    "elapsed_s": None,
                    "error": str(e),
                }
            results[idx] = rec
            done += 1
            print(f"\r{done}/{len(test_cases)} cases processed...", end="", flush=True)
    print(f"\n[INFO] All embeddings done in {time.time() - t0:.2f}s")

    df_out = pd.DataFrame(results)
    assert all(
        df_out.loc[i, "testCaseName"] == test_cases[i]["testCaseName"]
        for i in range(len(test_cases))
    ), "Order mismatch between input tests and results!"
    return df_out
