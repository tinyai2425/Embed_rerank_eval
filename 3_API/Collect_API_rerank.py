#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect rerank API results via generic /v1/reranks.

Usage:
  python Collect_API_rerank.py <input.jsonl> <output.txt> [api_url]

Default api_url: http://localhost:11434/v1/reranks

Each output sample is 3 lines: full request JSON, response JSON, API_total_time.
Only model + query + documents are sent to the server.
System prompt / instruct are applied server-side and are not sent.
"""

import json
import os
import sys
import time

import requests

DEFAULT_URL = "http://localhost:11434/v1/reranks"


def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python Collect_API_rerank.py <input.jsonl> <output.txt> [api_url]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    api_url = sys.argv[3] if len(sys.argv) == 4 else DEFAULT_URL
    if not os.path.isfile(input_file):
        sys.exit(f"Input file not found: {input_file}")

    out_dir = os.path.dirname(os.path.abspath(output_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    n = 0
    with open(input_file, "r", encoding="utf-8") as fin, open(
        output_file, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            req = json.loads(line)
            payload = {
                "model": req["model"],
                "query": req["query"],
                "documents": req["documents"],
            }
            n += 1
            print(f">> Sending rerank request {n} to {api_url} ...")
            t0 = time.time()
            resp = requests.post(api_url, json=payload, timeout=300)
            elapsed = time.time() - t0
            resp.raise_for_status()
            fout.write(json.dumps(req, ensure_ascii=False) + "\n")
            fout.write(json.dumps(resp.json(), ensure_ascii=False) + "\n")
            fout.write(f"API_total_time: {elapsed:.6f}\n")

    print(f"All requests completed. Responses saved to: {output_file}")


if __name__ == "__main__":
    main()
