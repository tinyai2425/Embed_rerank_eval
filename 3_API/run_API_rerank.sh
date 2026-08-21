#!/bin/bash
TEST_RESULT_PATH="../../model-eval-storage/qwen3-rerank-0.6B/project-1/project-1-RERANK-API-MAXQ100.txt"
python Eval_API_rerank.py "$TEST_RESULT_PATH" "[5,10]"
