#!/bin/bash
TEST_RESULT_PATH="../../model-eval-storage/Qwen3-Embedding-0.6B/project-1/project-1-test-C-METB-STSB-API-ALL.txt"
python Eval_API_embed_STSB.py "$TEST_RESULT_PATH"
