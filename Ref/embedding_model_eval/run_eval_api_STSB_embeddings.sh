#!/bin/bash

# 配置区域
TEST_RESULT_PATH="../../model-eval-storage/bge-m3/project-1/project-1-test-C-METB-STSB-API-ALL.txt"

# 执行评估（单次执行）
python Eval_API_STSB_embeddings.py "$TEST_RESULT_PATH"
