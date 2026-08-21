#!/bin/bash

# 配置区域
TOKENIZER_PATH="../Model_file/Qwen3-4B"
TEST_RESULT_PATH="../../model-eval-storage/Qwen3-4B/project-2/qwen3-4b-project-1-CEVAL-API-100-30.txt"
SHOW_DETAIL=true  # true或false

# 执行评估（单次执行）
if [ "$SHOW_DETAIL" = true ]; then
    python Eval_API_results.py "$TOKENIZER_PATH" "$TEST_RESULT_PATH" --show_detail
else
    python Eval_API_results.py "$TOKENIZER_PATH" "$TEST_RESULT_PATH"
fi