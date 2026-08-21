#!/bin/bash

# 固定参数配置（按需修改）
TOKENIZER_PATH="../Model_file/Qwen3-1.7B"
INPUT_JSON_PATH="../../model-eval-storage/Qwen3-17B/project-4/project-4-CEVAL-GPU-100-30.json"
VLLM_IP="10.93.64.30"
VLLM_PORT=8894
VLLM_MODEL_ID="qwen3_17b"
VERSION_FLAG=10  # 会生成 version_nums = [1, 2, 3]
SHOW_DETAIL="true"  # ✅ 改为 "none" 表示不展示图表

# 构造执行命令
CMD="python Eval_GPU_results.py \"$TOKENIZER_PATH\" \"$INPUT_JSON_PATH\" \"$VLLM_IP\" \"$VLLM_PORT\" \"$VLLM_MODEL_ID\" \"$VERSION_FLAG\""

# 如果开启详细图表分析，则添加 --show_detail 参数
if [ "$SHOW_DETAIL" = "true" ]; then
    CMD="$CMD --show_detail"
fi

# 执行命令
eval $CMD