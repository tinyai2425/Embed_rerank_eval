#!/bin/bash

# 固定参数配置（按需修改）
INPUT_JSON_PATH="../../model-eval-storage/Qwen3-Reranker-0.6B/project-2/project-2-RERANK-GPU-MAXQ100.json"  # 修改为您的输入文件路径
VLLM_IP="10.93.64.30"                # 修改为您的vLLM服务器IP
VLLM_PORT=8897                          # 修改为您的vLLM服务器端口
VLLM_MODEL_ID="qwen3-reranker"          # 修改为您的模型ID

# 构造执行命令
CMD="python GPU_run_rerank.py \"$INPUT_JSON_PATH\" \"$VLLM_IP\" \"$VLLM_PORT\" \"$VLLM_MODEL_ID\""

# 打印将要执行的命令
echo "Executing: $CMD"

# 执行命令
eval $CMD