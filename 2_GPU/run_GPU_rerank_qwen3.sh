#!/bin/bash
# Edit the variables below, then run from 2_GPU/
# Inference + evaluation_results.md are produced in one pass.

INPUT_JSON_PATH="../model-eval-storage/qwen3-rerank-0.6B/project-1/project-1-RERANK-GPU-MAXQ100.json"
VLLM_IP="10.93.64.30"
VLLM_PORT=8897
VLLM_MODEL_ID="qwen3-reranker"
K_VALUES="[5,10]"

CMD="python GPU_run_rerank_qwen3.py \"$INPUT_JSON_PATH\" \"$VLLM_IP\" \"$VLLM_PORT\" \"$VLLM_MODEL_ID\" \"$K_VALUES\""
echo "Executing: $CMD"
eval $CMD
