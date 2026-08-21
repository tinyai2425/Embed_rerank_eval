#!/bin/bash
# run_eval_gpu_perf_simple.sh

INPUT_JSON_PATH="../../model-eval-storage/Qwen3-Embedding-0.6B/project-4/project-4-perf-embedding-wiki-GPU-ALL.json"
VLLM_IP="10.93.64.30"
VLLM_PORT=8896
VLLM_MODEL_ID="Qwen3-Embedding-0.6B"

echo "[RUN] python Eval_GPU_perf_simple.py \"$INPUT_JSON_PATH\" \"$VLLM_IP\" \"$VLLM_PORT\" \"$VLLM_MODEL_ID\""
python Eval_GPU_perf_simple.py "$INPUT_JSON_PATH" "$VLLM_IP" "$VLLM_PORT" "$VLLM_MODEL_ID"
