#!/bin/bash
# run_eval_gpu_embeddings.sh

INPUT_JSON_PATH="../../model-eval-storage/Qwen3-Embedding-0.6B/project-1/project-1-test-C-METB-STSB-GPU-100.json"
VLLM_IP="10.93.64.30"
VLLM_PORT=8896
VLLM_MODEL_ID="Qwen3-Embedding-0.6B"
VERSION_FLAG=3

CMD="python Eval_GPU_STSB_embeddings.py \"$INPUT_JSON_PATH\" \"$VLLM_IP\" \"$VLLM_PORT\" \"$VLLM_MODEL_ID\" \"$VERSION_FLAG\""
echo "[RUN] $CMD"
eval $CMD
