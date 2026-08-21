#!/bin/bash
# Edit the variables below, then run from 2_GPU/

INPUT_JSON_PATH="../../model-eval-storage/Qwen3-Embedding-0.6B/project-1/project-1-perf-embedding-wiki-GPU-ALL.json"
VLLM_IP="10.93.64.30"
VLLM_PORT=8896
VLLM_MODEL_ID="Qwen3-Embedding-0.6B"

echo "[RUN] python Eval_GPU_embed_wiki.py \"$INPUT_JSON_PATH\" \"$VLLM_IP\" \"$VLLM_PORT\" \"$VLLM_MODEL_ID\""
python Eval_GPU_embed_wiki.py "$INPUT_JSON_PATH" "$VLLM_IP" "$VLLM_PORT" "$VLLM_MODEL_ID"
