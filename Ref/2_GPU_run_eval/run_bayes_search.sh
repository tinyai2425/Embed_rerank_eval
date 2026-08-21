#!/usr/bin/env bash
# run_bayes_search.sh

set -euo pipefail

# === 基本配置（按需修改） ===
INPUT_JSON_PATH="../../model-eval-storage/DeepSeek-R1-Distill-Qwen-15B/project-1/project-1-CEVAL-GPU-100-0.json"
VLLM_IP="10.93.64.30"
VLLM_PORT=8897
VLLM_MODEL_ID="ds15b"

# === 搜索规模 ===
N_CALLS=200
N_INIT=12
SEED=42

# === 重试策略 ===
RETRY_TIMES=2
RETRY_WAIT=3

# === 参数空间（可用固定值或范围：a:b:step / 用逗号列举离散值） ===
# 例如：TEMP="0.6:1.2:0.05"   或 TEMP="1.0"
TEMP="0.6"
TOPP="0.95"
TOPK="1:20:1"                # 固定 top_k=8
REPP="0.1:2.0:0.1"

# === 是否保存每次评估的明细 ===
SAVE_EACH="--save_each_eval"   # 留空则不保存

CMD="python Eval_Bayesian_par_search.py \
  \"$INPUT_JSON_PATH\" \"$VLLM_IP\" $VLLM_PORT \"$VLLM_MODEL_ID\" \
  --n_calls $N_CALLS --n_init $N_INIT --seed $SEED \
  --retry_times $RETRY_TIMES --retry_wait $RETRY_WAIT \
  --temp \"$TEMP\" --topp \"$TOPP\" --topk \"$TOPK\" --repp \"$REPP\" \
  $SAVE_EACH"

echo "[RUN] $CMD"
eval $CMD
