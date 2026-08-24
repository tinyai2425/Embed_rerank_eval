#!/bin/bash
# Generate rerank cases (GPU JSON + API JSONL).
# Run from 1_Data_gen/.
# MAX_QUERY: number of queries to take from the dataset.
# MAX_PROMPT_LEN: keep only pairs whose assembled GPU prompt
#   (system + instruct + query + document + chat template) is <= this many tokens.
#   0 = no length filter. Typical values: 1024 or 8192.
# Length filter needs TOKENIZER_CONFIG_PATH in the config json.

CONFIG="configs/rerank_qwen3.example.json"
MAX_QUERY=100
MAX_PROMPT_LEN=0
# MAX_PROMPT_LEN=1024
# MAX_PROMPT_LEN=8192

echo "[RUN] python Gen_rerank_cases.py \"$CONFIG\" \"$MAX_QUERY\" \"$MAX_PROMPT_LEN\""
python Gen_rerank_cases.py "$CONFIG" "$MAX_QUERY" "$MAX_PROMPT_LEN"
