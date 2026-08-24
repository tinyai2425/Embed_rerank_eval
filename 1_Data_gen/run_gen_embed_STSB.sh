#!/bin/bash
# Generate STSB embedding cases (GPU JSON + API JSONL).
# Run from 1_Data_gen/. 0 MAX_CASES = full dataset.

CONFIG="configs/embed.example.json"
MAX_CASES=0

echo "[RUN] python Gen_embed_STSB_cases.py \"$CONFIG\" \"$MAX_CASES\""
python Gen_embed_STSB_cases.py "$CONFIG" "$MAX_CASES"
