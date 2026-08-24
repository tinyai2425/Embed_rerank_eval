#!/bin/bash
# Generate wiki embedding perf cases (GPU JSON + API JSONL).
# Run from 1_Data_gen/.
# MAX_CASES=0 means full dataset.
# TARGET_LENGTHS is comma-separated token lengths; no need to edit the Python file.

CONFIG="configs/embed.example.json"
MAX_CASES=0
TARGET_LENGTHS="1000"
# TARGET_LENGTHS="512,1024,2048"

echo "[RUN] python Gen_embed_wiki_cases.py \"$CONFIG\" \"$MAX_CASES\" \"$TARGET_LENGTHS\""
python Gen_embed_wiki_cases.py "$CONFIG" "$MAX_CASES" "$TARGET_LENGTHS"
