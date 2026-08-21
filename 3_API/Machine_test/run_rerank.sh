#!/bin/sh
# Usage: sh run_rerank.sh <input.jsonl> <output.txt>
# POST http://localhost:11434/v1/reranks
# Prefer ../Collect_API_rerank.py (sends only model/query/documents).

if [ "$#" -ne 2 ]; then
    echo "Usage: sh run_rerank.sh <input.jsonl> <output.txt>"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"
API_URL="http://localhost:11434/v1/reranks"

echo "$INPUT_FILE" | grep -qE '\.jsonl$' || { echo "Input must end with .jsonl"; exit 1; }
echo "$OUTPUT_FILE" | grep -qE '\.txt$'   || { echo "Output must end with .txt";  exit 1; }
[ -f "$INPUT_FILE" ] || { echo "Input file not found: $INPUT_FILE"; exit 1; }

TMP_RESP="tmp_response.json"
: > "$OUTPUT_FILE"
counter=1

while IFS= read -r line || [ -n "$line" ]; do
    [ -z "$line" ] && continue
    printf "%s\n" "$line" >> "$OUTPUT_FILE"
    echo ">> Sending rerank request $counter to $API_URL ..."
    response_time=$(curl -sS -w "%{time_total}" -o "$TMP_RESP" \
        -H "Content-Type: application/json" \
        -X POST "$API_URL" \
        -d "$line")
    cat "$TMP_RESP" >> "$OUTPUT_FILE"
    printf "\nAPI_total_time: %s\n" "$response_time" >> "$OUTPUT_FILE"
    counter=$((counter + 1))
done < "$INPUT_FILE"

rm -f "$TMP_RESP"
echo "All requests completed. Responses saved to: $OUTPUT_FILE"
