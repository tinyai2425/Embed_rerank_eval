#!/bin/sh

# === Argument check ===
if [ "$#" -ne 2 ]; then
    echo "Usage: sh run_chat.sh <input.jsonl> <output.txt>"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"

# Check input extension
if ! echo "$INPUT_FILE" | grep -qE '\.jsonl$'; then
    echo "Input file must have .jsonl extension: $INPUT_FILE"
    exit 1
fi

# Check output extension
if ! echo "$OUTPUT_FILE" | grep -qE '\.txt$'; then
    echo "Output file must have .txt extension: $OUTPUT_FILE"
    exit 1
fi

# Check file existence
if [ ! -f "$INPUT_FILE" ]; then
    echo "Input file does not exist: $INPUT_FILE"
    exit 1
fi

# === Temp variables and initialization ===
TMP_REQ="tmp_req.json"
TMP_RESP="tmp_response.json"

: > "$OUTPUT_FILE"

counter=1

# === Process each line (JSON) ===
while IFS= read -r line || [ -n "$line" ]; do
    if [ -z "$line" ]; then
        continue
    fi

    printf "%s" "$line" > "$TMP_REQ"
    echo ">> Sending request $counter..."

    response_time=$(curl -s -w "%{time_total}" -o "$TMP_RESP" http://localhost:11434/api/chat \
        -H "Content-Type: application/json" \
        -d @"$TMP_REQ")

    cat "$TMP_REQ" >> "$OUTPUT_FILE"
    echo >> "$OUTPUT_FILE"

    cat "$TMP_RESP" >> "$OUTPUT_FILE"

    printf "API_total_time: %s\n" "$response_time" >> "$OUTPUT_FILE"

    counter=$((counter + 1))
done < "$INPUT_FILE"

rm -f "$TMP_REQ" "$TMP_RESP"

echo "All requests completed. Responses saved to: $OUTPUT_FILE"