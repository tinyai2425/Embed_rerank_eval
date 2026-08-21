#!/bin/sh
# 用法: sh run_embed.sh <input.jsonl> <output.txt>
# 固定使用: http://localhost:11434/api/embed

# === Args ===
if [ "$#" -ne 2 ]; then
    echo "Usage: sh run_embed.sh <input.jsonl> <output.txt>"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"
API_URL="http://localhost:11434/api/embed"

# === Checks ===
echo "$INPUT_FILE" | grep -qE '\.jsonl$' || { echo "Input must end with .jsonl"; exit 1; }
echo "$OUTPUT_FILE" | grep -qE '\.txt$'   || { echo "Output must end with .txt";  exit 1; }
[ -f "$INPUT_FILE" ] || { echo "Input file not found: $INPUT_FILE"; exit 1; }

# === Temps ===
TMP_RESP="tmp_response.json"
: > "$OUTPUT_FILE"

counter=1

# === Main loop ===
while IFS= read -r line || [ -n "$line" ]; do
    [ -z "$line" ] && continue

    # 1) 原始用例行
    printf "%s\n" "$line" >> "$OUTPUT_FILE"

    echo ">> Sending embed request $counter to $API_URL ..."

    # 2) 直接原样发送，记录总耗时
    response_time=$(curl -sS -w "%{time_total}" -o "$TMP_RESP" \
        -H "Content-Type: application/json" \
        -X POST "$API_URL" \
        -d "$line")

    # 3) 响应 + 耗时
    cat "$TMP_RESP" >> "$OUTPUT_FILE"
    printf "\nAPI_total_time: %s\n" "$response_time" >> "$OUTPUT_FILE"

    counter=$((counter + 1))
done < "$INPUT_FILE"

rm -f "$TMP_RESP"
echo "All requests completed. Responses saved to: $OUTPUT_FILE"
