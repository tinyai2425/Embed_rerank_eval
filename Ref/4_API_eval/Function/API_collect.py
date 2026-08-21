import os
import json
import pandas as pd
import re
from typing import List, Dict, Union
from transformers import AutoTokenizer

# === 全局变量 ===
TOKENIZER_CONFIG_PATH = None
tokenizer = None

def init_tokenizer(path: str):
    """初始化全局 tokenizer"""
    global TOKENIZER_CONFIG_PATH, tokenizer
    TOKENIZER_CONFIG_PATH = path
    tokenizer = AutoTokenizer.from_pretrained(path)

def parse_llm_api_results(input_path: str) -> pd.DataFrame:
    """
    解析 API 日志，每 3 行为一组：
    第 1 行：请求 JSON
    第 2 行：响应 JSON
    第 3 行：API 总时长文本（单位：秒）
    """
    results: List[Dict[str, Union[str, float, int]]] = []
    lines = [line.strip() for line in iter_lines_safely(input_path) if line.strip()]
    
    assert len(lines) % 3 == 0, f"输入文件 {input_path} 行数不是3的倍数，请检查格式。"

    for i in range(0, len(lines), 3):
        req_line = json.loads(lines[i])
        resp_line = json.loads(lines[i+1])
        api_time_line = lines[i+2]

        case = {}

        # === 基本字段 ===
        case["testCaseName"] = req_line.get("testCaseName", "")
        case_name = case["testCaseName"]
        matched = TEST_CASE_NAME_PATTERN.match(case_name)
        if matched:
            case["project_name"] = matched.group("project_name")
            case["flavor"] = matched.group("flavor")
            case["vertical"] = matched.group("vertical")

        case["prompt"] = req_line["messages"][0]["content"]
        case["expect"] = req_line.get("expect", "")
        case["response"] = resp_line["message"]["content"]

        # === 模型设置参数 ===
        options = req_line.get("options", {})
        case["temperature"] = options.get("temperature")
        case["top_k"] = options.get("top_k")
        case["top_p"] = options.get("top_p")
        case["repetitionPenalty"] = options.get("repeat_penalty")
        case["max gen tokens"] = options.get("num_predict")

        # === Token 长度统计 ===
        case["prompt_token_len"] = len(tokenizer(case["prompt"])["input_ids"])
        case["response_token_len"] = len(tokenizer(case["response"])["input_ids"])

        # === 时间处理 ===
        decode_ns = resp_line.get("eval_duration", 0)
        api_total_time_str = api_time_line.replace("API_total_time:", "").strip()
        api_total_time = float(api_total_time_str)

        case["decode_time"] = decode_ns / 1e9  # 纳秒 → 秒
        case["first_token_time"] = api_total_time - case["decode_time"]
        case["total_time"] = api_total_time

        # 补充指标计算
        case["get_ans"] = GET_answer(case["response"])
        case["repeat"] = calculate_repetition_rate(case["response"])
        case["entropy"] = calculate_token_entropy(case["response"])     
        case["correct"] = verify_answer(case["expect"], case["response"])

        results.append(case)

    print(f"{len(results)} tests processed in results {input_path}")
    return pd.DataFrame(results)

def evaluate_llm_api_results(df_results: pd.DataFrame) -> Dict[str, float]:
    """返回与旧 evaluate_hiai_test_results() 相同结构的评估指标"""
    return {
        "prompt_token_mean": df_results["prompt_token_len"].mean(),
        "prompt_token_std": df_results["prompt_token_len"].std(),
        "response_token_mean": df_results["response_token_len"].mean(),
        "response_token_std": df_results["response_token_len"].std(),
        "TTFT_sec": df_results["first_token_time"].mean(),
        "TPS": df_results["response_token_len"].sum() / df_results["decode_time"].sum(),
        "accuracy": df_results["correct"].mean() if "correct" in df_results else None
    }

def save_jsonl_result(df_results, output_dir: str):

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "API_result.jsonl")

    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df_results.iterrows():
            tps = row["response_token_len"] / row["decode_time"]

            obj = {
                "testCaseName": row["testCaseName"],
                "prompt": row["prompt"],
                "response": row["response"],
                "prompt_token_len": row["prompt_token_len"],
                "response_token_len": row["response_token_len"],
                "first_token_time": row["first_token_time"],
                "decode_time": row["decode_time"],
                "TPS": tps,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[SAVE] API jsonl saved to: {out_path}")
    return out_path