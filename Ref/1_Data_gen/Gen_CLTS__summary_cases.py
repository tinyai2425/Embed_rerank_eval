import sys, os
import json
import re
from types import SimpleNamespace

sys.path.append(os.path.abspath("Function"))
import CLTS
import test_utils

# === 参数校验 ===
if len(sys.argv) < 2 or len(sys.argv) > 3:  # 最少需要2个参数，最多3个参数
    print("用法: python Gen_CLTS_cases.py <model_config.json> n_cases")
    sys.exit(1)

config_path = sys.argv[1]
n_cases = 1200
if len(sys.argv) == 3:
    try:
        n_cases = int(sys.argv[2])
    except ValueError:
        raise ValueError("n_cases 必须是整数")

sel_prompt = CLTS.summary_prompt

if not os.path.exists(config_path):
    raise FileNotFoundError(f"找不到配置文件: {config_path}")

# === 模型配置读取 ===
with open(config_path, "r", encoding="utf-8") as f:
    config_dict = json.load(f)
mp = SimpleNamespace(**config_dict)

# === 常量定义 ===
MAX_OUTPUT_TOKEN_LENGTH = 5000

# === 项目路径设置 ===
PROJECT_BASE = f"../../model-eval-storage/{mp.model_name}"
PROJECT_PREFIX = "project"

def get_unique_subdir(base_dir, prefix):
    counter = 1
    while os.path.exists(f"{base_dir}/{prefix}-{counter}"):
        counter += 1
    return f"{prefix}-{counter}"

PROJECT_NAME = get_unique_subdir(PROJECT_BASE, PROJECT_PREFIX)
PROJECT_PATH = f"{PROJECT_BASE}/{PROJECT_NAME}"
os.makedirs(PROJECT_PATH, exist_ok=True)

print(f"PROJECT_NAME {PROJECT_NAME}")
print(f"Full path to project dir: {PROJECT_PATH}")

# === 配置绑定 ===
test_utils.set_model_config(mp)

# ===== OMC 模式 =====
CLTS.create_test = test_utils.create_test
CLTS.process_prompt = test_utils.process_prompt

test_path_omc = f"{PROJECT_PATH}/{PROJECT_NAME}-TextSummary-OMC-{n_cases}.json"
with open(test_path_omc, 'w', encoding='utf-8') as f:
    summary_cases = CLTS.generate_CLTS_tests(PROJECT_NAME, sel_prompt, n_cases=n_cases, max_output_token_length= MAX_OUTPUT_TOKEN_LENGTH, is_gpu_base_model=0)
    json.dump(summary_cases, f, indent=4, ensure_ascii=False)
    print(f"{len(summary_cases)} OMC test cases saved to {test_path_omc}")

# ===== GPU 模式 =====
test_path_gpu = f"{PROJECT_PATH}/{PROJECT_NAME}-TextSummary-GPU-{n_cases}.json"
with open(test_path_gpu, 'w', encoding='utf-8') as f:
    summary_cases = CLTS.generate_CLTS_tests(PROJECT_NAME, sel_prompt, n_cases=n_cases, max_output_token_length= MAX_OUTPUT_TOKEN_LENGTH, is_gpu_base_model=1)
    json.dump(summary_cases, f, indent=4, ensure_ascii=False)
    print(f"{len(summary_cases)} GPU test cases saved to {test_path_gpu}")

# ===== API 模式 =====
CLTS.create_test = test_utils.create_API_test

test_path_api = f"{PROJECT_PATH}/{PROJECT_NAME}-TextSummary-API-{n_cases}.jsonl"
with open(test_path_api, 'w', encoding='utf-8') as f:
    summary_cases = CLTS.generate_CLTS_tests(PROJECT_NAME, sel_prompt, n_cases=n_cases, max_output_token_length= MAX_OUTPUT_TOKEN_LENGTH, is_gpu_base_model=1)
    for case in summary_cases:
        json.dump(case, f, ensure_ascii=False)
        f.write('\n')
    print(f"{len(summary_cases)} API test cases saved to {test_path_api}")
