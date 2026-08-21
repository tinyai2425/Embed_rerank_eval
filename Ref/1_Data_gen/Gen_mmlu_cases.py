# Gen_mmlu_cases.py
import sys, os
import json
import re
from types import SimpleNamespace

# === 把 Function/ 加到 sys.path，兼容你现有工程结构 ===
sys.path.append(os.path.abspath("Function"))

import mmlu as mml
import rand_perf_test as grpt
import test_utils

# === 参数校验 ===
# 用法：python Gen_mmlu_cases.py <model_config.json> <MAX_CASES_PER_SUBJECT> [slow]
if len(sys.argv) < 3 or len(sys.argv) > 4:
    print("用法: python Gen_mmlu_cases.py <model_config.json> <MAX_CASES_PER_SUBJECT> [slow]")
    sys.exit(1)

config_path = sys.argv[1]
try:
    MAX_CASES_PER_SUBJECT = int(sys.argv[2])
except ValueError:
    raise ValueError("MAX_CASES_PER_SUBJECT 必须是整数")

# 判断是否传入了 'slow' 参数
if len(sys.argv) == 4 and sys.argv[3] == "slow":
    sel_prompt = mml.slow_prompt
else:
    sel_prompt = mml.fast_prompt

if not os.path.exists(config_path):
    raise FileNotFoundError(f"找不到配置文件: {config_path}")

# === 模型配置读取 ===
with open(config_path, "r", encoding="utf-8") as f:
    config_dict = json.load(f)
mp = SimpleNamespace(**config_dict)

# === 常量定义（与你现有保持一致） ===
MAX_OUTPUT_TOKEN_LENGTH = 5000

# === 性能用例固定参数（按你要求固定 10 个）===
NUM_PERF_TEST_CASES = 10
PROMPT_TOKEN_LENGTH = 1024
PERF_OUTPUT_TOKEN_LENGTH = 50

# 若你的 perf 脚本需要 tokenizer 路径（与 CEVAL 保持一致）
if hasattr(mp, "TOKENIZER_CONFIG_PATH"):
    grpt.TOKENIZER_CONFIG_PATH = mp.TOKENIZER_CONFIG_PATH

# === 模型名称合法性检查 ===
if not re.fullmatch(r"[A-Za-z0-9_\-]+", mp.model_name):
    raise ValueError(f"非法的 MODEL_NAME: {mp.model_name}，只能包含字母、数字、- 和 _")

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

# === 配置绑定（与你现有 CEVAL 流水线保持一致） ===
test_utils.set_model_config(mp)

# ===== OMC 模式 =====
mml.create_test = test_utils.create_test
grpt.create_test = test_utils.create_test
mml.process_prompt = test_utils.process_prompt
grpt.process_prompt = test_utils.process_prompt

test_path_omc = f"{PROJECT_PATH}/{PROJECT_NAME}-MMLU-OMC-{MAX_CASES_PER_SUBJECT}-{NUM_PERF_TEST_CASES}.json"
with open(test_path_omc, 'w', encoding='utf-8') as f:
    acc = mml.generate_mmlu_tests(
        PROJECT_NAME,
        sel_prompt,
        MAX_CASES_PER_SUBJECT,
        max_output_token_length=MAX_OUTPUT_TOKEN_LENGTH,
        is_gpu_base_model=0
    )
    perf = grpt.generate_perf_tests(
        PROJECT_NAME,
        PROMPT_TOKEN_LENGTH,
        PERF_OUTPUT_TOKEN_LENGTH,
        NUM_PERF_TEST_CASES,
        is_gpu_base_model=0
    )
    all_cases = acc + perf
    json.dump(all_cases, f, indent=4, ensure_ascii=False)
    print(f"{len(all_cases)} OMC test cases saved to {test_path_omc}")

# ===== GPU 模式 =====
test_path_gpu = f"{PROJECT_PATH}/{PROJECT_NAME}-MMLU-GPU-{MAX_CASES_PER_SUBJECT}-{NUM_PERF_TEST_CASES}.json"
with open(test_path_gpu, 'w', encoding='utf-8') as f:
    acc = mml.generate_mmlu_tests(
        PROJECT_NAME,
        sel_prompt,
        MAX_CASES_PER_SUBJECT,
        max_output_token_length=MAX_OUTPUT_TOKEN_LENGTH,
        is_gpu_base_model=1
    )
    perf = grpt.generate_perf_tests(
        PROJECT_NAME,
        PROMPT_TOKEN_LENGTH,
        PERF_OUTPUT_TOKEN_LENGTH,
        NUM_PERF_TEST_CASES,
        is_gpu_base_model=1
    )
    all_cases = acc + perf
    json.dump(all_cases, f, indent=4, ensure_ascii=False)
    print(f"{len(all_cases)} GPU test cases saved to {test_path_gpu}")

# ===== API 模式 =====
mml.create_test = test_utils.create_API_test
grpt.create_test = test_utils.create_API_test
mml.process_prompt = test_utils.process_prompt
grpt.process_prompt = test_utils.process_prompt

test_path_api = f"{PROJECT_PATH}/{PROJECT_NAME}-MMLU-API-{MAX_CASES_PER_SUBJECT}-{NUM_PERF_TEST_CASES}.jsonl"
with open(test_path_api, 'w', encoding='utf-8') as f:
    acc = mml.generate_mmlu_tests(
        PROJECT_NAME,
        sel_prompt,
        MAX_CASES_PER_SUBJECT,
        max_output_token_length=MAX_OUTPUT_TOKEN_LENGTH,
        is_gpu_base_model=1
    )
    perf = grpt.generate_perf_tests(
        PROJECT_NAME,
        PROMPT_TOKEN_LENGTH,
        PERF_OUTPUT_TOKEN_LENGTH,
        NUM_PERF_TEST_CASES,
        is_gpu_base_model=1
    )
    all_cases = acc + perf
    for case in all_cases:
        json.dump(case, f, ensure_ascii=False)
        f.write('\n')
    print(f"{len(all_cases)} API test cases saved to {test_path_api}")
