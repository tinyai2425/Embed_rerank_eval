# Gen_specbench_cases.py
import sys, os
import json
import re
from types import SimpleNamespace

sys.path.append(os.path.abspath("Function"))

import specbench as spb
import test_utils_systempt as test_utils  # ✅ 换成新的 utils

# === 参数校验 ===
# 用法：python Gen_specbench_cases.py <model_config.json> <MAX_CASES_PER_CATEGORY>
if len(sys.argv) != 3:
    print("用法: python Gen_specbench_cases.py <model_config.json> <MAX_CASES_PER_CATEGORY>")
    sys.exit(1)

config_path = sys.argv[1]
try:
    MAX_CASES_PER_CATEGORY = int(sys.argv[2])
except ValueError:
    raise ValueError("MAX_CASES_PER_CATEGORY 必须是整数")

if not os.path.exists(config_path):
    raise FileNotFoundError(f"找不到配置文件: {config_path}")

# === 模型配置读取 ===
with open(config_path, "r", encoding="utf-8") as f:
    config_dict = json.load(f)
mp = SimpleNamespace(**config_dict)

# === 常量定义 ===
MAX_OUTPUT_TOKEN_LENGTH = 5000

# === SpecBench：不包含额外性能测试用例 ===
NUM_PERF_TEST_CASES = 0

# === 模型名称合法性检查 ===
if not re.fullmatch(r"[A-Za-z0-9_\-]+", mp.model_name):
    raise ValueError(f"非法的 MODEL_NAME: {mp.model_name}")

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
spb.create_test = test_utils.create_test
spb.process_prompt = test_utils.process_prompt
spb.create_API_test = test_utils.create_API_test  # ✅ 也一并绑定，specbench.py 会用到

test_path_omc = (
    f"{PROJECT_PATH}/"
    f"{PROJECT_NAME}-SPEC-OMC-{MAX_CASES_PER_CATEGORY}-0.json"
)

with open(test_path_omc, "w", encoding="utf-8") as f:
    all_cases = spb.generate_specbench_tests(
        project_name=PROJECT_NAME,
        n_per_vertical=MAX_CASES_PER_CATEGORY,
        max_output_token_length=MAX_OUTPUT_TOKEN_LENGTH,
        is_gpu_base_model=0,
    )
    json.dump(all_cases, f, indent=4, ensure_ascii=False)

print(f"{len(all_cases)} OMC test cases saved to {test_path_omc}")

# ===== API 模式 =====
spb.create_test = test_utils.create_test
spb.process_prompt = test_utils.process_prompt
spb.create_API_test = test_utils.create_API_test

test_path_api = (
    f"{PROJECT_PATH}/"
    f"{PROJECT_NAME}-SPEC-API-{MAX_CASES_PER_CATEGORY}-0.jsonl"
)

with open(test_path_api, "w", encoding="utf-8") as f:
    all_cases = spb.generate_specbench_tests(
        project_name=PROJECT_NAME,
        n_per_vertical=MAX_CASES_PER_CATEGORY,
        max_output_token_length=MAX_OUTPUT_TOKEN_LENGTH,
        is_gpu_base_model=1,
    )
    for case in all_cases:
        json.dump(case, f, ensure_ascii=False)
        f.write("\n")

print(f"{len(all_cases)} API test cases saved to {test_path_api}")
