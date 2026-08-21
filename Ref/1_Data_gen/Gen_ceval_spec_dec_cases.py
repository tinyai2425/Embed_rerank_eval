import sys, os
import json
import re
from types import SimpleNamespace
from transformers import AutoTokenizer

sys.path.append(os.path.abspath("Function"))
import spec_ceval_perf_test as gscpt
import test_utils
import ceval as cev

# === 命令行参数读取 ===
if len(sys.argv) < 3 or len(sys.argv) > 4:  # 最少需要4个参数，最多5个参数
    print("用法: python Gen_ceval_spec_test.py <model_config.json> <MAX_PERF_TEST_CASES_PER_VERTICAL> [slow]")
    sys.exit(1)

config_path = sys.argv[1]
try:
    MAX_PERF_TEST_CASES_PER_VERTICAL = int(sys.argv[2])
except ValueError:
    raise ValueError("MAX_PERF_TEST_CASES_PER_VERTICAL 必须是整数")

# 判断是否传入了'slow'参数
if len(sys.argv) == 4 and sys.argv[3] == "slow":
    sel_prompt = cev.slow_prompt
else:
    sel_prompt = cev.fast_prompt

if not os.path.exists(config_path):
    raise FileNotFoundError(f"找不到配置文件: {config_path}")

# === 加载模型配置 ===
with open(config_path, "r", encoding="utf-8") as f:
    config_dict = json.load(f)

mp = SimpleNamespace(**config_dict)

# === 固定参数 ===
PROMPT_TOKEN_LENGTH = 1024
PERF_OUTPUT_TOKEN_LENGTH = 50

# === 检查合法模型名 ===
if not re.fullmatch(r"[A-Za-z0-9_\-]+", mp.model_name):
    raise ValueError(f"非法的 MODEL_NAME: {mp.model_name}，只能包含字母、数字、- 和 _")

# === 路径与工程名 ===
PROJECT_BASE = f"../../model-eval-storage/{mp.model_name}"
PROJECT_PREFIX = "project"
gscpt.TOKENIZER_CONFIG_PATH = mp.TOKENIZER_CONFIG_PATH

def get_unique_subdir(base_dir, prefix):
    counter = 1
    while os.path.exists(f"{base_dir}/{prefix}-{counter}"):
        counter += 1
    return f"{prefix}-{counter}"

PROJECT_NAME = get_unique_subdir(PROJECT_BASE, PROJECT_PREFIX)
print(f"PROJECT_NAME {PROJECT_NAME}")
print(f"Full path to project dir: {PROJECT_BASE}/{PROJECT_NAME}")

# === 配置绑定 ===
test_utils.set_model_config(mp)

# 用于 OMC / GPU：标准 prompt 构造函数
gscpt.create_test = test_utils.create_test
gscpt.process_prompt = test_utils.process_prompt

# === 创建输出目录 ===
output_dir = f"{PROJECT_BASE}/{PROJECT_NAME}"
os.makedirs(output_dir, exist_ok=True)

# === OMC 模式测试用例生成 ===
path_omc = f"{output_dir}/{PROJECT_NAME}-CEVAL-Spec-perf-OMC-{MAX_PERF_TEST_CASES_PER_VERTICAL}.json"
with open(path_omc, 'w', encoding='utf-8') as f:
    perf_cases = gscpt.generate_speculative_decoding_perf_tests(
        PROJECT_NAME,
        sel_prompt,
        MAX_PERF_TEST_CASES_PER_VERTICAL,
        PROMPT_TOKEN_LENGTH,
        PERF_OUTPUT_TOKEN_LENGTH,
        is_gpu_base_model=0
    )
    json.dump(perf_cases, f, indent=4, ensure_ascii=False)
    print(f"{len(perf_cases)} OMC test cases saved to {path_omc}")

# === GPU 模式测试用例生成 ===
path_gpu = f"{output_dir}/{PROJECT_NAME}-CEVAL-Spec-perf-GPU-{MAX_PERF_TEST_CASES_PER_VERTICAL}.json"
with open(path_gpu, 'w', encoding='utf-8') as f:
    perf_cases = gscpt.generate_speculative_decoding_perf_tests(
        PROJECT_NAME,
        sel_prompt,
        MAX_PERF_TEST_CASES_PER_VERTICAL,
        PROMPT_TOKEN_LENGTH,
        PERF_OUTPUT_TOKEN_LENGTH,
        is_gpu_base_model=1
    )
    json.dump(perf_cases, f, indent=4, ensure_ascii=False)
    print(f"{len(perf_cases)} GPU test cases saved to {path_gpu}")

# === API 模式测试用例生成（create_test 替换为 API 风格）===
gscpt.create_test = test_utils.create_API_test  # 切换为 API 结构
path_api = f"{output_dir}/{PROJECT_NAME}-CEVAL-Spec-perf-API-{MAX_PERF_TEST_CASES_PER_VERTICAL}.jsonl"
with open(path_api, 'w', encoding='utf-8') as f:
    perf_cases = gscpt.generate_speculative_decoding_perf_tests(
        PROJECT_NAME,
        sel_prompt,
        MAX_PERF_TEST_CASES_PER_VERTICAL,
        PROMPT_TOKEN_LENGTH,
        PERF_OUTPUT_TOKEN_LENGTH,
        is_gpu_base_model=1
    )
    for case in perf_cases:
        json.dump(case, f, ensure_ascii=False)
        f.write("\n")
    print(f"{len(perf_cases)} API test cases saved to {path_api}")

print("[ALL DONE]")
