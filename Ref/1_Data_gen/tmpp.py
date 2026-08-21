import sys, os
import json
import re
from types import SimpleNamespace
from transformers import AutoTokenizer

sys.path.append(os.path.abspath("Function"))
import ceval as cev
import rand_perf_test as grpt
import test_utils

# === 参数校验 ===
if len(sys.argv) < 4 or len(sys.argv) > 5:  # 最少需要4个参数，最多5个参数
    print("用法: python Gen_ceval_cases.py <model_config.json> <MAX_CASES_PER_VERTICAL> <NUM_PERF_TEST_CASES> [slow]")
    sys.exit(1)

config_path = sys.argv[1]
try:
    MAX_CASES_PER_VERTICAL = int(sys.argv[2])
    NUM_PERF_TEST_CASES = int(sys.argv[3])
except ValueError:
    raise ValueError("MAX_CASES_PER_VERTICAL 和 NUM_PERF_TEST_CASES 必须是整数")

# 判断是否传入了'slow'参数
if len(sys.argv) == 5 and sys.argv[4] == "slow":
    sel_prompt = cev.slow_prompt
else:
    sel_prompt = cev.fast_prompt

if not os.path.exists(config_path):
    raise FileNotFoundError(f"找不到配置文件: {config_path}")

# === 模型配置读取 ===
with open(config_path, "r", encoding="utf-8") as f:
    config_dict = json.load(f)
mp = SimpleNamespace(**config_dict)

# === 常量定义 ===
MAX_OUTPUT_TOKEN_LENGTH = 5000
PROMPT_TOKEN_LENGTH = 4096
PERF_OUTPUT_TOKEN_LENGTH = 1024

# === 模型名称合法性检查 ===
if not re.fullmatch(r"[A-Za-z0-9_\-]+", mp.model_name):
    raise ValueError(f"非法的 MODEL_NAME: {mp.model_name}，只能包含字母、数字、- 和 _")

# === 项目路径设置 ===
PROJECT_BASE = f"../../model-eval-storage/{mp.model_name}"
PROJECT_PREFIX = "project"
grpt.TOKENIZER_CONFIG_PATH = mp.TOKENIZER_CONFIG_PATH

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
cev.create_test = test_utils.create_test
grpt.create_test = test_utils.create_test
cev.process_prompt = test_utils.process_prompt
grpt.process_prompt = test_utils.process_prompt

test_path_omc = f"{PROJECT_PATH}/{PROJECT_NAME}-CEVAL-OMC-{MAX_CASES_PER_VERTICAL}-{NUM_PERF_TEST_CASES}.json"
with open(test_path_omc, 'w', encoding='utf-8') as f:
    acc = cev.generate_ceval_tests(PROJECT_NAME, sel_prompt, MAX_CASES_PER_VERTICAL, is_gpu_base_model=0)
    perf = grpt.generate_perf_tests(PROJECT_NAME, PROMPT_TOKEN_LENGTH, PERF_OUTPUT_TOKEN_LENGTH, NUM_PERF_TEST_CASES, is_gpu_base_model=0)
    all_cases = acc + perf
    json.dump(all_cases, f, indent=4, ensure_ascii=False)
    print(f"{len(all_cases)} OMC test cases saved to {test_path_omc}")

# ===== GPU 模式 =====
test_path_gpu = f"{PROJECT_PATH}/{PROJECT_NAME}-CEVAL-GPU-{MAX_CASES_PER_VERTICAL}-{NUM_PERF_TEST_CASES}.json"
with open(test_path_gpu, 'w', encoding='utf-8') as f:
    acc = cev.generate_ceval_tests(PROJECT_NAME, sel_prompt, MAX_CASES_PER_VERTICAL, is_gpu_base_model=1)
    perf = grpt.generate_perf_tests(PROJECT_NAME, PROMPT_TOKEN_LENGTH, PERF_OUTPUT_TOKEN_LENGTH, NUM_PERF_TEST_CASES, is_gpu_base_model=1)
    all_cases = acc + perf
    json.dump(all_cases, f, indent=4, ensure_ascii=False)
    print(f"{len(all_cases)} GPU test cases saved to {test_path_gpu}")

# ===== API 模式 =====
cev.create_test = test_utils.create_API_test
grpt.create_test = test_utils.create_API_test
cev.process_prompt = test_utils.process_prompt
grpt.process_prompt = test_utils.process_prompt

test_path_api = f"{PROJECT_PATH}/{PROJECT_NAME}-CEVAL-API-{MAX_CASES_PER_VERTICAL}-{NUM_PERF_TEST_CASES}.jsonl"
with open(test_path_api, 'w', encoding='utf-8') as f:
    acc = cev.generate_ceval_tests(PROJECT_NAME, sel_prompt, MAX_CASES_PER_VERTICAL, is_gpu_base_model=1)
    perf = grpt.generate_perf_tests(PROJECT_NAME, PROMPT_TOKEN_LENGTH, PERF_OUTPUT_TOKEN_LENGTH, NUM_PERF_TEST_CASES, is_gpu_base_model=1)
    all_cases = acc + perf
    for case in all_cases:
        json.dump(case, f, ensure_ascii=False)
        f.write('\n')
    print(f"{len(all_cases)} API test cases saved to {test_path_api}")

# ===== API config 生成 =====
tokenizer = AutoTokenizer.from_pretrained(mp.TOKENIZER_CONFIG_PATH)
CHAT_TEMPLATE = tokenizer.chat_template

license_file = f"{mp.TOKENIZER_CONFIG_PATH}/LICENSE"
with open(license_file, "r", encoding="utf-8") as f:
    license_text = f.read()

api_config_path = f"{PROJECT_PATH}/api_config.json"
api_config = {
    "inferType": mp.INFER_TYPE,
    "tokenizerType": mp.TOKENIZER_TYPE,
    "tokenizerPath": mp.TOKENIZER_PATH,
    "modelType": mp.MODEL_TYPE,
    "modelPath": mp.MODEL_PATH,
    "weightDir": mp.WEIGHT_DIR,
    "prefixPrompt": mp.PREFIX_PROMPT,
    "pmtCacheOperation": mp.PMT_CACHE_OP,
    "pfxInitTokenLen": mp.PFX_INIT_TOKEN_LEN,
    "loraCfgPath": mp.LORA_CFG_PATH,
    "expect": "",
    "callbackFreq": mp.CALLBACK_FREQ,
    "sampleFlag": mp.SAMPLE_FLAG,
    "seed": mp.SEED,
    "topK": mp.TOPK,
    "topP": mp.TOPP,
    "temperature": mp.TEMPERATURE,
    "maxGenTokens": MAX_OUTPUT_TOKEN_LENGTH,
    "repetitionPenalty": mp.REPETITIONPENALTY,
    "initTokenLen": mp.INIT_TOKEN_LEN,
    "stopSeq": mp.STOP_SEQ,
    "lora_rank": [{"rank": r, "size": None} for r in mp.LORA_RANKS],
    "isAsync": mp.IS_ASYNC,
    "chatTemplate": CHAT_TEMPLATE,
    "modelInfo": {"license": license_text}
}
with open(api_config_path, 'w', encoding='utf-8') as f:
    json.dump(api_config, f, indent=4, ensure_ascii=False)
    print(f"api_config.json saved to {api_config_path}")

print("[ALL DONE]")
