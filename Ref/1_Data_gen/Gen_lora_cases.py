# Gen_lora_cases.py
import sys, os
import json
import re
from types import SimpleNamespace
from transformers import AutoTokenizer

# === 路径设置 ===
sys.path.append(os.path.abspath("Function"))
import lora_simple as lora
import rand_perf_test as grpt
import test_utils

# === 参数校验 ===
# 用法:
#   默认（OMC合并）：python Gen_lora_cases.py <model_config.json> <NUM_PERF_TEST_CASES>
#   旧版（OMC分散）：python Gen_lora_cases.py <model_config.json> <NUM_PERF_TEST_CASES> omcsep
if len(sys.argv) not in (3, 4):
    print("用法: python Gen_lora_cases.py <model_config.json> <NUM_PERF_TEST_CASES> [omcsep]")
    sys.exit(1)

config_path = sys.argv[1]
try:
    NUM_PERF_TEST_CASES = int(sys.argv[2])
except ValueError:
    raise ValueError("NUM_PERF_TEST_CASES 必须是整数")

OMC_SEPARATE = False
if len(sys.argv) == 4 and sys.argv[3].strip().lower() == "omcsep":
    OMC_SEPARATE = True  # 旧版输出：每条用例一个sentences

if not os.path.exists(config_path):
    raise FileNotFoundError(f"找不到配置文件: {config_path}")

# === 模型配置读取 ===
with open(config_path, "r", encoding="utf-8") as f:
    config_dict = json.load(f)
mp = SimpleNamespace(**config_dict)

# === 常量定义（保持与原脚本一致） ===
MAX_OUTPUT_TOKEN_LENGTH = 5000
PROMPT_TOKEN_LENGTH = 1024
PERF_OUTPUT_TOKEN_LENGTH = 50

# === 模型名称合法性检查 ===
if not re.fullmatch(r"[A-Za-z0-9_\\-]+", mp.model_name):
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

# 绑定 create_test / process_prompt（与 ceval 脚本相同用法）
lora.create_test = test_utils.create_test
grpt.create_test = test_utils.create_test
lora.process_prompt = test_utils.process_prompt
grpt.process_prompt = test_utils.process_prompt

def merge_omc_cases(cases):
    """
    将多条 OMC 用例（每条一个 sentences）合并为一条用例（多个 sentences）。
    - 以第一条用例为“头部模板”（保留其 testCaseName、各项 top-level 字段）
    - 将所有用例的 sentences 依次追加到 merged['sentences'] 中
    - 返回列表形式：[merged_case]
    """
    if not cases:
        return []
    # 深拷贝第一条头部（只保留其 top-level，后面重建 sentences）
    head = cases[0].copy()
    head["sentences"] = []
    for c in cases:
        # 安全获取 sentences
        ss = c.get("sentences") or []
        for s in ss:
            head["sentences"].append(s)
    return [head]

# ===== OMC 模式 =====
test_path_omc = f"{PROJECT_PATH}/{PROJECT_NAME}-LORA-OMC-{NUM_PERF_TEST_CASES}.json"
with open(test_path_omc, 'w', encoding='utf-8') as f:
    # Lora 准确性用例（整文件全量，不分 vertical）
    acc = lora.generate_lora_tests(PROJECT_NAME, is_gpu_base_model=0, max_output_token_length=MAX_OUTPUT_TOKEN_LENGTH)
    # 性能用例
    perf = grpt.generate_perf_tests(PROJECT_NAME, PROMPT_TOKEN_LENGTH, PERF_OUTPUT_TOKEN_LENGTH, NUM_PERF_TEST_CASES, is_gpu_base_model=0)
    all_cases = acc + perf

    if OMC_SEPARATE:
        # 旧版：分散（每条一个 sentences）
        json.dump(all_cases, f, indent=4, ensure_ascii=False)
        print(f"{len(all_cases)} OMC test cases (SEPARATED) saved to {test_path_omc}")
    else:
        # 新版默认：合并为一条用例，多条 sentences
        merged = merge_omc_cases(all_cases)
        json.dump(merged, f, indent=4, ensure_ascii=False)
        print(f"{len(all_cases)} OMC sentences merged into 1 case -> saved to {test_path_omc}")

# ===== GPU 模式（保持不变：分散）=====
test_path_gpu = f"{PROJECT_PATH}/{PROJECT_NAME}-LORA-GPU-{NUM_PERF_TEST_CASES}.json"
with open(test_path_gpu, 'w', encoding='utf-8') as f:
    acc = lora.generate_lora_tests(PROJECT_NAME, is_gpu_base_model=1, max_output_token_length=MAX_OUTPUT_TOKEN_LENGTH)
    perf = grpt.generate_perf_tests(PROJECT_NAME, PROMPT_TOKEN_LENGTH, PERF_OUTPUT_TOKEN_LENGTH, NUM_PERF_TEST_CASES, is_gpu_base_model=1)
    all_cases = acc + perf
    json.dump(all_cases, f, indent=4, ensure_ascii=False)
    print(f"{len(all_cases)} GPU test cases saved to {test_path_gpu}")

# ===== API 模式（保持不变：分离为 JSONL）=====
# 改为 API 用例生成器
lora.create_test = test_utils.create_API_test
grpt.create_test = test_utils.create_API_test
lora.process_prompt = test_utils.process_prompt
grpt.process_prompt = test_utils.process_prompt

test_path_api = f"{PROJECT_PATH}/{PROJECT_NAME}-LORA-API-{NUM_PERF_TEST_CASES}.jsonl"
with open(test_path_api, 'w', encoding='utf-8') as f:
    acc = lora.generate_lora_tests(PROJECT_NAME, is_gpu_base_model=1, max_output_token_length=MAX_OUTPUT_TOKEN_LENGTH)
    perf = grpt.generate_perf_tests(PROJECT_NAME, PROMPT_TOKEN_LENGTH, PERF_OUTPUT_TOKEN_LENGTH, NUM_PERF_TEST_CASES, is_gpu_base_model=1)
    all_cases = acc + perf
    for case in all_cases:
        json.dump(case, f, ensure_ascii=False)
        f.write('\n')
    print(f"{len(all_cases)} API test cases saved to {test_path_api}")

# ===== API config 生成（保持与原脚本一致）=====
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
    "isAsync": mp.IS_ASYNC,
    "chatTemplate": CHAT_TEMPLATE,
    "modelInfo": {"license": license_text}
}
with open(api_config_path, 'w', encoding='utf-8') as f:
    json.dump(api_config, f, indent=4, ensure_ascii=False)
    print(f"api_config.json saved to {api_config_path}")

print("[ALL DONE]")
