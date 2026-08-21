import sys, os
import json
from types import SimpleNamespace
import random

sys.path.append(os.path.abspath("Function"))
import human_eval
import test_utils

# === 参数校验 ===
# 用法: python Gen_human_eval_cases.py <model_config.json> [n_cases]
if len(sys.argv) < 2 or len(sys.argv) > 3:
    print("用法: python Gen_human_eval_cases.py <model_config.json> [n_cases]")
    sys.exit(1)

config_path = sys.argv[1]
n_cases = None  # None 表示默认使用“全部”用例
if len(sys.argv) == 3:
    try:
        n_cases = int(sys.argv[2])
        if n_cases <= 0:
            raise ValueError
    except ValueError:
        raise ValueError("n_cases 必须是正整数")

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

# === 绑定配置 ===
test_utils.set_model_config(mp)

# === 绑定 create/process 函数（接口与 CLTS 保持一致）===
human_eval.create_test = test_utils.create_test
human_eval.process_prompt = test_utils.process_prompt

# === 生成用例 ===
template = human_eval.humaneval_prompt_template
# is_gpu_base_model 强制为 1：输入已内含 chat 模板，不再叠加模型模板
IS_GPU_BASE_MODEL = 1

# 先把全量索引取出，再依需求裁剪
total = human_eval.count_humaneval_items()
if n_cases is None or n_cases >= total:
    selected_indices = list(range(total))
    count_str = "ALL"
else:
    random.seed(42)  # 固定随机子样
    selected_indices = sorted(random.sample(range(total), n_cases))
    count_str = str(n_cases)

# ===== OMC 模式 =====
test_path_omc = f"{PROJECT_PATH}/{PROJECT_NAME}-HumanEval-OMC-{count_str}.json"
with open(test_path_omc, 'w', encoding='utf-8') as f:
    he_cases = human_eval.generate_humaneval_tests(
        project_name=PROJECT_NAME,
        template=template,
        indices=selected_indices,
        max_output_token_length=MAX_OUTPUT_TOKEN_LENGTH,
        is_gpu_base_model=IS_GPU_BASE_MODEL
    )
    json.dump(he_cases, f, indent=4, ensure_ascii=False)
    print(f"{len(he_cases)} OMC test cases saved to {test_path_omc}")

# ===== GPU 模式 =====
test_path_gpu = f"{PROJECT_PATH}/{PROJECT_NAME}-HumanEval-GPU-{count_str}.json"
with open(test_path_gpu, 'w', encoding='utf-8') as f:
    he_cases = human_eval.generate_humaneval_tests(
        project_name=PROJECT_NAME,
        template=template,
        indices=selected_indices,
        max_output_token_length=MAX_OUTPUT_TOKEN_LENGTH,
        is_gpu_base_model=IS_GPU_BASE_MODEL
    )
    json.dump(he_cases, f, indent=4, ensure_ascii=False)
    print(f"{len(he_cases)} GPU test cases saved to {test_path_gpu}")

# ===== API 模式 =====
human_eval.create_test = test_utils.create_API_test  # 切换为 API 版本
test_path_api = f"{PROJECT_PATH}/{PROJECT_NAME}-HumanEval-API-{count_str}.jsonl"
with open(test_path_api, 'w', encoding='utf-8') as f:
    he_cases = human_eval.generate_humaneval_tests(
        project_name=PROJECT_NAME,
        template=template,
        indices=selected_indices,
        max_output_token_length=MAX_OUTPUT_TOKEN_LENGTH,
        is_gpu_base_model=IS_GPU_BASE_MODEL
    )
    for case in he_cases:
        json.dump(case, f, ensure_ascii=False)
        f.write('\n')
    print(f"{len(he_cases)} API test cases saved to {test_path_api}")
