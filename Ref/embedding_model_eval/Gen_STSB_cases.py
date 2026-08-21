

# 用法:
#   python Gen_embedding_cases.py <model_config.json> [MAX_CASES]
# 说明:
#   - 不提供 MAX_CASES -> 处理数据集全量
#   - 提供 MAX_CASES  -> 最多处理该数量（head(MAX_CASES)）

import sys, os, json, re
from types import SimpleNamespace
import pandas as pd

sys.path.append(os.path.abspath("Function"))

from test_utils_embedding import (
    set_model_config,
    create_gpu_embedding_test,
    create_api_embedding_test,
)

# ===== 固定数据集路径（相对本脚本目录）=====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.normpath(os.path.join(
    SCRIPT_DIR, "../data_set/C-MTEB/test-C-METB-STSB.parquet"
))

# ===== 参数解析 =====
if len(sys.argv) not in (2, 3):
    print("用法: python Gen_embedding_cases.py <model_config.json> [MAX_CASES]")
    sys.exit(1)

config_path = sys.argv[1]
MAX_CASES = None
if len(sys.argv) == 3:
    try:
        MAX_CASES = int(sys.argv[2])
        if MAX_CASES <= 0:
            # 非正数就视为全量
            MAX_CASES = None
    except ValueError:
        raise ValueError("MAX_CASES 必须是整数")

if not os.path.exists(config_path):
    raise FileNotFoundError(f"找不到配置文件: {config_path}")
if not os.path.exists(PARQUET_PATH):
    raise FileNotFoundError(f"找不到数据文件: {PARQUET_PATH}")

# ===== 读取配置 =====
with open(config_path, "r", encoding="utf-8") as f:
    mp = SimpleNamespace(**json.load(f))

if not getattr(mp, "model_name", None):
    raise ValueError("配置文件中必须包含 `model_name`（用于 /api/embed 的 model 名称）")
"""if not re.fullmatch(r"[A-Za-z0-9_\-]+", mp.model_name):
    raise ValueError(f"非法的 model_name: {mp.model_name}（只能包含字母、数字、-、_）")"""

set_model_config(mp)

# ===== 项目路径 =====
def get_unique_subdir(base_dir, prefix):
    cnt = 1
    while os.path.exists(f"{base_dir}/{prefix}-{cnt}"):
        cnt += 1
    return f"{prefix}-{cnt}"

PROJECT_BASE = f"../../model-eval-storage/{mp.model_name}"
PROJECT_PREFIX = "project"
PROJECT_NAME = get_unique_subdir(PROJECT_BASE, PROJECT_PREFIX)
PROJECT_PATH = f"{PROJECT_BASE}/{PROJECT_NAME}"
os.makedirs(PROJECT_PATH, exist_ok=True)

print(f"PROJECT_NAME {PROJECT_NAME}")
print(f"Full path to project dir: {PROJECT_PATH}")

# ===== 读取 STSB parquet（需要 sentence1/sentence2/score）=====
df = pd.read_parquet(PARQUET_PATH)
need = {"sentence1", "sentence2", "score"}
if not need.issubset(df.columns):
    raise ValueError(f"parquet 必须包含列 {need}，实际得到 {df.columns.tolist()}")

# 轻度清洗：去掉关键列缺失的行
df = df.dropna(subset=["sentence1", "sentence2", "score"])

base = os.path.splitext(os.path.basename(PARQUET_PATH))[0]

# 选取范围：全量或 head(MAX_CASES)
iter_df = df if MAX_CASES is None else df.head(MAX_CASES)

# ===== 生成两份用例 =====
gpu_cases, api_cases = [], []
for i, row in iter_df.iterrows():
    s1, s2 = str(row["sentence1"]), str(row["sentence2"])
    gold = float(row["score"])
    tc_name = f"{PROJECT_NAME}-{base}-{i}"
    gpu_cases.append(create_gpu_embedding_test(tc_name, [s1, s2], gold))
    api_cases.append(create_api_embedding_test(tc_name, [s1, s2], gold))

# 保存 GPU（JSON 数组）
suffix = "ALL" if MAX_CASES is None else str(len(iter_df))
gpu_json_path = f"{PROJECT_PATH}/{PROJECT_NAME}-{base}-GPU-{suffix}.json"
with open(gpu_json_path, "w", encoding="utf-8") as f:
    json.dump(gpu_cases, f, ensure_ascii=False, indent=2)
print(f"[GPU] {len(gpu_cases)} cases saved -> {gpu_json_path}")

# 保存 API（JSONL）
api_jsonl_path = f"{PROJECT_PATH}/{PROJECT_NAME}-{base}-API-{suffix}.jsonl"
with open(api_jsonl_path, "w", encoding="utf-8") as f:
    for case in api_cases:
        f.write(json.dumps(case, ensure_ascii=False) + "\n")
print(f"[API] {len(api_cases)} cases saved -> {api_jsonl_path}")
