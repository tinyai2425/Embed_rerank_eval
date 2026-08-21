# 用法:
#   python Gen_embedding_perf_cases.py <model_config.json> [MAX_CASES]

import sys, os, json, re
from types import SimpleNamespace
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

# ========== 你现有工具 ==========
sys.path.append(os.path.abspath("Function"))
try:
    from test_utils_embedding import set_model_config
except Exception:
    def set_model_config(mp):  # no-op
        pass

# ===== 固定数据集路径（相对本脚本目录）=====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.normpath(os.path.join(
    SCRIPT_DIR, "../data_set/WIKI_perf/wiki_CN_ENG_merged.parquet"
))

# 需要生成的 token 长度
TARGET_LENGTHS = [1000]

# ===== 参数解析 =====
if len(sys.argv) not in (2, 3):
    print("用法: python Gen_embedding_perf_cases.py <model_config.json> [MAX_CASES]")
    sys.exit(1)

config_path = sys.argv[1]
MAX_CASES = None
if len(sys.argv) == 3:
    try:
        MAX_CASES = int(sys.argv[2])
        if MAX_CASES <= 0:
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
if not getattr(mp, "TOKENIZER_CONFIG_PATH", None):
    raise ValueError("配置文件中必须包含 `TOKENIZER_CONFIG_PATH`（HF tokenizer 路径）")
if not os.path.exists(mp.TOKENIZER_CONFIG_PATH):
    raise FileNotFoundError(f"TOKENIZER_CONFIG_PATH 不存在: {mp.TOKENIZER_CONFIG_PATH}")

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

# ===== 读取 WIKI PERF parquet =====
df = pd.read_parquet(PARQUET_PATH)
need = {"text", "word_count", "source", "num_sources"}
if not need.issubset(df.columns):
    raise ValueError(f"parquet 必须包含列 {need}，实际得到 {df.columns.tolist()}")

# 轻度清洗：去掉 text 缺失
df = df.dropna(subset=["text"]).reset_index(drop=True)

# 将可能的 numpy/pandas 类型转为原生 Python
def _to_source_list(x):
    if isinstance(x, np.ndarray):
        return [str(v) for v in x.tolist()]
    if isinstance(x, (list, tuple, set)):
        return [str(v) for v in x]
    if pd.isna(x):
        return []
    return [str(x)]

def _to_int_or_none(x):
    return None if pd.isna(x) else int(x)

df["source"] = df["source"].apply(_to_source_list)
df["word_count"] = df["word_count"].apply(_to_int_or_none)
df["num_sources"] = df["num_sources"].apply(_to_int_or_none)

# 选取范围：全量或 head(MAX_CASES)
iter_df = df if MAX_CASES is None else df.head(MAX_CASES)
base = os.path.splitext(os.path.basename(PARQUET_PATH))[0]  # wiki_CN_ENG_merged

# ===== 准备 tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(mp.TOKENIZER_CONFIG_PATH, use_fast=True)
tokenizer.model_max_length = 10**12  # 静默“超长”告警

def truncate_to_exact_tokens(text: str, target_tokens: int) -> str:
    ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        max_length=target_tokens,
        truncation=True,
    )
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

# ===== JSON 兜底转换器（确保万无一失）=====
def json_safe(o):
    if o is None or isinstance(o, (str, int, float, bool)):
        return o
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [json_safe(x) for x in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return [json_safe(x) for x in o.tolist()]
    return str(o)

# ===== 聚合两个文件：一个 GPU JSON、一个 API JSONL =====
gpu_cases_all = []   # JSON 数组
api_lines_all = []   # JSONL 行对象（字符串稍后写入）

for i, row in iter_df.iterrows():
    raw_text = str(row["text"])

    for L in TARGET_LENGTHS:
        cut_text = truncate_to_exact_tokens(raw_text, L)
        tc_name = f"{PROJECT_NAME}-perf-test-prefill-{L}-embedding-wiki-{i}"

        # GPU 用例（仅一个数组文件；去掉 prompt_token_length、inputs）
        gpu_cases_all.append({
            "testCaseName": tc_name,
            "input": cut_text
        })

        # API 用例（仅 model/testCaseName/input；不含 prompt_token_length / meta）
        api_lines_all.append({
            "model": mp.model_name,
            "testCaseName": tc_name,
            "input": [cut_text]
        })

# ===== 保存：两个总文件 =====
suffix = "ALL" if MAX_CASES is None else str(len(iter_df))

gpu_json_path = f"{PROJECT_PATH}/{PROJECT_NAME}-perf-embedding-wiki-GPU-{suffix}.json"
with open(gpu_json_path, "w", encoding="utf-8") as f:
    json.dump(gpu_cases_all, f, ensure_ascii=False, indent=2, default=json_safe)

api_jsonl_path = f"{PROJECT_PATH}/{PROJECT_NAME}-perf-embedding-wiki-API-{suffix}.jsonl"
with open(api_jsonl_path, "w", encoding="utf-8") as f:
    for obj in api_lines_all:
        f.write(json.dumps(obj, ensure_ascii=False, default=json_safe) + "\n")

print(f"[GPU] {len(gpu_cases_all)} cases saved -> {gpu_json_path}")
print(f"[API] {len(api_lines_all)} cases saved -> {api_jsonl_path}")
print("Done.")
