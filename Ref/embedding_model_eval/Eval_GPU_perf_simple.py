# Eval_GPU_embeddings_simple.py
# 用法:
#   python Eval_GPU_embeddings_simple.py <input_json_path> <ip> <port> <model_id>
#
# 说明：
# - <input_json_path> 是用 Gen_embedding_cases.py 生成的 GPU-*.json（数组，每条 {testCaseName,input,expect}）
# - 产物写到同级目录 GPU/ 下（只保留落盘，不做任何评估/报告）：
#     GPU-Results-1.parquet
#     GPU-Results-1.jsonl

import sys
import os
import json
from typing import Tuple

# 通用组件
sys.path.append(os.path.abspath("Function"))
import emb_gpu_runner  # GPU并行执行+客户端

VERSION_NUM = 1  # 固定为 1

def parse_args():
    if len(sys.argv) != 5:
        print("Usage: python Eval_GPU_embeddings_simple.py <input_json_path> <ip> <port> <model_id>")
        sys.exit(1)
    input_json_path = sys.argv[1]
    ip = sys.argv[2]
    try:
        port = int(sys.argv[3])
    except ValueError:
        print("[ERR] <port> 必须是整数")
        sys.exit(1)
    model_id = sys.argv[4]
    return input_json_path, ip, port, model_id

def parse_project_base_and_filename(input_json_path: str) -> Tuple[str, str]:
    abs_path = os.path.abspath(input_json_path)
    return os.path.dirname(abs_path), os.path.basename(abs_path)

if __name__ == "__main__":
    input_json_path, ip, port, model_id = parse_args()

    # 项目路径/命名
    project_base, file_name = parse_project_base_and_filename(input_json_path)

    # GPU 输出目录
    gpu_output_dir = os.path.join(project_base, "GPU")
    os.makedirs(gpu_output_dir, exist_ok=True)

    print("Notice: Proxy settings should be disabled before proceeding.")
    print(f"[INFO] Using server http://{ip}:{port}/v1  model={model_id}")
    print(f"[INFO] Processing version {VERSION_NUM}...")

    # 并行拉取 embeddings（保持输入顺序）
    df_results = emb_gpu_runner.run_gpu_embeddings_from_tests(
        tests_json_path=os.path.join(project_base, file_name),
        server_ip=ip,
        server_port=port,
        model_id=model_id,
        output_dir=gpu_output_dir,   # 结果目录（也可不传，默认 tests 同级/GPU）
    )

    # 仅落盘（命名与老框架一致，版本固定为 1）
    parquet_path = os.path.join(gpu_output_dir, f"GPU-Results-{VERSION_NUM}.parquet")
    jsonl_path   = os.path.join(gpu_output_dir, f"GPU-Results-{VERSION_NUM}.jsonl")

    # 保存 parquet
    try:
        df_results.to_parquet(parquet_path, index=False)
        print(f"[SAVE] {parquet_path}")
    except Exception as e:
        print(f"[WARN] 保存 Parquet 失败：{e}\n       请确认已安装 pyarrow 或 fastparquet")

    # 保存 jsonl
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df_results.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    print(f"[SAVE] {jsonl_path}")

    print(f"[DONE] Version {VERSION_NUM} finished (parquet & jsonl only).")