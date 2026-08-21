# 用法:
#   python Eval_GPU_STSB_embeddings.py <input_json_path> <ip> <port> <model_id> <version_flag>
#
# 说明：
# - <input_json_path> 是用 Eval_GPU_STSB_embeddings.py 生成的 GPU-*.json（数组，每条 {testCaseName,input,expect}）
# - 产物写到同级目录 GPU/ 下：
#     GPU-Results-<version>.parquet / .jsonl
#     GPU_eval_v<version>.txt（覆盖）
#     GPU_eval_overall.txt（追加）

import sys
import os
import json
from typing import Tuple

# 通用组件
sys.path.append(os.path.abspath("Function"))
import emb_gpu_runner     # GPU并行执行+客户端
import emb_metrics        # STSB 评估（Spearman/Pearson）
import emb_report         # 写 GPU/*.txt

def parse_args():
    if len(sys.argv) != 6:
        print("Usage: python Eval_GPU_embeddings.py <input_json_path> <ip> <port> <model_id> <version_flag>")
        sys.exit(1)
    input_json_path = sys.argv[1]
    ip = sys.argv[2]
    port = int(sys.argv[3])
    model_id = sys.argv[4]
    version_flag = int(sys.argv[5])
    return input_json_path, ip, port, model_id, version_flag

def parse_project_base_and_filename(input_json_path: str) -> Tuple[str, str]:
    abs_path = os.path.abspath(input_json_path)
    return os.path.dirname(abs_path), os.path.basename(abs_path)

def get_version_nums(version_flag: int):
    if version_flag < 1:
        raise ValueError("version_flag must be >= 1")
    return list(range(1, version_flag + 1))

if __name__ == "__main__":
    input_json_path, ip, port, model_id, version_flag = parse_args()

    # 项目路径/命名
    project_base, file_name = parse_project_base_and_filename(input_json_path)
    version_nums = get_version_nums(version_flag)

    # GPU 输出目录
    gpu_output_dir = os.path.join(project_base, "GPU")
    os.makedirs(gpu_output_dir, exist_ok=True)

    print("Notice: Proxy settings should be disabled before proceeding.")
    print(f"[INFO] Using server http://{ip}:{port}/v1  model={model_id}")

    for version_num in version_nums:
        print(f"[INFO] Processing version {version_num}...")

        # 并行拉取 embeddings（保持输入顺序）
        df_results = emb_gpu_runner.run_gpu_embeddings_from_tests(
            tests_json_path=os.path.join(project_base, file_name),
            server_ip=ip,
            server_port=port,
            model_id=model_id,
            output_dir=gpu_output_dir,   # 结果目录（也可不传，默认 tests 同级/GPU）
        )

        # 落盘（命名与老框架一致）
        parquet_path = os.path.join(gpu_output_dir, f"GPU-Results-{version_num}.parquet")
        jsonl_path   = os.path.join(gpu_output_dir, f"GPU-Results-{version_num}.jsonl")
        df_results.to_parquet(parquet_path, index=False)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for _, row in df_results.iterrows():
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        print(f"[SAVE] {parquet_path}")
        print(f"[SAVE] {jsonl_path}")

        # 评估（STSB：cosine vs gold -> Spearman/Pearson）
        metrics = emb_metrics.evaluate_stsb_from_df(df_results, l2norm=True)

        # 写 GPU/*.txt（版本覆盖 + 总览追加）
        emb_report.write_eval_txt(
            gpu_dir=gpu_output_dir,
            version_num=version_num,
            metrics=metrics,
            df=df_results,
            tests_json_path=os.path.join(project_base, file_name),
            server_ip=ip,
            server_port=port,
            model_id=model_id,
        )

        print(f"[DONE] Evaluation for version {version_num}")
