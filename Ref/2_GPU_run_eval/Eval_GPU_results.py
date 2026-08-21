import sys
import os
import pandas as pd

sys.path.append(os.path.abspath("Function"))
import verify_ans
import extend_metrics
import extract_name

import parallel_fetch_gpu
import measure_perf_gpu

import eval_results
import anlz_cont_quality
import write_average_sheet
import write_GPU_json

def parse_args():
    if len(sys.argv) < 7:
        print("Usage: python Eval_GPU_results.py <tokenizer_path> <input_json_path> <ip> <port> <model_id> <version_flag> [--show_detail]")
        sys.exit(1)

    tokenizer_path = sys.argv[1]
    input_json_path = sys.argv[2]
    ip = sys.argv[3]
    port = int(sys.argv[4])
    model_id = sys.argv[5]
    version_flag = int(sys.argv[6])
    show_detail = "--show_detail" in sys.argv

    return tokenizer_path, input_json_path, ip, port, model_id, version_flag, show_detail

def get_version_nums(version_flag):
    if version_flag < 1:
        raise ValueError("version_flag must be >= 1")
    return list(range(1, version_flag + 1))

if __name__ == "__main__":
    tokenizer_path, input_json_path, ip, port, model_id, version_flag, show_detail = parse_args()
    project_base, file_name = extract_name.parse_project_base_and_filename(input_json_path)
    version_nums = get_version_nums(version_flag)
    num_perf_cases = extract_name.extract_last_number_from_filename(file_name)

    # 创建 GPU 结果输出目录
    gpu_output_dir = os.path.join(project_base, "GPU")
    os.makedirs(gpu_output_dir, exist_ok=True)
    save_summary_path = os.path.join(gpu_output_dir, f"GPU_Summary.xlsx")

    # 初始化 tokenizer
    measure_perf_gpu.init_tokenizer(tokenizer_path)
    extend_metrics.init_tokenizer(tokenizer_path)

    measure_perf_gpu.verify_ans = verify_ans
    measure_perf_gpu.extend_metrics = extend_metrics

    parallel_fetch_gpu.verify_ans = verify_ans
    parallel_fetch_gpu.measure_perf_gpu = measure_perf_gpu

    print("Notice: Proxy settings should be disabled before proceeding.")

    df_ref_results = {}
    evaluated_results = {}

    with pd.ExcelWriter(save_summary_path) as writer:
        for version_num in version_nums:
            print(f"[INFO] Processing version {version_num}...")

            ref_result_parquet_path = os.path.join(gpu_output_dir, f"GPU-Results-{version_num}.parquet")

            # 拉取结果
            df_ref_results[version_num] = parallel_fetch_gpu.parallel_fetch_reference_model(
                os.path.join(project_base, file_name),
                ip,
                port,
                model_id
            )

            # 保存 parquet 到 GPU 目录
            df_ref_results[version_num].to_parquet(ref_result_parquet_path)
            print(f"[SAVE] Results saved to {ref_result_parquet_path}")

            write_GPU_json.save_jsonl_selected_fields(ref_result_parquet_path)

            # 评估
            ref_results = pd.read_parquet(ref_result_parquet_path)
            evaluated_results[version_num] = eval_results.evaluate_reference_results(
                ref_results,
                verbal=show_detail,
                save_summary_path=save_summary_path if show_detail else None,
                save_writer=writer if show_detail else None,
                sheet_prefix=f"v{version_num}"
            )
            print(f"[DONE] Evaluation for version {version_num}")

            # 图像分析
            if show_detail:
                num_total = len(evaluated_results[version_num])
                num_valid = num_total - num_perf_cases
                print(f"[ANALYZE] analyzing first {num_valid} samples (excluding last {num_perf_cases} performance test cases)")

                fig_save_path = os.path.join(gpu_output_dir, f"analysis_v{version_num}.png")
                anlz_cont_quality.analyze_entropy_and_repeat(
                    evaluated_results[version_num][:num_valid],
                    save_fig_path=fig_save_path
                )
        
    if show_detail:
        write_average_sheet.save_overall_summary(save_summary_path, "summary", "GPU_avg")
        write_average_sheet.save_overall_summary(save_summary_path, "breakdown", "GPU_avg")

