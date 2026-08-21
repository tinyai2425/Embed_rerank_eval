import sys
import os

# 添加模块搜索路径
sys.path.append(os.path.abspath("Function"))
sys.path.append(os.path.abspath("../2_GPU_run_eval/Function"))
sys.path.append(os.path.abspath("../3_OMC_eval/Function"))

# 导入相关模块
import API_collect
import eval_results
import verify_ans
import extend_metrics
import file_utils
import extract_name
import anlz_cont_quality
import write_average_sheet

def parse_args():
    if len(sys.argv) < 3:  # 确保至少有3个参数（tokenizer_path 和 test_result_path）
        print("Usage: python Eval_OMC_results.py <tokenizer_path> <test_result_path> [--show_detail]")
        sys.exit(1)

    tokenizer_path = sys.argv[1]
    test_result_path = sys.argv[2]
    show_detail = "--show_detail" in sys.argv  # 如果有--show_detail参数，就为True，否则为False

    return tokenizer_path, test_result_path, show_detail

def get_next_api_dir(project_base: str) -> str:
    existing = []
    for name in os.listdir(project_base) if os.path.exists(project_base) else []:
        if name.startswith("API-"):
            try:
                existing.append(int(name.split("-")[-1]))
            except ValueError:
                pass
    next_idx = max(existing) + 1 if existing else 1
    return os.path.join(project_base, f"API-{next_idx}")

if __name__ == "__main__":
    # 命令行参数解析
    tokenizer_path, test_result_path, show_detail = parse_args()
    project_base, file_name = extract_name.parse_project_base_and_filename(test_result_path)
    num_perf_cases = extract_name.extract_last_number_from_filename(file_name)
    
    # 创建 API 结果输出目录
    api_output_dir = get_next_api_dir(project_base)
    os.makedirs(api_output_dir, exist_ok=True)
    print(f"[API] output dir: {api_output_dir}")

    # 初始化 tokenizer（必须双重初始化）
    API_collect.init_tokenizer(tokenizer_path)
    extend_metrics.init_tokenizer(tokenizer_path)

    # 函数挂载（与旧流程一致）
    API_collect.GET_answer = extend_metrics.GET_answer
    API_collect.calculate_repetition_rate = extend_metrics.calculate_repetition_rate
    API_collect.calculate_token_entropy = extend_metrics.calculate_token_entropy
    API_collect.verify_answer = verify_ans.verify_answer
    API_collect.iter_lines_safely = file_utils.iter_lines_safely
    API_collect.TEST_CASE_NAME_PATTERN = verify_ans.TEST_CASE_NAME_PATTERN

    # 解析 API 测试日志为 DataFrame
    df_api_results = API_collect.parse_llm_api_results(test_result_path)
    API_collect.save_jsonl_result(df_api_results, api_output_dir)

    # 输出摘要指标
    summary = API_collect.evaluate_llm_api_results(df_api_results)

    # 若开启详细模式，则生成摘要图像
    save_summary_path = None
    if show_detail:
        save_summary_path = os.path.join(api_output_dir, f"API_summary.xlsx")

    # 执行可视化评估并打印结果
    df_ref_results = eval_results.evaluate_reference_results(
        df_api_results,
        verbal=show_detail,
        save_summary_path=save_summary_path
    )
    # 图像分析
    if show_detail:
        num_total = len(df_ref_results)
        num_valid = num_total - num_perf_cases
        print(f"[ANALYZE] analyzing first {num_valid} samples (excluding last {num_perf_cases} performance test cases)")

        fig_save_path = os.path.join(api_output_dir, f"analysis.png")
        anlz_cont_quality.analyze_entropy_and_repeat(
            df_api_results[:num_valid],
            save_fig_path=fig_save_path
        )

    if show_detail:
        write_average_sheet.save_overall_summary(save_summary_path, "summary", "API")
        write_average_sheet.save_overall_summary(save_summary_path, "breakdown", "API")