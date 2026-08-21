# Eval_Bayesian_par_search.py
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import json

# 让我们能 import Function/ 下的模块
sys.path.append(os.path.abspath("Function"))
import Bayesian_search as B


def parse_specs_from_cli(args) -> dict:
    """
    将命令行里的 --temp/--topp/--topk/--repp 转成 dict，
    固定值或范围字符串（如 0.6:1.2:0.05 / 8 / 2:64:2 / 0.8,1.0,1.2）。
    没有提供的键将不会出现（即不参与，也不固定）。
    """
    specs = {}
    if args.temp is not None:
        specs["temperature"] = args.temp
    if args.topp is not None:
        specs["top_p"] = args.topp
    if args.topk is not None:
        specs["top_k"] = args.topk
    if args.repp is not None:
        specs["repetition_penalty"] = args.repp
    return specs


def main():
    parser = argparse.ArgumentParser(
        description="自动化基于精度的贝叶斯参数搜索（vLLM OpenAI 兼容）"
    )
    parser.add_argument("input_json_path", type=str, help="测试集 JSON 路径（单句式用例数组）")
    parser.add_argument("ip", type=str, help="vLLM 服务器 IP")
    parser.add_argument("port", type=int, help="vLLM 服务器端口")
    parser.add_argument("model_id", type=str, help="模型名（vLLM served-model-name）")

    # 搜索规模
    parser.add_argument("--n_calls", type=int, default=50, help="优化总调用次数")
    parser.add_argument("--n_init", type=int, default=10, help="初始随机探索次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # 重试策略
    parser.add_argument("--retry_times", type=int, default=2, help="评估失败重试次数")
    parser.add_argument("--retry_wait", type=int, default=3, help="评估失败重试等待秒数")

    # 参数规格（固定值或范围）
    parser.add_argument("--temp", type=str, help="temperature 规格，如 0.6:1.2:0.05 或 1.0")
    parser.add_argument("--topp", type=str, help="top_p 规格，如 0.7:1.0:0.02 或 1.0")
    parser.add_argument("--topk", type=str, help="top_k 规格，如 2:64:1 或 8")
    parser.add_argument("--repp", type=str, help="repetition_penalty 规格，如 0.8:1.4:0.02 或 1.05")

    # 结果保存
    parser.add_argument("--save_each_eval", action="store_true", help="每次评估都保存 df 结果")
    parser.add_argument("--out_dir", type=str, default=None, help="输出目录（默认：<测试集同目录>/Bayes_Search）")

    args = parser.parse_args()

    # 输出目录
    if args.out_dir:
        out_dir = args.out_dir
    else:
        base_dir = os.path.dirname(os.path.abspath(args.input_json_path))
        out_dir = os.path.join(base_dir, "Bayes_Search")
    os.makedirs(out_dir, exist_ok=True)

    # 参数规格
    param_specs = parse_specs_from_cli(args)
    if not param_specs:
        print("[WARN] 未提供任何参数规格（temp/topp/topk/repp），将不会进行参数优化，仅执行一次评估。")

    save_each_dir = os.path.join(out_dir, "trials") if args.save_each_eval else None

    result, trials_df = B.bayesian_search(
        input_json_path=args.input_json_path,
        server_ip=args.ip,
        server_port=args.port,
        model_id=args.model_id,
        param_specs=param_specs,
        n_calls=args.n_calls,
        n_init=args.n_init,
        seed=args.seed,
        retry_times=args.retry_times,
        retry_wait=args.retry_wait,
        save_each_eval_dir=save_each_dir,
        verbose=True,
    )

    # 保存总结果
    B.save_search_artifacts(out_dir, result, trials_df)

    # 友好打印
    if not trials_df.empty and "accuracy" in trials_df.columns:
        best_row = trials_df.iloc[trials_df["accuracy"].idxmax()]
        best_acc = float(best_row["accuracy"])
        best_params = {k: best_row.get(k, None) for k in ["temperature", "top_p", "top_k", "repetition_penalty"]}
        print("\n🎯 最佳准确率:", f"{best_acc:.4f}")
        print("🔧 最佳参数组合:", best_params)
    else:
        print("\n[INFO] 未获得有效评估结果。")


if __name__ == "__main__":
    main()
