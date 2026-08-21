import json
import math
import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from EvaluationRetrieval import EvaluateRetrieval

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True, help="Path to the JSONL results file from rerank model")
    parser.add_argument("--k_values", type=str, default="[5,10]", help="List of k values for evaluation, e.g., '[5,10]' or '5,10'")
    return parser.parse_args()

def parse_k_values(k_values_str: str) -> List[int]:
    """解析k_values字符串，支持[5,10]或5,10格式"""
    k_values_str = k_values_str.strip()
    
    # 移除可能的方括号
    if k_values_str.startswith('[') and k_values_str.endswith(']'):
        k_values_str = k_values_str[1:-1]
    
    # 分割并转换为整数
    k_values = []
    for k in k_values_str.split(','):
        k = k.strip()
        if k:  # 确保不是空字符串
            k_values.append(int(k))
    
    return k_values

def load_results(results_file: str) -> List[Dict]:
    """加载JSONL格式的结果文件"""
    results = []
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

def calculate_score_from_response(response: List[Dict]) -> float:
    """
    根据response中的logprobs计算score
    只关心精确的"yes"和"no"这两个token
    """
    # 构建token到logprob的映射
    logprobs_dict = {}
    for item in response:
        token = item["token"]
        logprob = item["logprob"]
        logprobs_dict[token] = logprob
    
    # 只获取精确的"yes"和"no"的logit，默认-10.0
    true_logit = logprobs_dict.get("yes", -10.0)
    false_logit = logprobs_dict.get("no", -10.0)
    
    # 计算概率分数
    true_score = math.exp(true_logit)
    false_score = math.exp(false_logit)
    
    score = true_score / (true_score + false_score)
    return score

def build_qrels_and_run(results: List[Dict]) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, float]]]:
    """
    从结果文件构建qrels和run
    qrels: {queryName: {corpusName: expect}}
    run: {queryName: {corpusName: score}}
    """
    qrels = defaultdict(dict)
    run = defaultdict(dict)
    
    for item in results:
        query_name = item.get("queryName", "")
        corpus_name = item.get("corpusName", "")
        expect = item.get("expect", 0)
        response = item.get("response", [])
        
        if not query_name or not corpus_name:
            continue
        
        # 构建qrels
        qrels[query_name][corpus_name] = expect
        
        # 计算分数并构建run
        if response:
            score = calculate_score_from_response(response)
            run[query_name][corpus_name] = score
    
    # 转换为普通字典
    qrels = dict(qrels)
    run = dict(run)
    
    return qrels, run

def save_results_to_markdown(output_path: str, k_values: List[int], ndcg: Dict, _map: Dict, recall: Dict, precision: Dict):
    """将评估结果保存为markdown格式"""
    md_content = []
    md_content.append("# Rerank Model Evaluation Results")
    md_content.append("")
    
    # 动态生成所有k值的输出
    for k in k_values:
        md_content.append(f"NDCG@{k}: {ndcg[f'NDCG@{k}']:.4f}")
    
    for k in k_values:
        md_content.append(f"MAP@{k}: {_map[f'MAP@{k}']:.4f}")
    
    for k in k_values:
        md_content.append(f"Recall@{k}: {recall[f'Recall@{k}']:.4f}")
    
    for k in k_values:
        md_content.append(f"P@{k}: {precision[f'P@{k}']:.4f}")
    
    # 写入markdown文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_content))

def main():
    args = parse_args()
    
    # 解析k_values
    k_values = parse_k_values(args.k_values)
    print(f"Using k values: {k_values}")
    
    print(f"Loading results from: {args.results_file}")
    results = load_results(args.results_file)
    print(f"Loaded {len(results)} results")
    
    print("Building qrels and run...")
    qrels, run = build_qrels_and_run(results)
    print(f"Built qrels with {len(qrels)} queries")
    print(f"Built run with {len(run)} queries")
    
    # 验证qrels和run的一致性
    missing_in_run = set(qrels.keys()) - set(run.keys())
    if missing_in_run:
        print(f"Warning: {len(missing_in_run)} queries in qrels but not in run")
    
    print("Evaluating...")
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, run, k_values)
    
    # 打印结果到控制台
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    # 动态打印所有k值的评估结果
    ndcg_str = "  ".join([f"NDCG@{k}: {ndcg[f'NDCG@{k}']:.4f}" for k in k_values])
    map_str = "  ".join([f"MAP@{k}: {_map[f'MAP@{k}']:.4f}" for k in k_values])
    recall_str = "  ".join([f"Recall@{k}: {recall[f'Recall@{k}']:.4f}" for k in k_values])
    precision_str = "  ".join([f"P@{k}: {precision[f'P@{k}']:.4f}" for k in k_values])
    
    print(ndcg_str)
    print(map_str)
    print(recall_str)
    print(precision_str)
    
    # 在结果文件同目录下生成markdown文件
    results_dir = os.path.dirname(args.results_file)
    output_markdown = os.path.join(results_dir, "evaluation_results.md")
    save_results_to_markdown(output_markdown, k_values, ndcg, _map, recall, precision)
    
    print(f"\nEvaluation results saved to: {output_markdown}")

if __name__ == "__main__":
    main()