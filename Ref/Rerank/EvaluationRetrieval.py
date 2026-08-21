from typing import Dict, List, Tuple, Optional
import pytrec_eval

def mrr(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
) -> Dict[str, float]:
    """MRR@k 计算。"""
    mrr_scores = {f"MRR@{k}": 0.0 for k in k_values}
    k_max = max(k_values)

    top_hits = {}
    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]

    for query_id in top_hits:
        query_relevant_docs = {doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0}
        for k in k_values:
            for rank, hit in enumerate(top_hits[query_id][0:k]):
                if hit[0] in query_relevant_docs:
                    mrr_scores[f"MRR@{k}"] += 1.0 / (rank + 1)
                    break

    for k in k_values:
        mrr_scores[f"MRR@{k}"] = round(mrr_scores[f"MRR@{k}"] / len(qrels), 5)
    return mrr_scores


def recall_cap(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
) -> Dict[str, float]:
    """R_cap@k 计算。"""
    capped_recall = {f"R_cap@{k}": 0.0 for k in k_values}
    k_max = max(k_values)

    for query_id, doc_scores in results.items():
        top_hits = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]
        query_relevant_docs = [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
        for k in k_values:
            retrieved_docs = [row[0] for row in top_hits[0:k] if qrels[query_id].get(row[0], 0) > 0]
            denominator = min(len(query_relevant_docs), k) if query_relevant_docs else 1
            capped_recall[f"R_cap@{k}"] += len(retrieved_docs) / denominator

    for k in k_values:
        capped_recall[f"R_cap@{k}"] = round(capped_recall[f"R_cap@{k}"] / len(qrels), 5)
    return capped_recall


def hole(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
) -> Dict[str, float]:
    """Hole@k 计算。"""
    hole_scores = {f"Hole@{k}": 0.0 for k in k_values}
    annotated_corpus = set()
    for _, docs in qrels.items():
        for doc_id in docs:
            annotated_corpus.add(doc_id)

    k_max = max(k_values)
    for _, scores in results.items():
        top_hits = sorted(scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]
        for k in k_values:
            hole_docs = [row[0] for row in top_hits[0:k] if row[0] not in annotated_corpus]
            hole_scores[f"Hole@{k}"] += len(hole_docs) / k

    for k in k_values:
        hole_scores[f"Hole@{k}"] = round(hole_scores[f"Hole@{k}"] / len(qrels), 5)
    return hole_scores


def top_k_accuracy(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
) -> Dict[str, float]:
    """Accuracy@k 计算。"""
    top_k_acc = {f"Accuracy@{k}": 0.0 for k in k_values}
    k_max = max(k_values)

    top_hits = {}
    for query_id, doc_scores in results.items():
        top_hits[query_id] = [
            item[0] for item in sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]
        ]

    for query_id in top_hits:
        query_relevant_docs = {doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0}
        for k in k_values:
            for relevant_doc_id in query_relevant_docs:
                if relevant_doc_id in top_hits[query_id][0:k]:
                    top_k_acc[f"Accuracy@{k}"] += 1.0
                    break

    for k in k_values:
        top_k_acc[f"Accuracy@{k}"] = round(top_k_acc[f"Accuracy@{k}"] / len(qrels), 5)
    return top_k_acc


class EvaluateRetrieval:
    """内置 BEIR EvaluateRetrieval，实现标准与自定义指标。"""

    def __init__(self, retriever=None, k_values: List[int] = None, score_function: Optional[str] = "cos_sim"):
        self.k_values = k_values or [1, 3, 5, 10, 100, 1000]
        self.top_k = max(self.k_values)
        self.retriever = retriever
        self.score_function = score_function

    @staticmethod
    def evaluate(
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int],
        ignore_identical_ids: bool = True,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        if ignore_identical_ids:
            popped = []
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
                        popped.append(pid)

        ndcg = {f"NDCG@{k}": 0.0 for k in k_values}
        _map = {f"MAP@{k}": 0.0 for k in k_values}
        recall = {f"Recall@{k}": 0.0 for k in k_values}
        precision = {f"P@{k}": 0.0 for k in k_values}

        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])

        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, {map_string, ndcg_string, recall_string, precision_string}
        )
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
                precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)

        return ndcg, _map, recall, precision

    @staticmethod
    def evaluate_custom(
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int],
        metric: str,
    ) -> Dict[str, float]:
        metric_lower = metric.lower()
        if metric_lower in ["mrr", "mrr@k", "mrr_cut"]:
            return mrr(qrels, results, k_values)
        if metric_lower in ["recall_cap", "r_cap", "r_cap@k"]:
            return recall_cap(qrels, results, k_values)
        if metric_lower in ["hole", "hole@k"]:
            return hole(qrels, results, k_values)
        if metric_lower in ["acc", "top_k_acc", "accuracy", "accuracy@k", "top_k_accuracy"]:
            return top_k_accuracy(qrels, results, k_values)
        raise ValueError(f"Unsupported metric: {metric}")


def evaluate_with_beir(qrels: Dict, run: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    使用 BEIR 指标进行评估，返回关键指标汇总。
    """
    print("Evaluating with BEIR metrics...")

    qrels_converted = {}
    needs_conversion = False
    for query_id, doc_scores in qrels.items():
        qrels_converted[query_id] = {}
        for doc_id, score in doc_scores.items():
            if isinstance(score, dict):
                actual_score = score.get("score") or score.get("relevance") or score.get("rel")
                if actual_score is None:
                    values = [v for v in score.values() if isinstance(v, (int, float))]
                    actual_score = values[0] if values else 0
                score = actual_score
                needs_conversion = True
            if isinstance(score, (int, float)):
                qrels_converted[query_id][doc_id] = int(score)
            elif isinstance(score, str):
                try:
                    qrels_converted[query_id][doc_id] = int(float(score))
                    needs_conversion = True
                except (ValueError, TypeError):
                    qrels_converted[query_id][doc_id] = 0
            else:
                qrels_converted[query_id][doc_id] = 0
    if needs_conversion:
        qrels = qrels_converted

    qrels_query_ids = set(qrels.keys())
    run_filtered = {qid: run[qid] for qid in run.keys() if qid in qrels_query_ids}
    if len(run_filtered) == 0:
        raise ValueError("No matching query_ids found between run and qrels.")

    k_values = [1, 10, 20, 50]
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, run_filtered, k_values)
    results2 = EvaluateRetrieval.evaluate_custom(qrels, run_filtered, k_values, metric="r_cap")
    results3 = EvaluateRetrieval.evaluate_custom(qrels, run_filtered, k_values, metric="mrr@k")

    return {
        "NDCG@10": ndcg["NDCG@10"],
        "Recall@10": recall["Recall@10"],
        "MAP@10": _map["MAP@10"],
        "P@10": precision["P@10"],
        "R_cap@10": results2["R_cap@10"],
        "MRR@10": results3["MRR@10"],
        "NDCG@20": ndcg["NDCG@20"],
        "Recall@20": recall["Recall@20"],
        "MAP@20": _map["MAP@20"],
        "P@20": precision["P@20"],
        "R_cap@20": results2["R_cap@20"],
        "MRR@20": results3["MRR@20"],
    }
