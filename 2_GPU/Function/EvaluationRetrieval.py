from typing import Dict, List, Optional, Tuple

import pytrec_eval


def mrr(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
) -> Dict[str, float]:
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
