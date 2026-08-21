import math
from typing import Dict, List, Optional, Tuple


def _dcg(rels: List[float], k: int) -> float:
    """trec_eval / pytrec_eval ndcg_cut: (2^rel - 1) / log2(rank + 1)."""
    dcg = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        if rel <= 0:
            continue
        dcg += (2.0 ** rel - 1.0) / math.log2(i + 1)
    return dcg


def _ndcg_at_k(ranked_ids: List[str], qrel: Dict[str, int], k: int) -> float:
    gains = [float(qrel.get(doc_id, 0)) for doc_id in ranked_ids[:k]]
    ideal = sorted((float(v) for v in qrel.values()), reverse=True)
    ideal_dcg = _dcg(ideal, k)
    if ideal_dcg == 0.0:
        return 0.0
    return _dcg(gains, k) / ideal_dcg


def _average_precision_at_k(ranked_ids: List[str], relevant: set, k: int) -> float:
    """trec_eval map_cut: divide by total relevant count, not min(R, k)."""
    if not relevant:
        return 0.0
    hit = 0
    ap_sum = 0.0
    for i, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant:
            hit += 1
            ap_sum += hit / i
    return ap_sum / len(relevant)


def _recall_at_k(ranked_ids: List[str], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant) / len(relevant)


def _precision_at_k(ranked_ids: List[str], relevant: set, k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant) / k


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

        k_max = max(k_values)
        eval_qids = [qid for qid in results if qid in qrels]
        if not eval_qids:
            raise ValueError("No overlapping query ids between qrels and results.")

        for query_id in eval_qids:
            qrel = qrels[query_id]
            relevant = {doc_id for doc_id, rel in qrel.items() if rel > 0}
            ranked_ids = [
                doc_id
                for doc_id, _ in sorted(
                    results[query_id].items(), key=lambda item: item[1], reverse=True
                )[:k_max]
            ]
            for k in k_values:
                ndcg[f"NDCG@{k}"] += _ndcg_at_k(ranked_ids, qrel, k)
                _map[f"MAP@{k}"] += _average_precision_at_k(ranked_ids, relevant, k)
                recall[f"Recall@{k}"] += _recall_at_k(ranked_ids, relevant, k)
                precision[f"P@{k}"] += _precision_at_k(ranked_ids, relevant, k)

        n_queries = len(eval_qids)
        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / n_queries, 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / n_queries, 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / n_queries, 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"] / n_queries, 5)

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
