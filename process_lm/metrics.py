"""Self-scoring metrics for Task 1 (next-step) and Task 2 (completion).

Reimplemented from the eval-protocol spec (generation_rules.md §5.2) so we can
iterate before the organizers' eval_metrics.py and eval files arrive.
"""
from __future__ import annotations


def edit_distance(a: list, b: list) -> int:
    """Levenshtein distance over two token lists."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def nextstep_metrics(ranked_preds: list[list[str]], truths: list[str]) -> dict:
    """Top-1/3/5 accuracy and MRR against the single true next step."""
    n = len(truths)
    if n == 0:
        return {"top1": 0.0, "top3": 0.0, "top5": 0.0, "mrr": 0.0, "n": 0}
    top1 = top3 = top5 = 0
    mrr = 0.0
    for preds, truth in zip(ranked_preds, truths):
        if truth in preds[:1]:
            top1 += 1
        if truth in preds[:3]:
            top3 += 1
        if truth in preds[:5]:
            top5 += 1
        if truth in preds:
            mrr += 1.0 / (preds.index(truth) + 1)
    return {"top1": top1 / n, "top3": top3 / n, "top5": top5 / n, "mrr": mrr / n, "n": n}


def completion_metrics(preds: list[list[str]], truths: list[list[str]]) -> dict:
    """Exact match, normalized edit distance, and token accuracy of the remainder."""
    n = len(truths)
    if n == 0:
        return {"exact_match": 0.0, "norm_edit_distance": 0.0, "token_accuracy": 0.0, "n": 0}
    exact = 0
    ned_sum = 0.0
    tok_correct = tok_total = 0
    for pred, truth in zip(preds, truths):
        if pred == truth:
            exact += 1
        ned_sum += edit_distance(pred, truth) / max(len(truth), len(pred), 1)
        for a, b in zip(pred, truth):
            if a == b:
                tok_correct += 1
        tok_total += len(truth)
    return {
        "exact_match": exact / n,
        "norm_edit_distance": ned_sum / n,
        "token_accuracy": tok_correct / max(tok_total, 1),
        "n": n,
    }
