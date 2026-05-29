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


# --- Block-level (process-logic) view: synonym- and optional-step-tolerant ---

def step_category(step: str) -> str:
    """Coarse functional category of a step (heuristic proxy for 'block').

    Collapses synonyms (DEPOSIT TOP METAL / DEPOSIT METAL 1 -> DEPOSIT) so we
    measure whether the *process logic* is right, not the exact wording. This is
    our own proxy until the organizers' eval_metrics.py defines block accuracy.
    """
    s = step.upper()
    if s in {"RECEIVE WAFER LOT", "LOT IDENTIFICATION", "LOT RELEASE",
             "FINAL LOT RELEASE", "SHIP LOT", "PACKAGE PREPARATION"}:
        return "LOGISTICS"
    if s.startswith("MEASURE") or "INSPECT" in s or s.startswith("FINAL ") or s.endswith("CHECK"):
        return "METROLOGY"
    if "TEST" in s or s == "YIELD ANALYSIS":
        return "TEST"
    if "IMPLANT" in s:
        return "IMPLANT"
    if "ETCH" in s and "CLEAN" not in s:
        return "ETCH"
    if any(k in s for k in ("CLEAN", "RCA", "HF DIP", "RINSE", "DRY WAFER",
                            "OXIDE STRIP", "SURFACE PREP", "PRE CLEAN")):
        return "CLEAN"
    if "STRIP" in s:
        return "STRIP"
    if "CMP" in s:
        return "CMP"
    if "FILL VIA" in s:
        return "VIA_FILL"
    if any(k in s for k in ("SPIN COAT", "SOFT BAKE", "ALIGN MASK", "EXPOSE LITHO",
                            "POST EXPOSE BAKE", "DEVELOP", "HARD BAKE", "PAD WINDOW LITHO")):
        return "LITHO"
    if any(k in s for k in ("ANNEAL", "DRIVE IN DIFFUSION", "DENSIFY", "CURE")):
        return "THERMAL"
    if any(k in s for k in ("DEPOSIT", "THERMAL OXIDATION", "GATE OXIDE",
                            "EPITAXIAL DEPOSITION", "FIELD OXIDE", "PAD OXIDE", "PASSIVATION")):
        return "DEPOSIT"
    if any(k in s for k in ("EPITAX", "SUBSTRATE", "GRIND", "BACKSIDE")):
        return "PREP"
    return "OTHER"


def block_sequence(steps: list[str]) -> list[str]:
    """Map steps to categories and collapse consecutive duplicates -> block flow."""
    out: list[str] = []
    for s in steps:
        c = step_category(s)
        if not out or out[-1] != c:
            out.append(c)
    return out


def blocklevel_metrics(preds: list[list[str]], truths: list[list[str]]) -> dict:
    """Block-flow exact match and normalized edit distance (synonym-tolerant)."""
    n = len(truths)
    if n == 0:
        return {"block_exact_match": 0.0, "block_norm_edit_distance": 0.0, "n": 0}
    exact = 0
    ned = 0.0
    for p, t in zip(preds, truths):
        pb, tb = block_sequence(p), block_sequence(t)
        if pb == tb:
            exact += 1
        ned += edit_distance(pb, tb) / max(len(tb), len(pb), 1)
    return {"block_exact_match": exact / n, "block_norm_edit_distance": ned / n, "n": n}
