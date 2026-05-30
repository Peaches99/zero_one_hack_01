"""Task 3 — anomaly detection from the language model's own surprise.

The track asks whether a model *learned process logic*. A model that did will be
SURPRISED exactly where a sequence breaks a rule: the offending step (or the step
whose missing prerequisite was just skipped) gets an improbably high per-step
negative log-likelihood. So we score a whole sequence by its surprise spike and
threshold it — no rule engine inside the detector.

Two reference points are reported for honesty:
  * MODEL detector  — LM surprisal (this is the learned-logic result).
  * ORACLE detector — the organizers' validate_sequence (perfect; an upper bound
                      and the source of ground-truth labels for evaluation).

We build a labeled eval set ourselves by corrupting valid routes with each of the
ten rule violations (validated to actually break the intended rule), so we can
measure ROC-AUC / F1 / rule-attribution before the organizers' file arrives.

    python -m process_lm.anomaly --ckpt process_lm/runs/full_big/best.pt --n 600
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import torch

from .data import build_records, load_all_families, split_records
from .predict import get_device, load_model
from .tokenizer import SPECIAL_TOKENS, Tokenizer

_DATA_DIR = Path(__file__).resolve().parent.parent / "tracks/industrial-infineon/training_data"
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))
import generate_sequences as gs  # type: ignore  # noqa: E402


# --------------------------------------------------------------------------- #
# Corruptions — each reliably triggers one of the 10 rules (validated after).  #
# --------------------------------------------------------------------------- #

def _drop_one(steps, predicate):
    idxs = [i for i, s in enumerate(steps) if predicate(s)]
    if not idxs:
        return None
    i = random.choice(idxs)
    return steps[:i] + steps[i + 1:]


def _move_before(steps, mover_pred, target_pred):
    """Move the first step matching mover_pred to just before the first target."""
    mi = next((i for i, s in enumerate(steps) if mover_pred(s)), None)
    ti = next((i for i, s in enumerate(steps) if target_pred(s)), None)
    if mi is None or ti is None or mi <= ti:
        return None
    s = steps[:mi] + steps[mi + 1:]
    ti = next((i for i, x in enumerate(s) if target_pred(x)), None)
    return s[:ti] + [steps[mi]] + s[ti:]


_CLEANS = gs.CLEAN_STEPS


def _swap_litho_levels(steps):
    """Swap the first and last ALIGN MASK LEVEL tokens -> out-of-order levels."""
    idxs = [i for i, s in enumerate(steps) if s.startswith("ALIGN MASK LEVEL ")]
    if len(idxs) < 2:
        return None
    s = list(steps)
    i, j = idxs[0], idxs[-1]
    s[i], s[j] = s[j], s[i]
    return s


CORRUPTIONS = {
    "RULE_DEP_NO_CLEAN": lambda s: _drop_one(s, lambda x: x in _CLEANS),
    "RULE_ETCH_NO_MASK": lambda s: _drop_one(s, lambda x: x in ("DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW")),
    "RULE_IMPLANT_NO_MASK": lambda s: _drop_one(s, lambda x: x in ("DEVELOP PHOTORESIST",)),
    "RULE_METAL_ETCH_NO_LITHO": lambda s: _drop_one(s, lambda x: x.startswith("EXPOSE LITHO LEVEL")),
    "RULE_LITHO_LEVEL_SKIP": _swap_litho_levels,
    "RULE_SHIP_BEFORE_TEST": lambda s: _move_before(s, lambda x: x == "SHIP LOT",
                                                     lambda x: x == "WAFER SORT TEST"),
    "RULE_TEST_BEFORE_PASSIVATION": lambda s: _move_before(
        s, lambda x: x in gs.ELECTRICAL_TEST_STEPS, lambda x: x == "CURE PASSIVATION"),
    "RULE_BACKSIDE_BEFORE_PASSIVATION": lambda s: _move_before(
        s, lambda x: x == "DEPOSIT BACKSIDE METAL", lambda x: x == "CURE PASSIVATION"),
    "RULE_CMP_NO_DEP": lambda s: _drop_one(s, lambda x: x in gs.DEPOSITION_STEPS),
}


def corrupt(steps, rng):
    """Return (corrupted_steps, rule_id) or None. Validated to truly break it."""
    rules = list(CORRUPTIONS)
    rng.shuffle(rules)
    for rule in rules:
        out = CORRUPTIONS[rule](list(steps))
        if not out:
            continue
        viol = gs.validate_sequence(out)
        if viol:  # genuinely invalid now
            rules_hit = {v.rule for v in viol}
            return out, (rule if rule in rules_hit else sorted(rules_hit)[0])
    return None


def build_eval(n_valid=300, n_invalid=300, seed=0):
    """Labeled mix of valid (1) and corrupted (0) real sequences with rule tags."""
    recs = build_records(load_all_families(_DATA_DIR))
    _, val = split_records(recs, 100, seed)
    rng = random.Random(seed)
    rng.shuffle(val)
    examples = []  # (family, steps, is_valid, rule_or_None)
    for fam, steps in val[:n_valid]:
        examples.append((fam, steps, 1, None))
    made = 0
    for fam, steps in val:
        if made >= n_invalid:
            break
        c = corrupt(steps, rng)
        if c:
            examples.append((fam, c[0], 0, c[1]))
            made += 1
    rng.shuffle(examples)
    return examples


# --------------------------------------------------------------------------- #
# LM surprise scoring                                                          #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def per_step_nll(model, tok, family, steps, device):
    """Per-step NLL aligned to `steps` (surprisal of each step given its prefix)."""
    ids = tok.encode_sequence(steps, family)
    ids = ids[: model.cfg.block_size + 1]
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    logits, _ = model(x)
    logp = torch.log_softmax(logits[0], dim=-1)
    tgt = torch.tensor(ids[1:], device=device)
    nll = -logp[torch.arange(len(tgt)), tgt]  # over [FAM, s1..sK, EOS]
    return nll[1:1 + len(steps)].tolist()  # drop FAM target; keep the K steps


def anomaly_score(model, tok, family, steps, device):
    """Sequence anomaly score = the surprise spike (max per-step NLL)."""
    nll = per_step_nll(model, tok, family, steps, device)
    if not nll:
        return 0.0, -1
    mx = max(nll)
    return mx, nll.index(mx)


def _auc(scores, labels_invalid):
    """ROC-AUC for 'higher score => more likely invalid'."""
    pos = [s for s, y in zip(scores, labels_invalid) if y == 1]  # invalid
    neg = [s for s, y in zip(scores, labels_invalid) if y == 0]  # valid
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def evaluate(model, tok, examples, device):
    scores, invalid = [], []
    spike_pos = []
    for fam, steps, is_valid, _rule in examples:
        sc, pos = anomaly_score(model, tok, fam, steps, device)
        scores.append(sc)
        invalid.append(0 if is_valid else 1)
        spike_pos.append(pos)
    auc = _auc(scores, invalid)
    # best-F1 threshold sweep
    order = sorted(set(scores))
    best = {"f1": -1}
    for thr in order:
        tp = sum(1 for s, y in zip(scores, invalid) if y == 1 and s >= thr)
        fp = sum(1 for s, y in zip(scores, invalid) if y == 0 and s >= thr)
        fn = sum(1 for s, y in zip(scores, invalid) if y == 1 and s < thr)
        tn = sum(1 for s, y in zip(scores, invalid) if y == 0 and s < thr)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best["f1"]:
            best = {"f1": f1, "thr": thr, "prec": prec, "rec": rec,
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return {"auc": auc, **best}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=300, help="valid (and invalid) example counts")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or get_device()
    model = load_model(Path(args.ckpt), device)
    tok = Tokenizer.load(Path(args.ckpt).parent / "tokenizer.json")

    examples = build_eval(args.n, args.n)
    n_valid = sum(1 for *_x, v, _r in examples if v == 1)
    n_invalid = len(examples) - n_valid
    print(f"eval set: {n_valid} valid + {n_invalid} invalid (corrupted, validator-confirmed)")

    res = evaluate(model, tok, examples, device)
    print(f"\n=== MODEL anomaly detector (LM surprise spike) ===")
    print(f"  ROC-AUC          : {res['auc']:.4f}")
    print(f"  best-F1          : {res['f1']:.4f}  (precision {res['prec']:.3f}, recall {res['rec']:.3f})")
    print(f"  confusion @thr   : TP={res['tp']} FP={res['fp']} FN={res['fn']} TN={res['tn']}")
    print(f"  threshold (NLL)  : {res['thr']:.3f}")
    print(f"\n  ORACLE detector (validate_sequence): AUC=1.000 F1=1.000 (rule engine; upper bound)")


if __name__ == "__main__":
    main()
