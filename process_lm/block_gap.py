"""Where exactly is Task-2 headroom? Model vs token-conditioned ceiling, split by cut.

block_headroom.py conditioned the oracle on the prefix BLOCK-signature, which de-dups
consecutive blocks and so destroys cycle-count state -> it underestimated the ceiling
(our model, seeing real tokens, beat it). This fixes that: the oracle conditions on the
longest exact TOKEN suffix of the prefix with enough bank matches, preserving cycle
state. That is a fair estimate of the Bayes ceiling -- the best any predictor could do
given the prefix. (ceiling - model) at each cut = genuinely recoverable headroom.

    python -m process_lm.block_gap --queries 150 --bank 3000

Reports, per family x cut (60/80): model greedy Block-acc / NED / Token-acc / Exact,
and the token-conditioned Block-acc ceiling. A gap only at 80% => the 60% cuts are
information-limited (route uncommitted), 80% cuts are decoder-limited (recoverable).
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "tracks/industrial-infineon/training_data"
_PF = _ROOT / "tracks/industrial-infineon/participant_files"
for _p in (_DATA, _PF):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import generate_sequences as gs  # noqa: E402
from eval_metrics import (_block_signature, block_level_accuracy,  # noqa: E402
                          normalized_edit_distance, token_accuracy)

from .predict import complete, get_device, load_model  # noqa: E402
from .tokenizer import Tokenizer  # noqa: E402

FAMILIES = ("mosfet", "igbt", "ic")
CUTS = (0.6, 0.8)


def gen(fam, n, seed):
    rng = random.Random(seed)
    return [gs.generate_sequence(fam, rng) for _ in range(n)]


def per_position_mode(sigs):
    """Bayes-optimal position-wise predictor: most common block at each index."""
    if not sigs:
        return []
    out = []
    for i in range(max(len(s) for s in sigs)):
        col = [s[i] for s in sigs if i < len(s)]
        if col:
            out.append(Counter(col).most_common(1)[0][0])
    return out


def build_tok_index(bank):
    """frac -> last-4-token bucket -> [(prefix_tokens, remaining_block_sig)]."""
    idx = {f: defaultdict(list) for f in CUTS}
    for steps in bank:
        for frac in CUTS:
            cut = min(len(steps) - 1, max(2, int(len(steps) * frac)))
            pre, rem = steps[:cut], steps[cut:]
            idx[frac][tuple(pre[-4:])].append((pre, tuple(_block_signature(rem))))
    return idx


def ceiling_pred(idx_frac, prefix, min_n=8):
    """Per-position modal block over bank seqs sharing the longest token suffix."""
    bucket = idx_frac.get(tuple(prefix[-4:]), [])
    best = None
    for k in (12, 8, 6, 4):
        if len(prefix) < k:
            continue
        tail = tuple(prefix[-k:])
        pool = [rem for pre, rem in bucket if tuple(pre[-k:]) == tail]
        if len(pool) >= min_n:
            best = pool
            break
    if best is None:
        best = [rem for _pre, rem in bucket] or \
               [rem for lst in idx_frac.values() for _pre, rem in lst]
    return per_position_mode([list(s) for s in best])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(_ROOT / "process_lm/runs/final/best.pt"))
    ap.add_argument("--bank", type=int, default=3000)
    ap.add_argument("--queries", type=int, default=150)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or get_device()
    model = load_model(Path(args.ckpt), device)
    tok = Tokenizer.load(Path(args.ckpt).parent / "tokenizer.json")

    print(f"Model vs token-conditioned ceiling  (ckpt={Path(args.ckpt).parent.name}, "
          f"bank={args.bank}/fam, queries={args.queries}/fam)\n")
    print(f"{'family':7} {'cut':4} {'mBlk':>7} {'ceilBlk':>8} {'gap':>7} "
          f"{'mNED':>7} {'mTok':>7} {'mExact':>7}")
    grand = defaultdict(lambda: defaultdict(float))
    for fam in FAMILIES:
        bank = gen(fam, args.bank, seed=1000)
        bkeys = {tuple(s) for s in bank}
        queries = [s for s in gen(fam, args.queries * 2, seed=777)
                   if tuple(s) not in bkeys][:args.queries]
        tidx = build_tok_index(bank)
        for frac in CUTS:
            acc = defaultdict(float)
            for steps in queries:
                cut = min(len(steps) - 1, max(2, int(len(steps) * frac)))
                pre, rem = steps[:cut], steps[cut:]
                rem_sig = _block_signature(rem)
                pred = complete(model, tok, fam, pre, device)
                acc["blk"] += block_level_accuracy(pred, rem)
                acc["ned"] += normalized_edit_distance(pred, rem)
                acc["tok"] += token_accuracy(pred, rem)
                acc["exact"] += float(pred == rem)
                ceil_sig = ceiling_pred(tidx[frac], pre)
                acc["ceil"] += token_accuracy(ceil_sig, rem_sig)
                acc["n"] += 1
            n = acc["n"]
            gap = (acc["ceil"] - acc["blk"]) / n
            print(f"{fam:7} {frac:<4} {acc['blk']/n:>7.4f} {acc['ceil']/n:>8.4f} "
                  f"{gap:>+7.4f} {acc['ned']/n:>7.4f} {acc['tok']/n:>7.4f} "
                  f"{acc['exact']/n:>7.4f}")
            for k in ("blk", "ceil", "ned", "tok", "exact", "n"):
                grand[frac][k] += acc[k]
        del bank
        if device == "cuda":
            torch.cuda.empty_cache()
    print("-" * 64)
    for frac in CUTS:
        g = grand[frac]
        n = g["n"]
        print(f"{'ALL':7} {frac:<4} {g['blk']/n:>7.4f} {g['ceil']/n:>8.4f} "
              f"{(g['ceil']-g['blk'])/n:>+7.4f} {g['ned']/n:>7.4f} {g['tok']/n:>7.4f} "
              f"{g['exact']/n:>7.4f}")
    tb = sum(grand[f]["blk"] for f in CUTS) / sum(grand[f]["n"] for f in CUTS)
    tc = sum(grand[f]["ceil"] for f in CUTS) / sum(grand[f]["n"] for f in CUTS)
    print(f"\nOverall model Block-acc {tb:.4f}  vs ceiling {tc:.4f}  "
          f"=> headroom {tc-tb:+.4f}")


if __name__ == "__main__":
    main()
