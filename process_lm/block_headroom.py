"""Is Task-2 Block-level Accuracy at a ceiling, or is 0.711 decoder slack?

The 0.328-nat entropy floor is a TOKEN-level fact. The official block-level metric
collapses steps into ~11 coarse blocks (LITHO/ETCH/DEPOSITION/...) and de-dups
consecutive ones, so synonym coin-flips ("MEASURE THICKNESS" vs "MEASURE INITIAL
THICKNESS" -> both METROLOGY_TEST) vanish. This script estimates the Bayes-optimal
ceiling for block-level completion accuracy WITHOUT a model: build a big reference
bank of generated sequences, and for each held-out (prefix, true-remaining) predict
the per-position MODAL block among bank sequences that share the prefix's block
state. That per-position mode is the Bayes-optimal predictor for position-wise
accuracy given the prefix; its score is the ceiling our decoder could chase.

    python -m process_lm.block_headroom --bank 4000 --queries 300

Contrast: it also reports the same oracle at the TOKEN level (should stay near the
floor -> proves the gap is block-specific, i.e. genuinely recoverable headroom).
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "tracks/industrial-infineon/training_data"
_PF = _ROOT / "tracks/industrial-infineon/participant_files"
for _p in (_DATA, _PF):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import generate_sequences as gs  # noqa: E402
from eval_metrics import _block_signature, token_accuracy  # noqa: E402

FAMILIES = ("mosfet", "igbt", "ic")
CUTS = (0.6, 0.8)


def gen(fam, n, seed):
    rng = random.Random(seed)
    return [gs.generate_sequence(fam, rng) for _ in range(n)]


def per_position_mode(sigs: list[tuple[str, ...]]) -> list[str]:
    """Bayes-optimal position-wise predictor: most common block at each index."""
    if not sigs:
        return []
    out = []
    for i in range(max(len(s) for s in sigs)):
        col = [s[i] for s in sigs if i < len(s)]
        out.append(Counter(col).most_common(1)[0][0])
    return out


def build_index(bank, fam):
    """key = prefix block-signature tuple -> list of remaining block-sig tuples.
    Built at both cut fractions so prefixes match held-out queries."""
    idx = defaultdict(lambda: defaultdict(list))  # frac -> key -> [remaining_sig]
    for steps in bank:
        full_sig = tuple(_block_signature(steps))
        for frac in CUTS:
            cut = min(len(steps) - 1, max(2, int(len(steps) * frac)))
            pre_sig = tuple(_block_signature(steps[:cut]))
            rem_sig = tuple(_block_signature(steps[cut:]))
            idx[frac][pre_sig].append(rem_sig)
    return idx


def lookup(idx_frac, pre_sig, min_n=8):
    """Exact prefix-block match, backing off to shorter block suffixes."""
    if len(idx_frac.get(pre_sig, [])) >= min_n:
        return idx_frac[pre_sig], "exact"
    # backoff: any bank prefix sharing the last-k blocks
    for k in (6, 4, 3, 2):
        if len(pre_sig) < k:
            continue
        tail = pre_sig[-k:]
        pool = [r for key, lst in idx_frac.items() if key[-k:] == tail for r in lst]
        if len(pool) >= min_n:
            return pool, f"tail{k}"
    pool = [r for lst in idx_frac.values() for r in lst]
    return pool, "family"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=int, default=4000, help="reference seqs per family")
    ap.add_argument("--queries", type=int, default=300, help="held-out seqs per family")
    args = ap.parse_args()

    print(f"Block-level completion CEILING probe (no model)")
    print(f"bank={args.bank}/fam  queries={args.queries}/fam  cuts={CUTS}\n")
    print(f"{'family':7} {'cut':4} {'blk-ceiling':>11} {'tok-ceiling':>11} "
          f"{'#rem-blocks':>11} {'distinct-sig':>12}")

    grand = defaultdict(lambda: defaultdict(float))
    for fam in FAMILIES:
        bank = gen(fam, args.bank, seed=1000)
        bank_keys = {tuple(s) for s in bank}
        queries = [s for s in gen(fam, args.queries * 2, seed=777)
                   if tuple(s) not in bank_keys][:args.queries]
        bidx = build_index(bank, fam)
        tidx = defaultdict(lambda: defaultdict(list))  # token-level analogue
        for steps in bank:
            for frac in CUTS:
                cut = min(len(steps) - 1, max(2, int(len(steps) * frac)))
                tidx[frac][tuple(steps[:cut][-6:])].append(tuple(steps[cut:]))

        for frac in CUTS:
            blk_scores, tok_scores, rem_lens = [], [], []
            distinct = set()
            for steps in queries:
                cut = min(len(steps) - 1, max(2, int(len(steps) * frac)))
                pre, rem = steps[:cut], steps[cut:]
                rem_sig = _block_signature(rem)
                rem_lens.append(len(rem_sig))
                distinct.add(tuple(rem_sig))
                # block-level oracle
                pre_sig = tuple(_block_signature(pre))
                pool, _lvl = lookup(bidx[frac], pre_sig)
                pred_sig = per_position_mode(pool)
                blk_scores.append(token_accuracy(pred_sig, rem_sig))
                # token-level oracle (contrast — should stay near floor)
                tpool = tidx[frac].get(tuple(pre[-6:]), [])
                if len(tpool) < 4:
                    tpool = [r for lst in tidx[frac].values() for r in lst]
                pred_tok = per_position_mode([list(t) for t in tpool])
                tok_scores.append(token_accuracy(pred_tok, rem))
            b = sum(blk_scores) / len(blk_scores)
            t = sum(tok_scores) / len(tok_scores)
            rl = sum(rem_lens) / len(rem_lens)
            print(f"{fam:7} {frac:<4} {b:>11.4f} {t:>11.4f} {rl:>11.1f} "
                  f"{len(distinct):>12}")
            grand[frac]["blk"] += b
            grand[frac]["tok"] += t
            grand[frac]["n"] += 1
    print("-" * 60)
    for frac in CUTS:
        g = grand[frac]
        print(f"{'ALL':7} {frac:<4} {g['blk']/g['n']:>11.4f} {g['tok']/g['n']:>11.4f}")
    allb = sum(grand[f]["blk"] for f in CUTS) / sum(grand[f]["n"] for f in CUTS)
    print(f"\nBlock-level ceiling (mean over families/cuts): {allb:.4f}")
    print(f"Our model's submitted block-acc (held-out)   : 0.711")
    print(f"=> recoverable headroom if ceiling > 0.711.")


if __name__ == "__main__":
    main()
