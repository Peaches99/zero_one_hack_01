"""Is the OOD block-acc gap (ID ~0.70 -> OOD ~0.50) recoverable, or is the 4th family
just genuinely different? Decides whether an OOD training campaign is worth it.

Two model-free ceilings per held-out family F, keyed on the (family-agnostic) prefix
block-signature with tail backoff, scored position-wise like the official metric:
  * transfer ceiling : predict F's remaining blocks from a bank of the OTHER families
                       only -> the best a model trained WITHOUT F could hope for.
  * oracle-F ceiling : predict from an F bank -> the "if we'd trained on F" upper bound.

LOFO model today ~0.50. If transfer >> 0.50 there is recoverable transfer headroom; if
transfer ~ 0.50 the gap is irreducible family difference (don't train, just report it).

    python -m process_lm.transfer_ceiling --bank 3000 --queries 250
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


def per_position_mode(sigs):
    if not sigs:
        return []
    out = []
    for i in range(max(len(s) for s in sigs)):
        col = [s[i] for s in sigs if i < len(s)]
        if col:
            out.append(Counter(col).most_common(1)[0][0])
    return out


def build_index(bank):
    """frac -> prefix-block-sig tuple -> [remaining-block-sig tuple]."""
    idx = {f: defaultdict(list) for f in CUTS}
    for steps in bank:
        for frac in CUTS:
            cut = min(len(steps) - 1, max(2, int(len(steps) * frac)))
            idx[frac][tuple(_block_signature(steps[:cut]))].append(
                tuple(_block_signature(steps[cut:])))
    return idx


def lookup(idx_frac, pre_sig, min_n=8):
    if len(idx_frac.get(pre_sig, [])) >= min_n:
        return idx_frac[pre_sig]
    for k in (8, 6, 4, 3, 2):
        if len(pre_sig) < k:
            continue
        tail = pre_sig[-k:]
        pool = [r for key, lst in idx_frac.items() if key[-k:] == tail for r in lst]
        if len(pool) >= min_n:
            return pool
    return [r for lst in idx_frac.values() for r in lst]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=int, default=3000, help="seqs per family in each bank")
    ap.add_argument("--queries", type=int, default=250)
    args = ap.parse_args()

    print(f"OOD transfer vs oracle-F block-acc ceiling  (bank={args.bank}/fam, "
          f"queries={args.queries}/held-out-fam)\n")
    print(f"{'heldout':8} {'cut':4} {'transfer':>9} {'oracleF':>9} {'LOFO~':>7}")
    lofo_ref = {("mosfet", 0.6): None}  # printed for context only
    grand = defaultdict(lambda: defaultdict(float))
    for F in FAMILIES:
        others = [f for f in FAMILIES if f != F]
        bank_other = []
        for j, o in enumerate(others):
            bank_other += gen(o, args.bank, seed=1000 + j)
        bank_F = gen(F, args.bank, seed=2000)
        idx_other = build_index(bank_other)
        idx_F = build_index(bank_F)
        bkeys = {tuple(s) for s in bank_F}
        queries = [s for s in gen(F, args.queries * 2, seed=777)
                   if tuple(s) not in bkeys][:args.queries]
        for frac in CUTS:
            tr, orc = [], []
            for steps in queries:
                cut = min(len(steps) - 1, max(2, int(len(steps) * frac)))
                pre_sig = tuple(_block_signature(steps[:cut]))
                rem_sig = _block_signature(steps[cut:])
                tr.append(token_accuracy(per_position_mode(lookup(idx_other[frac], pre_sig)), rem_sig))
                orc.append(token_accuracy(per_position_mode(lookup(idx_F[frac], pre_sig)), rem_sig))
            t = sum(tr) / len(tr)
            o = sum(orc) / len(orc)
            print(f"{F:8} {frac:<4} {t:>9.4f} {o:>9.4f} {'~0.50':>7}")
            grand[frac]["tr"] += t
            grand[frac]["orc"] += o
            grand[frac]["n"] += 1
    print("-" * 42)
    for frac in CUTS:
        g = grand[frac]
        print(f"{'ALL':8} {frac:<4} {g['tr']/g['n']:>9.4f} {g['orc']/g['n']:>9.4f}")
    tr = sum(grand[f]["tr"] for f in CUTS) / sum(grand[f]["n"] for f in CUTS)
    orc = sum(grand[f]["orc"] for f in CUTS) / sum(grand[f]["n"] for f in CUTS)
    print(f"\nTransfer ceiling {tr:.4f}  |  oracle-F {orc:.4f}  |  LOFO model ~0.50")
    print(f"Recoverable OOD headroom (transfer - model) ~ {tr - 0.50:+.4f}")


if __name__ == "__main__":
    main()
