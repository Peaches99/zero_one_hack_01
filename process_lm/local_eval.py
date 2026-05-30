"""Local clone of the organizers' eval — on FRESH sequences we hold the truth for.

The official eval keeps the answer key, so we cannot self-score next-step (Task 1)
or completion (Task 2) on it. Here we build our OWN answer key: draw N fresh
sequences from the organizers' generator (a seed independent of training, deduped
against the provided training data so every test sequence is genuinely unseen),
cut each at a RANDOM point, have the model predict the next step and complete the
rest, then score against the known truth with the ORGANIZERS' OWN metric functions
(``participant_files/eval_metrics.py``):

  Task 1 next-step   : Top-1 / Top-3 / Top-5 Accuracy, MRR
  Task 2 completion  : Normalized Edit Distance, Exact Match, Token Accuracy,
                       Block-level Accuracy, plus % valid (their validator)

This is exactly what the organizers did to build their held-out eval — same grammar,
fresh draws — so the numbers are directly comparable to how they will score us.

    python -m process_lm.local_eval --ckpt process_lm/runs/final/best.pt --n 1000
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

from .data import build_records, load_all_families
from .metrics import nextstep_metrics
from .predict import complete, get_device, load_model, predict_next
from .tokenizer import Tokenizer

_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "tracks/industrial-infineon/training_data"
_PF = _ROOT / "tracks/industrial-infineon/participant_files"
for _p in (_DATA, _PF):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import generate_sequences as gs  # noqa: E402
from eval_metrics import (  # noqa: E402  -- organizers' official metric functions
    block_level_accuracy,
    normalized_edit_distance,
    token_accuracy,
)

FAMS = ["mosfet", "igbt", "ic"]


def generate_fresh(n: int, seed: int, exclude: set) -> tuple[list, int]:
    """Draw n fresh grammar sequences, skipping any that collide with training."""
    rng = random.Random(seed)
    out, collisions, attempts = [], 0, 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        fam = FAMS[len(out) % len(FAMS)]
        steps = gs.generate_sequence(fam, rng)
        if tuple(steps) in exclude:
            collisions += 1
            continue
        out.append((fam, steps))
    return out, collisions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=99991,
                    help="held-out generation seed (independent of training)")
    ap.add_argument("--cut-lo", type=float, default=0.5)
    ap.add_argument("--cut-hi", type=float, default=0.9)
    ap.add_argument("--guided", action="store_true",
                    help="validity-guided decoding for completion (default: plain, matches submission)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or get_device()
    model = load_model(args.ckpt, device)
    tok = Tokenizer.load(Path(args.ckpt).parent / "tokenizer.json")

    train = {tuple(s) for _f, s in build_records(load_all_families(_DATA))}
    seqs, collisions = generate_fresh(args.n, args.seed, train)
    print(f"Generated {len(seqs)} fresh sequences (seed {args.seed}); {collisions} "
          f"regenerated a training sequence and were skipped -> every test seq is unseen.")
    print(f"Random cut fraction in [{args.cut_lo}, {args.cut_hi}] per sequence; "
          f"decoding = {'guided' if args.guided else 'plain (matches submission)'}.\n")

    if args.guided:
        from .guided import complete_guided, repair_route

    cut_rng = random.Random(args.seed + 1)
    ranked, truths = [], []
    agg: dict = defaultdict(float)
    per: dict = defaultdict(lambda: defaultdict(float))

    for fam, steps in seqs:
        frac = cut_rng.uniform(args.cut_lo, args.cut_hi)
        cut = min(len(steps) - 1, max(2, int(len(steps) * frac)))
        partial, truth_next, ref = steps[:cut], steps[cut], steps[cut:]

        preds = predict_next(model, tok, fam, partial, device, k=5)
        ranked.append(preds)
        truths.append(truth_next)

        if args.guided:
            comp = repair_route(list(partial) + complete_guided(model, tok, fam, partial, device))[len(partial):]
        else:
            comp = complete(model, tok, fam, partial, device)

        vals = {
            "ned": normalized_edit_distance(comp, ref),
            "ex": float(comp == ref),
            "ta": token_accuracy(comp, ref),
            "blk": block_level_accuracy(comp, ref),
            "valid": float(len(gs.validate_sequence(partial + comp)) == 0),
            "ship": float(bool(comp) and comp[-1] == "SHIP LOT"),
            "t1": float(bool(preds) and preds[0] == truth_next),
            "n": 1.0,
        }
        for k, v in vals.items():
            agg[k] += v
            per[fam][k] += v

    ns = nextstep_metrics(ranked, truths)
    n = agg["n"]
    print("=" * 72)
    print(f"LOCAL EVAL — {int(n)} fresh held-out sequences, official metric functions")
    print("=" * 72)
    print("TASK 1  next-step prediction:")
    print(f"  Top-1 {ns['top1']:.4f}   Top-3 {ns['top3']:.4f}   "
          f"Top-5 {ns['top5']:.4f}   MRR {ns['mrr']:.4f}")
    print("TASK 2  sequence completion (vs the known true ending):")
    print(f"  Block-level Accuracy     {agg['blk']/n:.4f}")
    print(f"  Token Accuracy           {agg['ta']/n:.4f}")
    print(f"  Normalized Edit Distance {agg['ned']/n:.4f}   (lower = better)")
    print(f"  Exact Match Rate         {agg['ex']/n:.4f}")
    print(f"  Valid completion rate    {agg['valid']/n:.4f}   (reaches SHIP LOT {agg['ship']/n:.4f})")
    print("\nby family (n / Top-1 / Block-acc / Valid):")
    for f in FAMS:
        d = per[f]
        m = d["n"]
        if m:
            print(f"  {f:7} n={int(m):4}   Top-1 {d['t1']/m:.3f}   "
                  f"Block-acc {d['blk']/m:.3f}   Valid {d['valid']/m:.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
