"""Build held-out eval sets in the OFFICIAL eval_metrics.py format + flagship
predictions, so the official scorer prints per-family / per-fraction breakdowns for
Task 1 (next-step) and Task 2 (completion). The shipped official eval inputs carry no
ground-truth continuations, so we mirror the exact 60/80 protocol on fresh held-out
routes (deduped vs training) that we can score.

    python -m process_lm.make_official_eval --n 100
Writes nextstep_{gt,pred}.csv and completion_{gt,pred}.csv to extras/results/heldout/.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tracks/industrial-infineon/training_data"))
import generate_sequences as gs  # noqa: E402

from .data import build_records, load_all_families  # noqa: E402
from .predict import complete, get_device, load_model, predict_next  # noqa: E402
from .tokenizer import Tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "submission/model/best.pt"))
    ap.add_argument("--n", type=int, default=100, help="held-out routes per family")
    ap.add_argument("--seed", type=int, default=99991)
    ap.add_argument("--out", default=str(ROOT / "extras/results/heldout"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = get_device()
    m = load_model(args.ckpt, dev)
    tok = Tokenizer.load(Path(args.ckpt).parent / "tokenizer.json")
    train_keys = {tuple(s) for _f, s in build_records(
        load_all_families(ROOT / "tracks/industrial-infineon/training_data"))}

    ns_gt, ns_pred, cp_gt, cp_pred = [], [], [], []
    eid = 0
    for fam in ("mosfet", "igbt", "ic"):
        rng = random.Random(args.seed + hash(fam) % 1000)
        got = 0
        while got < args.n:
            s = gs.generate_sequence(fam, rng)
            if tuple(s) in train_keys:
                continue
            got += 1
            for frac in (0.6, 0.8):
                cut = min(len(s) - 1, max(2, int(len(s) * frac)))
                e = f"{fam}_{eid:04d}_{int(frac*100)}"
                ns_gt.append((e, fam, frac, s[cut]))
                ranks = (predict_next(m, tok, fam, s[:cut], dev, k=5) + [""] * 5)[:5]
                ns_pred.append((e, *ranks))
                cp_gt.append((e, fam, frac, "|".join(s[:cut]), "|".join(s)))
                cp_pred.append((e, "|".join(complete(m, tok, fam, s[:cut], dev))))
            eid += 1

    def w(path, header, rows):
        with open(out / path, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(header)
            wr.writerows(rows)

    w("nextstep_gt.csv", ["EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION", "NEXT_STEP"], ns_gt)
    w("nextstep_pred.csv", ["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"], ns_pred)
    w("completion_gt.csv", ["EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION", "PARTIAL_SEQUENCE", "FULL_SEQUENCE"], cp_gt)
    w("completion_pred.csv", ["EXAMPLE_ID", "PREDICTED_SEQUENCE"], cp_pred)
    print(f"wrote held-out official-format eval ({len(ns_gt)} rows) to {out}")


if __name__ == "__main__":
    main()
