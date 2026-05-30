"""Does GROKKING occur in process-sequence learning?

Grokking = a model memorizes a tiny training set (train loss collapses below the
generalizing floor) and only MUCH later does held-out loss suddenly drop to the floor
— delayed generalization. It needs (a) data small enough to memorize, (b) weight decay,
(c) long training. Our normal runs never grok because thousands of sequences make the
model generalize immediately (train == val from the start). So we deliberately starve
the data and train long, watching the in-distribution held-out curve for the snap.

Each size still SEES the grammar: 100 seqs ~= 33/family x ~130 steps ~= 4k transitions
per family. The grid brackets the window: 100/300 can memorize (grok candidates), 800
should generalize immediately (control). Plus a weight-decay ablation at 300.

    python -m process_lm.grokking --epochs 12000
Then: python -m process_lm.plots   (renders runs/figures/grokking.png from the logs)
Resumable: a config whose train_log.csv already reached --epochs rows is skipped.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
RUNS = _ROOT / "process_lm/runs/grok"

# (name, train-limit sequences, weight decay)
CONFIGS = [
    ("d100_wd1.0", 100, 1.0),
    ("d300_wd1.0", 300, 1.0),
    ("d800_wd1.0", 800, 1.0),   # control: sees plenty -> expect immediate generalization
    ("d300_wd0.1", 300, 0.1),   # weight-decay ablation (grokking is wd-sensitive)
]


def done(out, epochs):
    log = out / "train_log.csv"
    if not log.exists():
        return False
    return sum(1 for _ in open(log)) - 1 >= epochs  # minus header


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12000)
    ap.add_argument("--save-every", type=int, default=3000)
    ap.add_argument("--n-layer", type=int, default=8)
    ap.add_argument("--n-embd", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-per-family", type=int, default=50,
                    help="held-out val size/family (smaller = faster epochs, still smooth curve)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)

    for name, limit, wd in CONFIGS:
        out = RUNS / name
        if done(out, args.epochs):
            print(f"[skip] {name} already has {args.epochs} epochs logged")
            continue
        cmd = [sys.executable, "-m", "process_lm.train",
               "--out-dir", str(out), "--train-limit", str(limit),
               "--weight-decay", str(wd), "--epochs", str(args.epochs),
               "--save-every", str(args.save_every), "--n-layer", str(args.n_layer),
               "--n-embd", str(args.n_embd), "--batch-size", str(args.batch_size),
               "--val-per-family", str(args.val_per_family),
               "--lr", "3e-4", "--seed", str(args.seed), "--force"]
        env = {**os.environ, "PYTHONUTF8": "1"}
        print(f"[run ] {name}  (limit={limit}, wd={wd}, {args.epochs} epochs)")
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            print(f"[FAIL] {name}\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
            continue
        print(f"[done] {name}  ({time.time()-t0:.0f}s)  {r.stdout.strip().splitlines()[-1]}")
    print(f"\ngrokking runs -> {RUNS}   (now run: python -m process_lm.plots)")


if __name__ == "__main__":
    main()
