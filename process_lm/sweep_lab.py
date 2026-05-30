"""Adversarial sweep — train + eval + compare every lever, hunt a verified gain.

The model sits at the loss floor, so the point of this is (1) to PROVE we are at
the in-distribution ceiling rather than assert it, and (2) to catch any real gain
(completion Block-acc, or MRR via a multi-seed ensemble) the floor argument might
miss. Each config is trained (subprocess -> train.py, exact same path as our real
model) and evaluated on a FIXED fresh held-out set with the official metric
functions, so every row is directly comparable. Resumable: a config whose result
is already logged is skipped.

Anti-self-deception: selection is on seed 99991; the winner is later confirmed on a
DIFFERENT fresh seed (run with --confirm) so a lucky-on-one-set config can't pass.

    python -m process_lm.sweep_lab                 # run the grid (sequential)
    python -m process_lm.sweep_lab --eval-seed 7   # confirmation set
"""
from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

from .data import build_records, load_all_families
from .local_eval import generate_fresh
from .metrics import nextstep_metrics
from .predict import get_device, load_model, predict_next
from .tokenizer import Tokenizer

_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "tracks/industrial-infineon/training_data"
RUNS = _ROOT / "process_lm/runs/sweep"
RESULTS = _ROOT / "process_lm/runs/sweep_results.csv"

# (name, train-arg overrides). Baseline arch = final (8L/512/8).
BASE = dict(n_layer=8, n_embd=512, n_head=8, dropout=0.1, family_dropout=0.0,
            extra=8000, epochs=20, lr=3e-4, batch=64, seed=0)


def _cfg(name, **kw):
    d = dict(BASE)
    d.update(kw)
    return (name, d)


CONFIGS = [
    # --- size sweep (no dropout, 8k data) ---
    _cfg("size_tiny", n_layer=4, n_embd=256),
    _cfg("size_small", n_layer=6, n_embd=384),
    _cfg("size_med", n_layer=8, n_embd=512),            # ~ final arch
    _cfg("size_large", n_layer=10, n_embd=640),
    _cfg("size_xl", n_layer=12, n_embd=768, batch=32),
    # --- regularization ---
    _cfg("fdrop15", family_dropout=0.15),               # the submitted handicap
    _cfg("drop0", dropout=0.0),
    _cfg("drop2", dropout=0.2),
    # --- data volume ---
    _cfg("data1k", extra=1000),
    _cfg("data20k", extra=20000),
    _cfg("data40k", extra=40000),
    # --- training length / lr ---
    _cfg("ep40", epochs=40),
    _cfg("ep60", epochs=60),
    _cfg("lr6e4", lr=6e-4),
    _cfg("lr1e3", lr=1e-3),
    # --- seeds of the strong config (for an ensemble) ---
    _cfg("seed1", seed=1),
    _cfg("seed2", seed=2),
    _cfg("seed3", seed=3),
    _cfg("seed4", seed=4),
]


def build_eval_set(eval_seed, n, train_keys):
    seqs, _ = generate_fresh(n, eval_seed, train_keys)
    cuts = []
    for fam, steps in seqs:
        for frac in (0.6, 0.8):
            cut = min(len(steps) - 1, max(2, int(len(steps) * frac)))
            cuts.append((fam, steps[:cut], steps[cut]))
    return cuts


@torch.no_grad()
def eval_nextstep(ckpt_dir, cuts, device):
    model = load_model(ckpt_dir / "best.pt", device)
    tok = Tokenizer.load(ckpt_dir / "tokenizer.json")
    ranked = [predict_next(model, tok, fam, partial, device, k=5) for fam, partial, _ in cuts]
    truths = [t for _f, _p, t in cuts]
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return nextstep_metrics(ranked, truths)


def train_config(name, args_dict):
    out = RUNS / name
    if (out / "best.pt").exists():
        return out, "cached"
    cmd = [sys.executable, "-m", "process_lm.train",
           "--out-dir", str(out),
           "--n-layer", str(args_dict["n_layer"]), "--n-embd", str(args_dict["n_embd"]),
           "--n-head", str(args_dict["n_head"]), "--dropout", str(args_dict["dropout"]),
           "--family-dropout", str(args_dict["family_dropout"]),
           "--extra-per-family", str(args_dict["extra"]), "--epochs", str(args_dict["epochs"]),
           "--lr", str(args_dict["lr"]), "--batch-size", str(args_dict["batch"]),
           "--seed", str(args_dict["seed"])]
    env = {**__import__("os").environ, "PYTHONUTF8": "1"}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print(f"[FAIL] {name}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
        return out, "fail"
    vl = ""
    for line in r.stdout.splitlines():
        if "best val loss" in line:
            parts = line.split("best val loss")[1].strip().split()
            if parts:
                vl = parts[0]
    return out, vl or "ok"


def already_done(name):
    if not RESULTS.exists():
        return False
    with open(RESULTS, newline="") as f:
        return any(row.get("name") == name for row in csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-seed", type=int, default=99991)
    ap.add_argument("--eval-n", type=int, default=1500)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or get_device()
    RUNS.mkdir(parents=True, exist_ok=True)

    train_keys = {tuple(s) for _f, s in build_records(load_all_families(_DATA))}
    cuts = build_eval_set(args.eval_seed, args.eval_n, train_keys)
    print(f"eval set: {len(cuts)} cases (seed {args.eval_seed}, 60/80 cuts)")

    new = not RESULTS.exists()
    fh = open(RESULTS, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["name", "eval_seed", "val_loss", "top1", "top3", "top5", "mrr",
                    "n_layer", "n_embd", "family_dropout", "dropout", "extra", "epochs", "lr", "seed"])
        fh.flush()

    for name, cfg in CONFIGS:
        tag = f"{name}@{args.eval_seed}"
        if already_done(tag):
            print(f"[skip] {tag} already logged")
            continue
        t0 = time.time()
        out, vl = train_config(name, cfg)
        if not (out / "best.pt").exists():
            print(f"[skip] {name}: no checkpoint")
            continue
        m = eval_nextstep(out, cuts, device)
        w.writerow([tag, args.eval_seed, vl, f"{m['top1']:.4f}", f"{m['top3']:.4f}",
                    f"{m['top5']:.4f}", f"{m['mrr']:.4f}", cfg["n_layer"], cfg["n_embd"],
                    cfg["family_dropout"], cfg["dropout"], cfg["extra"], cfg["epochs"],
                    cfg["lr"], cfg["seed"]])
        fh.flush()
        print(f"[done] {name:12} val={vl:>7} top1={m['top1']:.4f} top3={m['top3']:.4f} "
              f"mrr={m['mrr']:.4f}  ({time.time()-t0:.0f}s)")
    fh.close()
    print(f"\nsweep complete -> {RESULTS}")


if __name__ == "__main__":
    main()
