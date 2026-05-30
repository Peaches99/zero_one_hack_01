"""Does FAMILY DIVERSITY (not size/data volume) drive OOD generalization?

Size and data-volume scaling are proven flat (sweep_lab). This isolates a different
axis: the NUMBER of distinct product families in training, at a FIXED data volume.
For each held-out family X (OOD target) we train, on the same total #sequences:
  * 1fam_A / 1fam_B : one other family only
  * 2fam            : both other families
  * 2fam_fd15       : both + family-dropout 0.15 (the research's "domain-dropout")
then measure OOD block-acc on fresh X. If 2fam > 1fam at equal volume, diversity
helps generalization -> our real 3-family submission should beat this 2-family proxy
on the hidden 4th family. Either way it is the Level-3 "scaling effects on
generalization" deliverable the brief asks for.

    python -m process_lm.diversity_ood --volume 4000 --epochs 20 --eval-n 150
Resumable: a condition whose checkpoint + logged result exist is skipped.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "tracks/industrial-infineon/training_data"
_PF = _ROOT / "tracks/industrial-infineon/participant_files"
for _p in (_DATA, _PF):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import generate_sequences as gs  # noqa: E402
from eval_metrics import (block_level_accuracy, normalized_edit_distance,  # noqa: E402
                          token_accuracy)

from .predict import complete, get_device, load_model  # noqa: E402
from .tokenizer import Tokenizer  # noqa: E402

FAMILIES = ("mosfet", "igbt", "ic")
RUNS = _ROOT / "process_lm/runs/diversity"
RESULTS = _ROOT / "process_lm/runs/diversity_results.csv"
N_LAYER, N_EMBD = 6, 384  # size is proven irrelevant; small = fast


def conditions(held_out):
    others = sorted(f for f in FAMILIES if f != held_out)
    a, b = others
    return [
        (f"1fam_{a}", a, 0.0),
        (f"1fam_{b}", b, 0.0),
        ("2fam", f"{a},{b}", 0.0),
        ("2fam_fd15", f"{a},{b}", 0.15),
    ]


def train_one(held_out, name, train_families, fd, volume, epochs, seed):
    out = RUNS / f"{held_out}_{name}"
    if (out / "best.pt").exists():
        return out, "cached"
    cmd = [sys.executable, "-m", "process_lm.train",
           "--out-dir", str(out), "--hold-out-family", held_out,
           "--train-families", train_families, "--family-dropout", str(fd),
           "--extra-per-family", str(volume), "--train-limit", str(volume),
           "--n-layer", str(N_LAYER), "--n-embd", str(N_EMBD),
           "--epochs", str(epochs), "--lr", "3e-4", "--batch-size", "64", "--seed", str(seed)]
    env = {**os.environ, "PYTHONUTF8": "1"}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0 or not (out / "best.pt").exists():
        print(f"[FAIL] {held_out}/{name}\n{r.stdout[-1200:]}\n{r.stderr[-800:]}")
        return out, "fail"
    vl = ""
    for line in r.stdout.splitlines():
        if "best val loss" in line:
            vl = line.split("best val loss")[1].strip().split()[0].rstrip(".")
    return out, vl or "ok"


@torch.no_grad()
def eval_ood(ckpt_dir, held_out, n, seed, device):
    model = load_model(ckpt_dir / "best.pt", device)
    tok = Tokenizer.load(ckpt_dir / "tokenizer.json")
    rng = random.Random(seed)
    seqs = [gs.generate_sequence(held_out, rng) for _ in range(n)]
    agg = defaultdict(float)
    for steps in seqs:
        for frac in (0.6, 0.8):
            cut = min(len(steps) - 1, max(2, int(len(steps) * frac)))
            pre, rem = steps[:cut], steps[cut:]
            pred = complete(model, tok, held_out, pre, device)
            agg[f"blk{int(frac*100)}"] += block_level_accuracy(pred, rem)
            agg["blk"] += block_level_accuracy(pred, rem)
            agg["tok"] += token_accuracy(pred, rem)
            agg["ned"] += normalized_edit_distance(pred, rem)
            agg["valid"] += float(len(gs.validate_sequence(pre + pred)) == 0)
            agg["n"] += 1
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    half = agg["n"] / 2
    return {"blk": agg["blk"] / agg["n"], "blk60": agg["blk60"] / half,
            "blk80": agg["blk80"] / half, "tok": agg["tok"] / agg["n"],
            "ned": agg["ned"] / agg["n"], "valid": agg["valid"] / agg["n"]}


def already(tag):
    if not RESULTS.exists():
        return False
    with open(RESULTS, newline="") as f:
        return any(r.get("tag") == tag for r in csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", type=int, default=4000, help="fixed total train seqs/condition")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--eval-n", type=int, default=150)
    ap.add_argument("--eval-seed", type=int, default=99991)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or get_device()
    RUNS.mkdir(parents=True, exist_ok=True)

    new = not RESULTS.exists()
    fh = open(RESULTS, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["tag", "held_out", "condition", "train_families", "fd", "volume",
                    "val_loss", "ood_blk", "ood_blk60", "ood_blk80", "ood_tok",
                    "ood_ned", "ood_valid"])
        fh.flush()

    for held_out in FAMILIES:
        for name, tf, fd in conditions(held_out):
            tag = f"{held_out}/{name}"
            if already(tag):
                print(f"[skip] {tag}")
                continue
            t0 = time.time()
            out, vl = train_one(held_out, name, tf, fd, args.volume, args.epochs, args.seed)
            if not (out / "best.pt").exists():
                continue
            m = eval_ood(out, held_out, args.eval_n, args.eval_seed, device)
            w.writerow([tag, held_out, name, tf, fd, args.volume, vl,
                        f"{m['blk']:.4f}", f"{m['blk60']:.4f}", f"{m['blk80']:.4f}",
                        f"{m['tok']:.4f}", f"{m['ned']:.4f}", f"{m['valid']:.3f}"])
            fh.flush()
            print(f"[done] {tag:22} val={vl:>7} oodBlk={m['blk']:.4f} "
                  f"(60={m['blk60']:.3f} 80={m['blk80']:.3f}) tok={m['tok']:.3f} "
                  f"({time.time()-t0:.0f}s)")
    fh.close()
    print(f"\ndiversity sweep -> {RESULTS}")


if __name__ == "__main__":
    main()
