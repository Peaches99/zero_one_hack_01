"""Overnight scaling + OOD experiment grid — built to run on the RTX 5090.

Returns RESULTS, not just a checkpoint. Three studies, all scored on the same
leave-one-family-out (OOD) proxy plus an in-distribution (ID) val:

  1. DATA SCALING   — how do ID and OOD move as training data grows?
                      (real sequences: 200 / 1k / 5k / 20k via --extra-per-family)
  2. MODEL SCALING  — tiny / small / medium / large at fixed data.
  3. HYBRID DOSE    — OOD vs amount of hybrid pseudo-family augmentation
                      (0 / 2k / 8k / 20k), the key generalization study.

Each run trains a model and immediately evaluates it; one JSON row per run is
appended to runs/overnight/results.jsonl so a crash never loses finished work
(re-running skips any run whose checkpoint already exists).

Usage on the 5090 (CUDA auto-detected):
    python -m process_lm.overnight --hold-out ic --epochs 25
    # smaller smoke first:
    python -m process_lm.overnight --hold-out ic --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from .lofo_analysis import (
    COMPLETION_LIMIT, NEXTSTEP_LIMIT, eval_completion, eval_nextstep,
    held_out, load_validator, mean_nll,
)
from .predict import get_device, load_model
from .tokenizer import Tokenizer

RUNS = Path("process_lm/runs/overnight")
RESULTS = RUNS / "results.jsonl"

# (name, model-flags, data-flags) — extended for the real run, trimmed for --smoke
SIZES = {
    "tiny":   ["--n-layer", "2", "--n-embd", "128", "--n-head", "4"],
    "small":  ["--n-layer", "6", "--n-embd", "256", "--n-head", "8"],
    "medium": ["--n-layer", "8", "--n-embd", "512", "--n-head", "8"],
    "large":  ["--n-layer", "12", "--n-embd", "768", "--n-head", "12"],
}


def run_one(name: str, hold_out: str, epochs: int, size_flags: list[str],
            data_flags: list[str], batch: int, device: str, validate) -> dict:
    out_dir = RUNS / name
    if not (out_dir / "best.pt").exists():
        env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
        cmd = [sys.executable, "-m", "process_lm.train",
               "--epochs", str(epochs), "--out-dir", str(out_dir),
               "--hold-out-family", hold_out, "--batch-size", str(batch),
               *size_flags, *data_flags]
        print(f"\n[train:{name}] {' '.join(cmd[4:])}")
        t0 = time.time()
        subprocess.run(cmd, check=True, env=env)
        train_secs = time.time() - t0
    else:
        print(f"[have:{name}]")
        train_secs = None

    model = load_model(out_dir / "best.pt", device)
    tok = Tokenizer.load(out_dir / "tokenizer.json")
    n_params = sum(p.numel() for p in model.parameters())

    # OOD = held-out family; ID = a slice of the trained families' held-out val.
    ood_ns = eval_nextstep(model, tok, held_out(hold_out, NEXTSTEP_LIMIT), device)
    ood_cm = eval_completion(model, tok, held_out(hold_out, COMPLETION_LIMIT), device, validate)
    ood_ppl = math.exp(mean_nll(model, tok, held_out(hold_out, NEXTSTEP_LIMIT), device))
    id_fam = next(f for f in ("mosfet", "igbt", "ic") if f != hold_out)
    id_ns = eval_nextstep(model, tok, held_out(id_fam, NEXTSTEP_LIMIT), device)

    row = {
        "name": name, "hold_out": hold_out, "params": n_params, "train_secs": train_secs,
        "id_top1": id_ns["overall"]["top1"],
        "ood_top1": ood_ns["overall"]["top1"], "ood_mrr": ood_ns["overall"]["mrr"],
        "ood_valid_completion": ood_cm.get("valid_completion_rate"),
        "ood_block_ned": ood_cm["block_norm_edit_distance"],
        "ood_token_acc": ood_cm["token_accuracy"], "ood_ppl": ood_ppl,
    }
    with open(RESULTS, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[done:{name}] params={n_params/1e6:.1f}M  ID_top1={row['id_top1']:.3f}  "
          f"OOD_top1={row['ood_top1']:.3f}  OOD_valid={row['ood_valid_completion']:.3f}  "
          f"OOD_ppl={ood_ppl:.2f}")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold-out", default="ic", choices=["mosfet", "igbt", "ic"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--smoke", action="store_true", help="tiny/fast grid to validate the harness")
    args = ap.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    device = get_device()
    validate = load_validator()
    print(f"device={device}  hold_out={args.hold_out}  epochs={args.epochs}")
    if device == "cpu":
        print("[warn] no GPU detected — this grid is meant for the 5090 (CUDA).")

    grid: list[tuple[str, list[str], list[str]]] = []

    if args.smoke:
        grid += [
            (f"smoke_data200_{args.hold_out}", SIZES["tiny"], ["--train-limit", "200"]),
            (f"smoke_hybrid2k_{args.hold_out}", SIZES["tiny"], ["--add-hybrids", "2000", "--hybrid-tag", "random"]),
        ]
        epochs = 3
    else:
        epochs = args.epochs
        # 1. DATA SCALING (small model, vary real data volume)
        for n in (200, 1000, 5000, 20000):
            flags = ["--extra-per-family", str(max(0, (n - 2000) // 2))] if n > 2000 \
                else ["--train-limit", str(n)]
            grid.append((f"data{n}_{args.hold_out}", SIZES["small"], flags))
        # 2. MODEL SCALING (fixed ~5k data, vary model size)
        for sz in ("tiny", "small", "medium", "large"):
            grid.append((f"model_{sz}_{args.hold_out}", SIZES[sz], ["--extra-per-family", "1500"]))
        # 3. HYBRID DOSE (small model, vary hybrid augmentation) — the key OOD study
        for h in (0, 2000, 8000, 20000):
            flags = ["--extra-per-family", "1500"]
            if h:
                flags += ["--add-hybrids", str(h), "--hybrid-tag", "random"]
            grid.append((f"hybrid{h}_{args.hold_out}", SIZES["small"], flags))

    print(f"grid: {len(grid)} runs")
    for name, size_flags, data_flags in grid:
        try:
            run_one(name, args.hold_out, epochs, size_flags, data_flags,
                    args.batch_size, device, validate)
        except subprocess.CalledProcessError as e:
            print(f"[FAIL:{name}] {e} — continuing")

    print(f"\nALL DONE. Results: {RESULTS}")
    print("Summarize in the morning with:  python -m process_lm.overnight_report")


if __name__ == "__main__":
    main()
