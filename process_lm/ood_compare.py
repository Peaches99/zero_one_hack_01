"""Decisive OOD-lever comparison: which data/regularization strategy generalizes?

The scaling grid showed that more *same-family* real data does not help the
held-out family — it just overfits faster. This isolates the levers that should:
validated hybrid pseudo-families, v2 structural diversity (variable cycle count),
family-token dropout, and their combinations. Same held-out family, same model
size, same epochs — only the training-data strategy changes — so the OOD
valid-completion-rate / perplexity differences are attributable to the lever.

Honest by design: every config is evaluated on the SAME 100 real held-out
sequences of the held-out family (never trained on, never leaked via hybrids/v2).

    python -m process_lm.ood_compare --hold-out ic --epochs 30
    python -m process_lm.ood_compare --hold-out mosfet --configs best
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

RUNS = Path("process_lm/runs/ood")
SMALL = ["--n-layer", "6", "--n-embd", "256", "--n-head", "8"]

# Each config: (name, extra train flags). Model size + epochs are shared.
CONFIG_SETS = {
    "levers": [
        ("real", []),
        ("real_fd15", ["--family-dropout", "0.15"]),
        ("real_fd30", ["--family-dropout", "0.30"]),
        ("hyb8k", ["--add-hybrids", "8000", "--hybrid-tag", "random"]),
        ("v2_8k", ["--add-v2", "8000"]),
        ("v2_8k_fd15", ["--add-v2", "8000", "--family-dropout", "0.15"]),
        ("v2_24k_fd15", ["--add-v2", "24000", "--family-dropout", "0.15"]),
    ],
    "best": [
        ("v2_8k_fd15", ["--add-v2", "8000", "--family-dropout", "0.15"]),
    ],
}


def train_and_eval(name: str, hold_out: str, epochs: int, extra: list[str],
                   batch: int, device: str, validate) -> dict:
    out_dir = RUNS / f"{hold_out}_{name}"
    if not (out_dir / "best.pt").exists():
        env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1", "PYTHONUTF8": "1"}
        cmd = [sys.executable, "-m", "process_lm.train",
               "--epochs", str(epochs), "--out-dir", str(out_dir),
               "--hold-out-family", hold_out, "--batch-size", str(batch),
               *SMALL, *extra]
        print(f"\n[train:{name}] {' '.join(extra) or '(real only)'}")
        t0 = time.time()
        subprocess.run(cmd, check=True, env=env)
        secs = time.time() - t0
    else:
        print(f"[have:{name}]")
        secs = None

    model = load_model(out_dir / "best.pt", device)
    tok = Tokenizer.load(out_dir / "tokenizer.json")
    ns, cm = held_out(hold_out, NEXTSTEP_LIMIT), held_out(hold_out, COMPLETION_LIMIT)
    ood_ns = eval_nextstep(model, tok, ns, device)
    ood_ns_unk = eval_nextstep(model, tok, ns, device, family_override="unk")
    ood_cm = eval_completion(model, tok, cm, device, validate)
    ood_cm_unk = eval_completion(model, tok, cm, device, validate, family_override="unk")
    ood_ppl = math.exp(mean_nll(model, tok, ns, device))
    shared = ood_ns.get("shared_vocab_only") or {}
    row = {
        "name": name, "hold_out": hold_out, "train_secs": secs,
        "ood_top1": ood_ns["overall"]["top1"], "ood_mrr": ood_ns["overall"]["mrr"],
        "ood_top1_shared": shared.get("top1"),            # logic on nameable steps
        "ood_top1_unk": ood_ns_unk["overall"]["top1"],    # 4th-family proxy (no family token)
        "ood_valid_completion": ood_cm.get("valid_completion_rate"),
        "ood_valid_unk": ood_cm_unk.get("valid_completion_rate"),
        "ood_block_ned": ood_cm["block_norm_edit_distance"],
        "ood_token_acc": ood_cm["token_accuracy"], "ood_ppl": ood_ppl,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    with open(RUNS / "results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[done:{name}] valid={row['ood_valid_completion']:.3f}(unk {row['ood_valid_unk']:.3f})  "
          f"top1={row['ood_top1']:.3f} shared={(row['ood_top1_shared'] or 0):.3f} "
          f"unk={row['ood_top1_unk']:.3f}  ppl={ood_ppl:.2f}")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold-out", default="ic", choices=["mosfet", "igbt", "ic"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--configs", default="levers", choices=list(CONFIG_SETS))
    args = ap.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    device = get_device()
    validate = load_validator()
    print(f"device={device}  hold_out={args.hold_out}  epochs={args.epochs}  "
          f"configs={args.configs}")

    rows = []
    for name, extra in CONFIG_SETS[args.configs]:
        try:
            rows.append(train_and_eval(f"{name}", args.hold_out, args.epochs, extra,
                                       args.batch_size, device, validate))
        except subprocess.CalledProcessError as e:
            print(f"[FAIL:{name}] {e} — continuing")

    rows.sort(key=lambda r: r.get("ood_top1_unk") or 0, reverse=True)
    print(f"\n=== OOD LEVER COMPARISON (hold-out {args.hold_out}); "
          f"unk-token columns = true 4th-family proxy ===")
    print(f"  {'config':13} {'valid':>6} {'vUNK':>6} {'top1':>6} {'shared':>6} {'t1UNK':>6} {'ppl':>7}")
    for r in rows:
        print(f"  {r['name']:13} {r['ood_valid_completion']:6.3f} {r['ood_valid_unk']:6.3f} "
              f"{r['ood_top1']:6.3f} {(r['ood_top1_shared'] or 0):6.3f} {r['ood_top1_unk']:6.3f} "
              f"{r['ood_ppl']:7.2f}")
    if rows:
        best = rows[0]
        print(f"\n  WINNER (by unk-token top1): {best['name']}  "
              f"(top1_unk {best['ood_top1_unk']:.3f}, valid_unk {best['ood_valid_unk']:.3f})")


if __name__ == "__main__":
    main()
