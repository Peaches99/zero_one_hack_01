"""Does hybrid augmentation close the OOD gap? Before/after on each held-out family.

For each family F:
  baseline = runs/lofo_{F}            (trained on the other 2 real families)
  hybrid   = runs/lofo_{F}_hybrid     (same, plus N validated hybrid routes drawn
                                       ONLY from the other 2 families)
Both are evaluated on the SAME 100 held-out F sequences. The headline question:
does hybrid augmentation raise OOD valid-completion-rate and cut OOD perplexity?

Single foreground run (trains the hybrid models if missing, one at a time):
    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m process_lm.hybrid_experiment --n-hybrid 4000
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

from .lofo_analysis import (
    COMPLETION_LIMIT, NEXTSTEP_LIMIT, eval_completion, eval_nextstep,
    held_out, load_validator, mean_nll,
)
from .predict import get_device, load_model
from .tokenizer import Tokenizer

FAMILIES = ["mosfet", "igbt", "ic"]
RUNS = Path("process_lm/runs")
EPOCHS = 15


def ensure(out_dir: Path, extra: list[str]) -> None:
    if (out_dir / "best.pt").exists():
        print(f"[have] {out_dir}")
        return
    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    cmd = [sys.executable, "-m", "process_lm.train",
           "--epochs", str(EPOCHS), "--out-dir", str(out_dir), *extra]
    print(f"[train] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def evaluate(run_dir: Path, family: str, device, validate) -> dict:
    model = load_model(run_dir / "best.pt", device)
    tok = Tokenizer.load(run_dir / "tokenizer.json")
    ns = eval_nextstep(model, tok, held_out(family, NEXTSTEP_LIMIT), device)
    cm = eval_completion(model, tok, held_out(family, COMPLETION_LIMIT), device, validate)
    nll = mean_nll(model, tok, held_out(family, NEXTSTEP_LIMIT), device)
    return {
        "top1": ns["overall"]["top1"], "mrr": ns["overall"]["mrr"],
        "valid": cm.get("valid_completion_rate", float("nan")),
        "blkNED": cm["block_norm_edit_distance"], "tokAcc": cm["token_accuracy"],
        "ppl": math.exp(nll),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-hybrid", type=int, default=4000)
    ap.add_argument("--hybrid-tag", default="random",
                    help="'random' varies the family tag per hybrid route (dropout-like)")
    args = ap.parse_args()

    device = get_device()
    validate = load_validator()

    print("=== train baseline + hybrid-augmented LOFO models (foreground) ===")
    for f in FAMILIES:
        ensure(RUNS / f"lofo_{f}", ["--hold-out-family", f])
        ensure(RUNS / f"lofo_{f}_hybrid",
               ["--hold-out-family", f, "--add-hybrids", str(args.n_hybrid),
                "--hybrid-tag", args.hybrid_tag])

    print("\n=== OOD before/after (held-out family, +%d hybrids) ===" % args.n_hybrid)
    hdr = f"{'family':7} {'cond':8} {'top1':>6} {'mrr':>6} {'validCmp':>9} {'blkNED':>7} {'ppl':>7}"
    print(hdr)
    deltas = []
    for f in FAMILIES:
        base = evaluate(RUNS / f"lofo_{f}", f, device, validate)
        hyb = evaluate(RUNS / f"lofo_{f}_hybrid", f, device, validate)
        for cond, r in (("baseline", base), ("hybrid", hyb)):
            print(f"{f:7} {cond:8} {r['top1']:6.3f} {r['mrr']:6.3f} {r['valid']:9.3f} "
                  f"{r['blkNED']:7.3f} {r['ppl']:7.2f}")
        deltas.append((f, base, hyb))

    print("\n=== Δ (hybrid - baseline); + valid / - blkNED / - ppl = better ===")
    for f, base, hyb in deltas:
        print(f"  {f:7}: validCmp {hyb['valid']-base['valid']:+.3f}  "
              f"blkNED {hyb['blkNED']-base['blkNED']:+.3f}  "
              f"ppl {hyb['ppl']-base['ppl']:+.2f}  "
              f"top1 {hyb['top1']-base['top1']:+.3f}")
    # Aggregate verdict
    vavg = sum(h['valid']-b['valid'] for _, b, h in deltas) / len(deltas)
    pavg = sum(h['ppl']-b['ppl'] for _, b, h in deltas) / len(deltas)
    print(f"\n  MEAN: valid-completion {vavg:+.3f}, perplexity {pavg:+.2f}  "
          f"-> hybrids {'HELP' if (vavg > 0 and pavg < 0) else 'MIXED/HURT'}")


if __name__ == "__main__":
    main()
