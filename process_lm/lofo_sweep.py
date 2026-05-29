"""Full leave-one-family-out sweep → ID-vs-OOD table.

Trains a full (all-families) baseline plus one model per held-out family, then
compares quality on each held-out family:
  ID  = full model (trained on all 3) evaluated on family F's held-out sequences
  OOD = the model that never trained on F, evaluated on F (proxy for the hidden 4th family)

Run from the repo root:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m process_lm.lofo_sweep

Re-runnable: training is skipped for any run dir that already has best.pt.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .data import build_records, load_all_families, split_records
from .metrics import blocklevel_metrics, completion_metrics, nextstep_metrics
from .predict import _cuts, complete, get_device, load_model, predict_next
from .tokenizer import Tokenizer

FAMILIES = ["mosfet", "igbt", "ic"]
DATA_DIR = "tracks/industrial-infineon/training_data"
EPOCHS = 15
EVAL_LIMIT = 50  # sequences per family for evaluation (each yields 2 cut points)


def train(out_dir: str, extra: list[str] | None = None) -> None:
    if (Path(out_dir) / "best.pt").exists():
        print(f"[skip] {out_dir} already trained")
        return
    cmd = [sys.executable, "-m", "process_lm.train",
           "--epochs", str(EPOCHS), "--out-dir", out_dir, *(extra or [])]
    print(f"[train] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def evaluate(ckpt: str, family: str, device: str) -> dict:
    tok = Tokenizer.load(Path(ckpt).parent / "tokenizer.json")
    model = load_model(ckpt, device)
    _, val = split_records(build_records(load_all_families(DATA_DIR)), 100, 0)
    recs = [r for r in val if r[0] == family][:EVAL_LIMIT]

    ranked, ns_truth = [], []
    preds, cmp_truth = [], []
    for fam, steps in recs:
        for cut in _cuts(steps):
            ranked.append(predict_next(model, tok, fam, steps[:cut], device))
            ns_truth.append(steps[cut])
            preds.append(complete(model, tok, fam, steps[:cut], device))
            cmp_truth.append(steps[cut:])
    ns = nextstep_metrics(ranked, ns_truth)
    bm = blocklevel_metrics(preds, cmp_truth)
    cm = completion_metrics(preds, cmp_truth)
    return {"top1": ns["top1"], "top5": ns["top5"], "mrr": ns["mrr"],
            "block_ned": bm["block_norm_edit_distance"], "tok_ned": cm["norm_edit_distance"]}


def main() -> None:
    device = get_device()
    print(f"device={device}\n--- training (full + leave-one-out per family) ---")
    train("process_lm/runs/full")
    for f in FAMILIES:
        train(f"process_lm/runs/lofo_{f}", ["--hold-out-family", f])

    print("\n==== ID vs OOD per family ====")
    print(f"{'family':7} {'cond':4} {'top1':>6} {'top5':>6} {'mrr':>6} {'blkNED':>7} {'tokNED':>7}")
    rows = []
    for f in FAMILIES:
        idr = evaluate("process_lm/runs/full/best.pt", f, device)
        ood = evaluate(f"process_lm/runs/lofo_{f}/best.pt", f, device)
        for cond, r in (("ID", idr), ("OOD", ood)):
            print(f"{f:7} {cond:4} {r['top1']:6.3f} {r['top5']:6.3f} {r['mrr']:6.3f} "
                  f"{r['block_ned']:7.3f} {r['tok_ned']:7.3f}")
        rows.append((f, idr, ood))

    print("\n==== generalization gap (ID - OOD) ====")
    for f, idr, ood in rows:
        print(f"{f:7} drop  top1 {idr['top1'] - ood['top1']:+.3f}  "
              f"top5 {idr['top5'] - ood['top5']:+.3f}  mrr {idr['mrr'] - ood['mrr']:+.3f}  "
              f"blkNED {ood['block_ned'] - idr['block_ned']:+.3f}")


if __name__ == "__main__":
    main()
