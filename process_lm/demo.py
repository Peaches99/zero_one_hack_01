"""Before/after demonstrator — baseline (untrained) vs trained model.

The track asks for exactly this: identical inputs, baseline output beside trained
output, for next-step prediction and sequence completion. We add a process-logic
check (does each completion pass the organizers' validator?) so the difference is
not just "looks better" but "is actually a legal process route".

    python -m process_lm.demo --ckpt process_lm/runs/full_big/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .data import build_records, load_all_families, split_records
from .model import GPT, GPTConfig
from .predict import complete, get_device, load_model, predict_next
from .tokenizer import Tokenizer

_DATA_DIR = Path(__file__).resolve().parent.parent / "tracks/industrial-infineon/training_data"
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))
import generate_sequences as gs  # type: ignore  # noqa: E402


def _fresh_baseline(model, device):
    """An untrained model with the same architecture (random init) = the baseline."""
    base = GPT(GPTConfig(**{k: getattr(model.cfg, k) for k in
                            ("vocab_size", "block_size", "n_layer", "n_head", "n_embd", "dropout")}))
    return base.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frac", type=float, default=0.6)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or get_device()
    model = load_model(Path(args.ckpt), device)
    tok = Tokenizer.load(Path(args.ckpt).parent / "tokenizer.json")
    base = _fresh_baseline(model, device)

    recs = build_records(load_all_families(_DATA_DIR))
    _, val = split_records(recs, 100, 0)
    picks = []
    for fam in ("mosfet", "igbt", "ic"):
        picks.append(next((r for r in val if r[0] == fam)))

    for fam, steps in picks:
        cut = max(1, int(len(steps) * args.frac))
        partial = steps[:cut]
        truth_next = steps[cut]
        print("=" * 78)
        print(f"FAMILY {fam.upper()}   (showing {cut}/{len(steps)} steps, predict the next)")
        print(f"  context tail : ...{' | '.join(partial[-4:])}")
        print(f"  TRUE next    : {truth_next}")
        b5 = predict_next(base, tok, fam, partial, device)
        t5 = predict_next(model, tok, fam, partial, device)
        print(f"  baseline top5: {b5}   {'HIT' if truth_next in b5 else 'miss'}")
        print(f"  trained  top5: {t5}   {'HIT' if truth_next in t5 else 'miss'}")

        # completion (the rest of the route) + validity
        b_comp = complete(base, tok, fam, partial, device)
        t_comp = complete(model, tok, fam, partial, device)
        b_full, t_full = partial + b_comp, partial + t_comp
        b_ok = len(gs.validate_sequence(b_full)) == 0
        t_ok = len(gs.validate_sequence(t_full)) == 0
        print(f"  completion (baseline): {len(b_comp)} steps -> "
              f"{'VALID route' if b_ok else 'INVALID (' + str(len(gs.validate_sequence(b_full))) + ' violations)'}")
        print(f"  completion (trained) : {len(t_comp)} steps -> "
              f"{'VALID route' if t_ok else 'INVALID (' + str(len(gs.validate_sequence(t_full))) + ' violations)'}")
        print(f"  trained completion head: {' | '.join(t_comp[:6])} ...")
    print("=" * 78)


if __name__ == "__main__":
    main()
