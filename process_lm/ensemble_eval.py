"""Ensemble + confirmation evaluator.

Averaging the probabilities of several independently-seeded floor-models reduces
variance and can lift Top-1/MRR even when each single model is at the loss floor —
the most plausible real gain. To not fool ourselves, we report single-vs-ensemble
on a SELECTION set and a separate CONFIRMATION set; a gain counts only if it holds
on both.

    python -m process_lm.ensemble_eval --ckpts a/best.pt,b/best.pt,c/best.pt \
        --eval-seed 99991 --confirm-seed 7 --n 1000 --completion
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .data import build_records, load_all_families
from .decode_lab import ensemble_complete, ensemble_predict_next
from .local_eval import generate_fresh
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
from eval_metrics import block_level_accuracy, normalized_edit_distance, token_accuracy  # noqa: E402


def eval_cuts(seed, n, exclude):
    seqs, _ = generate_fresh(n, seed, exclude)
    cuts = []
    for fam, steps in seqs:
        for frac in (0.6, 0.8):
            cut = min(len(steps) - 1, max(2, int(len(steps) * frac)))
            cuts.append((fam, steps[:cut], steps[cut], steps[cut:]))
    return cuts


@torch.no_grad()
def run(models, tok, cuts, device, ensemble, completion):
    ranked, truths = [], []
    blk = ned = ta = valid = nc = 0.0
    for fam, partial, tn, ref in cuts:
        if ensemble:
            preds = ensemble_predict_next(models, tok, fam, partial, device)
        else:
            preds = predict_next(models[0], tok, fam, partial, device, k=5)
        ranked.append(preds)
        truths.append(tn)
        if completion:
            comp = (ensemble_complete(models, tok, fam, partial, device) if ensemble
                    else complete(models[0], tok, fam, partial, device))
            blk += block_level_accuracy(comp, ref)
            ned += normalized_edit_distance(comp, ref)
            ta += token_accuracy(comp, ref)
            valid += float(len(gs.validate_sequence(partial + comp)) == 0)
            nc += 1
    out = dict(nextstep_metrics(ranked, truths))
    if completion and nc:
        out.update(blk=blk / nc, ned=ned / nc, ta=ta / nc, valid=valid / nc)
    return out


def _line(tag, m):
    s = f"  {tag:18} Top-1 {m['top1']:.4f}  Top-3 {m['top3']:.4f}  Top-5 {m['top5']:.4f}  MRR {m['mrr']:.4f}"
    if "blk" in m:
        s += f"  | Block {m['blk']:.4f}  NED {m['ned']:.4f}  Tok {m['ta']:.4f}  Valid {m['valid']:.3f}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="comma-separated best.pt paths")
    ap.add_argument("--eval-seed", type=int, default=99991)
    ap.add_argument("--confirm-seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--completion", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or get_device()

    paths = [Path(p) for p in args.ckpts.split(",") if p.strip()]
    models = [load_model(p, device) for p in paths]
    tok = Tokenizer.load(paths[0].parent / "tokenizer.json")
    print(f"{len(models)} models | completion={args.completion}")

    train_keys = {tuple(s) for _f, s in build_records(load_all_families(_DATA))}
    for label, seed in (("SELECTION", args.eval_seed), ("CONFIRM", args.confirm_seed)):
        cuts = eval_cuts(seed, args.n, train_keys)
        single = run(models, tok, cuts, device, ensemble=False, completion=args.completion)
        ens = run(models, tok, cuts, device, ensemble=True, completion=args.completion)
        print(f"\n=== {label} set (seed {seed}, {len(cuts)} cases) ===")
        print(_line("single(best)", single))
        print(_line(f"ensemble({len(models)})", ens))
        print(f"  delta              Top-1 {ens['top1']-single['top1']:+.4f}  "
              f"MRR {ens['mrr']-single['mrr']:+.4f}"
              + (f"  Block {ens['blk']-single['blk']:+.4f}" if "blk" in ens else ""))


if __name__ == "__main__":
    main()
