"""Free OOD lever? A LOFO model never trained on its held-out family's <FAM:*> token,
so feeding that token at inference hands the model a random, never-updated embedding.

This tests, with PURE INFERENCE (no training), whether feeding a *trained* family
signal instead lifts OOD block-acc. For each LOFO model (held out family F) completing
fresh F sequences, we try the family signal = {F itself (untrained, current baseline),
UNK, and each of the two families the model DID train on}. If a trained signal beats
F's own token, that's a free Task-4 gain and a clean story.

    python -m process_lm.ood_familytoken --n 100
"""
from __future__ import annotations

import argparse
import random
import sys
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

LOFO = {
    "mosfet": _ROOT / "process_lm/runs/lofo_mosfet",
    "igbt": _ROOT / "process_lm/runs/lofo_igbt",
    "ic": _ROOT / "process_lm/runs/lofo_ic",
}
ALL_FAMS = ("mosfet", "igbt", "ic")
CUTS = (0.6, 0.8)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=99991)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or get_device()

    print(f"OOD family-token probe (LOFO model on held-out family), n={args.n}/family\n")
    print(f"{'heldout':8} {'family-signal':14} {'Block-acc':>10} {'NED':>8} {'TokenAcc':>9}")
    grand = defaultdict(lambda: defaultdict(float))
    for fam, run in LOFO.items():
        if not (run / "best.pt").exists():
            print(f"  [skip] {fam}")
            continue
        model = load_model(run / "best.pt", device)
        tok = Tokenizer.load(run / "tokenizer.json")
        rng = random.Random(args.seed)
        seqs = [gs.generate_sequence(fam, rng) for _ in range(args.n)]
        # signals: the held-out family's own (untrained) token, UNK, and the 2 trained
        signals = [fam, "unknown"] + [f for f in ALL_FAMS if f != fam]
        labels = {fam: f"{fam}(untrained)", "unknown": "UNK"}
        agg = defaultdict(lambda: defaultdict(float))
        for steps in seqs:
            for frac in CUTS:
                cut = min(len(steps) - 1, max(2, int(len(steps) * frac)))
                pre, rem = steps[:cut], steps[cut:]
                for sig in signals:
                    pred = complete(model, tok, sig, pre, device)
                    a = agg[sig]
                    a["blk"] += block_level_accuracy(pred, rem)
                    a["ned"] += normalized_edit_distance(pred, rem)
                    a["tok"] += token_accuracy(pred, rem)
                    a["n"] += 1
        for sig in signals:
            a = agg[sig]
            n = a["n"]
            lab = labels.get(sig, f"{sig}(trained)")
            print(f"{fam:8} {lab:14} {a['blk']/n:>10.4f} {a['ned']/n:>8.4f} {a['tok']/n:>9.4f}")
            kind = ("own" if sig == fam else "unk" if sig == "unknown" else "trained")
            grand[kind]["blk"] += a["blk"]
            grand[kind]["n"] += a["n"]
            if kind == "trained":  # track best trained-family signal per held-out fam
                grand["best_trained"]["blk"] += max(
                    agg[s]["blk"] for s in ALL_FAMS if s != fam)
                grand["best_trained"]["n"] += a["n"]
        print()
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    print("-" * 54)
    for kind in ("own", "unk", "trained"):
        g = grand[kind]
        if g["n"]:
            print(f"{'ALL':8} {kind:14} block-acc {g['blk']/g['n']:.4f}")
    own = grand["own"]["blk"] / grand["own"]["n"] if grand["own"]["n"] else 0
    unk = grand["unk"]["blk"] / grand["unk"]["n"] if grand["unk"]["n"] else 0
    trn = grand["trained"]["blk"] / grand["trained"]["n"] if grand["trained"]["n"] else 0
    print(f"\nvs current baseline (own untrained token {own:.4f}):  "
          f"UNK {unk-own:+.4f}   any-trained-family {trn-own:+.4f}")


if __name__ == "__main__":
    main()
