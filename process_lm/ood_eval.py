"""OOD completion eval (Task-4 proxy): does validity-guided decoding lift the
SCORED completion metrics (Block-acc / NED), not just validity, on a held-out family?

For each family F we use the LOFO model that never trained on F, generate fresh F
sequences (genuinely OOD for that model), cut at 60/80, and compare plain greedy vs
guided+repair completion with the official metric functions. On in-distribution data
guided == greedy (already valid); the question is whether on an UNFAMILIAR family it
also wins the scored metrics, where plain greedy drifts off-grammar.

    python -m process_lm.ood_eval --n 100
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

from .guided import complete_guided, repair_route
from .predict import complete, get_device, load_model
from .tokenizer import Tokenizer

_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "tracks/industrial-infineon/training_data"
_PF = _ROOT / "tracks/industrial-infineon/participant_files"
for _p in (_DATA, _PF):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import generate_sequences as gs  # noqa: E402
from eval_metrics import block_level_accuracy, normalized_edit_distance, token_accuracy  # noqa: E402

LOFO = {
    "mosfet": _ROOT / "process_lm/runs/lofo_mosfet",
    "igbt": _ROOT / "process_lm/runs/lofo_igbt",
    "ic": _ROOT / "process_lm/runs/lofo_ic",
}


def fresh_family(fam, n, seed):
    rng = random.Random(seed)
    return [gs.generate_sequence(fam, rng) for _ in range(n)]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=99991)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or get_device()

    print(f"OOD completion (LOFO models on their held-out family), n={args.n}/family, 60/80 cuts")
    print(f"{'family':8} {'decode':8} {'Block-acc':>10} {'NED':>8} {'TokenAcc':>9} {'Valid':>7}")
    grand = defaultdict(lambda: defaultdict(float))
    for fam, run in LOFO.items():
        if not (run / "best.pt").exists():
            print(f"  [skip] {fam}: no LOFO checkpoint")
            continue
        model = load_model(run / "best.pt", device)
        tok = Tokenizer.load(run / "tokenizer.json")
        seqs = fresh_family(fam, args.n, args.seed)
        agg = defaultdict(lambda: defaultdict(float))
        for steps in seqs:
            for frac in (0.6, 0.8):
                cut = min(len(steps) - 1, max(2, int(len(steps) * frac)))
                partial, ref = steps[:cut], steps[cut:]
                g = complete(model, tok, fam, partial, device)
                gd = repair_route(list(partial) + complete_guided(model, tok, fam, partial, device))[len(partial):]
                for name, pred in (("greedy", g), ("guided", gd)):
                    a = agg[name]
                    a["blk"] += block_level_accuracy(pred, ref)
                    a["ned"] += normalized_edit_distance(pred, ref)
                    a["ta"] += token_accuracy(pred, ref)
                    a["valid"] += float(len(gs.validate_sequence(partial + pred)) == 0)
                    a["n"] += 1
        for name in ("greedy", "guided"):
            a = agg[name]
            n = a["n"]
            print(f"{fam:8} {name:8} {a['blk']/n:10.4f} {a['ned']/n:8.4f} {a['ta']/n:9.4f} {a['valid']/n:7.3f}")
            for k in ("blk", "ned", "ta", "valid", "n"):
                grand[name][k] += a[k]
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    print("-" * 56)
    for name in ("greedy", "guided"):
        a = grand[name]
        n = a["n"]
        if n:
            print(f"{'ALL':8} {name:8} {a['blk']/n:10.4f} {a['ned']/n:8.4f} {a['ta']/n:9.4f} {a['valid']/n:7.3f}")
    g, gd = grand["greedy"], grand["guided"]
    if g["n"] and gd["n"]:
        print(f"\n  guided - greedy:  dBlock {gd['blk']/gd['n']-g['blk']/g['n']:+.4f}  "
              f"dNED {gd['ned']/gd['n']-g['ned']/g['n']:+.4f}  "
              f"dValid {gd['valid']/gd['n']-g['valid']/g['n']:+.4f}")


if __name__ == "__main__":
    main()
