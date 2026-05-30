"""Validity-guided decoding — a model+validator hybrid for harder OOD families.

Greedy decoding can wander into a rule violation on an unfamiliar family. Here, at
each step we take the model's ranked next-step candidates and emit the
highest-probability one that does NOT increase the number of rule violations
(checked incrementally with the organizers' validator). The model still drives
*what* comes next; the validator only vetoes locally-illegal choices. This is a
legitimate inference-time tool (grammar-constrained decoding), and it targets the
families where plain greedy struggles most (IGBT/MOSFET OOD valid-completion ~0.66
/ 0.76).

    python -m process_lm.guided --ckpt process_lm/runs/ood/igbt_real/best.pt --family igbt
"""
from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import torch

from .data import build_records, load_all_families, split_records
from .predict import complete, get_device, load_model
from .tokenizer import SPECIAL_TOKENS, Tokenizer

_DATA_DIR = Path(__file__).resolve().parent.parent / "tracks/industrial-infineon/training_data"
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))
import generate_sequences as gs  # type: ignore  # noqa: E402


def _amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()


@torch.no_grad()
def complete_guided(model, tok, family, partial, device, k=8, max_new=250):
    cur = list(partial)
    base = len(gs.validate_sequence(cur))
    ids = tok.encode_sequence(cur, family, add_bos=True, add_eos=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    block = model.cfg.block_size
    specials = set(SPECIAL_TOKENS)
    for _ in range(max_new):
        with _amp(device):
            logits, _ = model(x[:, -block:])
        order = torch.argsort(logits[0, -1].float(), descending=True).tolist()
        chosen = None
        for tid in order[: k + 6]:
            s = tok.itos[tid]
            if tid == tok.eos_id:
                chosen = None  # model wants to stop
                break
            if s in specials:
                continue
            if len(gs.validate_sequence(cur + [s])) <= base:  # no NEW violation
                chosen = s
                break
        if chosen is None:
            # model's top choice was EOS (stop) — but only stop if route looks done
            if "SHIP LOT" in cur:
                break
            # else force the top legal-ish step to keep going
            for tid in order:
                if tok.itos[tid] not in specials:
                    chosen = tok.itos[tid]
                    break
            if chosen is None:
                break
        cur.append(chosen)
        base = len(gs.validate_sequence(cur))
        x = torch.cat([x, torch.tensor([[tok.stoi[chosen]]], dtype=torch.long, device=device)], dim=1)
        if chosen == "SHIP LOT":
            break
    return cur[len(partial):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--family", required=True, choices=["mosfet", "igbt", "ic"])
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or get_device()
    model = load_model(Path(args.ckpt), device)
    tok = Tokenizer.load(Path(args.ckpt).parent / "tokenizer.json")
    recs = build_records(load_all_families(_DATA_DIR))
    _, val = split_records(recs, 100, 0)
    held = [r for r in val if r[0] == args.family][: args.limit]

    g_ok = gd_ok = n = 0
    for fam, steps in held:
        for frac in (0.6, 0.8):
            cut = max(1, int(len(steps) * frac))
            part = steps[:cut]
            greedy = complete(model, tok, fam, part, device)
            guided = complete_guided(model, tok, fam, part, device)
            n += 1
            g_ok += len(gs.validate_sequence(part + greedy)) == 0
            gd_ok += len(gs.validate_sequence(part + guided)) == 0
    print(f"OOD valid-completion on held-out {args.family.upper()} (n={n}):")
    print(f"  greedy decoding         : {g_ok/n:.3f}")
    print(f"  validity-guided decoding: {gd_ok/n:.3f}")
    print(f"  -> guided {'+' if gd_ok>=g_ok else ''}{(gd_ok-g_ok)/n:.3f}")


if __name__ == "__main__":
    main()
