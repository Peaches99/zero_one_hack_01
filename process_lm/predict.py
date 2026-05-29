"""Self-scoring and demo: next-step, completion, and a process-validity sanity check.

Run from the repo root, e.g.:
    python -m process_lm.predict --mode nextstep
    python -m process_lm.predict --mode completion --limit 50
    python -m process_lm.predict --mode sanity
    python -m process_lm.predict --mode demo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .data import build_records, load_all_families, split_records
from .metrics import blocklevel_metrics, completion_metrics, nextstep_metrics
from .model import GPT, GPTConfig
from .tokenizer import SPECIAL_TOKENS, Tokenizer


def get_device(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def load_validator(data_dir):
    """Import the organizers' validate_sequence so we can score generated routes."""
    sys.path.insert(0, str(Path(data_dir).resolve()))
    try:
        from generate_sequences import validate_sequence  # type: ignore
        return validate_sequence
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"[warn] could not import validate_sequence ({e}); sanity check disabled")
        return None


def predict_next(model, tok, family, partial_steps, device, k=5):
    ids = tok.encode_sequence(partial_steps, family, add_bos=True, add_eos=False)
    x = torch.tensor([ids[-model.cfg.block_size:]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(x)
    logits = logits[0, -1, :].clone()
    for s in SPECIAL_TOKENS:           # never predict a special token as a "step"
        logits[tok.stoi[s]] = -float("inf")
    topk = torch.topk(logits, k).indices.tolist()
    return [tok.itos[i] for i in topk]


def complete(model, tok, family, partial_steps, device, max_new=220):
    ids = tok.encode_sequence(partial_steps, family, add_bos=True, add_eos=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_new, eos_id=tok.eos_id, greedy=True)
    steps = []
    for i in out[0, len(ids):].tolist():
        if i == tok.eos_id:
            break
        s = tok.itos[i]
        if s not in SPECIAL_TOKENS:
            steps.append(s)
    return steps


def _cuts(steps):
    for frac in (0.6, 0.8):
        cut = max(1, int(len(steps) * frac))
        if cut < len(steps):
            yield cut


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="process_lm/runs/v1/best.pt")
    p.add_argument("--data-dir", default="tracks/industrial-infineon/training_data")
    p.add_argument("--mode", choices=["nextstep", "completion", "sanity", "demo"], default="nextstep")
    p.add_argument("--eval-family", default=None, choices=["mosfet", "igbt", "ic"],
                   help="restrict evaluation to one family's held-out sequences (ID vs OOD)")
    p.add_argument("--val-per-family", type=int, default=100)
    p.add_argument("--limit", type=int, default=0, help="cap #val sequences (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = get_device(args.device)
    tok = Tokenizer.load(Path(args.ckpt).parent / "tokenizer.json")
    model = load_model(args.ckpt, device)

    records = build_records(load_all_families(Path(args.data_dir)))
    _, val_recs = split_records(records, args.val_per_family, args.seed)
    if args.eval_family:
        val_recs = [r for r in val_recs if r[0] == args.eval_family.lower()]
    if args.limit:
        val_recs = val_recs[:args.limit]

    if args.mode == "nextstep":
        ranked, truths = [], []
        for fam, steps in val_recs:
            for cut in _cuts(steps):
                ranked.append(predict_next(model, tok, fam, steps[:cut], device))
                truths.append(steps[cut])
        print(nextstep_metrics(ranked, truths))

    elif args.mode == "completion":
        validate = load_validator(args.data_dir)
        preds, truths, partials = [], [], []
        for fam, steps in val_recs:
            for cut in _cuts(steps):
                preds.append(complete(model, tok, fam, steps[:cut], device))
                truths.append(steps[cut:])
                partials.append(steps[:cut])
        print(completion_metrics(preds, truths))
        print(blocklevel_metrics(preds, truths))
        if validate is not None and preds:
            matched = valid_diff = invalid = 0
            for part, pr, tr in zip(partials, preds, truths):
                if pr == tr:
                    matched += 1
                elif len(validate(part + pr)) == 0:
                    valid_diff += 1
                else:
                    invalid += 1
            n = len(preds)
            print(f"correctness breakdown over {n} completions:")
            print(f"  exact match            : {matched / n:.3f}")
            print(f"  valid but different    : {valid_diff / n:.3f}   <- legal route, different coin-flips")
            print(f"  invalid (rule-breaking): {invalid / n:.3f}")
            print(f"  -> process-valid total : {(matched + valid_diff) / n:.3f}")

    elif args.mode == "sanity":
        validate = load_validator(args.data_dir)
        if validate is None:
            return
        valid = total = 0
        for fam, steps in val_recs:
            gen = complete(model, tok, fam, ["RECEIVE WAFER LOT"], device, max_new=model.cfg.block_size)
            full = ["RECEIVE WAFER LOT"] + gen
            total += 1
            if len(validate(full)) == 0:
                valid += 1
        print(f"process-valid generated routes: {valid}/{total} = {valid / max(total, 1):.3f}")

    elif args.mode == "demo":
        fam, steps = val_recs[0]
        cut = max(1, int(len(steps) * 0.6))
        print(f"family={fam}  cut {cut}/{len(steps)}")
        print("context tail:", steps[max(0, cut - 4):cut])
        print("top-5 next  :", predict_next(model, tok, fam, steps[:cut], device))
        print("true next   :", steps[cut])


if __name__ == "__main__":
    main()
