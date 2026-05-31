"""Decode lab — try to beat greedy on the official completion/next-step metrics.

Runs several decoding strategies on the SAME fresh, unseen sequences (paired) and
scores each with the ORGANIZERS' OWN metric functions, so any gain is real, not
noise. The model is at the loss floor, so next-step is near its ceiling; the real
headroom is sequence-level completion metrics where greedy (per-token-optimal) is
not metric-optimal.

Strategies (completion):
  greedy    : argmax each step (baseline = what we submit)
  guided    : validity-guided greedy (validator vetoes illegal next steps)
  beam      : length-normalized beam search (width W)
  mbr       : sample K, keep the consensus (min mean edit-distance) -> targets NED
  ensemble  : probability-average of final + full (both all-3-family) greedy

    python -m process_lm.decode_lab --strategies greedy,beam,mbr,ensemble --n 300
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

from .data import build_records, load_all_families
from .local_eval import generate_fresh
from .metrics import nextstep_metrics
from .predict import complete, get_device, load_model, predict_next
from .tokenizer import SPECIAL_TOKENS, EOS, Tokenizer

_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "tracks/industrial-infineon/training_data"
_PF = _ROOT / "tracks/industrial-infineon/participant_files"
for _p in (_DATA, _PF):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import generate_sequences as gs  # noqa: E402
from eval_metrics import (  # noqa: E402
    _block_signature,
    block_level_accuracy,
    normalized_edit_distance,
    token_accuracy,
)
from .guided import complete_guided  # noqa: E402

FAMS = ["mosfet", "igbt", "ic"]
_SPECIAL_IDS = None  # filled once tok is known


def _trim(ids_after, tok):
    """Token ids after the prefix -> step strings, stop at EOS or SHIP LOT."""
    out = []
    for i in ids_after:
        if i == tok.eos_id:
            break
        s = tok.itos[i]
        if s not in SPECIAL_TOKENS:
            out.append(s)
        if s == "SHIP LOT":
            break
    return out


@torch.no_grad()
def beam_complete(model, tok, fam, partial, device, width=5, max_new=240):
    ids = tok.encode_sequence(partial, fam, add_bos=True, add_eos=False)
    block = model.cfg.block_size
    mask = [tok.stoi[s] for s in SPECIAL_TOKENS if s != EOS]
    beams = [(ids, 0.0, False)]  # (ids, logprob, done)
    for _ in range(max_new):
        if all(d for _, _, d in beams):
            break
        cand = []
        for seq, lp, done in beams:
            if done:
                cand.append((seq, lp, True))
                continue
            x = torch.tensor([seq[-block:]], dtype=torch.long, device=device)
            logits, _ = model(x)
            logp = torch.log_softmax(logits[0, -1].float(), dim=-1)
            logp[mask] = -float("inf")
            topv, topi = torch.topk(logp, width)
            for v, i in zip(topv.tolist(), topi.tolist()):
                done2 = (i == tok.eos_id) or (tok.itos[i] == "SHIP LOT")
                cand.append((seq + [i], lp + v, done2))
        # length-normalized score keeps it from collapsing to short routes
        cand.sort(key=lambda c: c[1] / max(1, len(c[0]) - len(ids)), reverse=True)
        beams = cand[:width]
    best = max(beams, key=lambda c: c[1] / max(1, len(c[0]) - len(ids)))
    return _trim(best[0][len(ids):], tok)


@torch.no_grad()
def mbr_complete(model, tok, fam, partial, device, k=8, temperature=0.7, max_new=240):
    ids = tok.encode_sequence(partial, fam, add_bos=True, add_eos=False)
    x = torch.tensor([ids] * k, dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_new, eos_id=tok.eos_id,
                         temperature=temperature, greedy=False)
    comps = [_trim(out[r, len(ids):].tolist(), tok) for r in range(k)]
    # consensus: the sample with the smallest mean edit-distance to the others
    best_i, best_cost = 0, float("inf")
    for i in range(k):
        cost = sum(normalized_edit_distance(comps[i], comps[j]) for j in range(k) if j != i)
        if cost < best_cost:
            best_cost, best_i = cost, i
    return comps[best_i]


@torch.no_grad()
def mbr_block_complete(model, tok, fam, partial, device, k=8, temperature=0.7, max_new=240):
    """Like MBR but consensus on BLOCK structure (min mean block-edit-dist) -> block-acc."""
    ids = tok.encode_sequence(partial, fam, add_bos=True, add_eos=False)
    x = torch.tensor([ids] * k, dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_new, eos_id=tok.eos_id,
                         temperature=temperature, greedy=False)
    comps = [_trim(out[r, len(ids):].tolist(), tok) for r in range(k)]
    sigs = [_block_signature(c) for c in comps]
    best_i, best_cost = 0, float("inf")
    for i in range(k):
        cost = sum(normalized_edit_distance(sigs[i], sigs[j]) for j in range(k) if j != i)
        if cost < best_cost:
            best_cost, best_i = cost, i
    return comps[best_i]


@torch.no_grad()
def ensemble_complete(models, tok, fam, partial, device, max_new=240):
    ids = tok.encode_sequence(partial, fam, add_bos=True, add_eos=False)
    cur = list(ids)
    mask = [tok.stoi[s] for s in SPECIAL_TOKENS if s != EOS]
    out = []
    for _ in range(max_new):
        avg = None
        for m in models:
            x = torch.tensor([cur[-m.cfg.block_size:]], dtype=torch.long, device=device)
            logits, _ = m(x)
            p = torch.softmax(logits[0, -1].float(), dim=-1)
            avg = p if avg is None else avg + p
        avg /= len(models)
        avg[mask] = 0.0
        nid = int(avg.argmax())
        s = tok.itos[nid]
        if nid == tok.eos_id:
            break
        cur.append(nid)
        if s not in SPECIAL_TOKENS:
            out.append(s)
        if s == "SHIP LOT":
            break
    return out


@torch.no_grad()
def ensemble_predict_next(models, tok, fam, partial, device, k=5):
    ids = tok.encode_sequence(partial, fam, add_bos=True, add_eos=False)
    avg = None
    for m in models:
        x = torch.tensor([ids[-m.cfg.block_size:]], dtype=torch.long, device=device)
        logits, _ = m(x)
        p = torch.softmax(logits[0, -1].float(), dim=-1)
        avg = p if avg is None else avg + p
    avg /= len(models)
    for s in SPECIAL_TOKENS:
        avg[tok.stoi[s]] = -1.0
    return [tok.itos[i] for i in torch.topk(avg, k).indices.tolist()]


@torch.no_grad()
def grammar_masked_predict_next(model, tok, fam, partial, device, k=5, pool=12):
    """Top-k next steps, but drop candidates whose append creates a NEW violation."""
    ids = tok.encode_sequence(partial, fam, add_bos=True, add_eos=False)
    x = torch.tensor([ids[-model.cfg.block_size:]], dtype=torch.long, device=device)
    logits, _ = model(x)
    logp = logits[0, -1].float().clone()
    for s in SPECIAL_TOKENS:
        logp[tok.stoi[s]] = -float("inf")
    order = torch.argsort(logp, descending=True).tolist()
    base = len(gs.validate_sequence(partial))
    valid, fallback = [], []
    for tid in order[:pool]:
        s = tok.itos[tid]
        fallback.append(s)
        if len(gs.validate_sequence(partial + [s])) <= base:
            valid.append(s)
        if len(valid) >= k:
            break
    out = valid + [s for s in fallback if s not in valid]
    return out[:k]


def score(pred, ref):
    return {
        "blk": block_level_accuracy(pred, ref),
        "ned": normalized_edit_distance(pred, ref),
        "ta": token_accuracy(pred, ref),
        "ex": float(pred == ref),
        "valid_pending": None,  # filled by caller (needs partial)
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="submission/model/best.pt")
    ap.add_argument("--ckpt2", default="process_lm/runs/full/best.pt",
                    help="second all-3-family model for the ensemble")
    ap.add_argument("--strategies", default="greedy,guided,beam,mbr,ensemble")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=99991)
    ap.add_argument("--cut-lo", type=float, default=0.5)
    ap.add_argument("--cut-hi", type=float, default=0.9)
    ap.add_argument("--beam-width", type=int, default=5)
    ap.add_argument("--mbr-k", type=int, default=8)
    ap.add_argument("--mbr-temp", type=float, default=0.7)
    ap.add_argument("--nextstep", action="store_true", help="also compare next-step (greedy vs ensemble vs grammar-masked)")
    ap.add_argument("--official-cuts", action="store_true", help="use 60%/80% cuts (the real eval condition) instead of random")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or get_device()
    model = load_model(args.ckpt, device)
    tok = Tokenizer.load(Path(args.ckpt).parent / "tokenizer.json")
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    models2 = None
    if "ensemble" in strategies or args.nextstep:
        models2 = [model, load_model(args.ckpt2, device)]

    train = {tuple(s) for _f, s in build_records(load_all_families(_DATA))}
    seqs, _ = generate_fresh(args.n, args.seed, train)
    cut_rng = random.Random(args.seed + 1)
    cuts = []
    for fam, steps in seqs:
        fracs = [0.6, 0.8] if args.official_cuts else [cut_rng.uniform(args.cut_lo, args.cut_hi)]
        for frac in fracs:
            cut = min(len(steps) - 1, max(2, int(len(steps) * frac)))
            cuts.append((fam, steps[:cut], steps[cut], steps[cut:]))
    cutdesc = "60/80 official" if args.official_cuts else f"random[{args.cut_lo},{args.cut_hi}]"
    print(f"{len(cuts)} fresh paired cases ({cutdesc} cuts) | strategies={strategies}\n")

    agg = {s: defaultdict(float) for s in strategies}
    ns_ranked, ns_truth, ens_ranked, mask_ranked = [], [], [], []

    for fam, partial, truth_next, ref in cuts:
        if args.nextstep:
            ns_ranked.append(predict_next(model, tok, fam, partial, device, k=5))
            ns_truth.append(truth_next)
            ens_ranked.append(ensemble_predict_next(models2, tok, fam, partial, device))
            mask_ranked.append(grammar_masked_predict_next(model, tok, fam, partial, device))
        for s in strategies:
            if s == "greedy":
                pred = complete(model, tok, fam, partial, device)
            elif s == "guided":
                pred = complete_guided(model, tok, fam, partial, device)
            elif s == "beam":
                pred = beam_complete(model, tok, fam, partial, device, width=args.beam_width)
            elif s == "mbr":
                pred = mbr_complete(model, tok, fam, partial, device, k=args.mbr_k, temperature=args.mbr_temp)
            elif s == "mbrblk":
                pred = mbr_block_complete(model, tok, fam, partial, device, k=args.mbr_k, temperature=args.mbr_temp)
            elif s == "ensemble":
                pred = ensemble_complete(models2, tok, fam, partial, device)
            else:
                raise SystemExit(f"unknown strategy {s}")
            a = agg[s]
            a["blk"] += block_level_accuracy(pred, ref)
            a["ned"] += normalized_edit_distance(pred, ref)
            a["ta"] += token_accuracy(pred, ref)
            a["ex"] += float(pred == ref)
            a["valid"] += float(len(gs.validate_sequence(partial + pred)) == 0)
            a["ship"] += float(bool(pred) and pred[-1] == "SHIP LOT")
            a["n"] += 1

    n = len(cuts)
    print("=" * 84)
    print(f"COMPLETION — {n} fresh paired sequences (official metrics)")
    print(f"{'strategy':10} {'Block-acc':>10} {'NED(lo)':>9} {'TokenAcc':>9} {'Exact':>7} {'Valid':>7} {'Ship':>6}")
    base = agg.get("greedy")
    for s in strategies:
        a = agg[s]
        blk, ned, ta = a["blk"]/n, a["ned"]/n, a["ta"]/n
        d = ""
        if base is not None and s != "greedy":
            d = f"   dBlk={blk-base['blk']/n:+.4f} dNED={ned-base['ned']/n:+.4f}"
        print(f"{s:10} {blk:10.4f} {ned:9.4f} {ta:9.4f} {a['ex']/n:7.4f} {a['valid']/n:7.4f} {a['ship']/n:6.3f}{d}")
    print("=" * 84)

    if args.nextstep:
        g = nextstep_metrics(ns_ranked, ns_truth)
        e = nextstep_metrics(ens_ranked, ns_truth)
        m = nextstep_metrics(mask_ranked, ns_truth)
        print(f"NEXT-STEP ({len(ns_truth)} cases):")
        print(f"  single        Top-1 {g['top1']:.4f}  Top-3 {g['top3']:.4f}  Top-5 {g['top5']:.4f}  MRR {g['mrr']:.4f}")
        print(f"  ensemble      Top-1 {e['top1']:.4f}  Top-3 {e['top3']:.4f}  Top-5 {e['top5']:.4f}  MRR {e['mrr']:.4f}   (dTop1 {e['top1']-g['top1']:+.4f})")
        print(f"  grammar-mask  Top-1 {m['top1']:.4f}  Top-3 {m['top3']:.4f}  Top-5 {m['top5']:.4f}  MRR {m['mrr']:.4f}   (dTop1 {m['top1']-g['top1']:+.4f})")


if __name__ == "__main__":
    main()
