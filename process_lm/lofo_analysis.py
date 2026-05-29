"""Thorough leave-one-family-out (OOD) analysis — the Task-4 proxy.

For each family F we compare, on the SAME 100 held-out F sequences:
  * ID         = the full model (trained on all 3 families)
  * OOD(fam)   = the model that never trained on F, given F's real family token
  * OOD(unk)   = the same OOD model, but told the family is <UNK> (true 4th-family analog)

Beyond headline metrics, it decomposes the OOD gap into:
  * VOCAB gap  — targets that are family-unique and therefore <UNK> to the OOD
                 model (it literally cannot name them; an irreducible floor)
  * LOGIC gap  — on the SHARED vocabulary both models can express, how much
                 worse the OOD model applies process logic in unfamiliar context
and locates where the drop happens (accuracy vs. position in the sequence).

Run from the repo root (single foreground run; trains missing LOFO models first):
    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m process_lm.lofo_analysis
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch

from .data import FAMILY_FILES, build_records, load_all_families, read_csv_sequences, split_records
from .metrics import blocklevel_metrics, completion_metrics, nextstep_metrics
from .predict import _cuts, complete, get_device, load_model, predict_next
from .tokenizer import Tokenizer

FAMILIES = ["mosfet", "igbt", "ic"]
DATA_DIR = "tracks/industrial-infineon/training_data"
RUNS = Path("process_lm/runs")
EPOCHS = 15
NEXTSTEP_LIMIT = 100   # held-out sequences per family for next-step (cheap)
COMPLETION_LIMIT = 40  # held-out sequences per family for completion (autoregressive)


# ---------------------------------------------------------------------------
# Phase A — make sure every checkpoint we need exists (train missing, foreground)
# ---------------------------------------------------------------------------

def ensure_checkpoint(out_dir: Path, extra: list[str]) -> None:
    if (out_dir / "best.pt").exists():
        print(f"[have] {out_dir}")
        return
    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    cmd = [sys.executable, "-m", "process_lm.train",
           "--epochs", str(EPOCHS), "--out-dir", str(out_dir), *extra]
    print(f"[train] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


# ---------------------------------------------------------------------------
# Phase B — how OOD is each family, really? (vocabulary overlap)
# ---------------------------------------------------------------------------

def vocab_analysis() -> dict:
    fam_steps: dict[str, set] = {}
    for fam, fname in FAMILY_FILES.items():
        seqs = read_csv_sequences(Path(DATA_DIR) / fname)
        steps: set = set()
        for s in seqs.values():
            steps.update(s)
        fam_steps[fam] = steps
    out = {}
    for fam in FAMILIES:
        others = set().union(*(fam_steps[o] for o in FAMILIES if o != fam))
        unique = fam_steps[fam] - others
        out[fam] = {"n_steps": len(fam_steps[fam]), "n_unique": len(unique),
                    "unique": sorted(unique)}
    return out


# ---------------------------------------------------------------------------
# Phase C/D — evaluators
# ---------------------------------------------------------------------------

def held_out(family: str, limit: int) -> list:
    _, val = split_records(build_records(load_all_families(DATA_DIR)), 100, 0)
    return [r for r in val if r[0] == family][:limit]


def eval_nextstep(model, tok, recs, device, family_override=None) -> dict:
    """Next-step metrics, plus a shared-vs-unique-vocabulary decomposition.

    `family_override` lets us feed the wrong/unknown family token (e.g. 'unk').
    """
    rows = []  # (ranked_preds, truth, cut_frac, truth_in_vocab)
    for fam, steps in recs:
        fam_used = family_override or fam
        for cut in _cuts(steps):
            preds = predict_next(model, tok, fam_used, steps[:cut], device)
            truth = steps[cut]
            in_vocab = truth in tok.stoi  # can the model even output this token?
            rows.append((preds, truth, round(len(steps[:cut]) / len(steps), 1), in_vocab))

    def subset(pred_keep):
        r = [(p, t) for p, t, _c, iv in rows if pred_keep(iv)]
        if not r:
            return None
        return nextstep_metrics([p for p, _ in r], [t for _, t in r])

    overall = subset(lambda iv: True)
    shared = subset(lambda iv: iv)             # only targets the model could name
    unique_frac = sum(1 for *_x, iv in rows if not iv) / len(rows)
    by_cut = {}
    for cf in (0.6, 0.8):
        r = [(p, t) for p, t, c, _iv in rows if c == cf]
        if r:
            by_cut[cf] = nextstep_metrics([p for p, _ in r], [t for _, t in r])
    return {"overall": overall, "shared_vocab_only": shared,
            "unique_target_frac": unique_frac, "by_cut": by_cut}


def eval_completion(model, tok, recs, device, validate, family_override=None) -> dict:
    preds, truths, partials = [], [], []
    for fam, steps in recs:
        fam_used = family_override or fam
        for cut in _cuts(steps):
            preds.append(complete(model, tok, fam_used, steps[:cut], device))
            truths.append(steps[cut:])
            partials.append(steps[:cut])
    out = {**completion_metrics(preds, truths), **blocklevel_metrics(preds, truths)}
    if validate is not None and preds:
        valid = sum(1 for part, pr in zip(partials, preds) if len(validate(part + pr)) == 0)
        out["valid_completion_rate"] = valid / len(preds)
    return out


def mean_nll(model, tok, recs, device, family_override=None) -> float:
    """Mean per-token negative log-likelihood (surprisal) over full sequences."""
    tot, n = 0.0, 0
    with torch.no_grad():
        for fam, steps in recs:
            ids = tok.encode_sequence(steps, family_override or fam)
            ids = ids[:model.cfg.block_size + 1]
            x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
            _, loss = model(x, y)
            tot += loss.item() * (len(ids) - 1)
            n += len(ids) - 1
    return tot / max(n, 1)


def position_accuracy(model, tok, recs, device, bins=10, family_override=None) -> list:
    """Top-1 next-step accuracy bucketed by normalized position in the sequence."""
    hit = [0] * bins
    tot = [0] * bins
    for fam, steps in recs:
        fam_used = family_override or fam
        for cut in range(2, len(steps)):
            b = min(bins - 1, int(cut / len(steps) * bins))
            preds = predict_next(model, tok, fam_used, steps[:cut], device, k=1)
            tot[b] += 1
            if preds and preds[0] == steps[cut]:
                hit[b] += 1
    return [(hit[i] / tot[i]) if tot[i] else float("nan") for i in range(bins)]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load_validator():
    sys.path.insert(0, str(Path(DATA_DIR).resolve()))
    try:
        from generate_sequences import validate_sequence  # type: ignore
        return validate_sequence
    except Exception as e:
        print(f"[warn] validator unavailable ({e})")
        return None


def fmt(m):
    return "n/a" if not m else (f"top1={m['top1']:.3f} top3={m['top3']:.3f} "
                                f"top5={m['top5']:.3f} mrr={m['mrr']:.3f}")


def main() -> None:
    device = get_device()
    print(f"device={device}\n")

    print("=== Phase A: ensure checkpoints (foreground, one at a time) ===")
    ensure_checkpoint(RUNS / "full", [])
    for f in FAMILIES:
        ensure_checkpoint(RUNS / f"lofo_{f}", ["--hold-out-family", f])

    print("\n=== Phase B: vocabulary overlap (how OOD is each family?) ===")
    vocab = vocab_analysis()
    for f in FAMILIES:
        v = vocab[f]
        print(f"  {f:6}: {v['n_steps']} distinct steps, {v['n_unique']} unique to it "
              f"(unseen if held out)")
        if v["unique"]:
            print(f"          unique: {', '.join(v['unique'][:8])}"
                  f"{' …' if len(v['unique']) > 8 else ''}")

    validate = load_validator()
    full_model = load_model(RUNS / "full" / "best.pt", device)
    full_tok = Tokenizer.load(RUNS / "full" / "tokenizer.json")

    results = {"vocab": vocab, "families": {}}
    print("\n=== Phase C/D: per-family ID vs OOD ===")
    for f in FAMILIES:
        ns_recs = held_out(f, NEXTSTEP_LIMIT)
        cm_recs = held_out(f, COMPLETION_LIMIT)
        lofo_model = load_model(RUNS / f"lofo_{f}" / "best.pt", device)
        lofo_tok = Tokenizer.load(RUNS / f"lofo_{f}" / "tokenizer.json")

        id_ns = eval_nextstep(full_model, full_tok, ns_recs, device)
        ood_ns = eval_nextstep(lofo_model, lofo_tok, ns_recs, device)
        ood_ns_unk = eval_nextstep(lofo_model, lofo_tok, ns_recs, device, family_override="unk")

        id_cm = eval_completion(full_model, full_tok, cm_recs, device, validate)
        ood_cm = eval_completion(lofo_model, lofo_tok, cm_recs, device, validate)

        id_nll = mean_nll(full_model, full_tok, ns_recs, device)
        ood_nll = mean_nll(lofo_model, lofo_tok, ns_recs, device)

        id_pos = position_accuracy(full_model, full_tok, cm_recs, device)
        ood_pos = position_accuracy(lofo_model, lofo_tok, cm_recs, device)

        results["families"][f] = {
            "id_nextstep": id_ns, "ood_nextstep": ood_ns, "ood_nextstep_unk": ood_ns_unk,
            "id_completion": id_cm, "ood_completion": ood_cm,
            "id_nll": id_nll, "ood_nll": ood_nll,
            "id_pos_acc": id_pos, "ood_pos_acc": ood_pos,
        }

        print(f"\n--- held-out family: {f.upper()} "
              f"({vocab[f]['n_unique']} unique steps, "
              f"{ood_ns['unique_target_frac']*100:.1f}% of targets unseen-vocab) ---")
        print(f"  next-step ID            : {fmt(id_ns['overall'])}")
        print(f"  next-step OOD (fam tok) : {fmt(ood_ns['overall'])}")
        print(f"  next-step OOD (unk tok) : {fmt(ood_ns_unk['overall'])}")
        # Decomposition on the shared vocabulary both models can express:
        ids = id_ns["shared_vocab_only"]
        oods = ood_ns["shared_vocab_only"]
        if ids and oods:
            print(f"  -- decomposition (shared-vocab targets only) --")
            print(f"     ID  top1={ids['top1']:.3f}   OOD top1={oods['top1']:.3f}   "
                  f"LOGIC gap={ids['top1']-oods['top1']:+.3f}")
            print(f"     VOCAB gap (unique targets the OOD model cannot name): "
                  f"{ood_ns['unique_target_frac']:.3f}")
        print(f"  completion ID           : tokAcc={id_cm['token_accuracy']:.3f} "
              f"blkNED={id_cm['block_norm_edit_distance']:.3f} "
              f"valid={id_cm.get('valid_completion_rate', float('nan')):.3f}")
        print(f"  completion OOD          : tokAcc={ood_cm['token_accuracy']:.3f} "
              f"blkNED={ood_cm['block_norm_edit_distance']:.3f} "
              f"valid={ood_cm.get('valid_completion_rate', float('nan')):.3f}")
        print(f"  mean NLL  ID={id_nll:.3f}  OOD={ood_nll:.3f}  (ppl {math.exp(id_nll):.2f} "
              f"-> {math.exp(ood_nll):.2f})")
        print(f"  top1 by position (deciles):")
        print(f"     ID : {' '.join(f'{x:.2f}' for x in id_pos)}")
        print(f"     OOD: {' '.join(f'{x:.2f}' for x in ood_pos)}")

    out_path = RUNS / "lofo_analysis.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved full results -> {out_path}")

    # Headline
    print("\n=== HEADLINE: generalization gap (ID - OOD, next-step top1) ===")
    for f in FAMILIES:
        r = results["families"][f]
        idt = r["id_nextstep"]["overall"]["top1"]
        oodt = r["ood_nextstep"]["overall"]["top1"]
        oodu = r["ood_nextstep_unk"]["overall"]["top1"]
        ids = r["id_nextstep"]["shared_vocab_only"]
        oods = r["ood_nextstep"]["shared_vocab_only"]
        logic = (ids["top1"] - oods["top1"]) if (ids and oods) else float("nan")
        print(f"  {f:6}: ID {idt:.3f} -> OOD {oodt:.3f} (drop {idt-oodt:+.3f}) | "
              f"unk-tok {oodu:.3f} | logic-only drop {logic:+.3f} | "
              f"vocab floor {r['ood_nextstep']['unique_target_frac']:.3f}")


if __name__ == "__main__":
    main()
