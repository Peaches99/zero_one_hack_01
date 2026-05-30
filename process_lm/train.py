"""Train the process-step GPT. Runs on Apple MPS / CUDA / CPU.

Run from the repo root:
    python -m process_lm.train --epochs 15
"""
from __future__ import annotations

import argparse
import contextlib
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import (
    SequenceDataset, build_records, load_all_families, lofo_split, make_collate, split_records,
)
from .model import GPT, GPTConfig
from .runguard import acquire
from .tokenizer import Tokenizer


def _amp_ctx(device: str):
    """bf16 autocast on CUDA: enables flash attention (O(T) attention memory) and
    the 5090's tensor cores. fp32 forced SDPA to the math kernel — O(B*H*T^2)
    memory (an 85M model OOM-thrashed ~30 GB and ran 23x slow). No-op off CUDA."""
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def get_device(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def evaluate(model, dl, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            with _amp_ctx(device):
                _, loss = model(x, y)
            tot += loss.item() * x.size(0)
            n += x.size(0)
    return tot / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="tracks/industrial-infineon/training_data")
    p.add_argument("--out-dir", default="process_lm/runs/v1")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1,
                   help="AdamW weight decay; grokking is sensitive to this (try 1.0)")
    p.add_argument("--save-every", type=int, default=1,
                   help="checkpoint cadence (epochs); large values avoid I/O dominating "
                        "very long runs. Logging stays per-epoch; best.pt still tracks best.")
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--family-dropout", type=float, default=0.0,
                   help="prob of masking the family token (OOD robustness; try 0.15)")
    p.add_argument("--val-per-family", type=int, default=100)
    p.add_argument("--hold-out-family", default=None, choices=["mosfet", "igbt", "ic"],
                   help="leave-one-family-out: train on the other two, validate on this (OOD)")
    p.add_argument("--train-families", default=None,
                   help="comma-separated subset of families to TRAIN on (diversity axis; "
                        "compose with --hold-out-family + --train-limit to vary #families "
                        "at a fixed data volume)")
    p.add_argument("--add-hybrids", type=int, default=0,
                   help="add N validated hybrid pseudo-family routes to training (OOD robustness)")
    p.add_argument("--hybrid-tag", default="hybrid",
                   help="family tag for hybrid routes (set 'random' to vary per-route)")
    p.add_argument("--add-v2", type=int, default=0,
                   help="add N validated v2 max-diversity routes (variable cycle count) to training")
    p.add_argument("--v2-tag", default="random",
                   help="family tag for v2 routes ('random' varies per-route)")
    p.add_argument("--train-limit", type=int, default=0,
                   help="cap real training sequences before augmentation (0=all; data-scaling axis)")
    p.add_argument("--extra-per-family", type=int, default=0,
                   help="generate N extra real validated sequences per family (scale data volume up)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true",
                   help="bypass the single-instance training lock (use with care)")
    args = p.parse_args()

    acquire(force=args.force)  # refuse to run a 2nd concurrent MPS job (RAM safety)
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    by_family = load_all_families(Path(args.data_dir))
    records = build_records(by_family)
    if args.hold_out_family:
        train_recs, val_recs = lofo_split(records, args.hold_out_family, args.seed)
        print(f"LOFO: holding out '{args.hold_out_family}' "
              f"(train {len(train_recs)} / OOD-val {len(val_recs)})")
    else:
        train_recs, val_recs = split_records(records, args.val_per_family, args.seed)

    # Diversity axis: restrict TRAIN to a subset of families (val/OOD unaffected).
    train_fams = ({f.strip().lower() for f in args.train_families.split(",")}
                  if args.train_families else None)
    if train_fams:
        train_recs = [r for r in train_recs if r[0] in train_fams]
        print(f"train-families: restricted to {sorted(train_fams)} -> {len(train_recs)} real seqs")

    # Scale real data UP: generate extra validated per-family sequences from the
    # organizers' grammar (combinatoric space is billions+, so these are fresh).
    if args.extra_per_family > 0:
        from .diversify import _DATA_DIR  # noqa: F401  (ensures generator import path)
        from generate_sequences import generate_dataset  # type: ignore
        pool_fams = [f for f in ("mosfet", "igbt", "ic")
                     if f != args.hold_out_family and (train_fams is None or f in train_fams)]
        extra = []
        for fam in pool_fams:
            for s in generate_dataset(fam, args.extra_per_family, seed=args.seed + hash(fam) % 1000):
                extra.append((fam, s))
        train_recs = list(train_recs) + extra
        print(f"scale-up: +{len(extra)} extra real sequences "
              f"({args.extra_per_family}/family across {pool_fams})")

    # Scale real data DOWN: cap before augmentation (the data-volume scaling axis).
    if args.train_limit > 0:
        random.Random(args.seed).shuffle(train_recs)
        train_recs = train_recs[:args.train_limit]
        print(f"train-limit: capped to {len(train_recs)} real sequences")

    # Optionally augment TRAIN with validated hybrid pseudo-family routes. During
    # LOFO, draw hybrids only from the trained families so the held-out family
    # stays genuinely unseen (honest OOD test).
    if args.add_hybrids > 0:
        from .diversify import FAMILIES, generate_diverse_records
        pool = [f for f in FAMILIES if f != args.hold_out_family] if args.hold_out_family else None
        hybrids = generate_diverse_records(args.add_hybrids, seed=args.seed, pool=pool)
        if args.hybrid_tag == "random":
            import random as _r
            rng_tag = _r.Random(args.seed)
            hybrids = [(rng_tag.choice(FAMILIES), s) for _, s in hybrids]
        elif args.hybrid_tag != "hybrid":
            hybrids = [(args.hybrid_tag, s) for _, s in hybrids]
        train_recs = list(train_recs) + hybrids
        random.Random(args.seed).shuffle(train_recs)
        print(f"augment: +{len(hybrids)} hybrid routes "
              f"(pool={pool or 'all'}, tag={args.hybrid_tag}) -> train {len(train_recs)}")

    # Optionally augment TRAIN with v2 max-diversity routes (variable cycle count,
    # mixed-family blocks, optional 2nd metal). Pool-restricted during LOFO so the
    # held-out family stays unseen (structural generalization, not family leakage).
    if args.add_v2 > 0:
        from .diversify2 import FAMILIES as V2_FAMILIES, generate_v2
        v2_pool = [f for f in V2_FAMILIES if f != args.hold_out_family] if args.hold_out_family else None
        v2recs = generate_v2(args.add_v2, seed=args.seed + 7, pool=v2_pool, tag=args.v2_tag)
        train_recs = list(train_recs) + v2recs
        random.Random(args.seed + 1).shuffle(train_recs)
        print(f"augment-v2: +{len(v2recs)} v2 routes "
              f"(pool={v2_pool or 'all'}, tag={args.v2_tag}) -> train {len(train_recs)}")
    # Build the tokenizer from TRAIN ONLY so a held-out family's unique steps are
    # genuinely unknown (<UNK>) — the honest OOD setup.
    tok = Tokenizer.from_sequences({i: s for i, (_, s) in enumerate(train_recs)})
    tok.save(out / "tokenizer.json")
    print(f"vocab={tok.vocab_size}  train={len(train_recs)}  val={len(val_recs)}  device={device}")

    collate = make_collate(tok.pad_id)
    train_dl = DataLoader(
        SequenceDataset(train_recs, tok, family_dropout=args.family_dropout, seed=args.seed),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate,
    )
    val_dl = DataLoader(
        SequenceDataset(val_recs, tok, family_dropout=0.0, seed=args.seed),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate,
    )

    cfg = GPTConfig(
        vocab_size=tok.vocab_size, block_size=args.block_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout,
    )
    model = GPT(cfg).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    log_path = out / "train_log.csv"
    log_path.write_text("epoch,train_loss,val_loss,seconds\n")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            with _amp_ctx(device):
                _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = tot / n
        val_loss = evaluate(model, val_dl, device)
        dt = time.time() - t0
        print(f"epoch {epoch:2d}  train {train_loss:.4f}  val {val_loss:.4f}  ({dt:.1f}s)")
        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{dt:.1f}\n")

        improved = val_loss < best_val
        if improved or epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = {"model": model.state_dict(), "config": vars(cfg), "args": vars(args)}
            if epoch % args.save_every == 0 or epoch == args.epochs:
                torch.save(ckpt, out / "last.pt")
            if improved:
                best_val = val_loss
                torch.save(ckpt, out / "best.pt")

    print(f"done. best val loss {best_val:.4f}. checkpoints + log in {out}")


if __name__ == "__main__":
    main()
