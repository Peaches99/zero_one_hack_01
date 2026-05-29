"""Train the process-step GPT. Runs on Apple MPS / CUDA / CPU.

Run from the repo root:
    python -m process_lm.train --epochs 15
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import (
    SequenceDataset, build_records, load_all_families, make_collate, split_records,
)
from .model import GPT, GPTConfig
from .tokenizer import Tokenizer


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
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--family-dropout", type=float, default=0.0,
                   help="prob of masking the family token (OOD robustness; try 0.15)")
    p.add_argument("--val-per-family", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    by_family = load_all_families(Path(args.data_dir))
    records = build_records(by_family)
    tok = Tokenizer.from_sequences({i: s for i, (_, s) in enumerate(records)})
    tok.save(out / "tokenizer.json")

    train_recs, val_recs = split_records(records, args.val_per_family, args.seed)
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
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))

    log_path = out / "train_log.csv"
    log_path.write_text("epoch,train_loss,val_loss,seconds\n")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
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

        ckpt = {"model": model.state_dict(), "config": vars(cfg), "args": vars(args)}
        torch.save(ckpt, out / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, out / "best.pt")

    print(f"done. best val loss {best_val:.4f}. checkpoints + log in {out}")


if __name__ == "__main__":
    main()
