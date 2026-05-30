"""Word-level tokenization — the one lever with a higher OOD ceiling.

Step-level tokenization makes a held-out family's unique steps `<UNK>`: the model
literally cannot emit them, capping OOD next-step top-1 at ~0.79 (the vocab
ceiling). If instead each step is a sequence of WORD tokens terminated by
`<ENDSTEP>`, the model can *compose* an unseen step from words it has seen — e.g.
IC's unique `MEASURE CD LEVEL 2` from MEASURE / CD / LEVEL / 2, all present in
MOSFET+IGBT. This directly attacks the vocabulary gap that step-level cannot.

Honest LOFO test: train on two families (word vocab from train only), evaluate
next-step + valid-completion on the held-out family, and compare to the step-level
baseline on the SAME held-out sequences.

    python -m process_lm.wordlevel --hold-out ic --epochs 20
"""
from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .data import build_records, load_all_families, lofo_split
from .model import GPT, GPTConfig
from .predict import _cuts

_DATA_DIR = Path(__file__).resolve().parent.parent / "tracks/industrial-infineon/training_data"
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))
import generate_sequences as gs  # type: ignore  # noqa: E402

PAD, BOS, EOS, UNK, ENDSTEP = "<PAD>", "<BOS>", "<EOS>", "<UNK>", "<ENDSTEP>"
FAM = {"mosfet": "<FAM:MOSFET>", "igbt": "<FAM:IGBT>", "ic": "<FAM:IC>", "unk": "<FAM:UNK>"}
SPECIALS = [PAD, BOS, EOS, UNK, ENDSTEP, FAM["mosfet"], FAM["igbt"], FAM["ic"], FAM["unk"]]


class WordTok:
    def __init__(self, words):
        self.itos = list(SPECIALS) + sorted(words)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        for name in ("PAD", "BOS", "EOS", "UNK", "ENDSTEP"):
            setattr(self, f"{name.lower()}_id", self.stoi[globals()[name]])

    @property
    def vocab_size(self):
        return len(self.itos)

    def fam_id(self, family):
        return self.stoi[FAM.get(family.lower(), FAM["unk"])]

    def encode(self, steps, family, add_eos=True):
        ids = [self.bos_id, self.fam_id(family)]
        for st in steps:
            for w in st.split():
                ids.append(self.stoi.get(w, self.unk_id))
            ids.append(self.endstep_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    @classmethod
    def from_steps(cls, records):
        words = set()
        for _fam, steps in records:
            for st in steps:
                words.update(st.split())
        return cls(words)


def _amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()


class WordDataset(torch.utils.data.Dataset):
    def __init__(self, records, tok, family_dropout=0.0, seed=0):
        import random
        self.records = records
        self.tok = tok
        self.family_dropout = family_dropout
        self._rng = random.Random(seed)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        fam, steps = self.records[idx]
        if self.family_dropout and self._rng.random() < self.family_dropout:
            fam = "unk"
        return torch.tensor(self.tok.encode(steps, fam), dtype=torch.long)


def make_collate(pad_id):
    def collate(batch):
        padded = pad_sequence(batch, batch_first=True, padding_value=pad_id)
        x = padded[:, :-1]
        y = padded[:, 1:].clone()
        y[y == pad_id] = -100
        return x, y
    return collate


@torch.no_grad()
def predict_next_step(model, tok, family, partial_steps, device, max_words=10):
    ids = tok.encode(partial_steps, family, add_eos=False)  # ends after last <ENDSTEP>
    x = torch.tensor([ids], dtype=torch.long, device=device)
    block = model.cfg.block_size
    words, banned = [], {tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id, tok.fam_id(family)}
    for _ in range(max_words):
        with _amp(device):
            logits, _ = model(x[:, -block:])
        logit = logits[0, -1].float()
        for b in banned:
            logit[b] = -float("inf")
        nxt = int(logit.argmax())
        if nxt == tok.endstep_id:
            break
        words.append(tok.itos[nxt])
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return " ".join(words)


@torch.no_grad()
def complete_route(model, tok, family, partial_steps, device, max_new=400):
    ids = tok.encode(partial_steps, family, add_eos=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    block = model.cfg.block_size
    out_ids = []
    for _ in range(max_new):
        with _amp(device):
            logits, _ = model(x[:, -block:])
        nxt = int(logits[0, -1].argmax())
        if nxt == tok.eos_id:
            break
        out_ids.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    steps, cur = [], []
    for i in out_ids:
        if i == tok.endstep_id:
            if cur:
                steps.append(" ".join(cur)); cur = []
        elif i not in (tok.pad_id, tok.bos_id, tok.eos_id):
            cur.append(tok.itos[i])
    if cur:
        steps.append(" ".join(cur))
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold-out", default="ic", choices=["mosfet", "igbt", "ic"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--family-dropout", type=float, default=0.15)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=768)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    recs = build_records(load_all_families(_DATA_DIR))
    train_recs, val_recs = lofo_split(recs, args.hold_out, 0)
    tok = WordTok.from_steps(train_recs)  # word vocab from TRAIN ONLY (honest OOD)
    print(f"word vocab={tok.vocab_size}  train={len(train_recs)}  OOD-val={len(val_recs)}  device={device}")

    collate = make_collate(tok.pad_id)
    train_dl = DataLoader(WordDataset(train_recs, tok, args.family_dropout),
                          batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size,
                    n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, dropout=0.1)
    model = GPT(cfg).to(device)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))

    for epoch in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tot = n = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            with _amp(device):
                _, loss = model(x, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        print(f"epoch {epoch:2d}  train {tot/n:.4f}  ({time.time()-t0:.1f}s)")

    # OOD eval on the held-out family (real-family token AND unk token)
    model.eval()
    ood = val_recs[:100]
    for fam_tok in (args.hold_out, "unk"):
        hit = tot = 0
        valid = vtot = 0
        for _fam, steps in ood[:60]:
            for cut in _cuts(steps):
                pred = predict_next_step(model, tok, fam_tok, steps[:cut], device)
                tot += 1
                if pred == steps[cut]:
                    hit += 1
            comp = complete_route(model, tok, fam_tok, steps[: max(1, int(len(steps) * 0.6))], device)
            full = steps[: max(1, int(len(steps) * 0.6))] + comp
            vtot += 1
            if len(gs.validate_sequence(full)) == 0:
                valid += 1
        tag = "real-fam-token" if fam_tok == args.hold_out else "UNK-token (4th-fam proxy)"
        print(f"  OOD [{tag:26}] next-step top1={hit/max(tot,1):.3f}  "
              f"valid-completion={valid/max(vtot,1):.3f}  (n={tot})")
    print("  compare to step-level baseline: OOD top1 ~0.635, valid ~1.0 (real+fd, hold-out ic)")


if __name__ == "__main__":
    main()
