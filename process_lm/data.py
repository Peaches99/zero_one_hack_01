"""CSV loading, dataset construction, and batching.

The provided `*_variants.csv` files are long-format: one (SEQUENCE_ID, STEP)
row per step. The reader is BOM/quote tolerant to match the organizers' files.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

FAMILY_FILES = {
    "mosfet": "MOSFET_variants.csv",
    "igbt": "IGBT_variants.csv",
    "ic": "IC_variants.csv",
}


def read_csv_sequences(path) -> dict[str, list[str]]:
    """Read a SEQUENCE_ID,STEP long-format CSV into {seq_id: [steps]}."""

    def _norm(name: str) -> str:
        return name.lstrip("﻿").strip().strip('"').strip()

    sequences: dict[str, list[str]] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        norm_map = {_norm(h): h for h in (reader.fieldnames or [])}
        if "STEP" not in norm_map:
            raise ValueError(f"{path}: no STEP column (headers={reader.fieldnames})")
        step_key = norm_map["STEP"]
        seq_key = norm_map.get("SEQUENCE_ID")
        for row in reader:
            step = row[step_key].strip().strip('"')
            if not step:
                continue
            sid = row[seq_key].strip() if seq_key else "seq_0001"
            sequences.setdefault(sid, []).append(step)
    return sequences


def load_all_families(data_dir) -> dict[str, dict[str, list[str]]]:
    """Return {family: {seq_id: [steps]}} for all three provided families."""
    data_dir = Path(data_dir)
    return {fam: read_csv_sequences(data_dir / fname) for fam, fname in FAMILY_FILES.items()}


def build_records(by_family: dict) -> list[tuple[str, list[str]]]:
    """Flatten to a list of (family, steps) records."""
    records: list[tuple[str, list[str]]] = []
    for fam, seqs in by_family.items():
        for steps in seqs.values():
            records.append((fam, steps))
    return records


def split_records(records, val_per_family: int = 100, seed: int = 0):
    """Hold out `val_per_family` sequences per family (mirrors the eval split)."""
    rng = random.Random(seed)
    by_fam: dict[str, list] = {}
    for rec in records:
        by_fam.setdefault(rec[0], []).append(rec)
    train, val = [], []
    for recs in by_fam.values():
        recs = list(recs)
        rng.shuffle(recs)
        val.extend(recs[:val_per_family])
        train.extend(recs[val_per_family:])
    rng.shuffle(train)
    return train, val


def lofo_split(records, hold_out: str, seed: int = 0):
    """Leave-one-family-out: train on every family except `hold_out`, evaluate on
    `hold_out` — a stand-in for the hidden 4th family (OOD / Task 4)."""
    hold_out = hold_out.lower()
    train = [r for r in records if r[0] != hold_out]
    test = [r for r in records if r[0] == hold_out]
    random.Random(seed).shuffle(train)
    return train, test


class SequenceDataset(torch.utils.data.Dataset):
    """Yields a 1D LongTensor of token ids ([BOS, FAM, steps..., EOS]) per record."""

    def __init__(self, records, tokenizer, family_dropout: float = 0.0, seed: int = 0):
        self.records = records
        self.tok = tokenizer
        self.family_dropout = family_dropout
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        fam, steps = self.records[idx]
        # Optionally drop the family token so the model degrades gracefully on
        # an unseen 4th family (Task 4 / OOD). Inert at the default 0.0.
        if self.family_dropout and self._rng.random() < self.family_dropout:
            fam = "unk"
        ids = self.tok.encode_sequence(steps, fam)
        return torch.tensor(ids, dtype=torch.long)


def make_collate(pad_id: int):
    """Right-pad a batch and build (input, target) with shifted targets.

    Right-padding + causal attention means real tokens never attend to PAD, and
    PAD targets are set to -100 so they are ignored by cross-entropy.
    """

    def collate(batch):
        padded = pad_sequence(batch, batch_first=True, padding_value=pad_id)
        x = padded[:, :-1]
        y = padded[:, 1:].clone()
        y[y == pad_id] = -100
        return x, y

    return collate
