"""Vocabulary and tokenizer.

Each process step string is a single token. Special tokens and family
conditioning tokens are reserved up front so their IDs are stable across runs.
"""
from __future__ import annotations

import json
from pathlib import Path

PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"            # unknown step string (robustness on eval / OOD family)
UNK_FAMILY = "<FAM:UNK>"  # used when the family is unknown or dropped out

FAMILY_TOKENS = {
    "mosfet": "<FAM:MOSFET>",
    "igbt": "<FAM:IGBT>",
    "ic": "<FAM:IC>",
}

# Order matters: these occupy the first, fixed IDs.
SPECIAL_TOKENS = [
    PAD, BOS, EOS, UNK, UNK_FAMILY,
    FAMILY_TOKENS["mosfet"], FAMILY_TOKENS["igbt"], FAMILY_TOKENS["ic"],
]


class Tokenizer:
    """Maps step strings <-> integer ids."""

    def __init__(self, steps: list[str]):
        # `steps` should be the sorted unique list of real step strings.
        self.itos: list[str] = list(SPECIAL_TOKENS) + list(steps)
        self.stoi: dict[str, int] = {s: i for i, s in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]
        self.unk_id = self.stoi[UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def family_id(self, family: str) -> int:
        return self.stoi[FAMILY_TOKENS.get(family.lower(), UNK_FAMILY)]

    def encode_step(self, step: str) -> int:
        return self.stoi.get(step, self.unk_id)

    def encode_sequence(
        self, steps: list[str], family: str,
        add_bos: bool = True, add_eos: bool = True,
    ) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.append(self.family_id(family))
        ids.extend(self.encode_step(s) for s in steps)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> list[str]:
        return [self.itos[i] for i in ids]

    @classmethod
    def from_sequences(cls, sequences: dict) -> "Tokenizer":
        """Build the vocabulary from {id: [step, ...]} sequences."""
        steps: set[str] = set()
        for seq in sequences.values():
            steps.update(seq)
        return cls(sorted(steps))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.itos, ensure_ascii=False))

    @classmethod
    def load(cls, path) -> "Tokenizer":
        itos = json.loads(Path(path).read_text())
        obj = cls.__new__(cls)
        obj.itos = itos
        obj.stoi = {s: i for i, s in enumerate(itos)}
        obj.pad_id = obj.stoi[PAD]
        obj.bos_id = obj.stoi[BOS]
        obj.eos_id = obj.stoi[EOS]
        obj.unk_id = obj.stoi[UNK]
        return obj
