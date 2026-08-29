"""Spec section 7 - text normalisation and vocabulary.

Normalisation runs before anything else touches the text: the manifest already
holds normalised strings, so no downstream stage may normalise a second time.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Iterable

BLANK_INDEX = 0
UNK_TOKEN = "<unk>"

ZERO_WIDTH = {"\u200d", "\u200c"}  # ZWJ, ZWNJ


def normalise(text: str, strip_zero_width: bool = True) -> str:
    """NFC, collapse whitespace, optionally drop ZWJ/ZWNJ (spec 7.1)."""
    text = unicodedata.normalize("NFC", text)
    if strip_zero_width:
        # Stripped before the whitespace collapse: a zero-width character
        # sitting between two spaces would otherwise leave a double space
        # behind, making normalise() non-idempotent.
        text = "".join(ch for ch in text if ch not in ZERO_WIDTH)
        # Removing it can also expose a newly composable pair.
        text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


class Vocab:
    """Index 0 is the CTC blank. Real characters take 1..C, <unk> takes C+1.

    ``size`` (V) counts the non-blank symbols including <unk>, so the model's
    final layer is ``V + 1`` wide exactly as spec 10.1 states, while spec 7.2's
    requirement of a single <unk> index is still met.
    """

    def __init__(self, chars: Iterable[str]) -> None:
        self.chars: list[str] = sorted(set(chars))
        self.itos: list[str] = ["<blank>"] + self.chars + [UNK_TOKEN]
        self.stoi: dict[str, int] = {c: i + 1 for i, c in enumerate(self.chars)}
        self.unk_index: int = len(self.chars) + 1

    @property
    def n_chars(self) -> int:
        """Number of real characters (C) - what spec 7.2 asks you to eyeball."""
        return len(self.chars)

    @property
    def size(self) -> int:
        """V: non-blank symbols, real characters plus <unk>."""
        return len(self.chars) + 1

    @property
    def num_classes(self) -> int:
        """V + 1, including the CTC blank."""
        return self.size + 1

    def encode(self, text: str) -> list[int]:
        """Unseen characters collapse to the single <unk> index, never crash."""
        return [self.stoi.get(ch, self.unk_index) for ch in text]

    def decode(self, indices: Iterable[int]) -> str:
        out = []
        for i in indices:
            if i == BLANK_INDEX:
                continue
            out.append(self.itos[i] if 0 <= i < len(self.itos) else UNK_TOKEN)
        return "".join(out)

    def coverage(self, texts: Iterable[str]) -> tuple[int, int, set[str]]:
        """(n_unk_chars, n_total_chars, unseen character set) for a split."""
        unseen: set[str] = set()
        n_unk = n_tot = 0
        for t in texts:
            for ch in t:
                n_tot += 1
                if ch not in self.stoi:
                    n_unk += 1
                    unseen.add(ch)
        return n_unk, n_tot, unseen

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chars": self.chars,
            "n_chars": self.n_chars,
            "size_V": self.size,
            "num_classes": self.num_classes,
            "blank_index": BLANK_INDEX,
            "unk_index": self.unk_index,
            "unk_token": UNK_TOKEN,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "Vocab":
        return cls(json.loads(path.read_text(encoding="utf-8"))["chars"])


def build_vocab(texts: Iterable[str]) -> Vocab:
    """Charset from the TRAINING SPLIT ONLY, after normalisation (spec 7.2)."""
    return Vocab({ch for t in texts for ch in t})


def describe_char(ch: str) -> str:
    try:
        name = unicodedata.name(ch)
    except ValueError:
        name = "<unnamed>"
    return f"U+{ord(ch):04X} {unicodedata.category(ch)} {name}"
