"""Spec section 10 - CTC decoding, CER and WER.

CER is reported corpus-level: total edit distance divided by total reference
characters. The mean of per-line CERs is a different statistic - it weights a
five-character line as heavily as a fifty-character one - and the two disagree
enough to matter when comparing runs. Both are returned; `cer` is the headline.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch

try:  # editdistance is C-backed and much faster; the fallback keeps tests runnable.
    import editdistance as _ed

    def _distance(a: Sequence, b: Sequence) -> int:
        return int(_ed.eval(a, b))

except ImportError:  # pragma: no cover - exercised only where the wheel is absent

    def _distance(a: Sequence, b: Sequence) -> int:
        """Levenshtein distance, two-row dynamic programme."""
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        previous = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            current = [i]
            for j, cb in enumerate(b, start=1):
                current.append(
                    min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
                )
            previous = current
        return previous[-1]


BLANK_INDEX = 0


def greedy_decode(
    log_probs: torch.Tensor, input_lengths: torch.Tensor | None = None
) -> list[list[int]]:
    """(T, B, C) log-probabilities -> index sequences.

    Best path per timestep, then the standard CTC collapse: drop repeats first,
    then drop blanks. Doing it in the other order would merge a genuine double
    character that the model correctly separated with a blank.
    """
    best = log_probs.argmax(dim=2).transpose(0, 1).cpu()  # (B, T)
    lengths = (
        [best.size(1)] * best.size(0) if input_lengths is None else input_lengths.tolist()
    )

    out: list[list[int]] = []
    for row, length in zip(best, lengths):
        seq: list[int] = []
        previous = None
        for index in row[: int(length)].tolist():
            if index != previous:
                if index != BLANK_INDEX:
                    seq.append(index)
                previous = index
        out.append(seq)
    return out


def cer_counts(prediction: str, reference: str) -> tuple[int, int]:
    """(edit distance, reference length) at character level."""
    return _distance(prediction, reference), len(reference)


def wer_counts(prediction: str, reference: str) -> tuple[int, int]:
    """(edit distance, reference length) over whitespace-separated tokens."""
    p, r = prediction.split(), reference.split()
    return _distance(p, r), len(r)


def evaluate_strings(
    predictions: Iterable[str], references: Iterable[str]
) -> dict[str, float]:
    """Corpus-level CER and WER, plus the per-line mean CER for comparison."""
    char_edits = char_total = word_edits = word_total = 0
    per_line: list[float] = []

    n = 0
    for prediction, reference in zip(predictions, references):
        n += 1
        ce, ct = cer_counts(prediction, reference)
        we, wt = wer_counts(prediction, reference)
        char_edits += ce
        char_total += ct
        word_edits += we
        word_total += wt
        per_line.append(ce / ct if ct else float(ce > 0))

    return {
        "cer": char_edits / char_total if char_total else 0.0,
        "wer": word_edits / word_total if word_total else 0.0,
        "cer_per_line_mean": sum(per_line) / len(per_line) if per_line else 0.0,
        "n_lines": n,
        "n_ref_chars": char_total,
    }
