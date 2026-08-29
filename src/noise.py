"""Spec section 8 - label noise injection.

Operates on the TRAINING SPLIT ONLY. Validation and test labels stay clean:
measuring against corrupted labels is meaningless.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

NOISE_TYPES = ("none", "char_flip", "line_swap")


def _alphabet(texts: Iterable[str]) -> list[str]:
    """The training charset. The CTC blank is not a character and cannot appear."""
    return sorted({ch for t in texts for ch in t})


def _char_flip(
    texts: list[str], noise_rate: float, rng: np.random.Generator, charset: list[str]
) -> tuple[list[str], np.ndarray]:
    """Symmetric character-level noise: diagonal 1-r, off-diagonal r/(V-1)."""
    v = len(charset)
    if v < 2:
        return list(texts), np.zeros(len(texts), dtype=bool)
    index = {c: i for i, c in enumerate(charset)}

    flat = np.array([index[ch] for t in texts for ch in t], dtype=np.int64)
    lengths = np.array([len(t) for t in texts], dtype=np.int64)

    # One draw per character, in one pass: order is fixed, so the seed fully
    # determines the output.
    hit = rng.random(flat.size) < noise_rate
    offset = rng.integers(0, v - 1, size=flat.size)
    # Skip the character's own index so a "flip" can never be a no-op.
    replacement = offset + (offset >= flat)
    out_flat = np.where(hit, replacement, flat)

    chars = np.array(charset, dtype=object)[out_flat]
    noisy: list[str] = []
    mask = np.zeros(len(texts), dtype=bool)
    pos = 0
    for i, n in enumerate(lengths):
        noisy.append("".join(chars[pos:pos + n]))
        mask[i] = bool(hit[pos:pos + n].any())
        pos += n
    return noisy, mask


def _derange(k: int, rng: np.random.Generator) -> np.ndarray:
    """Sattolo's algorithm: one cycle, therefore no fixed point."""
    perm = np.arange(k)
    for i in range(k - 1, 0, -1):
        j = int(rng.integers(0, i))
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def _line_swap(
    texts: list[str], noise_rate: float, rng: np.random.Generator
) -> tuple[list[str], np.ndarray]:
    """Permute the labels of a selected subset among themselves (a derangement)."""
    n = len(texts)
    k = int(round(n * noise_rate))
    noisy = list(texts)
    mask = np.zeros(n, dtype=bool)
    if k < 2:
        return noisy, mask

    selected = np.sort(rng.choice(n, size=k, replace=False))
    perm = _derange(k, rng)

    # A derangement is fixed-point-free by index, but duplicate transcriptions
    # (page numbers, "लखनऊ") can still hand a row back its own string. Repair
    # those so every selected row really does carry a different label.
    sel_texts = [texts[i] for i in selected]
    for a in range(k):
        if sel_texts[perm[a]] != sel_texts[a]:
            continue
        for b in rng.permutation(k):
            if b == a:
                continue
            if sel_texts[perm[b]] != sel_texts[a] and sel_texts[perm[a]] != sel_texts[b]:
                perm[a], perm[b] = perm[b], perm[a]
                break

    for a, row in enumerate(selected):
        noisy[row] = sel_texts[perm[a]]
        mask[row] = noisy[row] != texts[row]
    return noisy, mask


def inject_noise(
    df: pd.DataFrame,
    noise_type: str,
    noise_rate: float,
    seed: int,
    charset: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (df_noisy, noise_mask). Never mutates the input (spec 8.1)."""
    if noise_type not in NOISE_TYPES:
        raise ValueError(f"noise_type must be one of {NOISE_TYPES}, got {noise_type!r}")

    df_noisy = df.copy(deep=True)
    texts = [str(t) for t in df["text"].tolist()]

    if noise_type == "none" or noise_rate <= 0.0:
        return df_noisy, np.zeros(len(df), dtype=bool)

    rng = np.random.default_rng(seed)
    if noise_type == "char_flip":
        noisy, mask = _char_flip(texts, noise_rate, rng, charset or _alphabet(texts))
    else:
        noisy, mask = _line_swap(texts, noise_rate, rng)

    df_noisy["text"] = noisy
    df_noisy["n_chars"] = [len(t) for t in noisy]
    return df_noisy, mask


def measured_corruption(
    original: pd.DataFrame, noisy: pd.DataFrame, mask: np.ndarray
) -> dict[str, float]:
    """Measured, not requested - spec 14 logs these beside the config values."""
    a = [str(t) for t in original["text"]]
    b = [str(t) for t in noisy["text"]]
    n_chars = sum(len(t) for t in a)
    changed = sum(
        sum(1 for x, y in zip(u, v) if x != y) + abs(len(u) - len(v))
        for u, v in zip(a, b)
    )
    return {
        "frac_labels_actually_corrupted": float(mask.mean()) if len(mask) else 0.0,
        "frac_chars_actually_corrupted": changed / n_chars if n_chars else 0.0,
        "n_lines_corrupted": int(mask.sum()),
        "n_chars_corrupted": int(changed),
    }


def save_corruption_artifacts(
    original: pd.DataFrame,
    noisy: pd.DataFrame,
    mask: np.ndarray,
    results_dir,
    run_name: str,
    n_examples: int = 20,
) -> None:
    """Persist the mask and a side-by-side sample (spec 8.4)."""
    from pathlib import Path

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    np.save(results_dir / f"{run_name}_noise_mask.npy", mask)
    idx = np.flatnonzero(mask)[:n_examples]
    pd.DataFrame({
        "line_path": original.iloc[idx]["line_path"].values,
        "original_text": original.iloc[idx]["text"].values,
        "corrupted_text": noisy.iloc[idx]["text"].values,
    }).to_csv(results_dir / f"{run_name}_corrupted_examples.csv", index=False, encoding="utf-8")
