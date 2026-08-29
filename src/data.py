"""Spec section 6.5 - page-level splitting, and the scaling-axis subsample.

Everything here reads the manifest only. No XML, no page geometry (spec 18).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONFIG, PATHS  # noqa: E402

SPLITS = ("train", "val", "test")


def load_manifest(dataset: str | None = None) -> pd.DataFrame:
    dataset = dataset or CONFIG["dataset"]
    path = PATHS["manifests_dir"] / f"{dataset}_lines.csv"
    df = pd.read_csv(path, encoding="utf-8", keep_default_na=False, na_values=[])
    df["text"] = df["text"].astype(str)
    return df


def split_by_page(
    df: pd.DataFrame,
    val_split: float,
    test_split: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Assign whole pages to train/val/test. Lines never cross a split (spec 6.5)."""
    pages = np.array(sorted(df["page_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(pages)

    n = len(pages)
    n_val = int(round(n * val_split))
    n_test = int(round(n * test_split))
    page_sets = {
        "val": set(pages[:n_val]),
        "test": set(pages[n_val:n_val + n_test]),
        "train": set(pages[n_val + n_test:]),
    }

    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                overlap = page_sets[a] & page_sets[b]
                assert not overlap, f"page_id in both {a} and {b}: {sorted(overlap)[:5]}"

    return {s: df[df["page_id"].isin(page_sets[s])].reset_index(drop=True) for s in SPLITS}


def subsample_train(
    df_train: pd.DataFrame,
    data_fraction: float,
    max_train_lines: int | None,
    seed: int,
) -> pd.DataFrame:
    """The scaling-law axis. Seeded, so a fraction is reproducible across runs."""
    n = len(df_train)
    target = n if data_fraction >= 1.0 else int(round(n * data_fraction))
    if max_train_lines is not None:
        target = min(target, max_train_lines)
    if target >= n:
        return df_train.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    keep = rng.choice(n, size=target, replace=False)
    return df_train.iloc[np.sort(keep)].reset_index(drop=True)


def split_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Line, page and character counts per split - the measured numbers."""
    return pd.DataFrame([
        {
            "split": s,
            "pages": splits[s]["page_id"].nunique(),
            "lines": len(splits[s]),
            "chars": int(splits[s]["n_chars"].sum()),
            "mean_chars_per_line": round(float(splits[s]["n_chars"].mean()), 2),
        }
        for s in SPLITS
    ])
