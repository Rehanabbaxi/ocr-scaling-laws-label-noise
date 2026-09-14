"""Spec section 8 (dataset / DataLoader) - manifest rows to padded CTC batches.

Reads the manifest and the line PNGs only. No XML, no page geometry.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONFIG  # noqa: E402
from vocab import Vocab  # noqa: E402

# Pad value in normalised space. Crops are dark ink on light paper, so the
# padding must be paper-white or the model learns a black bar as a feature.
PAD_VALUE = 1.0


def resize_width(orig_w: int, orig_h: int, img_height: int, img_max_width: int) -> int:
    """Target width after scaling to img_height, preserving aspect ratio."""
    w = int(round(orig_w * img_height / max(1, orig_h)))
    return max(1, min(w, img_max_width))


class LineDataset(Dataset):
    """One manifest row -> (image, encoded target).

    `df` must already carry whatever label noise the run calls for: this class
    reads the `text` column as given and never modifies it.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        lines_dir: Path,
        vocab: Vocab,
        img_height: int | None = None,
        img_max_width: int | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.lines_dir = Path(lines_dir)
        self.vocab = vocab
        self.img_height = img_height or CONFIG["img_height"]
        self.img_max_width = img_max_width or CONFIG["img_max_width"]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[index]
        path = self.lines_dir / row["line_path"]
        with Image.open(path) as im:
            im = im.convert("L")
            w = resize_width(im.width, im.height, self.img_height, self.img_max_width)
            im = im.resize((w, self.img_height), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0

        # [0, 1] -> [-1, 1]; PAD_VALUE sits at the white end of this range.
        image = torch.from_numpy(arr).unsqueeze(0) * 2.0 - 1.0
        text = str(row["text"])
        target = torch.tensor(self.vocab.encode(text), dtype=torch.long)
        return image, target, text


def collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]], width_downsample: int | None = None
) -> dict[str, object]:
    """Pad to the widest crop in the batch, rounded up to a whole timestep.

    Rounding matters: the conv stack divides width by exactly
    `width_downsample`, so a batch width that is not a multiple of it would
    silently drop a partial column.
    """
    down = width_downsample or CONFIG["cnn_width_downsample"]
    images, targets, texts = zip(*batch)

    widths = [im.shape[-1] for im in images]
    height = images[0].shape[-2]
    padded_w = -(-max(widths) // down) * down

    canvas = torch.full((len(images), 1, height, padded_w), PAD_VALUE, dtype=torch.float32)
    for i, im in enumerate(images):
        canvas[i, :, :, : im.shape[-1]] = im

    # Valid timesteps per sample, from the true width rather than the padded one.
    input_lengths = torch.tensor([-(-w // down) for w in widths], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)

    return {
        "images": canvas,
        "targets": torch.cat(targets) if targets else torch.empty(0, dtype=torch.long),
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
        "texts": list(texts),
    }


def make_loader(
    df: pd.DataFrame,
    lines_dir: Path,
    vocab: Vocab,
    batch_size: int | None = None,
    shuffle: bool = False,
    num_workers: int | None = None,
    seed: int | None = None,
) -> DataLoader:
    """DataLoader with a seeded generator, so shuffling is reproducible."""
    dataset = LineDataset(df, lines_dir, vocab)
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(CONFIG["seed"] if seed is None else seed)

    workers = CONFIG["num_workers"] if num_workers is None else num_workers
    return DataLoader(
        dataset,
        batch_size=batch_size or CONFIG["batch_size"],
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=False,
        generator=generator,
        persistent_workers=workers > 0,
    )
