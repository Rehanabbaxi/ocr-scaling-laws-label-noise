"""Spec section 9/10 - the CRNN.

Convolutional feature extractor -> bidirectional LSTM -> CTC head.

The horizontal downsampling factor is fixed at CONFIG["cnn_width_downsample"]
(4) and is load-bearing: extract.py's CTC feasibility filter decided which
lines to keep by assuming a width-W crop yields exactly ceil(W / 4) timesteps.
If this architecture's downsampling changes, that filter is wrong and the
manifest must be rebuilt. test_model.py asserts the two agree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONFIG  # noqa: E402

# Channel widths before cnn_width_mult is applied.
BASE_CHANNELS = (64, 128, 256, 256, 512, 512)


def _scaled(channels: int, mult: float) -> int:
    """Scale a channel count, keeping it a positive multiple of 8."""
    return max(8, int(round(channels * mult / 8)) * 8)


class CRNN(nn.Module):
    """Input (B, 1, H, W) -> log-probabilities (T, B, num_classes), T = W / 4.

    Height must be CONFIG["img_height"] (32): the convolutional stack collapses
    it to 1 by design, so a different height silently changes the feature
    dimension fed to the LSTM.
    """

    def __init__(
        self,
        num_classes: int,
        img_height: int = 32,
        width_mult: float = 1.0,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if img_height != 32:
            raise ValueError(
                f"CRNN is specified for img_height=32, got {img_height}. The "
                "convolutional stack collapses height to 1 by construction."
            )
        self.num_classes = num_classes
        self.img_height = img_height
        self.width_downsample = CONFIG["cnn_width_downsample"]

        c1, c2, c3, c4, c5, c6 = (_scaled(c, width_mult) for c in BASE_CHANNELS)

        def block(in_c: int, out_c: int) -> list[nn.Module]:
            return [
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ]

        # Height 32 -> 1 over five pools; width -> W/4 over the first two only.
        # Every conv is padded, so the width arithmetic is exactly W/2, W/4.
        self.cnn = nn.Sequential(
            *block(1, c1),
            nn.MaxPool2d(2, 2),                      # H/2,  W/2
            *block(c1, c2),
            nn.MaxPool2d(2, 2),                      # H/4,  W/4
            *block(c2, c3),
            *block(c3, c4),
            nn.MaxPool2d((2, 1), (2, 1)),            # H/8,  W/4
            *block(c4, c5),
            *block(c5, c6),
            nn.MaxPool2d((2, 1), (2, 1)),            # H/16, W/4
            *block(c6, c6),
            nn.MaxPool2d((2, 1), (2, 1)),            # H/32, W/4  -> height 1
        )

        self.rnn = nn.LSTM(
            input_size=c6,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=False,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(rnn_hidden * 2, num_classes)

    def timesteps_for_width(self, width: int) -> int:
        """Timesteps this network emits for a crop of the given pixel width."""
        return -(-width // self.width_downsample)  # ceil

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 1, 32, W) -> log_softmax (T, B, num_classes), ready for CTCLoss."""
        feats = self.cnn(images)                     # (B, C, 1, T)
        if feats.size(2) != 1:
            raise RuntimeError(
                f"expected the conv stack to collapse height to 1, got {feats.size(2)}"
            )
        feats = feats.squeeze(2).permute(2, 0, 1)    # (T, B, C)
        seq, _ = self.rnn(feats)
        logits = self.head(self.dropout(seq))        # (T, B, num_classes)
        # CTC loss is computed in fp32 even under autocast: the log-sum-exp
        # recursion underflows in half precision on long sequences.
        return logits.float().log_softmax(dim=2)


def build_model(vocab_size_with_blank: int, cfg: dict | None = None) -> CRNN:
    """Construct the CRNN from CONFIG. `vocab_size_with_blank` is Vocab.num_classes."""
    cfg = cfg or CONFIG
    return CRNN(
        num_classes=vocab_size_with_blank,
        img_height=cfg["img_height"],
        width_mult=cfg["cnn_width_mult"],
        rnn_hidden=cfg["rnn_hidden"],
        rnn_layers=cfg["rnn_layers"],
        dropout=cfg["dropout"],
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
