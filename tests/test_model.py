"""Spec 15 sanity checks for the model, batching and metrics layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from config import CONFIG
from dataset import PAD_VALUE, LineDataset, collate, resize_width
from extract import ctc_feasible
from metrics import evaluate_strings, greedy_decode
from model import CRNN, build_model, count_parameters
from vocab import build_vocab

torch.manual_seed(0)


@pytest.fixture(scope="module")
def vocab():
    return build_vocab(["अब c", "कखग"])


@pytest.fixture(scope="module")
def model(vocab):
    # A narrow model keeps the test fast; the width arithmetic is unaffected.
    return CRNN(num_classes=vocab.num_classes, width_mult=0.125, rnn_hidden=16, rnn_layers=1)


# --- architecture -----------------------------------------------------------

def test_output_is_time_major_with_one_step_per_four_pixels(model, vocab):
    out = model(torch.zeros(2, 1, 32, 128))
    assert out.shape == (128 // 4, 2, vocab.num_classes)


@pytest.mark.parametrize("width", [32, 64, 100, 256, 512])
def test_timestep_count_matches_the_filter_that_built_the_manifest(model, width):
    """extract.ctc_feasible assumed ceil(W/4) timesteps. If the model disagrees,
    the manifest was filtered against the wrong number and must be rebuilt."""
    padded = -(-width // 4) * 4
    actual = model(torch.zeros(1, 1, 32, padded)).shape[0]
    assert actual == model.timesteps_for_width(padded)
    assert actual >= model.timesteps_for_width(width)


def test_log_probabilities_are_normalised_and_fp32(model):
    out = model(torch.zeros(1, 1, 32, 64))
    assert out.dtype == torch.float32
    assert torch.allclose(out.exp().sum(dim=2), torch.ones(out.shape[:2]), atol=1e-5)


def test_non_32_height_is_rejected():
    with pytest.raises(ValueError, match="img_height=32"):
        CRNN(num_classes=10, img_height=64)


def test_width_multiplier_changes_capacity(vocab):
    small = count_parameters(CRNN(vocab.num_classes, width_mult=0.25, rnn_hidden=16))
    large = count_parameters(CRNN(vocab.num_classes, width_mult=1.0, rnn_hidden=16))
    assert large > small


def test_build_model_uses_config(vocab):
    m = build_model(vocab.num_classes)
    assert m.num_classes == vocab.num_classes
    assert m.img_height == CONFIG["img_height"]


# --- batching ---------------------------------------------------------------

def test_resize_preserves_aspect_and_caps_width():
    assert resize_width(100, 50, 32, 512) == 64
    assert resize_width(10_000, 50, 32, 512) == 512
    assert resize_width(1, 5000, 32, 512) >= 1


def test_collate_pads_to_a_whole_number_of_timesteps():
    batch = [
        (torch.zeros(1, 32, 30), torch.tensor([1, 2]), "ab"),
        (torch.zeros(1, 32, 61), torch.tensor([3]), "c"),
    ]
    out = collate(batch, width_downsample=4)
    assert out["images"].shape == (2, 1, 32, 64)   # 61 -> 64, a multiple of 4
    assert out["input_lengths"].tolist() == [8, 16]  # ceil(30/4), ceil(61/4)
    assert out["target_lengths"].tolist() == [2, 1]
    assert out["targets"].tolist() == [1, 2, 3]


def test_collate_pads_with_paper_white_not_black():
    batch = [
        (torch.full((1, 32, 8), -1.0), torch.tensor([1]), "a"),
        (torch.full((1, 32, 16), -1.0), torch.tensor([1]), "a"),
    ]
    out = collate(batch, width_downsample=4)
    assert out["images"][0, 0, 0, 8:].eq(PAD_VALUE).all()


def test_input_lengths_never_exceed_the_padded_timesteps():
    batch = [(torch.zeros(1, 32, w), torch.tensor([1]), "a") for w in (7, 33, 64)]
    out = collate(batch, width_downsample=4)
    assert out["input_lengths"].max().item() <= out["images"].shape[-1] // 4


def test_dataset_reads_a_crop_and_encodes_its_label(tmp_path, vocab):
    from PIL import Image

    Image.new("L", (80, 40), color=255).save(tmp_path / "x.png")
    df = pd.DataFrame([{"line_path": "x.png", "text": "अब"}])
    image, target, text = LineDataset(df, tmp_path, vocab)[0]

    assert image.shape == (1, 32, resize_width(80, 40, 32, CONFIG["img_max_width"]))
    assert image.min() >= -1.0 and image.max() <= 1.0
    assert target.tolist() == vocab.encode("अब")
    assert text == "अब"


# --- CTC decoding -----------------------------------------------------------

def _one_hot(sequence, num_classes):
    """(T, 1, C) log-probabilities that argmax to `sequence`."""
    out = torch.full((len(sequence), 1, num_classes), -20.0)
    for t, index in enumerate(sequence):
        out[t, 0, index] = 0.0
    return out


def test_greedy_decode_collapses_repeats_then_strips_blanks():
    assert greedy_decode(_one_hot([1, 1, 0, 1, 2, 2], 5))[0] == [1, 1, 2]


def test_blank_separated_double_characters_survive():
    """Collapsing blanks before repeats would turn this into a single '1'."""
    assert greedy_decode(_one_hot([1, 0, 1], 5))[0] == [1, 1]


def test_decode_respects_input_lengths():
    log_probs = _one_hot([1, 2, 3, 4], 6)
    assert greedy_decode(log_probs, torch.tensor([2]))[0] == [1, 2]


def test_all_blank_decodes_to_empty():
    assert greedy_decode(_one_hot([0, 0, 0], 5))[0] == []


# --- metrics ----------------------------------------------------------------

def test_cer_is_zero_for_an_exact_match():
    assert evaluate_strings(["अबक"], ["अबक"])["cer"] == 0.0


def test_cer_counts_edits_over_total_reference_characters():
    # one substitution in four characters, plus a perfect eight-character line
    scores = evaluate_strings(["abxd", "abcdefgh"], ["abcd", "abcdefgh"])
    assert scores["cer"] == pytest.approx(1 / 12)


def test_corpus_cer_differs_from_the_per_line_mean():
    """Short lines dominate a per-line mean; the headline metric must not."""
    scores = evaluate_strings(["x", "abcdefghij"], ["a", "abcdefghij"])
    assert scores["cer"] == pytest.approx(1 / 11)
    assert scores["cer_per_line_mean"] == pytest.approx(0.5)


def test_wer_is_token_level():
    assert evaluate_strings(["the cat sat"], ["the dog sat"])["wer"] == pytest.approx(1 / 3)


def test_empty_prediction_scores_one():
    assert evaluate_strings([""], ["abcd"])["cer"] == 1.0


# --- the model and the manifest filter agree --------------------------------

def test_feasible_lines_really_do_fit_their_labels(model):
    """Every line the filter kept must give the model more timesteps than the
    label has characters, or CTC cannot align it."""
    for text_len, w, h in [(10, 200, 50), (40, 1400, 100), (5, 90, 60)]:
        if not ctc_feasible(text_len, w, h):
            continue
        resized = min(
            int(round(w * CONFIG["img_height"] / h)), CONFIG["img_max_width"]
        )
        assert model.timesteps_for_width(resized) >= text_len
