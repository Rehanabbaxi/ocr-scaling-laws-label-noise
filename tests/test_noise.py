"""Spec 8.5 - the noise unit tests. Do not skip these."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from noise import inject_noise, measured_corruption

ALPHABET = list("abcdefghijklmnopqrstuvwxyz")


def make_df(n: int = 500, line_len: int = 40, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    texts = ["".join(rng.choice(ALPHABET, size=line_len)) for _ in range(n)]
    return pd.DataFrame({
        "line_path": [f"l{i}.png" for i in range(n)],
        "page_id": [f"p{i // 10}" for i in range(n)],
        "line_id": [f"line{i}" for i in range(n)],
        "text": texts,
        "n_chars": [len(t) for t in texts],
    })


def char_frac_changed(a: pd.DataFrame, b: pd.DataFrame) -> float:
    tot = sum(len(t) for t in a["text"])
    ch = sum(sum(1 for x, y in zip(u, v) if x != y) for u, v in zip(a["text"], b["text"]))
    return ch / tot


@pytest.mark.parametrize("noise_type", ["char_flip", "line_swap"])
def test_rate_zero_is_identity(noise_type: str) -> None:
    """noise_rate=0.0: df_noisy identical to df, mask all False."""
    df = make_df()
    out, mask = inject_noise(df, noise_type, 0.0, seed=1234)
    pd.testing.assert_frame_equal(out, df)
    assert not mask.any()
    assert mask.dtype == bool and mask.shape == (len(df),)


def test_rate_one_char_flip_changes_every_character() -> None:
    """noise_rate=1.0 (char_flip): no character survives unchanged."""
    df = make_df()
    out, mask = inject_noise(df, "char_flip", 1.0, seed=1234)
    for original, corrupted in zip(df["text"], out["text"]):
        assert len(original) == len(corrupted)
        assert all(x != y for x, y in zip(original, corrupted))
    assert mask.all()


def test_char_flip_measured_rate_within_tolerance() -> None:
    """noise_rate=0.2 (char_flip): measured altered fraction is 0.2 +/- 0.02."""
    df = make_df(n=1000, line_len=50)
    out, _ = inject_noise(df, "char_flip", 0.2, seed=1234)
    assert abs(char_frac_changed(df, out) - 0.2) <= 0.02


def test_line_swap_exact_fraction_and_no_row_keeps_own_label() -> None:
    """noise_rate=0.2 (line_swap): exactly 20% of rows differ; none keeps its own."""
    df = make_df(n=500)
    out, mask = inject_noise(df, "line_swap", 0.2, seed=1234)
    differing = [i for i in range(len(df)) if df["text"][i] != out["text"][i]]
    assert len(differing) == 100
    assert mask.sum() == 100
    assert np.array_equal(np.flatnonzero(mask), np.array(differing))
    assert sorted(out["text"]) == sorted(df["text"])  # labels permuted, not invented


def test_line_swap_survives_duplicate_labels() -> None:
    """Duplicate transcriptions must not let a swapped row keep its own string."""
    texts = ["dup"] * 40 + [f"u{i}" for i in range(60)]
    df = pd.DataFrame({"text": texts, "n_chars": [len(t) for t in texts]})
    out, mask = inject_noise(df, "line_swap", 0.5, seed=99)
    assert mask.sum() == 50
    assert all(df["text"][i] != out["text"][i] for i in np.flatnonzero(mask))


@pytest.mark.parametrize("noise_type,rate", [("char_flip", 0.2), ("line_swap", 0.2)])
def test_same_seed_twice_is_byte_identical(noise_type: str, rate: float) -> None:
    """Same seed twice -> byte-identical output."""
    df = make_df()
    a, ma = inject_noise(df, noise_type, rate, seed=1234)
    b, mb = inject_noise(df, noise_type, rate, seed=1234)
    pd.testing.assert_frame_equal(a, b)
    assert np.array_equal(ma, mb)
    assert a["text"].str.cat().encode() == b["text"].str.cat().encode()


@pytest.mark.parametrize("noise_type", ["char_flip", "line_swap"])
def test_different_seed_gives_different_output(noise_type: str) -> None:
    df = make_df()
    a, _ = inject_noise(df, noise_type, 0.2, seed=1234)
    b, _ = inject_noise(df, noise_type, 0.2, seed=4321)
    assert list(a["text"]) != list(b["text"])


@pytest.mark.parametrize("noise_type", ["char_flip", "line_swap"])
def test_input_is_never_mutated(noise_type: str) -> None:
    df = make_df()
    before = df.copy(deep=True)
    inject_noise(df, noise_type, 0.3, seed=5)
    pd.testing.assert_frame_equal(df, before)


def test_char_flip_never_introduces_out_of_vocabulary_characters() -> None:
    df = make_df()
    out, _ = inject_noise(df, "char_flip", 0.5, seed=1234)
    assert set("".join(out["text"])) <= set(ALPHABET)


def test_measured_corruption_reports_reality_not_request() -> None:
    df = make_df(n=1000, line_len=50)
    out, mask = inject_noise(df, "char_flip", 0.2, seed=1234)
    m = measured_corruption(df, out, mask)
    assert abs(m["frac_chars_actually_corrupted"] - 0.2) <= 0.02
    assert m["n_lines_corrupted"] == int(mask.sum())


def test_unknown_noise_type_raises() -> None:
    with pytest.raises(ValueError):
        inject_noise(make_df(), "gaussian", 0.1, seed=1)
