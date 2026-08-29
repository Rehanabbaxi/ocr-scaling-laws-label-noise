"""Spec 15 sanity checks that live in the data layer, plus spec 7 normalisation."""

from __future__ import annotations

import unicodedata

import pandas as pd
import pytest

from config import CONFIG, PATHS
from data import SPLITS, load_manifest, split_by_page, split_summary, subsample_train
from vocab import BLANK_INDEX, UNK_TOKEN, Vocab, build_vocab, normalise


# --- spec 7.1 normalisation -------------------------------------------------

def test_nfc_is_applied() -> None:
    decomposed = "\u0915\u093c"          # KA + NUKTA
    composed = "\u0958"                   # KA WITH NUKTA
    assert normalise(decomposed) == normalise(composed)
    assert unicodedata.is_normalized("NFC", normalise(decomposed))


def test_whitespace_is_collapsed_and_stripped() -> None:
    assert normalise("  a\t\tb\n\nc  ") == "a b c"


def test_zero_width_joiners_are_stripped_when_configured() -> None:
    assert normalise("\u0915\u200d\u0916", strip_zero_width=True) == "\u0915\u0916"
    assert normalise("\u0915\u200d\u0916", strip_zero_width=False) == "\u0915\u200d\u0916"


def test_normalise_is_idempotent() -> None:
    s = "  \u0915\u093c \u200c \u0916  "
    assert normalise(normalise(s)) == normalise(s)


# --- spec 7.2 vocabulary ----------------------------------------------------

def test_blank_is_index_zero_and_chars_start_at_one() -> None:
    v = build_vocab(["abc"])
    assert BLANK_INDEX == 0
    assert v.stoi["a"] == 1
    assert v.num_classes == v.size + 1


def test_unseen_characters_map_to_a_single_unk_index_and_never_crash() -> None:
    v = build_vocab(["abc"])
    assert v.encode("xyz") == [v.unk_index] * 3
    assert v.itos[v.unk_index] == UNK_TOKEN


def test_encode_decode_roundtrip_for_known_characters() -> None:
    v = build_vocab(["abc "])
    assert v.decode(v.encode("cab")) == "cab"


def test_vocab_json_roundtrip(tmp_path) -> None:
    v = build_vocab(["\u0915\u0916 abc"])
    p = tmp_path / "vocab.json"
    v.to_json(p)
    assert Vocab.from_json(p).chars == v.chars


# --- spec 6.5 splitting -----------------------------------------------------

def _synthetic(n_pages: int = 100, per_page: int = 10) -> pd.DataFrame:
    rows = [
        {"page_id": f"p{p:03d}", "line_id": f"l{i}", "text": "ab", "n_chars": 2}
        for p in range(n_pages) for i in range(per_page)
    ]
    return pd.DataFrame(rows)


def test_split_page_sets_are_disjoint() -> None:
    s = split_by_page(_synthetic(), 0.1, 0.1, seed=42)
    sets = {k: set(v["page_id"]) for k, v in s.items()}
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                assert not sets[a] & sets[b]


def test_split_is_exhaustive_and_loses_no_line() -> None:
    df = _synthetic()
    s = split_by_page(df, 0.1, 0.1, seed=42)
    assert sum(len(v) for v in s.values()) == len(df)


def test_split_is_deterministic_for_a_fixed_seed() -> None:
    a = split_by_page(_synthetic(), 0.1, 0.1, seed=42)
    b = split_by_page(_synthetic(), 0.1, 0.1, seed=42)
    for k in SPLITS:
        assert list(a[k]["page_id"]) == list(b[k]["page_id"])


def test_split_changes_with_the_seed() -> None:
    a = split_by_page(_synthetic(), 0.1, 0.1, seed=42)
    b = split_by_page(_synthetic(), 0.1, 0.1, seed=7)
    assert set(a["test"]["page_id"]) != set(b["test"]["page_id"])


def test_no_line_from_one_page_appears_in_two_splits() -> None:
    s = split_by_page(_synthetic(), 0.1, 0.1, seed=42)
    seen: set[tuple[str, str]] = set()
    for k in SPLITS:
        keys = set(zip(s[k]["page_id"], s[k]["line_id"]))
        assert not keys & seen
        seen |= keys


# --- spec 3 scaling axis ----------------------------------------------------

@pytest.mark.parametrize("frac", [0.125, 0.25, 0.5, 1.0])
def test_subsample_returns_the_requested_fraction(frac: float) -> None:
    df = _synthetic()
    out = subsample_train(df, frac, None, seed=42)
    assert len(out) == (len(df) if frac >= 1.0 else round(len(df) * frac))


def test_subsample_is_deterministic() -> None:
    df = _synthetic()
    a = subsample_train(df, 0.25, None, seed=42)
    b = subsample_train(df, 0.25, None, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_max_train_lines_caps_the_subsample() -> None:
    assert len(subsample_train(_synthetic(), 1.0, 50, seed=42)) == 50


# --- spec 15 checks against the real extracted corpus -----------------------

manifest_exists = pytest.mark.skipif(
    not (PATHS["manifests_dir"] / f"{CONFIG['dataset']}_lines.csv").exists(),
    reason="manifest not built yet - run src/extract.py",
)


@manifest_exists
def test_real_split_page_sets_are_disjoint() -> None:
    s = split_by_page(load_manifest(), CONFIG["val_split"], CONFIG["test_split"], CONFIG["seed"])
    sets = {k: set(v["page_id"]) for k, v in s.items()}
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]
    assert sum(len(v) for v in s.values()) == len(load_manifest())


@manifest_exists
def test_real_charset_contains_no_control_characters() -> None:
    """Spec 15 vocabulary sanity."""
    s = split_by_page(load_manifest(), CONFIG["val_split"], CONFIG["test_split"], CONFIG["seed"])
    v = build_vocab(s["train"]["text"])
    offenders = [c for c in v.chars if unicodedata.category(c).startswith("C")]
    assert not offenders, f"control characters in charset: {offenders!r}"


@manifest_exists
def test_real_charset_latin_content_is_negligible_and_explained() -> None:
    """Latin here is genuine Roman imprint text, not a transcription artefact."""
    df = load_manifest()
    total = int(df["n_chars"].sum())
    latin = sum(1 for t in df["text"] for c in t if "LATIN" in unicodedata.name(c, ""))
    assert latin / total < 0.001, f"{latin}/{total} Latin characters - investigate"


@manifest_exists
def test_manifest_has_no_empty_labels_and_matching_char_counts() -> None:
    df = load_manifest()
    assert (df["text"].str.strip() != "").all()
    assert (df["text"].str.len() == df["n_chars"]).all()


@manifest_exists
def test_manifest_text_is_already_nfc_normalised() -> None:
    df = load_manifest()
    assert all(unicodedata.is_normalized("NFC", t) for t in df["text"])


@manifest_exists
def test_every_manifest_row_points_at_an_existing_image() -> None:
    df = load_manifest()
    root = PATHS["lines_dir"] / CONFIG["dataset"]
    missing = [p for p in df["line_path"] if not (root / p).exists()]
    assert not missing, f"{len(missing)} missing crops, e.g. {missing[:3]}"
