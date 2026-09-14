# Scaling Laws for OCR under Label Noise

How does optical character recognition error scale with **training set size** when a
known fraction of the training labels is **wrong**?

This repository holds the data pipeline and (in progress) the training code for a
controlled study on historical printed Devanagari. Everything is built so that a
single run is one point on a curve, and the curve is the result.

**Status:** data pipeline and training code complete, 65 tests passing. The baseline run
has not yet been executed, so no CER is reported here yet. See [Roadmap](#6-roadmap).

---

## 1. The research question

Real OCR ground truth is never clean — human transcribers mistype, and automatic
alignment tools misalign. The practical question this raises is:

> When labels are noisy, does collecting more data still help — and how much more
> data is needed to offset a given noise rate?

The experiment holds everything fixed and sweeps two axes:

| Axis | Config key | Values |
| --- | --- | --- |
| Training data seen | `data_fraction` | 0.125, 0.25, 0.5, 1.0 |
| Label noise rate | `noise_rate` | 0.0, 0.1, 0.2, 0.4 |

Each of the 16 cells produces one validation CER (Character Error Rate). Fitting a
curve through them gives the scaling law; comparing curves across noise rates answers
the question.

### Noise models

Noise is injected into the **training split only**. Validation and test labels are
never touched — measuring against corrupted targets is meaningless.

- **`char_flip`** — symmetric character-level corruption. Each character is replaced,
  independently with probability *r*, by a character drawn uniformly from the rest of
  the alphabet. Formally a symmetric noise transition matrix with diagonal `1 - r` and
  off-diagonal `r / (V - 1)`.
- **`line_swap`** — a selected subset of lines has its labels permuted among themselves
  via a derangement, so no line keeps its own transcription. Models misalignment rather
  than mistyping.

Both are seeded from `noise_seed`, kept **separate from** the training seed so noise
and initialisation can be varied independently.

### Design constraints

These are deliberate and load-bearing. Breaking any one of them invalidates the curve.

| Constraint | Reason |
| --- | --- |
| Train from scratch; no pretrained weights | A pretrained model has already consumed an unknown, large corpus. That destroys the "data seen" x-axis. |
| No data augmentation (`augment: False`) | Augmentation changes the effective dataset size — the exact quantity being measured. |
| Every run appends one row to a results file | A scaling law is fitted across many runs; results that live only in notebook output are lost on disconnect. |
| Every randomness source seeded and logged | Runs must be comparable across data sizes and noise rates. An unseeded run is a wasted run. |
| `dataset` is a task identifier, **not** a scaling axis | Each script has a different alphabet and different intrinsic difficulty. Devanagari and Malayalam are points on *different* curves and must never be pooled into one fit. |

---

## 2. Data

**Source:** [FID4SA-GT — Ground truth data for HTR on South Asian Scripts](https://heidata.uni-heidelberg.de/dataverse/FID4SA-GT),
heiDATA, Heidelberg University.
**Subset used:** Printed Devanagari — `doi:10.11588/data/EGOKEI`

Letterpress printings from the **Naval Kishore Press** (Lakhnau, North India), late 19th
to early 20th century, in Hindi, Sanskrit, Braj Bhasha and Awadhi. Each page ships as a
JPEG scan plus an XML transcription giving per-line coordinates and text.

The corpus is **not redistributed here** (approximately 1.5 GB). See
[Reproducing the data](#5-reproducing-the-data) to rebuild it from the DOI; the pipeline
is deterministic and reproduces the manifests byte-identically.

### As received

| | |
| --- | --- |
| Books | 19 |
| Pages (JPEG + XML pairs) | 247 |
| `TextLine` elements | 5,142 |
| XML format | ALTO v4 — 18 books, 4,858 lines |
| | PAGE 2013-07-15 — 1 book (`vyasa1906`), 284 lines |

Both formats are parsed. They differ structurally: ALTO stores geometry as a rectangle
and text in a `CONTENT` attribute; PAGE stores geometry as a polygon and text in a
`<Unicode>` child element. Namespaces are mandatory in both — plain tag matching
silently returns nothing.

---

## 3. Preprocessing

Run order, with the reason each step exists.

### 3.1 Schema inspection (before any parser was written)

`src/inspect_xml.py` dumps root tags, namespace maps and sample `TextLine` elements
across the corpus. A parser cannot be written correctly against an assumed schema —
this step is what discovered that one book uses PAGE rather than ALTO. Output is
committed at [`inspect/stage1_xml_inspection.txt`](inspect/stage1_xml_inspection.txt).

### 3.2 Line extraction

Pages are cropped into individual line images. A page-level image is not a usable
training unit for a CTC recogniser; a line is.

For PAGE polygons the **axis-aligned bounding box** is taken. No polygon masking and no
dewarping — both are accuracy tricks that add uncontrolled variables to a measurement
study.

Crops are converted to grayscale (colour carries no signal for character identity and
triples the I/O) and saved as lossless PNG.

### 3.3 Crop padding — a correction to the original specification

The specification called for a flat 2-pixel pad around each bounding box. **This was
wrong for this corpus.** The ground-truth boxes bound the base glyph band and exclude
the superscript vowel marks (*matras*) that sit above it.

![Crop comparison](inspect/crop_vs_padded.png)

*Each pair: original 2 px pad above, corrected pad below. The upper crop of each pair is
missing the marks that sit over the letter body.*

This matters more than it appears. A crop missing a matra, paired with a label that
contains that matra, asks the model to predict a character that is **not present in the
image**. It teaches noise, and it does so silently.

The fix: pad **proportionally to box height** — 55 % of the height above, 25 % below —
clamped at the midpoint to any vertically adjacent line sharing horizontal extent, so a
crop can never swallow its neighbour. Implemented in
[`compute_crop_boxes`](src/extract.py). Per-line audit at
[`inspect/crop_edge_audit.csv`](inspect/crop_edge_audit.csv).

### 3.4 Filtering

A line is dropped if any of the following hold:

| Condition | Reason | Dropped |
| --- | --- | ---: |
| Text is empty or whitespace only | No target to learn | 13 |
| CTC-infeasible (see below) | Label cannot be produced from the image | 76 |
| Height < 8 px or width < 16 px | Degenerate crop | 0 |
| Bounding box outside the image | Bad coordinates | 0 |
| Aspect ratio (w/h) > 60 | Almost certainly a merged multi-line region | 0 |
| Text length > 200 characters | Same | 0 |
| | **Total** | **89 / 5,142 (1.73 %)** |

Well inside the 10 % gate above which the specification says to suspect the parser
rather than the data.

**The CTC feasibility filter.** A CRNN reads a line image as a sequence of vertical
slices and emits one prediction per slice; CTC then collapses that sequence into the
output string. This imposes a hard requirement: **the number of slices must exceed the
number of target characters.** After resizing to `img_height = 32` and applying the
CNN's 4x horizontal downsampling, a crop of width *w* yields `ceil(w / 4)` timesteps.
Lines where timesteps fall below `1.2 * len(text)` are discarded.

![CTC-infeasible examples](inspect/ctc_infeasible_lines.png)

*Dropped lines. Each crop is a few characters wide; each label claims a full line of
text. These are annotation errors — the coordinates and the transcription disagree.*

Retaining them would feed the model impossible targets, producing infinite CTC loss that
`zero_infinity` silently masks — a failure mode that corrupts training without raising
an error.

Every drop is logged with its reason to `manifests/devanagari_drops.csv`.

### 3.5 Text normalisation

Applied **before anything else touches the text**, so the manifest stores
already-normalised strings and no downstream stage normalises a second time.

1. Unicode **NFC** normalisation
2. Whitespace runs collapsed to a single space; leading and trailing space stripped
3. Zero-width joiner / non-joiner (U+200D / U+200C) stripped — decision recorded in
   `strip_zero_width_joiners` and applied consistently

NFC is not optional for Indic scripts. The same visible Devanagari cluster can be
encoded as different codepoint sequences. Without normalisation the vocabulary inflates
with duplicates, identical strings compare as unequal, and CER is computed against an
inconsistent target.

### 3.6 Vocabulary

Built from the **training split only**, after normalisation. Index 0 is reserved for the
CTC blank; characters take 1..C; a single `<unk>` index absorbs anything unseen so
val/test cannot crash the encoder.

| | |
| --- | --- |
| Characters (C) | 119 |
| Symbols including `<unk>` (V) | 120 |
| Output classes (V + 1, including blank) | 121 |

Coverage of the held-out splits by the training charset:

| Split | Unseen characters | Rate |
| --- | --- | --- |
| val | 1 of 16,066 (the digit `4`) | 0.006 % |
| test | 0 of 17,898 | 0.000 % |

Saved to `manifests/devanagari_vocab.json`.

### 3.7 Splitting — by page, never by line

Lines from the same page share typeface, ink, scan quality and page-level artefacts.
Splitting randomly at line level places near-identical lines in both train and test,
which **leaks** and inflates the score.

Whole pages are therefore assigned to a split, and the three page sets are asserted
disjoint at runtime.

| Split | Pages | Lines | Characters | Mean chars/line |
| --- | ---: | ---: | ---: | ---: |
| train | 197 | 4,059 | 152,457 | 37.6 |
| val | 25 | 502 | 16,066 | 32.0 |
| test | 25 | 492 | 17,898 | 36.4 |
| **Total** | **247** | **5,053** | **186,421** | |

### 3.8 The scaling axis

`subsample_train` draws a seeded subset of the training split. Because the seed is
fixed, a given fraction is identical across runs and across noise rates:

| `data_fraction` | Train lines |
| --- | ---: |
| 0.125 | 507 |
| 0.25 | 1,015 |
| 0.5 | 2,030 |
| 1.0 | 4,059 |

**4,059 lines is the ceiling**, and it directly limits how far the curve can extend.
This is a small-data scaling study by construction.

---

## 4. Repository layout

```
src/
  config.py        every knob and path; nothing configurable is hard-coded elsewhere
  inspect_xml.py   schema inspection (detection only, extracts nothing)
  parse_xml.py     ALTO and PAGE parsers -> LineRecord
  extract.py       cropping, filtering, manifest construction
  vocab.py         NFC normalisation, charset, encode/decode
  data.py          page-level splitting, scaling-axis subsampling
  noise.py         label noise injection and measured-corruption reporting
  dataset.py       manifest rows -> padded CTC batches
  model.py         CRNN: conv stack -> bidirectional LSTM -> CTC head
  metrics.py       greedy CTC decode, corpus-level CER and WER
  train.py         training loop, evaluation, per-run results logging
notebook.ipynb     Colab driver: environment setup and run launching only
tests/             65 pytest cases
manifests/         committed pipeline output (labels, drops, vocabulary)
inspect/           committed inspection artefacts and visual evidence
```

`src/parse_xml.py` and `src/extract.py` are the only dataset-specific layer. Nothing
downstream of the manifest references XML or page geometry, so a second script is added
by writing a parser — not by touching the training code.

### Committed manifests

| File | Contents |
| --- | --- |
| `manifests/devanagari_lines.csv` | One row per kept line: path, page_id, source_id, line_id, normalised text, width, height, n_chars |
| `manifests/devanagari_drops.csv` | One row per dropped line, with reason, bbox and text |
| `manifests/devanagari_vocab.json` | Character list and index scheme |

These are the preprocessing result. Every experimental decision downstream — splits,
subsampling, noise injection — reads the manifest and nothing else.

The committed copy is a snapshot for review. At runtime the pipeline reads and writes
manifests under `OCR_PERSISTENT_ROOT` (see [section 5](#5-reproducing-the-data)), which
defaults to a directory outside the repository so that bulk outputs and checkpoints stay
untracked.

---

## 5. Reproducing the data

```bash
pip install -r requirements.txt
```

1. Download the Printed Devanagari archive for `doi:10.11588/data/EGOKEI` from
   [heiDATA](https://heidata.uni-heidelberg.de/dataverse/FID4SA-GT). It arrives as
   `dataverse_files.zip` containing 19 per-book archives.

2. Unpack so the layout is:

   ```
   data/raw/devanagari/<book>/<book>/{*.jpg, alto/*.xml}
   ```

3. Confirm the schema, then extract:

   ```bash
   python src/inspect_xml.py     # root tags, namespaces, TextLine samples
   python src/extract.py         # line crops + manifests
   ```

   Expect `lines extracted: 5053` and `lines dropped: 89 (1.73%)`.

4. Run the tests:

   ```bash
   python -m pytest -q           # 40 passed
   ```

Outputs are written to the directory given by `OCR_PERSISTENT_ROOT` (default: a sibling
`OCR_Model_Training_output/`); bulk line crops go to `OCR_WORK_ROOT/data/lines/`. Both
are overridable by environment variable.

> **Note:** `src/extract.py` does not clear stale crops. Empty `data/lines/devanagari/`
> before re-extracting, or files dropped by a later filter revision will linger.

---

## 6. Roadmap

**Complete** — the full pipeline: schema inspection, ALTO/PAGE parsing, line extraction,
filtering, normalisation, vocabulary, page-level splitting, scaling-axis subsampling,
label noise injection, CRNN model, greedy CTC decoding, CER/WER metrics, and the
training loop with per-run results logging. 65 tests passing.

Verified end to end on CPU: the model memorises a 16-line subset to **CER 0.0000,
16/16 exact matches**, confirming that image loading, label encoding, the CTC setup,
decoding and scoring are mutually consistent.

**Next**

1. **Baseline run** — `data_fraction = 1.0`, `noise_rate = 0.0`, on a Colab T4 via
   `notebook.ipynb`. Roughly 20–35 minutes.
2. **The sweep** — the remaining 15 cells of the 4x4 grid, which is the same call with
   two configuration values changed.
3. **Curve fitting** — fit error against training-set size per noise rate, and compare
   the exponents.

**Baseline success criterion:** validation CER below roughly 30 %; below 10 % is good.
The approximately 2.29 % CER reported by the Heidelberg team using Transkribus is **not**
the target — that is a mature production system, and matching it is not the purpose of
this study.

### Design decisions worth knowing

- **The vocabulary is built from the full training split, before subsampling.** Building
  it from each subsample would make the output layer's width a function of
  `data_fraction`, turning one axis into two. The alphabet is a property of the script,
  not of the sample.
- **Horizontal downsampling is fixed at 4x** and is load-bearing: the manifest was
  filtered on the assumption that a width-*W* crop yields `ceil(W/4)` timesteps. Change
  the architecture's downsampling and the manifest must be rebuilt. A test asserts the
  two agree.
- **Results are one JSON file per run**, not appended rows in a shared CSV. Parallel
  experiments on separate branches would otherwise conflict on every merge.
  `aggregate_results()` collects them into a table.
- **CER is corpus-level** (total edits over total reference characters). The mean of
  per-line CERs is also recorded, but it over-weights short lines and is not the
  headline number.

---

## 7. Data licence and citation

The FID4SA-GT corpus is the work of the Specialised Information Service South Asia
(FID4SA), Heidelberg University. **Verify the licence terms on the dataset page before
redistributing any part of the corpus.** This repository deliberately ships no page
images and no transcriptions from it — only derived statistics and the code that
produces them.

Cite the dataset as instructed on its heiDATA landing page: `doi:10.11588/data/EGOKEI`
