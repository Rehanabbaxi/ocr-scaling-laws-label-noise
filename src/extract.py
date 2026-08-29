"""Spec section 6 - line extraction: XML pages -> line images + manifest."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONFIG, PATHS  # noqa: E402
from parse_xml import LineRecord, detect_format, parse_any  # noqa: E402
from vocab import normalise  # noqa: E402

MIN_HEIGHT = 8
MIN_WIDTH = 16
MAX_ASPECT = 60.0
MAX_TEXT_LEN = 200

GT_SUBDIRS = ("alto", "page")
MANIFEST_COLUMNS = [
    "line_path",
    "page_id",
    "source_id",
    "line_id",
    "text",
    "width",
    "height",
    "n_chars",
]


def find_pages(raw_dir: Path) -> list[tuple[str, str, Path, Path]]:
    """(source_id, page_id, image_path, xml_path) for every paired page."""
    pages: list[tuple[str, str, Path, Path]] = []
    for book in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        inner = book / book.name
        if not inner.is_dir():
            inner = book
        images = {p.stem: p for p in inner.glob("*.jpg")}
        images.update({p.stem: p for p in inner.glob("*.JPG")})
        for sub in GT_SUBDIRS:
            if not (inner / sub).is_dir():
                continue
            for xml in sorted((inner / sub).glob("*.xml")):
                if xml.stem in images:
                    pages.append((book.name, f"{book.name}__{xml.stem}", images[xml.stem], xml))
    return pages


def _clip(box: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (max(0, x0), max(0, y0), min(w, x1), min(h, y1))


MIN_X_OVERLAP_FRAC = 0.3


def compute_crop_boxes(
    records: list[LineRecord], img_w: int, img_h: int
) -> list[tuple[int, int, int, int]]:
    """Vertical padding proportional to box height, clamped at the midpoint to
    vertically adjacent lines that share horizontal extent, so a crop can never
    swallow its neighbour. Still an axis-aligned box: no masking, no dewarping.
    """
    pad_x = CONFIG["crop_pad_x"]
    top_frac = CONFIG["crop_pad_top_frac"]
    bot_frac = CONFIG["crop_pad_bottom_frac"]
    clamp = CONFIG["crop_clamp_to_neighbours"]
    boxes = [r["bbox"] for r in records]

    out: list[tuple[int, int, int, int]] = []
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        h = y1 - y0
        top_limit, bottom_limit = 0, img_h
        if clamp:
            for j, (ox0, oy0, ox1, oy1) in enumerate(boxes):
                if i == j:
                    continue
                overlap = min(x1, ox1) - max(x0, ox0)
                if overlap <= MIN_X_OVERLAP_FRAC * min(x1 - x0, ox1 - ox0):
                    continue
                if oy1 <= y0:
                    top_limit = max(top_limit, (oy1 + y0) // 2)
                if oy0 >= y1:
                    bottom_limit = min(bottom_limit, (y1 + oy0) // 2)
        ny0 = max(0, top_limit, y0 - int(round(top_frac * h)))
        ny1 = min(img_h, bottom_limit, y1 + int(round(bot_frac * h)))
        out.append(_clip((x0 - pad_x, ny0, x1 + pad_x, ny1), img_w, img_h))
    return out


def classify(
    rec: LineRecord, box: tuple[int, int, int, int]
) -> tuple[str | None, tuple[int, int, int, int]]:
    """Return (drop_reason or None, crop box). Spec 6.3."""
    text = normalise(rec["text"], CONFIG["strip_zero_width_joiners"])
    if not text:
        return "empty_text", box

    cw, ch = box[2] - box[0], box[3] - box[1]
    if cw <= 0 or ch <= 0:
        return "bbox_outside_image", box
    if ch < MIN_HEIGHT or cw < MIN_WIDTH:
        return "degenerate_crop", box
    if cw / ch > MAX_ASPECT:
        return "aspect_ratio", box
    if len(text) > MAX_TEXT_LEN:
        return "text_too_long", box
    if not ctc_feasible(len(text), cw, ch):
        return "ctc_infeasible", box
    return None, box


def ctc_feasible(text_len: int, crop_w: int, crop_h: int) -> bool:
    """Will this crop yield enough CTC timesteps for its label? (spec 9.3)"""
    ratio = CONFIG["min_ctc_ratio"]
    if not ratio:
        return True
    w_resized = min(
        int(round(crop_w * CONFIG["img_height"] / crop_h)), CONFIG["img_max_width"]
    )
    timesteps = -(-w_resized // CONFIG["cnn_width_downsample"])  # ceil
    return timesteps >= text_len * ratio


def extract_lines(raw_dir: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, Counter]:
    """Crop every line to PNG and build the manifest (spec 6.2-6.4)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    drops: list[dict] = []
    stats: Counter = Counter()

    for source_id, page_id, img_path, xml_path in find_pages(raw_dir):
        fmt, records = parse_any(xml_path, page_id)
        stats[f"pages_{fmt}"] += 1
        stats["textlines_seen"] += len(records)
        with Image.open(img_path) as im:
            page = im.convert("L")  # grayscale on save (spec 6.2)
            img_w, img_h = page.size
            crop_boxes = compute_crop_boxes(records, img_w, img_h)
            for rec, box in zip(records, crop_boxes):
                reason, box = classify(rec, box)
                if (rec["bbox"][0] < 0 or rec["bbox"][1] < 0
                        or rec["bbox"][2] > img_w or rec["bbox"][3] > img_h):
                    stats["boxes_needing_clip"] += 1
                if reason is not None:
                    stats[f"drop_{reason}"] += 1
                    drops.append({
                        "page_id": page_id, "line_id": rec["line_id"], "reason": reason,
                        "bbox": str(rec["bbox"]), "text": rec["text"][:80],
                    })
                    continue
                text = normalise(rec["text"], CONFIG["strip_zero_width_joiners"])
                name = f"{page_id}__{rec['line_id']}.png"
                crop = page.crop(box)
                crop.save(out_dir / name, format="PNG", optimize=True)
                rows.append({
                    "line_path": name,
                    "page_id": page_id,
                    "source_id": source_id,
                    "line_id": rec["line_id"],
                    "text": text,
                    "width": crop.width,
                    "height": crop.height,
                    "n_chars": len(text),
                })

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    drop_log = pd.DataFrame(drops, columns=["page_id", "line_id", "reason", "bbox", "text"])
    return manifest, drop_log, stats


def main() -> None:
    from config import make_dirs, use_utf8_stdout

    use_utf8_stdout()
    make_dirs()
    dataset = CONFIG["dataset"]
    raw_dir = PATHS["raw_dir"]
    out_dir = PATHS["lines_dir"] / dataset

    print(f"raw_dir : {raw_dir}")
    print(f"out_dir : {out_dir}")
    print(f"strip_zero_width_joiners = {CONFIG['strip_zero_width_joiners']}  (logged decision, spec 7.1)")

    manifest, drop_log, stats = extract_lines(raw_dir, out_dir)

    seen = stats["textlines_seen"]
    kept = len(manifest)
    dropped = seen - kept
    print(f"\npages parsed        : alto={stats['pages_alto']} page={stats['pages_page']}")
    print(f"TextLine elements   : {seen}")
    print(f"lines extracted     : {kept}")
    print(f"lines dropped       : {dropped}  ({dropped / seen:.2%})")
    for reason in sorted(k for k in stats if k.startswith("drop_")):
        print(f"    {reason[5:]:22s} {stats[reason]}")
    print(f"boxes needing clip to image bounds: {stats['boxes_needing_clip']}")

    man_path = PATHS["manifests_dir"] / f"{dataset}_lines.csv"
    drop_path = PATHS["manifests_dir"] / f"{dataset}_drops.csv"
    manifest.to_csv(man_path, index=False, encoding="utf-8")
    drop_log.to_csv(drop_path, index=False, encoding="utf-8")
    print(f"\nmanifest -> {man_path}")
    print(f"drop log -> {drop_path}")

    if dropped / seen > 0.10:
        print("\n*** WARNING: more than 10% of lines dropped - spec 6.3 says investigate ***")


if __name__ == "__main__":
    main()


def save_sample_crops(
    manifest: pd.DataFrame,
    lines_dir: Path,
    out_dir: Path,
    n: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Copy n random crops out for the spec 15 visual check, plus a contact sheet."""
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("sample_*.png"):
        stale.unlink()

    rng = np.random.default_rng(seed)
    picks = manifest.iloc[np.sort(rng.choice(len(manifest), size=min(n, len(manifest)), replace=False))]

    crops = []
    for rank, (_, row) in enumerate(picks.iterrows()):
        src = lines_dir / row["line_path"]
        with Image.open(src) as im:
            crop = im.convert("L")
            crop.save(out_dir / f"sample_{rank:02d}_{row['line_path']}", format="PNG")
            crops.append(crop.copy())

    pad = 6
    width = max(c.width for c in crops) + 2 * pad
    height = sum(c.height + pad for c in crops) + pad
    sheet = Image.new("L", (width, height), color=255)
    y = pad
    for c in crops:
        sheet.paste(c, (pad, y))
        y += c.height + pad
    sheet.save(out_dir / "sample_contact_sheet.png", format="PNG")

    index = picks[["line_path", "page_id", "text", "width", "height", "n_chars"]].copy()
    index.to_csv(out_dir / "sample_lines.csv", index=False, encoding="utf-8")
    return index
