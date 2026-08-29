"""Spec section 5 - XML format inspection.

Confirms the schema before any parser is written. Detection only: this module
must not extract lines.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PATHS, use_utf8_stdout  # noqa: E402

GT_SUBDIRS = ("alto", "page")


def detect_format(xml_path: Path) -> str:
    """Return 'alto', 'page', or 'unknown' from the root tag (spec 5, 6.1)."""
    root = etree.parse(str(xml_path)).getroot()
    tag = etree.QName(root).localname.lower()
    ns = (root.tag[1:].split("}")[0] if root.tag.startswith("{") else "").lower()
    if "alto" in tag or "alto" in ns:
        return "alto"
    if "pcgts" in tag:
        return "page"
    return "unknown"


def find_images(directory: Path) -> list[Path]:
    """Unique page images. Resolved to a set: the filesystem is case-insensitive."""
    seen = {p.resolve() for p in directory.glob("*.jpg")}
    seen |= {p.resolve() for p in directory.glob("*.JPG")}
    return sorted(seen)


def find_ground_truth(directory: Path) -> list[Path]:
    """Ground-truth XML only - metadata.xml and mets.xml are not transcriptions."""
    out: list[Path] = []
    for sub in GT_SUBDIRS:
        if (directory / sub).is_dir():
            out.extend(sorted((directory / sub).glob("*.xml")))
    return out


def iter_books(raw_dir: Path):
    """Yield (book_name, inner_dir). Archives nest as <book>/<book>/."""
    for book in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        inner = book / book.name
        yield book.name, (inner if inner.is_dir() else book)


def directory_tree(raw_dir: Path) -> str:
    """Two levels deep, with unique jpg / xml counts."""
    out: list[str] = []
    tj = tx = 0
    for name, inner in iter_books(raw_dir):
        j = len(find_images(inner))
        x = len(list((raw_dir / name).rglob("*.xml")))
        tj += j
        tx += x
        out.append(f"  {name}/{inner.name}/  [{j} jpg, {x} xml]")
        subs = sorted(p.name for p in inner.iterdir() if p.is_dir())
        files = sorted(p.name for p in inner.iterdir() if p.is_file() and p.suffix == ".xml")
        if subs:
            out.append(f"      subdirs: {', '.join(subs)}")
        if files:
            out.append(f"      loose xml: {', '.join(files)}")
    out.insert(0, f"{raw_dir}/  [{tj} jpg, {tx} xml]")
    return "\n".join(out)


def format_census(raw_dir: Path) -> dict[str, list[str]]:
    """Map detected format -> book names, using the ground-truth XML only."""
    census: dict[str, list[str]] = defaultdict(list)
    for name, inner in iter_books(raw_dir):
        gt = find_ground_truth(inner)
        fmts = {detect_format(p) for p in gt}
        for f in sorted(fmts):
            census[f].append(name)
    return census


def dump_file(path: Path, n_lines: int, raw_dir: Path) -> None:
    print(f"\n--- {path.relative_to(raw_dir)} , first {n_lines} lines verbatim ---")
    with path.open(encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= n_lines:
                print(f"... [truncated at {n_lines} lines]")
                break
            print(f"{i + 1:3d}| {line.rstrip()}")


def dump_schema(path: Path, raw_dir: Path, n_textlines: int = 3) -> None:
    """Root tag, namespaces, detected format, TextLine count, sample elements."""
    root = etree.parse(str(path)).getroot()
    print(f"\n  root.tag      : {root.tag}")
    print(f"  local name    : {etree.QName(root).localname}")
    print(f"  namespace map : {root.nsmap}")
    fmt = detect_format(path)
    print(f"  detect_format -> {fmt!r}")
    if fmt == "unknown":
        tags = Counter(etree.QName(e).localname for e in root.iter())
        print("  UNKNOWN FORMAT - all distinct tags, then stopping (spec 5):")
        for t, n in tags.most_common():
            print(f"    {t:24s} {n}")
        return
    lines = root.findall(".//{*}TextLine")
    print(f"  TextLine count in this file: {len(lines)}")
    for i, el in enumerate(lines[:n_textlines]):
        print(f"\n  --- TextLine #{i + 1} in full " + "-" * 40)
        print(etree.tostring(el, pretty_print=True, encoding="unicode"))


def main() -> None:
    use_utf8_stdout()
    raw = PATHS["raw_dir"]

    print("=" * 78)
    print("1. DIRECTORY TREE (two levels deep)")
    print("=" * 78)
    print(directory_tree(raw))

    images = [p for _, inner in iter_books(raw) for p in find_images(inner)]
    gt = [p for _, inner in iter_books(raw) for p in find_ground_truth(inner)]
    all_xml = list(raw.rglob("*.xml"))
    print(f"\nunique page images      : {len(images)}")
    print(f"ground-truth XML files   : {len(gt)}")
    print(f"all XML files (incl. metadata.xml / mets.xml): {len(all_xml)}")

    print("\n" + "=" * 78)
    print("2. ROOT TAG CENSUS (every XML file)")
    print("=" * 78)
    census: Counter = Counter()
    for p in all_xml:
        census[etree.QName(etree.parse(str(p)).getroot()).localname] += 1
    for tag, n in census.most_common():
        print(f"  {tag:20s} {n}")

    print("\n" + "=" * 78)
    print("3. DETECTED FORMAT PER BOOK")
    print("=" * 78)
    for fmt, books in sorted(format_census(raw).items()):
        print(f"  {fmt:8s} {len(books):2d} books: {', '.join(books)}")

    print("\n" + "=" * 78)
    print("4. FIRST GROUND-TRUTH XML FILE")
    print("=" * 78)
    first = gt[0]
    dump_file(first, 80, raw)
    dump_schema(first, raw)

    others = [p for p in gt if detect_format(p) != detect_format(first)]
    if others:
        print("\n" + "=" * 78)
        print("5. A SECOND FORMAT IS ALSO PRESENT - first file of that format")
        print("=" * 78)
        dump_file(others[0], 40, raw)
        dump_schema(others[0], raw)

    print("\n" + "=" * 78)
    print("6. TOTALS ACROSS THE WHOLE CORPUS")
    print("=" * 78)
    per_fmt: Counter = Counter()
    for p in gt:
        per_fmt[detect_format(p)] += len(etree.parse(str(p)).getroot().findall(".//{*}TextLine"))
    for fmt, n in sorted(per_fmt.items()):
        print(f"  TextLine elements in {fmt:5s}: {n}")
    print(f"  TextLine elements TOTAL     : {sum(per_fmt.values())}")


if __name__ == "__main__":
    main()
