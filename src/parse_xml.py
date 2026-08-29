"""Spec section 6.1 - ALTO and PAGE parsers producing LineRecords.

This module and extract.py are the only dataset-specific layer. Nothing
downstream of the manifest may reference XML or page geometry (spec 18).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, TypedDict

from lxml import etree


class LineRecord(TypedDict):
    page_id: str
    line_id: str
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 - axis-aligned
    polygon: Optional[list[tuple[int, int]]]
    text: str


def detect_format(xml_path: Path | str) -> str:
    """Return 'alto', 'page' or 'unknown' from the root tag (spec 5, 6.1)."""
    root = etree.parse(str(xml_path)).getroot()
    tag = etree.QName(root).localname.lower()
    ns = (root.tag[1:].split("}")[0] if root.tag.startswith("{") else "").lower()
    if "alto" in tag or "alto" in ns:
        return "alto"
    if "pcgts" in tag:
        return "page"
    return "unknown"


def _parse_points(raw: str) -> list[tuple[int, int]]:
    """'x1,y1 x2,y2 ...' -> [(x1, y1), ...]"""
    pts: list[tuple[int, int]] = []
    for token in raw.split():
        x, _, y = token.partition(",")
        pts.append((int(round(float(x))), int(round(float(y)))))
    return pts


def _bbox_of(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _alto_line_text(line: etree._Element) -> str:
    """CONTENT of child String elements joined with spaces, honouring SP."""
    parts: list[str] = []
    for child in line:
        local = etree.QName(child).localname
        if local == "String":
            parts.append(child.get("CONTENT") or "")
        elif local == "SP":
            parts.append(" ")
    text = ""
    for i, part in enumerate(parts):
        if i and part != " " and not text.endswith(" "):
            text += " "
        text += part
    return text


def parse_alto(xml_path: Path | str, page_id: str) -> list[LineRecord]:
    """ALTO v4: geometry in HPOS/VPOS/WIDTH/HEIGHT, text in String/@CONTENT."""
    root = etree.parse(str(xml_path)).getroot()
    records: list[LineRecord] = []
    for i, line in enumerate(root.findall(".//{*}TextLine")):
        line_id = line.get("ID") or f"line{i:04d}"
        try:
            x0 = int(round(float(line.get("HPOS"))))
            y0 = int(round(float(line.get("VPOS"))))
            w = int(round(float(line.get("WIDTH"))))
            h = int(round(float(line.get("HEIGHT"))))
        except (TypeError, ValueError):
            continue  # no usable geometry; counted as a drop by the caller
        shape = line.find("./{*}Shape/{*}Polygon")
        polygon = _parse_points(shape.get("POINTS")) if shape is not None else None
        records.append(
            LineRecord(
                page_id=page_id,
                line_id=line_id,
                bbox=(x0, y0, x0 + w, y0 + h),
                polygon=polygon,
                text=_alto_line_text(line),
            )
        )
    return records


def parse_page(xml_path: Path | str, page_id: str) -> list[LineRecord]:
    """PAGE 2013-07-15: polygon in Coords/@points, text in TextEquiv/Unicode."""
    root = etree.parse(str(xml_path)).getroot()
    records: list[LineRecord] = []
    for i, line in enumerate(root.findall(".//{*}TextLine")):
        line_id = line.get("id") or line.get("ID") or f"line{i:04d}"
        coords = line.find("./{*}Coords")
        if coords is None or not coords.get("points"):
            continue
        polygon = _parse_points(coords.get("points"))
        # Direct child only: never pick up a nested Word-level TextEquiv.
        unicode_el = line.find("./{*}TextEquiv/{*}Unicode")
        text = unicode_el.text if unicode_el is not None and unicode_el.text else ""
        records.append(
            LineRecord(
                page_id=page_id,
                line_id=line_id,
                bbox=_bbox_of(polygon),  # axis-aligned box, no masking (spec 6.2)
                polygon=polygon,
                text=text,
            )
        )
    return records


def parse_any(xml_path: Path | str, page_id: str) -> tuple[str, list[LineRecord]]:
    """Dispatch on the detected format. Raises on an unknown schema."""
    fmt = detect_format(xml_path)
    if fmt == "alto":
        return fmt, parse_alto(xml_path, page_id)
    if fmt == "page":
        return fmt, parse_page(xml_path, page_id)
    raise ValueError(f"Unknown XML schema for {xml_path}")
