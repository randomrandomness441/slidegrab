"""deck.py — a reviewable PDF of the distinct slides, each footed with its timestamp so
a page you disagree with is one click from the moment in the video."""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz

from .frames import Frame


def write_pdf(frames: list[Frame], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    try:
        for f in frames:
            with fitz.open(f.path) as image:
                rect = image[0].rect
                page = doc.new_page(width=rect.width, height=rect.height + 16)
                page.insert_image(rect, filename=str(f.path))
                page.insert_text((6, rect.height + 12), f"{f.stamp}   slide {f.number}",
                                 fontsize=8, color=(0.45, 0.45, 0.45))
        doc.save(dest)
    finally:
        doc.close()
    return dest
