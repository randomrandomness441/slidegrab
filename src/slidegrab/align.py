"""align.py — attach to each slide the narration spoken while it was on screen.

A slide's narration is the captions from every moment it was showing: from each
showing's start to the next showing's start (any slide). Revisits count — a speaker who
flips back to slide 4 and explains it again is still explaining slide 4.

WHAT THIS ASSUMES: captions are timed against the same clock as the frames, and a
slide's narration begins when it appears. Speakers sometimes talk about the NEXT slide
before changing to it, and auto-captions lag speech by a second or two; both shift words
across a boundary. This is a time window, not understanding — it is context for a
reader, not a ground truth about which words belong to which slide.

NO CAPTIONS -> narration is "" for every slide. The slides are the product."""

from __future__ import annotations

from dataclasses import asdict

from .frames import Frame
from .read import Slide


def stamp(t: float) -> str:
    s = int(t)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def narration_windows(showings: list[tuple[Frame, int]], duration: float) -> dict[int, list[tuple[float, float]]]:
    """kept-slide index -> [(t0, t1)] over all its showings."""
    ordered = sorted(showings, key=lambda s: s[0].start)
    out: dict[int, list[tuple[float, float]]] = {}
    for i, (fr, k) in enumerate(ordered):
        end = ordered[i + 1][0].start if i + 1 < len(ordered) else max(duration, fr.end)
        out.setdefault(k, []).append((fr.start, end))
    return out


def narration_for(cues: list[tuple[float, str]], windows: list[tuple[float, float]]) -> str:
    return " ".join(txt for ts, txt in cues if any(a <= ts < b for a, b in windows))


def build_records(slides: list[Slide], kept: list[Frame], showings: list[tuple[Frame, int]],
                  cues: list[tuple[float, str]], duration: float) -> list[dict]:
    """One record per distinct slide, in order of first appearance.

    `slides` may be shorter than `kept` (a failed read is absent, never blank), so they
    are joined on the frame's file stem, not on position."""
    by_name = {s.name: s for s in slides}
    windows = narration_windows(showings, duration)
    records = []
    for k, fr in enumerate(kept):
        sl = by_name.get(fr.path.stem)
        if sl is None:
            continue
        shown = windows.get(k, [])
        rec = asdict(sl)
        rec.update({
            "index": len(records) + 1,
            "stamp": stamp(fr.start),
            "shown_at": [stamp(a) for a, _ in shown],
            "narration": narration_for(cues, shown),
        })
        records.append(rec)
    return records
