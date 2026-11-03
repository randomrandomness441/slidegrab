"""dedupe.py — collapse repeat showings of a slide, remembering when each one was shown.

A speaker revisits a slide while discussing it, and a quizmaster revisits a question.
Collapsing repeats makes the deck strictly linear — which is what lets quiz Q/A pairing
be structural instead of a guess — and stops a lecture from being read twice.

FULL-RESOLUTION MEAN ABSOLUTE DIFFERENCE, not a perceptual hash. Measured on one quiz
deck against transcript-derived labels:

    same slide       n=  17   mean MAE  0.348   worst   0.804
    different slide  n=2128   mean MAE 56.788   closest 6.238

A 7.8x gap with no overlap, so any threshold between them is exact. aHash/dHash/pHash
downsample to 8x8..32x32 to be invariant to re-encoding; the only thing separating two
question slides is TEXT, which that downsampling averages away (156 pairs of completely
different questions collapsed to identical hashes at every size tried).

PROVENANCE OF 3.5: read off that ONE deck (flat backgrounds, large text, hard cuts). A
lecture's noise floor differs, so sampled mode passes its own measured cutoff instead
(frames.settled_states). If dedup ever merges two different slides, this is the number
to suspect.

RE-MEASURED ON REAL FRAMES while porting (two already-extracted quiz videos, 71+117
distinct slides, 33+113 repeat showings; frames uncropped, half resolution). NOTE the
labels are circular — the old pipeline's own <3.5 rule produced the kept/repeat split —
so this shows how the scores behave, not that every merge was right:

    different slides, lowest global MAE      6.8   and   7.3     (n=2485, 6786 pairs)
    repeat vs its match, highest global MAE  2.8   and   6.3

The 7.3 vs 6.3 pair is a thin margin on the second video: a repeat whose webcam moved
scored 6.3 uncropped. Pass `compare_box` to exclude the webcam and the margin widens.

ALTERNATIVE TRIED AND REJECTED: the MAX over an 8x8 grid of tiles, meant to catch slides
that differ in a small region. Different slides separate far better (lowest 29 and 34)
but SAME-slide repeats blow up when anything else on screen moves (highest 33 and 193 —
a webcam), so without a crop box it splits one slide into many. Global mean tolerates a
webcam because the webcam is a small fraction of the frame. Reconsider only together with
a mandatory compare_box.

BLIND TO: SPARSE slides that differ in a small region. Global mean averages over the
whole frame; a synthetic 1280x720 pair with a one-line title and three short lines each,
text different, scored 3.1 — UNDER the threshold — and would be merged. Real quiz slides
here are dense, and none of the measured pairs were that sparse, so this is observed on
a constructed case, not a measured failure. If a text-light deck loses slides, check
deck.pdf and lower SAME_SLIDE_MAE for that deck.

KEEPS THE FIRST occurrence: decks run question, answer, question-again — keeping the
last would place a question after its own answer.

COMPARES AGAINST EVERY EARLIER KEEPER, not just the previous one. Comparing only to the
previous keeper (what the lecture extractor did) lets A-B-A produce three slides.

RETURNS EVERY SHOWING, not just the survivors: narration spoken while a slide was
revisited belongs to that slide (align.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .frames import Box, Frame, load_gray, mae

SAME_SLIDE_MAE = 3.5


def dedupe(frames: list[Frame], threshold: float = SAME_SLIDE_MAE, compare_box: Box | None = None,
           half: bool = False) -> tuple[list[Frame], list[tuple[Frame, int]]]:
    """-> (kept, showings) where showings[i] = (frame, index into kept) for EVERY input
    frame in order, kept ones included."""
    kept: list[Frame] = []
    seen: list[np.ndarray] = []
    showings: list[tuple[Frame, int]] = []
    for f in frames:
        img = load_gray(Path(f.path), compare_box, half)
        match = next((i for i, prev in enumerate(seen)
                      if prev.shape == img.shape and mae(prev, img) < threshold), None)
        if match is None:
            seen.append(img)
            kept.append(f)
            match = len(kept) - 1
        showings.append((f, match))
    return kept, showings
