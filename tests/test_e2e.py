"""End to end on a synthetic video: three distinct slides, one revisited, with narration.
Needs ffmpeg. Cuts mode also needs the scenedetect binary."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from slidegrab import pipeline
from helpers import make_slide

SCENEDETECT = os.environ.get("SLIDEGRAB_SCENEDETECT") or shutil.which("scenedetect")


@pytest.fixture
def video(tmp_path, have_ffmpeg):
    d = tmp_path / "src"; d.mkdir()
    # distinct BACKGROUNDS: cuts mode misses text appearing on a plain white panel (see
    # frames.py "BLIND TO"), which is what lecture/sampled mode is for.
    seq = [("ONE", ["alpha"] * 5, 255), ("TWO", ["beta"] * 7, 170), ("ONE", ["alpha"] * 5, 255),
           ("THREE", ["gamma"] * 9, 90)]
    lst = []
    for i, (t, ln, bg) in enumerate(seq):
        img = make_slide(d / f"{i}.png", f"SLIDE {t}", lines=ln, bg=bg)
        lst += [f"file '{img}'", "duration 6"]
    lst.append(f"file '{d / '3.png'}'")
    (d / "list.txt").write_text("\n".join(lst))
    out = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(d / "list.txt"), "-vf", "fps=10,format=yuv420p", str(out), "-y"], check=True)
    vtt = tmp_path / "v.vtt"
    vtt.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nfirst words\n\n"
                   "00:00:08.000 --> 00:00:09.000\nsecond words\n\n"
                   "00:00:14.000 --> 00:00:15.000\nback again\n")
    return out, vtt


def test_lecture_mode_finds_three_slides_and_narration(video, tmp_path, fake):
    v, vtt = video
    s = pipeline.run(str(v), tmp_path / "out", mode="lecture", provider=fake, captions=vtt, every=1.0)
    assert s["slides"] == 3 and s["slides_unread"] == 0
    import json
    recs = json.loads((tmp_path / "out" / "v" / "slides.json").read_text())
    assert [r["stamp"] for r in recs][0] == "00:00"
    assert "first words" in recs[0]["narration"] and "back again" in recs[0]["narration"]   # revisit
    assert "second words" in recs[1]["narration"]
    assert len(recs[0]["shown_at"]) == 2
    assert (tmp_path / "out" / "v" / "deck.pdf").stat().st_size > 0


def test_no_read_needs_no_provider(video, tmp_path):
    v, _ = video
    s = pipeline.run(str(v), tmp_path / "out", mode="lecture", provider=None, read=False, every=1.0)
    assert s["slides"] == 3 and s["read"] is False


@pytest.mark.skipif(not SCENEDETECT, reason="scenedetect binary not available")
def test_quiz_mode_cuts_finds_distinct_slides(video, tmp_path, fake):
    v, _ = video
    s = pipeline.run(str(v), tmp_path / "out", mode="quiz", provider=fake)
    assert s["slides"] == 3
