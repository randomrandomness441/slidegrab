"""frames.py — a video file in, an ordered list of candidate slide frames out.

Two strategies, because slides arrive in two different ways and a detector built for
one is blind to the other.

  "cuts"     — PySceneDetect AdaptiveDetector. For QUIZ DECKS, where one slide
               REPLACES another and the frame difference spikes.
  "sampled"  — a frame every N seconds, keep the last of each quiet run. For
               LECTURES and screencasts, where a slide BUILDS UP in place (bullets
               appear, code is typed) and there is no cut to detect.

WHY NOT ONE METHOD
ffmpeg `select=gt(scene,0.25)` and a perceptual hash were both tried on quiz decks
and failed SILENTLY: a slide that fills an empty white panel produces a tiny frame
delta, so the filter never fired and the deck's first question was lost — its ANSWER
slide was captured instead, output that still looks plausible. A 12x12 hash could not
tell two different question slides apart (9/144 bits differing against a tolerance of
6) because downsampling destroys the only thing separating them: the text. So cuts
mode compares full-resolution frames against a ROLLING average rather than a fixed
threshold. Sampled mode exists because that same detector sees ONE long scene for a
whole typed-in code build-up.

THRESHOLDS
Cuts mode's burst rule and the 3.5 same-slide cutoff were measured on one quiz deck
(see dedupe.py). Sampled mode derives its cutoff from THE VIDEO ITSELF: the median
consecutive-frame difference IS the still-frame noise (most sample pairs show nothing
happening — the speaker is talking over an unchanged slide), and a real change must
sit NOISE_MARGIN times above it. `calibrate()` prints the distribution so the rule
can be checked, not trusted.

BLIND TO (sampled): footage where the screen changes on most samples (continuous
animation, a fast scroll throughout) — there the median is not noise and the floor is
over-estimated, dropping real slides. A webcam inset that moves constantly does the
same unless `compare_box` excludes it. Two signatures: frac_above_cutoff near 1.0 (floor
too LOW, everything is a change) or cutoff above diff_p99 (floor too HIGH, nothing is) —
the second was seen on a video interview, where the 'slides' are live video and a median
frame difference of 7 is simply the speaker moving. Cropping out the webcam did not help.
BLIND TO (cuts): text appearing on, or replacing text on, a plain panel — the HSV delta
stays under MIN_CONTENT_VAL, so black-on-white slides that differ only in their words
produce ONE scene (seen on a synthetic 4-slide white deck with default thresholds too).
Use "sampled" for such decks. Also: a cut hidden inside high motion; and a slide shown for less than
SAMPLE_EVERY falls between samples in sampled mode. Neither method dominates.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ADAPTIVE_THRESHOLD = 1        # cut when a frame scores 1x the local rolling average
MIN_CONTENT_VAL = 10          # floor so sensor noise in a static shot can't trigger
MIN_SCENE_SECONDS = 1.5       # shorter than this is "brief" — see select_slides
BURST_RUN = 3                 # brief scenes in a run this long are playback, not slides
SETTLE_BACK = 0.25            # grab this far BEFORE a scene ends — see grab_frame
SAMPLE_EVERY = 3.0            # sampled mode: seconds between samples
NOISE_MARGIN = 6.0            # a change must sit this far above the measured floor
MIN_CUTOFF = 0.5              # UNMEASURED guard for a zero-noise median — see settled_states

Box = tuple[int, int, int, int]       # (x0, y0, x1, y1) in pixels of the full frame


@dataclass
class Frame:
    """One candidate slide: a time span on the video and the image grabbed from it."""
    number: int
    start: float
    end: float
    path: Path | None = None

    @property
    def length(self) -> float:
        return self.end - self.start

    @property
    def stamp(self) -> str:
        s = int(self.start)
        return f"{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def _run(cmd: list) -> None:
    try:
        subprocess.run([str(c) for c in cmd], check=True, capture_output=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"{cmd[0]} not found — see README → Requirements") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"{cmd[0]} failed: {(e.stderr or b'')[-300:].decode('utf-8', 'replace')}") from e


# ── shared: comparable images ────────────────────────────────────────────────

def load_gray(path: Path, box: Box | None = None, half: bool = False) -> np.ndarray:
    """Greyscale pixels of the part of a frame worth COMPARING (uint8).

    CROP IS FOR COMPARISON ONLY, NEVER FOR WHAT WE KEEP. Cropping the SAVED frame cost
    real content: at 1020 wide on a 1280 video, slide text was clipped mid-word ("the
    son of the director g|") and 42% of extracted questions carried damage that was
    blamed on the transcriber. A webcam strip only needs excluding when frames are
    compared, so a speaker shifting in their seat does not read as a slide change."""
    img = Image.open(path).convert("L")
    if box:
        x0, y0, x1, y1 = box
        img = img.crop((x0, y0, min(x1, img.width), min(y1, img.height)))
    if half:
        img = img.resize((max(1, img.width // 2), max(1, img.height // 2)))
    return np.asarray(img, dtype=np.uint8)


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


# ── cuts mode ────────────────────────────────────────────────────────────────

def _scenedetect_bin() -> str:
    found = os.environ.get("SLIDEGRAB_SCENEDETECT") or shutil.which("scenedetect")
    if not found:
        raise RuntimeError(
            "cuts mode needs the `scenedetect` binary: pip install 'slidegrab[cuts]' "
            "or set SLIDEGRAB_SCENEDETECT=/path/to/scenedetect")
    return found


def detect_scenes(video: Path, compare_box: Box | None = None) -> list[Frame]:
    """Every point where the screen changes. Full resolution — downscaling is what
    blinded the first attempt to text."""
    cmd = [_scenedetect_bin(), "--input", video]
    if compare_box:
        cmd += ["--crop", *compare_box]
    with tempfile.TemporaryDirectory() as tmp:
        _run(cmd + ["-d", 1, "detect-adaptive",
                    "--threshold", ADAPTIVE_THRESHOLD, "--min-content-val", MIN_CONTENT_VAL,
                    "list-scenes", "--filename", "scenes", "--output", tmp, "--skip-cuts"])
        csv_path = next(Path(tmp).rglob("scenes.csv"))
        rows = list(csv.reader(csv_path.open()))
    return [Frame(int(r[0]), float(r[3]), float(r[6]))
            for r in rows[1:] if r and r[0].isdigit()]


def select_slides(scenes: list[Frame], min_seconds: float = MIN_SCENE_SECONDS,
                  burst: int = BURST_RUN) -> list[Frame]:
    """Scenes that are slides rather than frames of a playing video.

    DURATION ALONE IS THE WRONG TEST. A 1.5s floor discarded the answer slide "A
    grenade (with its pin removed)", on screen 0.96s — presenters flash reveals. What
    separates them is BURSTINESS: a playing clip makes the detector cut continuously,
    so its brief scenes arrive in a long run; a flashed slide is one brief scene
    between two long ones. Measured on one deck: runs of 36, 7 and 4 were playback; 13
    isolated brief scenes were real slides, every one.

    A run of `burst`+ brief scenes is dropped EXCEPT its last scene: a clip can carry a
    static caption and the caption can be the answer ("Keanu Reeves shooting for the
    Bullet-Time sequence" sat over 36 brief scenes). If the clip had no caption this
    costs one junk frame that dedup or the reader discards; a missed slide cannot be
    recovered at all."""
    brief = [s.length < min_seconds for s in scenes]
    keep, i = [], 0
    while i < len(scenes):
        if not brief[i]:
            keep.append(scenes[i]); i += 1; continue
        run_end = i
        while run_end < len(scenes) and brief[run_end]:
            run_end += 1
        if run_end - i < burst:
            keep.extend(scenes[i:run_end])
        else:
            keep.append(scenes[run_end - 1])
        i = run_end
    return keep


def grab_frame(video: Path, scene: Frame, dest: Path) -> Path:
    """One FULL frame from near the END of the scene: a slide that builds up in place
    is least complete when it appears and fully rendered just before it is replaced."""
    at = max(scene.start, scene.end - SETTLE_BACK)
    if not dest.exists():
        _run(["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{at:.2f}",
              "-i", video, "-frames:v", 1, "-q:v", 2, dest, "-y"])
    return dest


def _cached_scenes(video: Path, cache: Path, compare_box: Box | None) -> list[Frame]:
    """Scene detection is the slow stage (~8 min for a 72-minute video), so a resume must
    not repeat it. Keyed on the video's size+mtime and the compare box: a different box
    changes what counts as a cut, so it must not reuse another box's scenes."""
    import json
    st = video.stat()
    key = {"size": st.st_size, "mtime": int(st.st_mtime), "box": list(compare_box) if compare_box else None}
    if cache.exists():
        got = json.loads(cache.read_text())
        if got.get("key") == key:
            return [Frame(n, a, b) for n, a, b in got["scenes"]]
    scenes = detect_scenes(video, compare_box)
    cache.write_text(json.dumps({"key": key, "scenes": [[s.number, s.start, s.end] for s in scenes]}))
    return scenes


def frames_from_cuts(video: Path, outdir: Path, compare_box: Box | None = None,
                     min_seconds: float = MIN_SCENE_SECONDS) -> tuple[list[Frame], dict]:
    outdir.mkdir(parents=True, exist_ok=True)
    scenes = _cached_scenes(video, outdir / "scenes.json", compare_box)
    kept = select_slides(scenes, min_seconds)
    for s in kept:
        s.path = grab_frame(video, s, outdir / f"s{s.number:04d}.jpg")
    return kept, {"mode": "cuts", "scenes": len(scenes), "kept": len(kept)}


# ── sampled mode ─────────────────────────────────────────────────────────────

def sample_frames(video: Path, outdir: Path, every: float = SAMPLE_EVERY) -> list[Frame]:
    """One WHOLE frame every `every` seconds. Resumable: skips what exists."""
    outdir.mkdir(parents=True, exist_ok=True)
    existing = sorted(outdir.glob("f_*.jpg"))
    if not existing:
        _run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", video,
              "-vf", f"fps=1/{every}", "-q:v", 2, outdir / "f_%05d.jpg", "-y"])
        existing = sorted(outdir.glob("f_*.jpg"))
    # ffmpeg numbers from 1; file N covers the sample at (N-1)*every
    return [Frame(i, i * every, (i + 1) * every, p) for i, p in enumerate(existing)]


def noise_floor(diffs: list[float]) -> float:
    return float(np.median(diffs)) if diffs else 0.0


def settled_states(frames: list[Frame], compare_box: Box | None = None) -> tuple[list[Frame], dict]:
    """Split the samples into SEGMENTS wherever consecutive frames differ by more than
    the cutoff, and keep each segment's LAST frame — stamped with the segment's START.

    WHY THE LAST FRAME: a slide that builds up in place (bullets appear, code is typed)
    is least complete when it appears and fully rendered just before it is replaced; a
    frame grabbed mid-typing shows half a statement. WHY THE SEGMENT'S START: the
    narration about this slide began when it first appeared, not when it finished
    building, and align.py windows on that time.

    The lecture extractor this was ported from documented "last frame of a quiet run"
    but its loop kept the FIRST quiet frame and skipped any frame whose successor
    differed — so a build-up whose final bullet was a small change could be represented
    by an earlier partial state. Segmenting on the change and taking the segment's end
    makes the code do what that docstring said.

    COST: build-ups whose steps each exceed the cutoff appear as successive states
    (title, then title+bullet 1, then +bullet 2...). Each is a true state of the screen;
    collapsing them would need a subset test pixels cannot give, so they are reported
    rather than guessed at.

    MIN_CUTOFF is an UNMEASURED guard, not a calibrated value: a losslessly static video
    has median diff 0, which would make every compression flicker a new slide."""
    if len(frames) < 2:
        return frames, {"n_frames": len(frames), "n_kept": len(frames),
                        "note": "too few frames to measure"}
    grays = [load_gray(f.path, compare_box, half=True) for f in frames]
    diffs = [mae(grays[i - 1], grays[i]) for i in range(1, len(grays))]
    floor = noise_floor(diffs)
    cutoff = max(floor * NOISE_MARGIN, MIN_CUTOFF)
    stats = {
        "n_frames": len(frames),
        "noise_floor_median": round(floor, 3), "cutoff": round(cutoff, 3),
        "diff_p90": round(float(np.percentile(diffs, 90)), 3),
        "diff_p99": round(float(np.percentile(diffs, 99)), 3),
        "diff_max": round(max(diffs), 3),
        "frac_above_cutoff": round(sum(d > cutoff for d in diffs) / len(diffs), 3),
    }
    kept: list[Frame] = []
    seg_start = 0
    for i in range(1, len(frames) + 1):
        if i == len(frames) or diffs[i - 1] > cutoff:       # segment [seg_start, i-1] ends
            first, last = frames[seg_start], frames[i - 1]
            kept.append(Frame(len(kept), first.start, last.end, last.path))
            seg_start = i
    stats["n_kept"] = len(kept)
    return kept, stats


def frames_from_samples(video: Path, outdir: Path, compare_box: Box | None = None,
                        every: float = SAMPLE_EVERY) -> tuple[list[Frame], dict]:
    frames = sample_frames(video, outdir, every)
    kept, stats = settled_states(frames, compare_box)
    stats["mode"] = "sampled"
    return kept, stats


def calibrate(video: Path, outdir: Path, compare_box: Box | None = None) -> dict:
    """The measured diff distribution, with no model calls. If frac_above_cutoff is
    near 1.0 the median is not measuring noise and the keeper rule does not apply."""
    _, stats = frames_from_samples(video, outdir, compare_box)
    return stats
