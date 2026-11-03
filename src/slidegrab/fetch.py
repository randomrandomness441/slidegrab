"""fetch.py — a YouTube URL in, a video file and its captions out. Cached.

Ported from FundaQuiz `video_frames.download` / `video_batch.playlist` and
`lecture.extract.parse_vtt`.

VIDEO-ONLY, 720p CAP. Slide text needs more than 360p; audio is not used (narration
comes from captions), so the audio track is never fetched. The format selector falls
back through progressively looser choices rather than naming one itag — the original
hard-coded format "136", which not every video offers.

CAPTIONS ARE BEST-EFFORT. YouTube auto-captions mangle technical terms ("co_await",
"std::pmr") and some videos have none. Missing captions yield an empty narration,
never a failure: the slides are the product, the narration is context."""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

VIDEO_FORMAT = "bv*[height<=720][ext=mp4]/bv*[height<=720]/b[height<=720]/b"


def _run(cmd: list) -> subprocess.CompletedProcess:
    try:
        return subprocess.run([str(c) for c in cmd], check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"{cmd[0]} not found on PATH — install it (see README → Requirements)") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"{cmd[0]} failed: {(e.stderr or '')[-300:]}") from e


def download_video(url: str, dest: Path) -> Path:
    """Video-only, <=720p. Skips if already present."""
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        _run(["yt-dlp", "--no-playlist", "--no-update", "-f", VIDEO_FORMAT,
              "--merge-output-format", "mp4", "-o", dest, url])
    return dest


def download_captions(url: str, dest_dir: Path, lang: str = "en") -> Path | None:
    """-> the .vtt path, or None when the video has no captions in `lang`."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    found = sorted(dest_dir.glob("*.vtt"))
    if found:
        return found[0]
    try:
        _run(["yt-dlp", "--no-playlist", "--no-update", "--skip-download",
              "--write-subs", "--write-auto-subs", "--sub-langs", f"{lang}.*,{lang}",
              "--sub-format", "vtt", "-o", dest_dir / "captions", url])
    except RuntimeError:
        return None
    found = sorted(dest_dir.glob("*.vtt"))
    return found[0] if found else None


def video_title(url: str) -> str:
    try:
        return _run(["yt-dlp", "--no-playlist", "--no-update", "--print", "%(title)s", url]).stdout.strip()
    except RuntimeError:
        return ""


def playlist(url: str, tries: int = 3) -> list[tuple[int, str, str, int]]:
    """-> [(index, video_id, title, duration_seconds)]. Retried: YouTube refuses
    intermittently under load and one refusal must not abandon a run of hours."""
    out = ""
    for attempt in range(tries):
        try:
            out = _run(["yt-dlp", "--no-update", "--flat-playlist", "--print",
                        "%(playlist_index)s|%(id)s|%(duration)s|%(title)s", url]).stdout
            break
        except RuntimeError:
            if attempt == tries - 1:
                raise
            time.sleep(20 * (attempt + 1))
    rows = []
    for line in out.strip().splitlines():
        parts = line.split("|", 3)
        if len(parts) == 4 and parts[0].isdigit():
            rows.append((int(parts[0]), parts[1], parts[3],
                         int(float(parts[2])) if parts[2] not in ("NA", "") else 0))
    return rows


_TS = re.compile(r"(\d\d):(\d\d):(\d\d)\.(\d\d\d) --> ")
_TAG = re.compile(r"<[^>]+>")


def parse_vtt(path: Path) -> list[tuple[float, str]]:
    """VTT -> [(start_seconds, line)]. Auto-captions repeat each line across cues as
    the caption scrolls, so a line is kept only the first time it is seen."""
    out: list[tuple[float, str]] = []
    seen: set[str] = set()
    start = 0.0
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _TS.match(raw)
        if m:
            h, mi, s, ms = (int(g) for g in m.groups())
            start = h * 3600 + mi * 60 + s + ms / 1000
            continue
        line = _TAG.sub("", raw).strip()
        if not line or line.startswith(("WEBVTT", "Kind:", "Language:")):
            continue
        if line in seen:
            continue
        seen.add(line)
        out.append((start, line))
    return out
