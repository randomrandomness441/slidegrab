"""pipeline.py — URL (or local file) in, a folder of slides out.

    fetch → frames (cuts | sampled) → dedupe → deck.pdf → read (provider) → align → [quiz]

Outputs, in <out>/<video id>/:
    slides.json    one record per DISTINCT slide: stamp, text, image, in_photo,
                   shown_at (every timestamp it was on screen), narration
    slides.md      the same, readable — skim this instead of watching the video
    deck.pdf       the slides as pages, timestamped
    qa_review.json (quiz mode) Q/A pairs; `problem` set where unresolved
    summary.json   counts, the measured noise stats, and which reader produced the text

MODES
    lecture  sampled frames, one slide per change, narration from captions
    quiz     scene cuts, then question/answer pairing

Resumable at every stage: the video, every frame image and every slide transcript are
cached on disk, and a finished video (summary.json) is skipped. Scene detection is the
one stage re-run on resume. Prints per stage, flushed.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from . import align, fetch, frames as fr, quiz
from .dedupe import SAME_SLIDE_MAE, dedupe
from .deck import write_pdf
from .errors import ProviderError
from .providers.base import Provider

MODES = {"lecture": "sampled", "quiz": "cuts"}


def _identify(source: str) -> tuple[str, str, bool]:
    """-> (id, title, is_local)."""
    p = Path(source)
    if p.exists():
        return p.stem, p.stem, True
    out = subprocess.run(["yt-dlp", "--no-playlist", "--no-update", "--print",
                          "%(id)s|%(title)s", source], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"yt-dlp could not resolve {source!r}: {out.stderr[-200:]}")
    vid, _, title = out.stdout.strip().partition("|")
    return vid, title, False


def run(source: str, out: Path, mode: str = "lecture", provider: Provider | None = None,
        captions: Path | None = None, compare_box: fr.Box | None = None,
        every: float = fr.SAMPLE_EVERY, read: bool = True, lang: str = "en") -> dict:
    """Process one video. `provider=None` (or read=False) stops after the frames and PDF —
    useful to inspect what was found before spending model calls."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}")
    vid, title, local = _identify(source)
    work = out / vid
    done = work / "summary.json"
    if done.exists() and (provider is not None or not read):
        return json.loads(done.read_text()) | {"skipped": True}
    started = time.time()
    work.mkdir(parents=True, exist_ok=True)

    print(f"  [{vid}] {title[:70]}", flush=True)
    video = Path(source) if local else fetch.download_video(source, work / "video.mp4")
    if captions is None and not local:
        captions = fetch.download_captions(source, work / "captions", lang)
    cues = fetch.parse_vtt(captions) if captions and Path(captions).exists() else []
    print(f"  captions: {len(cues)} lines" if cues else "  captions: none (narration will be empty)",
          flush=True)

    frames_dir = work / "frames"
    if MODES[mode] == "cuts":
        kept_frames, stats = fr.frames_from_cuts(video, frames_dir, compare_box)
        kept, showings = dedupe(kept_frames, SAME_SLIDE_MAE, compare_box)
    else:
        kept_frames, stats = fr.frames_from_samples(video, frames_dir, compare_box, every)
        # sampled mode dedups at THIS video's measured cutoff, at the same half-res scale
        kept, showings = dedupe(kept_frames, stats.get("cutoff", SAME_SLIDE_MAE), compare_box, half=True)
    for i, f in enumerate(kept, 1):
        f.number = i
    print(f"  frames: {stats.get('n_frames', stats.get('scenes'))} → {len(kept_frames)} candidates "
          f"→ {len(kept)} distinct slides", flush=True)
    if mode == "lecture" and "cutoff" in stats and stats["cutoff"] > stats["diff_p99"]:
        print(f"  WARNING: the change cutoff ({stats['cutoff']}) is above even the 99th-percentile "
              f"frame difference ({stats['diff_p99']}) — almost nothing counts as a slide change. "
              "Either this is mostly live footage (a talking head, a webcam, constant motion) and "
              "not slides, or it has very few slides. Check deck.pdf; --box can exclude a webcam.",
              flush=True)
    if mode == "lecture" and stats.get("frac_above_cutoff", 0) > 0.5:
        print(f"  WARNING: {stats['frac_above_cutoff']:.0%} of frame pairs exceed the noise cutoff — "
              "the median is probably not measuring noise (constant motion, a moving webcam). "
              "Pass --box to exclude it, and check the slides.", flush=True)
    write_pdf(kept, work / "deck.pdf")

    summary = {"video_id": vid, "title": title, "mode": mode, "frames": stats,
               "slides": len(kept), "read": False}
    if not (read and provider):
        (work / "summary.partial.json").write_text(json.dumps(summary, indent=2))
        print("  stopped before reading (no provider / --no-read). deck.pdf written.", flush=True)
        return summary

    from .read import read_all
    slides = read_all(kept, provider, work / "transcripts")
    duration = max((f.end for f in kept), default=0.0)
    records = align.build_records(slides, kept, showings, cues, duration)
    (work / "slides.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    (work / "slides.md").write_text(_markdown(title, records))

    summary.update({"read": True, "reader": f"{provider.name}/{provider.model}",
                    "slides_read": len(slides), "slides_unread": len(kept) - len(slides),
                    "truncated": sum(1 for s in slides if s.truncated)})
    if mode == "quiz":
        recs = [asdict(s) for s in slides]
        pairs = quiz.extract_pairs(recs, provider, vid)
        (work / "qa_review.json").write_text(json.dumps(pairs, indent=2, ensure_ascii=False))
        anchors = len(quiz.find_anchors(recs))
        summary.update({"anchors": anchors, "pairs": sum(1 for p in pairs if not p["problem"]),
                        "flagged": sum(1 for p in pairs if p["problem"])})
        if anchors == 0 and len(kept) > 10:
            print("  NO ANCHORS — either this is not a quiz, or its deck numbers questions "
                  "differently and quiz.py's rules need extending. Look before believing it.", flush=True)
    summary["minutes"] = round((time.time() - started) / 60, 1)
    done.write_text(json.dumps(summary, indent=2))
    return summary


def _markdown(title: str, records: list[dict]) -> str:
    lines = [f"# {title}", "", f"{len(records)} distinct slides", ""]
    for r in records:
        shown = ", ".join(r["shown_at"][1:]) if len(r["shown_at"]) > 1 else ""
        lines += [f"## {r['index']}. [{r['stamp']}]" + (f"  (revisited {shown})" if shown else ""), ""]
        if r["text"]:
            lines += ["```", r["text"], "```", ""]
        if r["image"]:
            lines += [f"*Picture:* {r['image']}", ""]
        if r["in_photo"]:
            lines += [f"*Text inside picture:* {r['in_photo']}", ""]
        if r["truncated"]:
            lines += ["*(transcript truncated)*", ""]
        if r["narration"]:
            lines += [f"> {r['narration']}", ""]
    return "\n".join(lines)


def run_playlist(url: str, out: Path, lo: int = 1, hi: int = 9999, **kw) -> dict:
    """Each video is independent: its own folder, its own failures. One that breaks does
    not stop the run, and a finished video is never redone."""
    items = [r for r in fetch.playlist(url) if lo <= r[0] <= hi]
    done, failed = [], []
    for i, vid, title, dur in items:
        print(f"\n  [{i}] {title[:58]}  ({dur // 60} min)", flush=True)
        try:
            done.append(run(f"https://www.youtube.com/watch?v={vid}", out, **kw))
        except (ProviderError, RuntimeError, OSError, subprocess.CalledProcessError) as e:
            failed.append({"index": i, "video_id": vid, "error": str(e)[:200]})
            print(f"      FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)
    return {"done": len(done), "failed": failed}
