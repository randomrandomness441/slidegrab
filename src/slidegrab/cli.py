"""slidegrab CLI.

    slidegrab URL --mode lecture --provider gemini
    slidegrab URL --mode quiz --provider zai
    slidegrab URL --no-read            # frames + PDF only; no key needed
    slidegrab FILE.mp4 --captions FILE.vtt
    slidegrab URL --calibrate          # print the noise distribution, no model calls
    slidegrab --playlist URL --from 3 --to 9
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import frames as fr
from .errors import ProviderError
from .pipeline import MODES, run, run_playlist
from .providers.base import KEY_VARS, get_provider


def _box(s: str) -> fr.Box:
    try:
        x0, y0, x1, y1 = (int(v) for v in s.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("--box wants x0,y0,x1,y1 in pixels, e.g. 0,0,1020,720")
    return (x0, y0, x1, y1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="slidegrab",
        description="Extract the distinct slides (text, pictures, narration) from a YouTube "
                    "lecture or quiz video, so you don't have to watch it.",
        epilog="Reading slides needs ONE provider key: " + ", ".join(
            f"{n} ({v})" for n, v in KEY_VARS.items()))
    ap.add_argument("source", nargs="?", help="YouTube URL or a local video file")
    ap.add_argument("--playlist", help="a playlist URL; each video is processed independently")
    ap.add_argument("--from", dest="lo", type=int, default=1)
    ap.add_argument("--to", dest="hi", type=int, default=9999)
    ap.add_argument("--mode", choices=sorted(MODES), default="lecture",
                    help="lecture: slides build up, narration matters. quiz: slides cut, "
                         "question/answer pairing")
    ap.add_argument("--provider", choices=sorted(KEY_VARS), help="default: first one whose key is set")
    ap.add_argument("--model", help="override the provider's default vision model")
    ap.add_argument("--out", default="slidegrab_out")
    ap.add_argument("--captions", type=Path, help="a .vtt file (local videos, or to override)")
    ap.add_argument("--box", type=_box, help="x0,y0,x1,y1 region to COMPARE frames in, to exclude "
                    "a webcam. Frames are always saved whole.")
    ap.add_argument("--every", type=float, default=fr.SAMPLE_EVERY, help="lecture mode: seconds between samples")
    ap.add_argument("--no-read", action="store_true", help="stop after frames + PDF (no key needed)")
    ap.add_argument("--calibrate", action="store_true", help="print the measured noise distribution and exit")
    a = ap.parse_args(argv)

    out = Path(a.out)
    try:
        if a.calibrate:
            if not a.source:
                ap.error("--calibrate needs a video")
            from . import fetch
            p = Path(a.source)
            video = p if p.exists() else fetch.download_video(a.source, out / "calibrate" / "video.mp4")
            print(json.dumps(fr.calibrate(video, out / "calibrate" / "frames", a.box), indent=2))
            return 0
        provider = None if a.no_read else get_provider(a.provider, a.model)
        kw = dict(mode=a.mode, provider=provider, compare_box=a.box, every=a.every, read=not a.no_read)
        if a.playlist:
            r = run_playlist(a.playlist, out, a.lo, a.hi, **kw)
            print(f"\n  {r['done']} videos processed, {len(r['failed'])} failed")
            return 1 if r["failed"] else 0
        if not a.source:
            ap.error("give a video URL/file or --playlist")
        s = run(a.source, out, captions=a.captions, **kw)
        print("\n" + json.dumps(s, indent=2))
        return 0
    except (ProviderError, RuntimeError) as e:
        print(f"error: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
