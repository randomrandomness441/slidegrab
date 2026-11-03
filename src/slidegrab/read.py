"""read.py — turn each slide image into a transcript, using whichever provider you set up.

Output per slide: the presenter's verbatim text, a one-line description of any
picture, text found INSIDE a photograph, and the question number if it carries one.

REQUIRES A PROVIDER KEY (see providers/base.py). This is the only stage that calls a
model.

WHY A VLM RATHER THAN OCR
OCR reads text well but marks pictures only as `![](bbox)`. That is fine for a question
slide, whose content is its text, and useless for a reveal that IS the picture ("Gandhi's
Three Monkeys" is sometimes just the sculpture) or a lecture diagram. A VLM gives text
and description in one pass.

WHY TEXT AND DESCRIPTION ARE KEPT APART
Different trustworthiness. Transcribed text is checkable character-by-character against
the image; a description is the model's account of a picture and is not. Downstream
prefers text and falls back to a description only when nothing else exists.

WHY IT DESCRIBES BUT NEVER INFERS
Asked "what is the answer here?" a VLM answers confidently even for a decorative photo.
That is the failure that cost the source project the most: an enrichment step fabricated
2 of 8 facts and an LLM asked to pair questions with answers invented 3 of 15 — twice a
CORRECT answer that appeared on no slide, which no output-only check can catch. The
prompt asks only for what is literally visible.

THE THREE-WAY SPLIT (SLIDE / IMAGE / INPHOTO)
An earlier version asked for "all text on the slide" and got the presenter's caption
glued to whatever was INSIDE a photograph — a tweet screenshot, a "(c)2015 Disney/Pixar"
watermark, the brand "Petzl" on climbing gear. Downstream that looked like a second
answer slide and made five quiz blocks look ambiguous.

PROMPT NOTE: the prompt is the source project's, plus one sentence extending "typed-in
text" to code and tables so lectures read correctly. That sentence has not been measured
against the quiz decks the original prompt was fitted to.

FAILURES ARE NEVER CACHED. An earlier version turned a failed call into "" and cached
it; eight slides became blank records indistinguishable from genuinely blank frames and
two of them were question anchors. A failure raises; nothing is written.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .errors import ProviderError, Truncated
from .providers.base import Provider

SYSTEM = (
    "You transcribe slides from a presentation. You report only what is "
    "literally on the slide. You never answer the question, never guess, and "
    "never add facts that are not printed or shown."
)

PROMPT = (
    "A slide has the presenter's OWN text — a title, caption, bullets, code or a table "
    "they typed in — and it may also contain PHOTOGRAPHS or DIAGRAMS, which often have "
    "text inside them: signs, screenshots, book pages, watermarks, brand names, axis labels.\n"
    "Output exactly three sections:\n"
    "SLIDE: the presenter's own typed text, verbatim, preserving line breaks, indentation "
    "(for code) and any leading number. Write none if there is none.\n"
    "IMAGE: one plain sentence describing what the photograph or diagram depicts — what is "
    "literally shown, not what it means. Write none if there is no picture.\n"
    "INPHOTO: any text appearing INSIDE the photograph or diagram, verbatim. Write none if "
    "there is none."
)

WORKERS = 2
# Escalating budgets: the first covers ordinary slides with headroom; the rest exist so
# a denser slide widens the budget instead of losing its tail.
TOKEN_BUDGETS = (1500, 3000, 6000)
_NONE = {"none", "no text", "no picture", "(none)", "n/a"}


@dataclass
class Slide:
    name: str
    text: str
    image: str
    number: int | None        # printed question number, when there is one (quiz decks)
    in_photo: str = ""        # text INSIDE a photograph — never the presenter's own
    truncated: bool = False   # transcript is PARTIAL — text ends mid-thought
    t: float = 0.0            # seconds into the video this slide first appeared
    reader: str = ""          # provider/model that produced it, for provenance

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())


def split_sections(raw: str) -> tuple[str, str, str]:
    """-> (slide text, image description, text inside the photo)."""
    out = {"SLIDE": "", "IMAGE": "", "INPHOTO": ""}
    parts = re.split(r"^(SLIDE|IMAGE|INPHOTO):\s*", raw.strip(), flags=re.M)
    for label, body in zip(parts[1::2], parts[2::2]):
        out[label] = body.strip()
    for k, v in out.items():
        if v.lower().strip(". ") in _NONE:
            out[k] = ""
    return out["SLIDE"], out["IMAGE"], out["INPHOTO"]


def question_number(text: str) -> int | None:
    """The printed question number, if the slide opens with one. Anchored to the FIRST
    line: a bare numeral at the top is the slide's own number, whereas '1912' inside a
    sentence is a fact in the question."""
    first = text.strip().splitlines()[0] if text.strip() else ""
    m = re.match(r"^\s*#*\s*(\d{1,2})\s*[.)]?\s*$", first)
    return int(m.group(1)) if m else None


def read_slide(path: Path, cache: Path, provider: Provider, t: float = 0.0) -> Slide:
    """Transcribe one slide. Cached on disk — re-runs cost nothing. A truncated
    transcript is retried at a larger budget; if it still truncates at the ceiling the
    record is marked `truncated` so downstream sees the text is partial."""
    hit = cache / f"{path.stem}.json"
    if hit.exists():
        return Slide(**json.loads(hit.read_text()))

    raw, truncated = "", False
    for budget in TOKEN_BUDGETS:
        try:
            raw = provider.vision(SYSTEM, PROMPT, path, max_tokens=budget)
            truncated = False
            break
        except Truncated as tr:
            raw, truncated = tr.text, True
            print(f"    {path.stem}: truncated at {budget} tokens, retrying larger", flush=True)
    else:
        print(f"    {path.stem}: STILL truncated at {TOKEN_BUDGETS[-1]} tokens", flush=True)

    text, image, in_photo = split_sections(raw)
    slide = Slide(name=path.stem, text=text, image=image, number=question_number(text),
                  in_photo=in_photo, truncated=truncated, t=t,
                  reader=f"{provider.name}/{provider.model}")
    hit.parent.mkdir(parents=True, exist_ok=True)
    hit.write_text(json.dumps(asdict(slide), indent=2, ensure_ascii=False))
    return slide


def read_all(frames: list, provider: Provider, cache: Path, workers: int = WORKERS) -> list[Slide]:
    """Read every frame (objects with .path and .start). Failures are loud, uncached,
    and absent from the output — never blank records. Order follows `frames`."""
    cache.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    done = 0

    def one(f) -> Slide | None:
        nonlocal done
        try:
            s = read_slide(Path(f.path), cache, provider, t=f.start)
        except ProviderError as e:
            failed.append(Path(f.path).stem)
            print(f"    {Path(f.path).stem}: FAILED — {e}", flush=True)
            s = None
        done += 1
        if done % 10 == 0 or done == len(frames):
            print(f"    read {done}/{len(frames)}", flush=True)
        return s

    with ThreadPoolExecutor(max_workers=workers) as pool:
        out = [s for s in pool.map(one, frames) if s is not None]
    if failed:
        print(f"\n  {len(failed)} slide(s) could not be read: {', '.join(failed)}\n"
              "  Re-run to retry them. They are absent from the output, not blank.")
    return out
