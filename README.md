# slidegrab

Turn a YouTube **lecture** or **quiz** video into its distinct slides — the text on each,
a description of any picture, and the narration spoken while it was on screen — so you
can skim a 90-minute talk instead of watching it. For quiz videos it can also pair each
question with its answer slide.

```
YouTube URL ──► video + captions ──► frames ──► distinct slides ──► read (model) ──► slides.md
                                       │                                └──► narration (captions)
                                       ├─ lecture: sample every N s, keep each change's fullest frame
                                       └─ quiz:    scene-cut detection ──► question/answer pairs
```

Extracted from [FundaQuiz](https://github.com/randomrandomness441/kapitsa)'s video pipeline.

## You need an API key

**Reading the text off a slide is done by a vision model, so you must supply a key for
one provider.** Without a key slidegrab still downloads, finds the slides and writes a
`deck.pdf` (`--no-read`), but it cannot tell you what the slides say.

| Provider | Env var | Default model | Status |
|---|---|---|---|
| `zai` (GLM) | `ZAI_API_KEY` | `glm-4.5v` (vision), `glm-5.3` (quiz pairing) | **verified** against the live API |
| `claude` (Anthropic) | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` | request format follows the Messages API; the **account used to build this had no credit**, so a successful read is not yet verified. The out-of-credit error path was verified. |
| `gemini` | `GEMINI_API_KEY` | `gemini-3.1-flash-lite` | request format follows the REST docs; **not verified live** — the keys available when this was built were rejected by Google (`API_KEY_INVALID`) |

Put the key in your environment or a `.env` in the working directory (`.env.example`).
`--provider` picks one; otherwise the first provider whose key is set is used. A provider
is a small class (`vision()` and `text()` — see `providers/base.py`), so adding another is
one file. Run `SLIDEGRAB_LIVE=1 pytest tests/test_live.py` to verify the providers you
configured.

Keys are only read from the environment, never logged, never written to output.

## Install

```bash
pip install "slidegrab @ git+https://github.com/randomrandomness441/slidegrab@v0.1.0"
pip install "slidegrab[cuts] @ git+https://github.com/randomrandomness441/slidegrab@v0.1.0"   # + quiz scene detection
```
(the repo is private: this needs `gh auth login` or a token that can read it.)

**Requirements:** Python ≥ 3.10, [`ffmpeg`](https://ffmpeg.org) and [`yt-dlp`](https://github.com/yt-dlp/yt-dlp)
on `PATH`. `--mode quiz` additionally needs the `scenedetect` binary (the `[cuts]` extra,
or `SLIDEGRAB_SCENEDETECT=/path/to/scenedetect`).

## Use

```bash
slidegrab "https://www.youtube.com/watch?v=..." --mode lecture --provider zai
slidegrab "https://www.youtube.com/watch?v=..." --mode quiz
slidegrab talk.mp4 --captions talk.en.vtt        # a local file
slidegrab URL --no-read                          # frames + PDF only, no key needed
slidegrab URL --calibrate                        # print the measured noise distribution
slidegrab --playlist URL --from 3 --to 9         # each video independent and resumable
```

```python
from pathlib import Path
from slidegrab import run, get_provider
summary = run("https://youtube.com/watch?v=...", Path("out"), mode="lecture",
              provider=get_provider("zai"))
```

Output, in `out/<video id>/`:

| File | What |
|---|---|
| `slides.md` | **start here** — each distinct slide with its text, picture description and narration |
| `slides.json` | the same, structured: `stamp`, `text`, `image`, `in_photo`, `shown_at[]`, `narration`, `reader` |
| `deck.pdf` | the slides as pages, each footed with its timestamp |
| `qa_review.json` | quiz mode: `{question, answer, from_slide, by, problem}`; `problem` is set where unresolved |
| `summary.json` | counts, the measured noise stats, which model read the text |

Everything is cached per frame and per slide; a finished video is skipped; failures are
never cached as blank slides (they are listed and absent, so a re-run retries them).

## Which mode

* **`lecture`** (default) — samples a frame every 3 s and splits the video wherever the
  picture really changes, keeping the *last* (fullest) frame of each stretch. Built for
  slides that **build up** (bullets appear, code is typed), which have no cut to detect.
  The change threshold is measured from the video itself (median frame-to-frame
  difference × 6), not fixed.
* **`quiz`** — scene-cut detection, then question/answer pairing. Built for decks where
  one slide **replaces** another. The pairing assumes a convention: questions are
  *numbered* on the slide and end in `?`, and answers are short captions. Another
  convention gives slides but **zero anchors and no pairs** — it says so rather than
  guessing.

## What it can't do (read this)

* **Narration is captions, not audio.** YouTube auto-captions mangle technical terms, lag
  speech by a second or two, and some videos have none (narration is then empty). A speaker
  who talks about the next slide before switching it will have those words on the previous one.
* **Slides that build up appear as successive states** (title, then title + bullet 1, then
  + bullet 2) when each step is a large change. Each is a true state of the screen; pixels
  cannot say one is a subset of another.
* **Sparse slides vs the default "same slide" cutoff.** Quiz mode treats two frames as the
  same slide when their mean pixel difference is under 3.5 — measured on dense real quiz
  decks (different slides ≥ 6.8, repeats ≤ 6.3). A constructed pair of text-light slides
  scored 3.1 and would be merged. Check `deck.pdf` on text-light decks.
* **Quiz scene detection misses words changing on a plain panel** (black-on-white slides
  that differ only in text produced one scene). Use `--mode lecture` for those.
* **A moving webcam** inflates the noise floor and can hide real changes (lecture) or split
  one slide into many. The tool warns when >50% of frame pairs exceed the cutoff; pass
  `--box x0,y0,x1,y1` to *compare* only the slide area. Frames are always saved whole.
* **Pictures are described, not understood.** Diagrams get one literal sentence; the model
  is told never to infer. Code is transcribed as the model sees it — look-alike characters
  (`l`/`1`, `O`/`0`) can be wrong and nothing here compiles or verifies it.
* **Quiz answers come from slides only.** If the answer is spoken, or is only inside a
  photograph, the pair is flagged `problem`, never filled from the model's own knowledge.
  Where several slides compete, the model *selects* one (index) and its quote must appear
  verbatim on that slide; those pairs carry a `review` list.
* **No audio, OCR-free by design:** no speech-to-text, no separate OCR engine.

## What has been verified (and what hasn't)

Run against a real 72-minute quiz deck that FundaQuiz's original pipeline had already
processed (same video, same compare box `0,0,1020,720`, same z.ai `glm-4.5v` reader):

| Stage | Original | slidegrab | |
|---|---|---|---|
| scenes → kept → distinct slides | 247 → 230 → 117 | 247 → 230 → 117 | identical |
| slide text vs the original's transcripts (117 slides) | — | 88% exact, 97% after ignoring bullet glyphs (`*`/`-`/`•`) and underscore runs | 4 slides differ in content (one title card read differently) |
| question anchors | 21 | 21 | same |
| Q/A pairs resolved by **structure** (no model) | 4 | 4 (+1 below) | agree |
| Q/A pairs where a **model picks** among competing slides | 3 | 2 | **not reproducible run to run**: 3 of 21 blocks flipped between answered and flagged |

* The model-pick step is non-deterministic even at temperature 0. Those pairs carry a
  `review` list; treat them as candidates, not facts.
* Porting found a bug in the original: the multi-part-question rule skipped the
  plausibility check, so a conference title card plus a rules slide were accepted as the
  answer to a music question. Fixed here (regression test in `tests/test_quiz.py`); the
  original `slide_qa.py` still has it. After the fix this deck gives **5 verified / 16
  flagged** — the original's 7 / 14 included that false pair.
* The same run lost 47 slide reads to a network outage. They were reported, not cached as
  blank, the video was not marked done, quiz pairing was skipped, and a re-run read only the
  missing ones. That is where the circuit breaker (stop after 5 consecutive failures) and the
  "incomplete" handling come from.

**Not verified:**
* **`lecture` mode on a real slide lecture.** It is covered by a synthetic-video test, and on
  one real video — an interview, i.e. a talking head with almost no slides — it produced 2
  slides and printed its "mostly live footage" warning. A CppCon-style talk has not been run.
  Try one with `--calibrate` first and check `deck.pdf` before trusting the slide count.
* **Gemini and Claude connectors** against the live APIs (see Providers).
* Narration alignment against real YouTube captions (tested on a hand-written `.vtt`).

## Tests

```bash
pip install -e ".[test]" && pytest            # offline: fake provider + a synthetic video
SLIDEGRAB_LIVE=1 pytest tests/test_live.py    # real providers, spends credit
```

Every module's docstring records what was tried, what failed and why — read those before
changing a threshold.

MIT licensed.
