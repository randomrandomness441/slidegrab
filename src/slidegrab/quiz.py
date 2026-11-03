"""quiz.py — slide transcripts become (question, answer) pairs. Optional; quiz decks only.

Ported from FundaQuiz `slide_qa.py`. Input: the records from read.py / align.py (dicts
with name, text, image, in_photo, number). Output: Q/A pairs each traceable to the slide
it was read off, with a `problem` field instead of a made-up answer when none is found.

CONVENTIONS IT ASSUMES — read this before trusting a run. The rules were fitted to one
quiz series: questions NUMBERED on the slide, a question mark on the question slide, and
answers shown as short captions. A deck that numbers differently produces slides but no
anchors, hence no pairs. `anchors == 0` with many slides is not a quiet failure to
average away: either it is not a quiz (an interview — zero is correct) or its conventions
differ and these rules need extending.

THE STRUCTURE. Question slides are the ANCHORS. Everything shown between one anchor and
the next belongs to that question: clue pictures, the answer slide, sometimes a round
title. ROUNDS come from the numbers: a decrease is a new round; the same number with
DIFFERENT text is a new round; the same number and text is a revisit.

WHY THE MODEL SELECTS AND NEVER WRITES. Asked to produce the answer, an LLM given a block
with no answer slide supplies one from its own knowledge — an earlier build invented 3 of
15, and TWICE the invention was CORRECT ("Broccoli", "The Berlin Wall"), read off no slide
at all. A correct fabrication passes every downstream check. So the model is handed
numbered candidates and returns an INDEX; the answer text is then taken from that slide
by us. The failure becomes "picked the wrong slide" — visible and checkable.

VALIDATION, in order: (1) the index must exist in the block; (2) the quoted answer must
appear VERBATIM in that slide; (3) a block with no answer is reported as missing, never
filled; (4) the text must be the right KIND of thing (plausible()).
"""

from __future__ import annotations

import re

from .errors import ProviderError
from .providers.base import Provider
from .retry import parse_json

SIMILAR = 0.60          # word overlap above which two anchors are the same question

SELECT_SYSTEM = (
    "You are reading the slides of a recorded quiz, in the order they appeared on screen.\n"
    "You are given one QUESTION slide, then the slides shown after it and before the next "
    "question. Exactly one of those may be the answer reveal. The others are clue images, "
    "round titles, or repeats of the question.\n"
    "Your job is to SELECT which slide is the answer. You never write an answer yourself.\n"
    "The answer you quote MUST appear word for word in the slide you select. If you cannot "
    "quote it from the slide, that slide is not the answer.\n"
    "An answer slide often also explains WHY the answer follows — capture that separately "
    "if present.\n"
    "If none of the slides reveals the answer, return index null. Do NOT supply the answer "
    "from your own knowledge: a missing answer is a useful result, an invented one is worse "
    "than nothing.\n"
    'Output ONLY JSON: {"index": <int or null>, "answer": "<verbatim>", "why": "<verbatim explanation or empty>"}'
)

PLAUSIBLE_SYSTEM = (
    "You are shown a quiz question and a piece of text taken from a slide that followed it. "
    "Decide whether that text is the KIND OF THING the question asks for.\n\n"
    "This is NOT about whether the answer is correct. You may not know the fact, and the "
    "answer may look wrong to you — say PLAUSIBLE anyway if it is the right kind of thing. "
    "A question asking for a nickname is answered by a name, even a surprising one.\n\n"
    "Say IMPLAUSIBLE only when the text could not answer ANY question of this shape: a "
    "conference title card, a round header, a sponsor credit, 'End of slide show', a page "
    "number, or text about a completely unrelated subject.\n"
    'Output ONLY JSON: {"verdict":"PLAUSIBLE"|"IMPLAUSIBLE","why":"<one line>"}'
)


def plausible(provider: Provider, question: str, answer: str) -> tuple[bool, str]:
    """Is this text the right KIND of thing to answer this question?

    A SHAPE check, deliberately not a correctness check. Asked whether an answer was
    RIGHT, a model refused "Berlin Wall" for the Haldeman/Ehrlichman nickname question —
    a knowledge test it could not pass. It exists because the verbatim guard cannot do
    this job: that guard confirms provenance, and a conference title card genuinely WAS
    on the slide — "INDUSTRY AND ALUMNI RELATIONS QuNITe MONTY QUIZ 3.0" was accepted as
    the answer to a motion-capture question. Provenance and plausibility are different.

    Failures return PLAUSIBLE: an unreachable provider is not evidence that an answer is
    furniture, and treating it so would silently delete pairs."""
    user = (f"QUESTION:\n{question[:900]}\n\nTEXT FROM THE FOLLOWING SLIDE:\n"
            f"{answer[:400]}\n\nIs this the kind of thing the question asks for?")
    try:
        got = parse_json(provider.text(PLAUSIBLE_SYSTEM, user, max_tokens=600, thinking=True))
    except (ProviderError, ValueError):
        return True, ""
    if "IMPLAUSIBLE" in str(got.get("verdict", "")).upper():
        return False, str(got.get("why", "")).strip()
    return True, ""


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z]{4,}", s.lower()))


def same_question(a: dict, b: dict) -> bool:
    """Two anchors showing the same question — a revisit or a build-up draft."""
    if a["number"] != b["number"]:
        return False
    wa, wb = _words(a["text"]), _words(b["text"])
    if not wa or not wb:
        return False
    return len(wa & wb) / min(len(wa), len(wb)) >= SIMILAR


def find_anchors(slides: list[dict]) -> list[int]:
    return [i for i, s in enumerate(slides) if s["number"] is not None and "?" in s["text"]]


def build_blocks(slides: list[dict]) -> list[dict]:
    """-> [{question, number, round, anchors, candidates}]"""
    groups: list[list[int]] = []
    for i in find_anchors(slides):
        if groups and same_question(slides[groups[-1][-1]], slides[i]):
            groups[-1].append(i)
        else:
            groups.append([i])
    blocks, rnd = [], 1
    for gi, group in enumerate(groups):
        first = group[0]
        if gi:
            prev, cur = slides[groups[gi - 1][-1]], slides[first]
            if cur["number"] < prev["number"] or (
                    cur["number"] == prev["number"] and not same_question(prev, cur)):
                rnd += 1
        stop = groups[gi + 1][0] if gi + 1 < len(groups) else len(slides)
        question = max((slides[i]["text"] for i in group), key=len)   # fullest build-up draft
        # A NUMBERED slide is a question slide even without a question mark (some are
        # captured mid-render). Leaving one in the pool let "10 - When this sporting
        # venue opened..." be chosen as another question's ANSWER, and it passed the
        # verbatim guard. Answer slides in these decks are never numbered.
        candidates = [j for j in range(first + 1, stop)
                      if j not in group and slides[j]["number"] is None]
        blocks.append({"round": rnd, "number": slides[first]["number"],
                       "question": question.strip(), "anchors": group, "candidates": candidates})
    return blocks


def pair_batched(blocks: list[dict], slides: list[dict]) -> None:
    """Decks that show ALL questions, then ALL answers (Q1 Q2 Q3 A1 A2 A3).

    That shape leaves the leading blocks empty and hands the last block every answer.
    Pairing is POSITIONAL — weaker than the rest of this module, since answers appearing
    in question order is a convention — so it applies only when the counts match exactly,
    the text is still taken verbatim off its slide, and the result is marked
    by="positional". Counts that disagree are left flagged rather than guessed at."""
    i = 0
    while i < len(blocks):
        run = i
        while run < len(blocks) and not blocks[run]["candidates"]:
            run += 1
        if run > i and run < len(blocks):
            answers = [j for j in blocks[run]["candidates"]
                       if slides[j]["text"].strip() and not _is_section(slides[j])]
            questions = blocks[i:run + 1]
            if len(answers) == len(questions):
                for block, j in zip(questions, answers):
                    block["batched_answer"] = j
            else:
                # Counts disagree, so nothing is paired by position — but the LAST block
                # would still take the single answer slide structurally, an answer that
                # may belong to the questions before it. Keep the pair, mark it for
                # review rather than presenting a guess as verified.
                blocks[run]["review_note"] = (
                    f"{run - i} question(s) just before this one have no answer slide and "
                    f"{len(answers)} answer slide(s) follow for {len(questions)} questions — "
                    "this answer may belong to an earlier question")
        i = max(run + 1, i + 1)


def _candidate_text(s: dict) -> str:
    parts = []
    if s["text"].strip():
        parts.append(s["text"].strip())
    if s["image"]:
        parts.append(f"[picture] {s['image']}")
    if s.get("in_photo", "").strip():
        parts.append(f"[text inside the picture] {s['in_photo'].strip()}")
    return "\n".join(parts) or "(blank slide)"


def _is_section(s: dict) -> bool:
    """A round title — short, shouted, no sentence. Never an answer."""
    t = s["text"].strip()
    return bool(t) and t.isupper() and len(t) < 40


def _fail(index, problem, by=None, answer="", review=None) -> dict:
    return {"index": index, "answer": answer, "why": "", "by": by, "review": review, "problem": problem}


def choose_answer(block: dict, slides: list[dict], provider: Provider) -> dict:
    """Pick the answer slide: structurally when that is unambiguous, otherwise by asking
    the model — always verifying the quote came from the slide named.

    THE STRUCTURAL CASE. Exactly one text slide in the block that is not a round title IS
    the reveal. Measured against the model's own picks on one deck: 9 agreements, 0
    disagreements — and it recovers a block the model REFUSED: shown a single candidate
    reading "Berlin Wall", it returned null because it was checking whether the answer was
    RIGHT, a knowledge test it cannot pass. Where structure determines the answer, no
    judgement should be invited."""
    if "batched_answer" in block:
        j = block["batched_answer"]
        ok, why = plausible(provider, block["question"], slides[j]["text"])
        if not ok:
            return _fail(j, f"not the kind of thing the question asks for: {why}", by="positional")
        return {"index": j, "answer": slides[j]["text"].strip(), "why": "", "problem": None,
                "by": "positional",
                "review": [{"slide": slides[j]["name"],
                            "text": "paired by position — answers were revealed together, in question order"}]}
    if not block["candidates"]:
        return _fail(None, "no slides after this question")

    texted = [j for j in block["candidates"]
              if slides[j]["text"].strip() and not _is_section(slides[j])]

    # TEXT INSIDE A PHOTOGRAPH is a CANDIDATE, never a verified answer. Validated on one
    # example, it failed at scale: across six videos 7 of 8 in-photo "answers" were junk
    # scraped off a picture ('1829 2', 'Hunero77 I&R'). A weak signal promoted to a
    # verdict is how a count improves while the data gets worse.
    if not texted:
        photo = [j for j in block["candidates"] if slides[j].get("in_photo", "").strip()]
        if not photo:
            return _fail(None, "no answer slide in this block")
        j = photo[0]
        return _fail(j, f"only text inside a photo is available — "
                        f"{slides[j]['in_photo'].strip()[:80]!r} — verify by hand", by="in-photo")

    if len(texted) == 1:
        j = texted[0]
        ok, why = plausible(provider, block["question"], slides[j]["text"])
        if not ok:
            return _fail(j, f"not the kind of thing the question asks for: {why}", by="structure")
        return {"index": j, "answer": slides[j]["text"].strip(), "why": "", "problem": None,
                "by": "structure",
                "review": ([{"slide": slides[j]["name"], "text": block["review_note"]}]
                           if block.get("review_note") else None)}

    # MULTI-PART questions have one answer slide per part and the answer is ALL of them.
    # The question's own '?' count says how many parts to expect.
    parts = block["question"].count("?")
    if parts > 1 and len(texted) == parts:
        return {"index": texted[0], "by": "structure", "problem": None, "why": "",
                "answer": "\n".join(slides[j]["text"].strip() for j in texted),
                "spans": [slides[j]["name"] for j in texted]}

    # More than one text slide does NOT mean two answers: over six such blocks four of the
    # extras were text inside a photo or deck chrome ("End of slide show"). So the model
    # still chooses, but the block is marked for review with its alternatives recorded.
    # When it picked the LATER of two genuine answers it quoted faithfully and the verbatim
    # guard passed — a silent wrong answer — which is why review is recorded, not skipped.
    review = ([{"slide": slides[j]["name"], "text": slides[j]["text"][:120]} for j in texted]
              if len(texted) > 1 else None)
    listing = "\n\n".join(f"--- slide {j} ---\n{_candidate_text(slides[j])[:700]}"
                          for j in block["candidates"])
    user = (f"QUESTION (slide is numbered {block['number']}):\n{block['question'][:1200]}\n\n"
            f"SLIDES SHOWN AFTER IT:\n{listing}\n\nWhich slide is the answer?")
    try:
        got = parse_json(provider.text(SELECT_SYSTEM, user, max_tokens=700))
    except (ProviderError, ValueError) as e:
        return _fail(None, f"model call failed: {e}")
    idx, answer = got.get("index"), (got.get("answer") or "").strip()
    if idx is None:
        return _fail(None, "no answer slide in this block")
    if idx not in block["candidates"]:
        return _fail(idx, "picked a slide outside the block", answer=answer)
    # THE GUARD: the quote must actually be on the slide it named.
    haystack = re.sub(r"\s+", " ", _candidate_text(slides[idx])).lower()
    needle = re.sub(r"\s+", " ", answer).lower()
    if not needle or needle[:40] not in haystack:
        return _fail(idx, "quoted text is not on the slide it named", answer=answer)
    ok, why = plausible(provider, block["question"], answer)
    if not ok:
        return _fail(idx, f"not the kind of thing the question asks for: {why}", by="model", review=review)
    return {"index": idx, "answer": answer, "why": (got.get("why") or "").strip(),
            "problem": None, "by": "model", "review": review}


def extract_pairs(slides: list[dict], provider: Provider, video_id: str = "") -> list[dict]:
    """Slide records -> Q/A pairs, flagged where unresolved. Never fills a gap."""
    blocks = build_blocks(slides)
    pair_batched(blocks, slides)
    pairs = []
    for b in blocks:
        res = choose_answer(b, slides, provider)
        pairs.append({
            "video_id": video_id, "round": b["round"], "number": b["number"],
            "question": b["question"], "answer": res["answer"], "why": res["why"],
            "from_slide": slides[res["index"]]["name"] if res["index"] is not None else None,
            "by": res.get("by"), "review": res.get("review"),
            "spans": res.get("spans"), "problem": res["problem"],
        })
    return pairs
