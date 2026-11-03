import numpy as np
import pytest

from slidegrab import align, fetch
from slidegrab.dedupe import dedupe
from slidegrab.errors import ProviderError, ProviderExhausted, Truncated
from slidegrab.frames import Frame, select_slides, settled_states
from slidegrab.providers.base import get_provider
from slidegrab.read import question_number, split_sections
from slidegrab.retry import parse_json, with_retry

from helpers import make_slide


def test_vtt_drops_rolling_repeats(tmp_path):
    v = tmp_path / "a.vtt"
    v.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nhello <c>there</c>\n\n"
                 "00:00:03.000 --> 00:00:05.000\nhello there\nsecond line\n")
    assert fetch.parse_vtt(v) == [(1.0, "hello there"), (3.0, "second line")]


def test_split_sections_three_way_and_none():
    s, i, p = split_sections("SLIDE: 3. Who?\nline two\nIMAGE: none\nINPHOTO: (c) Disney")
    assert s == "3. Who?\nline two" and i == "" and p == "(c) Disney"


@pytest.mark.parametrize("text,n", [("12\nWho?", 12), ("# 4.\nx", 4), ("In 1912 a ship", None), ("", None)])
def test_question_number_only_from_first_line(text, n):
    assert question_number(text) == n


def _scenes(lengths):
    t, out = 0.0, []
    for i, l in enumerate(lengths):
        out.append(Frame(i, t, t + l)); t += l
    return out


def test_select_slides_keeps_isolated_flash_drops_burst_but_keeps_its_last():
    sc = _scenes([5, 0.9, 5] + [0.5] * 6 + [5])     # flash, then a 6-run, then a slide
    kept = select_slides(sc)
    nums = [s.number for s in kept]
    assert 1 in nums                  # the 0.9s flash is a real slide
    assert 8 in nums                  # last scene of the burst survives (caption over clip)
    assert not {3, 4, 5, 6, 7} & set(nums)


def test_dedupe_compares_against_all_earlier_and_keeps_first(tmp_path):
    # explicit threshold: this tests the LOGIC (A-B-A), not the 3.5 default, which is
    # unsafe for slides this sparse — see dedupe.py "BLIND TO"
    a = make_slide(tmp_path / "a.jpg", "AAAA question one", lines=["x"] * 3)
    b = make_slide(tmp_path / "b.jpg", "BBBB question two", lines=["y"] * 3)
    a2 = make_slide(tmp_path / "a2.jpg", "AAAA question one", lines=["x"] * 3)
    fr = [Frame(0, 0, 5, a), Frame(1, 5, 10, b), Frame(2, 10, 15, a2)]
    kept, showings = dedupe(fr, threshold=1.0)
    assert [f.path.name for f in kept] == ["a.jpg", "b.jpg"]          # A-B-A is TWO slides
    assert [k for _, k in showings] == [0, 1, 0]


def test_sparse_slides_can_fall_under_the_default_threshold(tmp_path):
    """Documents a known limit rather than hiding it: see dedupe.py 'BLIND TO'."""
    a = make_slide(tmp_path / "a.jpg", "AAAA question one", lines=["x"] * 3)
    b = make_slide(tmp_path / "b.jpg", "BBBB question two", lines=["y"] * 3)
    kept, _ = dedupe([Frame(0, 0, 5, a), Frame(1, 5, 10, b)])
    assert len(kept) == 1                    # merged at the default 3.5 — the caveat


def test_settled_states_keeps_last_frame_of_segment_stamped_with_segment_start(tmp_path):
    # Like a real screencast most sample pairs are IDENTICAL, which is what makes the
    # median a noise measure: title (6 samples), title+bullets (6), unrelated slide (8).
    kinds = [("TITLE A", [])] * 6 + [("TITLE A", ["bullet one is long text"] * 4)] * 6 \
        + [("OTHER B", ["zzz"] * 6)] * 8
    f = [make_slide(tmp_path / f"f{i:02d}.jpg", t, lines=l) for i, (t, l) in enumerate(kinds)]
    frames = [Frame(i, i * 3.0, i * 3.0 + 3, p) for i, p in enumerate(f)]
    kept, stats = settled_states(frames)
    assert [k.path.name for k in kept] == ["f05.jpg", "f11.jpg", "f19.jpg"]   # LAST of each
    assert [k.start for k in kept] == [0.0, 18.0, 36.0]                       # START of each
    assert stats["noise_floor_median"] == 0.0


def test_align_narration_includes_revisits():
    kept = [Frame(1, 0, 10), Frame(2, 10, 20)]
    for k in kept:
        k.path = type("P", (), {"stem": f"s{k.number}"})()
    showings = [(Frame(0, 0, 10), 0), (Frame(1, 10, 20), 1), (Frame(2, 20, 30), 0)]
    cues = [(1, "one"), (12, "two"), (22, "back to one")]
    w = align.narration_windows(showings, 30)
    assert align.narration_for(cues, w[0]) == "one back to one"
    assert align.narration_for(cues, w[1]) == "two"


def test_parse_json_tolerates_fences_and_trailing_junk():
    assert parse_json('```json\n{"a": 1}\n``` and then]') == {"a": 1}
    with pytest.raises(ProviderError):
        parse_json("no json here")


def test_with_retry_stops_on_exhausted_account_and_on_truncation():
    calls = []

    def boom():
        calls.append(1)
        raise RuntimeError("HTTP 429 Insufficient balance or no resource package")
    with pytest.raises(ProviderExhausted):
        with_retry(boom, retries=4, base=0)
    assert len(calls) == 1                      # one try, not five

    def trunc():
        calls.append(2)
        raise Truncated("x", "partial", 10)
    with pytest.raises(Truncated):
        with_retry(trunc, retries=4, base=0)
    assert calls.count(2) == 1


def test_no_key_is_a_clear_error(monkeypatch, tmp_path):
    for v in ("ZAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.chdir(tmp_path)                 # no .env here
    with pytest.raises(ProviderError, match="No provider key"):
        get_provider()
    with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
        get_provider("gemini")
