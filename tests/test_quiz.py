import json

from slidegrab import quiz
from helpers import FakeProvider


def S(name, text="", number=None, image="", in_photo=""):
    return {"name": name, "text": text, "number": number, "image": image, "in_photo": in_photo}


def test_single_text_slide_is_the_answer_structurally(fake):
    slides = [S("q", "1\nWhich fruit?", 1), S("pic", image="a banana"), S("ans", "Banana")]
    pairs = quiz.extract_pairs(slides, fake)
    assert pairs[0]["answer"] == "Banana" and pairs[0]["by"] == "structure" and pairs[0]["problem"] is None
    assert not any(c[0] == "text" and "Which slide" in c[1] for c in fake.calls)   # no model pick


def test_no_answer_slide_is_reported_never_invented(fake):
    pairs = quiz.extract_pairs([S("q", "1\nWho painted it?", 1), S("pic", image="a painting")], fake)
    assert pairs[0]["answer"] == "" and pairs[0]["problem"] == "no answer slide in this block"


def test_quote_must_be_on_the_slide_it_named():
    p = FakeProvider([json.dumps({"index": 2, "answer": "Napoleon Bonaparte", "why": ""})])
    slides = [S("q", "1\nWho?", 1), S("a", "Wellington"), S("b", "Waterloo")]
    pairs = quiz.extract_pairs(slides, p)
    assert pairs[0]["problem"] == "quoted text is not on the slide it named"      # invented, caught


def test_in_photo_text_is_never_accepted_as_answer(fake):
    pairs = quiz.extract_pairs([S("q", "1\nWhat?", 1), S("pic", in_photo="Hunero77 I&R")], fake)
    assert pairs[0]["answer"] == "" and "verify by hand" in pairs[0]["problem"]


def test_batched_layout_pairs_by_position_only_when_counts_match(fake):
    slides = [S("q1", "1\nA?", 1), S("q2", "2\nB?", 2), S("a1", "Alpha"), S("a2", "Beta")]
    pairs = quiz.extract_pairs(slides, fake)
    assert [p["answer"] for p in pairs] == ["Alpha", "Beta"] and all(p["by"] == "positional" for p in pairs)
    short = quiz.extract_pairs(slides[:3], FakeProvider())      # 2 questions, 1 answer
    assert short[0]["problem"] and short[0]["answer"] == ""
    assert short[1]["review"] and "may belong to an earlier question" in short[1]["review"][0]["text"]


def test_rounds_from_numbers():
    slides = [S("a", "1\nOne thing?", 1), S("b", "2\nTwo thing?", 2), S("c", "1\nNew round?", 1)]
    assert [b["round"] for b in quiz.build_blocks(slides)] == [1, 1, 2]


def test_multipart_rule_still_runs_the_plausibility_check():
    """Regression, from a real deck: a question with two '?' followed by a title card and a
    rules slide used to be 'answered' by both, with the model never asked."""
    bad = '{"verdict":"IMPLAUSIBLE","why":"event title card"}'
    p = FakeProvider([bad])
    slides = [S("q", "20\nWho sings it? Which album?", 20),
              S("card", "INDUSTRY AND ALUMNI\nQuNiTe\nMONTY QUIZ 3.0"),
              S("rules", "The Deal\n* 20 questions")]
    pair = quiz.extract_pairs(slides, p)[0]
    assert pair["answer"] == "" and "title card" in pair["problem"]
    assert any(c[0] == "text" for c in p.calls)          # the model WAS consulted


def test_multipart_genuine_parts_still_joined(fake):
    slides = [S("q", "5\nWhat ritual? Who were these people?", 5),
              S("a1", "Cleaning the courtyard with brooms"), S("a2", "Anita Pratap")]
    pair = quiz.extract_pairs(slides, fake)[0]           # fake says PLAUSIBLE
    assert pair["problem"] is None and "brooms" in pair["answer"] and "Pratap" in pair["answer"]
