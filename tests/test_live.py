"""Real provider calls. Skipped unless SLIDEGRAB_LIVE=1 AND that provider's key is set,
because they spend money and a fake provider cannot show that a real API accepts our
request. Run: SLIDEGRAB_LIVE=1 pytest tests/test_live.py -v"""
import os

import pytest

from slidegrab.providers.base import KEY_VARS, get_provider
from slidegrab.read import PROMPT, SYSTEM, split_sections

from helpers import make_slide


@pytest.mark.parametrize("name", sorted(KEY_VARS))
def test_provider_reads_a_slide(name, tmp_path):
    if os.environ.get("SLIDEGRAB_LIVE") != "1":
        pytest.skip("set SLIDEGRAB_LIVE=1")
    if not os.environ.get(KEY_VARS[name]):
        pytest.skip(f"{KEY_VARS[name]} not set")
    img = make_slide(tmp_path / "s.jpg", "PHOTOSYNTHESIS", lines=["Light reactions", "Calvin cycle"])
    text, _, _ = split_sections(get_provider(name).vision(SYSTEM, PROMPT, img, max_tokens=1500))
    assert "PHOTOSYNTHESIS" in text.upper() and "calvin" in text.lower()
