import shutil

import pytest

from helpers import FakeProvider


@pytest.fixture
def fake():
    return FakeProvider()


@pytest.fixture
def have_ffmpeg():
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
