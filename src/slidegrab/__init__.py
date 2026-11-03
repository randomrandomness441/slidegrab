"""slidegrab — extract the distinct slides of a lecture or quiz video.

    from slidegrab import run, get_provider
    run("https://youtube.com/watch?v=...", Path("out"), mode="lecture",
        provider=get_provider("gemini"))

Reading slide text needs a provider API key (z.ai, Gemini or Claude) — see README.
"""
from .errors import ProviderError, ProviderExhausted, Truncated
from .pipeline import run, run_playlist
from .providers import get_provider

__all__ = ["run", "run_playlist", "get_provider", "ProviderError", "ProviderExhausted", "Truncated"]
__version__ = "0.1.0"
