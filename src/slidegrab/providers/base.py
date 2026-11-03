"""The connector interface.

slidegrab needs exactly two capabilities from a model provider:

    vision(system, prompt, image_path, max_tokens) -> text   read one slide image
    text(system, user, max_tokens)                 -> text   pick/validate (quiz mode)

Anything that implements `Provider` works. Adding one means a new file in this
package and a line in `get_provider`; nothing else in the library knows which
provider is in use.

REQUIRED: you must supply an API key for at least one provider. Without a key the
frame-extraction stages still run, but no slide text can be read.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from ..errors import ProviderError


@runtime_checkable
class Provider(Protocol):
    name: str
    model: str            # the vision model, recorded in outputs for provenance

    def vision(self, system: str, prompt: str, image_path, max_tokens: int) -> str: ...

    def text(self, system: str, user: str, max_tokens: int, *, thinking: bool = False) -> str: ...


# env var that must hold the key, per provider
KEY_VARS = {"zai": "ZAI_API_KEY", "gemini": "GEMINI_API_KEY", "claude": "ANTHROPIC_API_KEY"}


def get_provider(name: str | None = None, model: str | None = None) -> Provider:
    """Build a provider by name; with no name, use the first one whose key is set."""
    from dotenv import load_dotenv
    load_dotenv()                     # a .env in the working directory

    if name is None:
        name = next((n for n, var in KEY_VARS.items() if os.environ.get(var)), None)
        if name is None:
            raise ProviderError(
                "No provider key found. Set one of: "
                + ", ".join(KEY_VARS.values())
                + " (environment or .env). See README → Providers.")
    if name == "zai":
        from .zai import ZaiProvider
        return ZaiProvider(model=model)
    if name == "gemini":
        from .gemini import GeminiProvider
        return GeminiProvider(model=model)
    if name == "claude":
        from .claude import ClaudeProvider
        return ClaudeProvider(model=model)
    raise ProviderError(f"unknown provider {name!r}; available: {', '.join(KEY_VARS)}")
