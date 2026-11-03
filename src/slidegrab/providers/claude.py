"""Claude (Anthropic API) connector. Needs ANTHROPIC_API_KEY.

REST, no SDK dependency. Vision sends the slide as a base64 image block."""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

from ..errors import ProviderError, Truncated
from ..retry import semaphore, with_retry

URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-haiku-5-5"


class ClaudeProvider:
    name = "claude"

    def __init__(self, model: str | None = None, timeout: int = 180):
        self.key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.key:
            raise ProviderError("ANTHROPIC_API_KEY not set")
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout

    def _post(self, system: str, content, max_tokens: int) -> str:
        body = json.dumps({
            "model": self.model, "max_tokens": max_tokens, "temperature": 0.0,
            "system": system,
            "messages": [{"role": "user", "content": content}]}).encode()

        def once() -> str:
            with semaphore(f"claude/{self.model}"):
                req = urllib.request.Request(
                    URL, data=body,
                    headers={"x-api-key": self.key, "anthropic-version": "2023-06-01",
                             "content-type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.load(resp)
            text = "".join(b.get("text", "") for b in data.get("content", [])
                           if b.get("type") == "text").strip()
            if data.get("stop_reason") == "max_tokens":
                raise Truncated(f"claude/{self.model}", text, max_tokens)
            if not text:
                raise ProviderError(f"claude/{self.model}: empty reply ({data.get('stop_reason')})")
            return text

        return with_retry(once, label=f"claude/{self.model}")

    def vision(self, system: str, prompt: str, image_path, max_tokens: int) -> str:
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        return self._post(system, [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": prompt}], max_tokens)

    def text(self, system: str, user: str, max_tokens: int, *, thinking: bool = False) -> str:
        return self._post(system, user, max_tokens)     # `thinking` unused: off by default
