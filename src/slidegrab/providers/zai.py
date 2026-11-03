"""z.ai (GLM) connector. Needs ZAI_API_KEY.

Ported from FundaQuiz's llm_client.py. Two things here were each paid for by a bug:

  * `thinking` is DISABLED unless asked for. glm-4.5v is a reasoning model and its
    hidden reasoning tokens are drawn from max_tokens, so with thinking on the
    VISIBLE transcript was truncated mid-sentence with no error — slides lost their
    closing question mark and were silently reclassified as non-questions.
  * `finish_reason == "length"` is checked on every reply. It is the only signal
    separating a short answer from a decapitated one.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

from ..errors import ProviderError, Truncated
from ..retry import semaphore, with_retry

URL = "https://api.z.ai/api/coding/paas/v4/chat/completions"
DEFAULT_VISION_MODEL = "glm-4.5v"
DEFAULT_TEXT_MODEL = "glm-5.3"


class ZaiProvider:
    name = "zai"

    def __init__(self, model: str | None = None, text_model: str | None = None,
                 timeout: int = 180):
        self.key = os.environ.get("ZAI_API_KEY")
        if not self.key:
            raise ProviderError("ZAI_API_KEY not set")
        self.model = model or DEFAULT_VISION_MODEL
        self.text_model = text_model or DEFAULT_TEXT_MODEL
        self.timeout = timeout

    def _post(self, model: str, messages: list, max_tokens: int, thinking: bool) -> str:
        body = json.dumps({
            "model": model, "max_tokens": max_tokens, "temperature": 0.0,
            "thinking": {"type": "enabled"} if thinking else {"type": "disabled"},
            "messages": messages,
        }).encode()

        def once() -> str:
            with semaphore(f"zai/{model}"):
                req = urllib.request.Request(
                    URL, data=body,
                    headers={"Authorization": f"Bearer {self.key}",
                             "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.load(resp)
            choice = data["choices"][0]
            text = (choice["message"]["content"] or "").strip()
            if choice.get("finish_reason") == "length":
                raise Truncated(f"zai/{model}", text, max_tokens)
            return text

        return with_retry(once, label=f"zai/{model}")

    def vision(self, system: str, prompt: str, image_path, max_tokens: int) -> str:
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        return self._post(self.model, [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]},
        ], max_tokens, thinking=False)

    def text(self, system: str, user: str, max_tokens: int, *, thinking: bool = False) -> str:
        return self._post(self.text_model, [
            {"role": "system", "content": system},
            {"role": "user", "content": user}], max_tokens, thinking)
