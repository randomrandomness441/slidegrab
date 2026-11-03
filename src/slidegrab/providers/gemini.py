"""Gemini connector. Needs GEMINI_API_KEY.

Talks to the REST API directly (no SDK dependency). Gemini 2.5+/3 models can spend
the output allowance on hidden "thinking" tokens, which truncates the visible text
exactly as z.ai's reasoning models do; `thinkingBudget: 0` is sent unless thinking
is requested. If a model rejects that field the call is retried once without it.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from ..errors import ProviderError, Truncated
from ..retry import semaphore, with_retry

BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-3.1-flash-lite"


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str | None = None, timeout: int = 180):
        self.key = os.environ.get("GEMINI_API_KEY")
        if not self.key:
            raise ProviderError("GEMINI_API_KEY not set")
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout

    def _post(self, system: str, parts: list, max_tokens: int, thinking: bool) -> str:
        def body(with_thinking_cfg: bool) -> bytes:
            cfg = {"maxOutputTokens": max_tokens, "temperature": 0.0}
            if with_thinking_cfg and not thinking:
                cfg["thinkingConfig"] = {"thinkingBudget": 0}
            return json.dumps({
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": cfg}).encode()

        def call(payload: bytes) -> dict:
            req = urllib.request.Request(
                f"{BASE}/{self.model}:generateContent", data=payload,
                headers={"x-goog-api-key": self.key, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.load(resp)

        def once() -> str:
            with semaphore(f"gemini/{self.model}"):
                try:
                    data = call(body(True))
                except urllib.error.HTTPError as e:
                    if e.code != 400:
                        raise
                    detail = e.read().decode("utf-8", "replace")
                    if "thinking" not in detail.lower():
                        raise ProviderError(f"gemini/{self.model}: HTTP 400 {detail[:200]}") from e
                    data = call(body(False))        # model has no thinkingConfig
            cand = (data.get("candidates") or [{}])[0]
            text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", [])).strip()
            if cand.get("finishReason") == "MAX_TOKENS":
                raise Truncated(f"gemini/{self.model}", text, max_tokens)
            if not text:
                reason = cand.get("finishReason") or data.get("promptFeedback")
                raise ProviderError(f"gemini/{self.model}: empty reply ({reason})")
            return text

        return with_retry(once, label=f"gemini/{self.model}")

    def vision(self, system: str, prompt: str, image_path, max_tokens: int) -> str:
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        return self._post(system, [{"text": prompt},
                                   {"inline_data": {"mime_type": "image/jpeg", "data": b64}}],
                          max_tokens, thinking=False)

    def text(self, system: str, user: str, max_tokens: int, *, thinking: bool = False) -> str:
        return self._post(system, [{"text": user}], max_tokens, thinking)
