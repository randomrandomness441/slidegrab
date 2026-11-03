"""One backoff policy and one tolerant JSON parser, shared by every provider.

Ported from FundaQuiz's llm_client.py. They live in one place so a fix reaches every
call site — a fix applied to three of four hand-rolled copies is how one path
silently kept the bug."""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Callable, TypeVar

from .errors import ProviderError, ProviderExhausted, Truncated

T = TypeVar("T")
DEFAULT_RETRIES = 4
BACKOFF_BASE = 3
BACKOFF_MAX = 30

# Phrasings for "your account cannot make this call", read from the response BODY.
_EXHAUSTED = ("insufficient balance", "no resource package", "quota exceeded",
              "billing", "recharge", "insufficient_quota", "resource_exhausted_billing")


def exhausted_reason(e: Exception) -> str | None:
    body = ""
    read = getattr(e, "read", None)
    if callable(read):
        try:
            body = read().decode("utf-8", "replace")[:400]
        except Exception:
            body = ""
    haystack = f"{body} {e}".lower()
    return (body.strip() or str(e)) if any(m in haystack for m in _EXHAUSTED) else None


def _is_rate_limit(e: Exception) -> bool:
    return getattr(e, "code", None) == 429 or "429" in str(e) or "rate" in str(e).lower()


def with_retry(fn: Callable[[], T], retries: int = DEFAULT_RETRIES,
               base: int = BACKOFF_BASE, label: str = "llm") -> T:
    """Exponential backoff; rate limits wait longer. Never retries Truncated or an
    exhausted account."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Truncated:
            raise
        except ProviderError:
            raise
        except Exception as e:
            reason = exhausted_reason(e)
            if reason is not None:
                raise ProviderExhausted(f"{label}: {reason}") from e
            last = e
            if attempt < retries:
                wait = min(base * (2 ** attempt), BACKOFF_MAX)
                if _is_rate_limit(e):
                    wait = min(wait * 2, BACKOFF_MAX)
                time.sleep(wait)
    raise ProviderError(f"{label} failed after {retries + 1} attempts: "
                        f"{type(last).__name__}: {last}") from last


_FENCE = re.compile(r"^```[a-z]*|```$", re.MULTILINE)


def parse_json(raw: str) -> Any:
    """Parse JSON out of an LLM reply: code fences, a prose preamble and trailing
    junk after a balanced value are all tolerated."""
    if not raw:
        raise ProviderError("empty LLM response")
    text = _FENCE.sub("", raw).strip()
    start = min((i for i in (text.find("["), text.find("{")) if i != -1), default=-1)
    if start == -1:
        raise ProviderError(f"no JSON found in response: {text[:80]!r}")
    try:
        return json.JSONDecoder().raw_decode(text[start:])[0]
    except json.JSONDecodeError as e:
        raise ProviderError(f"malformed JSON: {e} | {text[start:start + 80]!r}") from e


# Per-model in-flight cap, so concurrency is bounded by prevention not recovery.
_sems: dict[str, threading.Semaphore] = {}
_sems_lock = threading.Lock()
DEFAULT_CONCURRENCY = 2


def semaphore(model: str, limit: int = DEFAULT_CONCURRENCY) -> threading.Semaphore:
    with _sems_lock:
        return _sems.setdefault(model, threading.Semaphore(limit))
