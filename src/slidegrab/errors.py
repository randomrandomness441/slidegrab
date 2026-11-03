"""Exception types. Provider failure is distinct from a programming error so callers
can degrade on one without masking the other."""


class ProviderError(RuntimeError):
    """A provider call failed (rate limit, timeout, bad response, missing key)."""


class ProviderExhausted(ProviderError):
    """The account has no credit or quota left. Retrying cannot help.

    HTTP 429 means two opposite things — "slow down" and "your balance is empty" —
    and only the response BODY tells them apart. Treating both as throttling burns
    every retry against an account that will never answer, and the failure gets
    misdiagnosed as a concurrency problem."""


class Truncated(ProviderError):
    """The model hit max_tokens and the output is cut off. Carries the partial text.

    Deterministic: the same call truncates at the same place every time, so it is
    never retried — the caller raises the budget."""

    def __init__(self, label: str, text: str, max_tokens: int):
        super().__init__(f"{label}: output truncated at {max_tokens} tokens ({len(text)} chars)")
        self.text = text
        self.max_tokens = max_tokens
