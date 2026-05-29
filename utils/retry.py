"""
Shared LLM call retry wrapper — used by every architecture.

Catches the full set of transient OpenAI-client errors that the FIM endpoint
can produce, plus httpx-level timeouts (which the OpenAI library sometimes
surfaces before it can wrap them).
"""
import time
import httpx

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)


try:
    _HTTPX_ERRORS = (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError)
except ImportError:  # pragma: no cover
    _HTTPX_ERRORS = ()

# Union of all error types we should retry on
_RETRYABLE = (
    APIError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
) + _HTTPX_ERRORS


def call_with_retry(client, *, max_retries: int = 5, base_wait: float = 2.0, **kwargs):
    """
    Call ``client.chat.completions.create(**kwargs)`` with exponential backoff.

    Retries on any transient network / server error.  Re-raises on the final
    attempt so the caller can see the original exception.

    Args:
        client:       An ``openai.OpenAI`` instance.
        max_retries:  Maximum number of attempts (default 5).
        base_wait:    Base seconds for backoff; attempt k waits base_wait**k
                      (so 1 s, 2 s, 4 s, 8 s with base_wait=2, max_retries=5).
        **kwargs:     Forwarded verbatim to ``chat.completions.create``.

    Returns:
        The ``ChatCompletion`` object from the successful call.

    4xx client errors (bad schema, auth, …) are *not* retried — only 429 and
    transient network / 5xx errors are.
    """
    if max_retries < 1:
        raise ValueError("Max number of retries must at least one.")

    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except _RETRYABLE as exc:
            # 4xx (except 429 rate limit) are client errors — retrying won't help.
            status = getattr(exc, "status_code", None)
            if isinstance(exc, APIStatusError) and status and 400 <= status < 500 and status != 429:
                raise
            if attempt == max_retries - 1:
                raise
            wait = base_wait ** attempt  # 1 s, 2 s, 4 s, 8 s …
            print(
                f"  [retry {attempt + 1}/{max_retries - 1}] "
                f"{type(exc).__name__}: {exc}. "
                f"Sleeping {wait:.0f}s…"
            )
            time.sleep(wait)

    raise Exception("Unexpected error occurred.")