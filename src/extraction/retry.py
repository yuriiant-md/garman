"""API errors worth another try, shared by the Claude and OpenRouter extractors.

The SDKs retry these themselves; whatever is still left after their retries becomes an `error`
result. Any other status means the request itself is wrong and stops the run.
"""

RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})


def is_retryable(status_code: int) -> bool:
    return status_code in RETRYABLE_STATUS_CODES or status_code >= 500
