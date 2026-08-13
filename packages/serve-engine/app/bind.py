"""Host bind + token policy for serve-engine."""
from __future__ import annotations

from typing import Mapping


class BindPolicyError(RuntimeError):
    pass


def is_loopback_host(host: str) -> bool:
    h = (host or "").strip().lower()
    return h in {"127.0.0.1", "localhost", "::1", "[::1]"}


def assert_safe_bind(host: str, token: str = "", *, allow_insecure: bool = False) -> None:
    if is_loopback_host(host):
        return
    if (token or "").strip():
        return
    if allow_insecure:
        return
    raise BindPolicyError(
        f"LAIL_HOST={host} is off-loopback and LAIL_TOKEN is unset. "
        "Bind 127.0.0.1, or set LAIL_TOKEN, or (compose only) LAIL_INSECURE_BIND=1 "
        "when host ports are published on 127.0.0.1."
    )


def token_from_headers(headers: Mapping[str, str], expected: str) -> bool:
    token = (expected or "").strip()
    if not token:
        return True
    # Starlette/FastAPI headers are case-insensitive; tests pass a plain dict.
    lower = {str(k).lower(): str(v) for k, v in headers.items()}
    auth = lower.get("authorization") or ""
    if auth.lower().startswith("bearer ") and auth[7:].strip() == token:
        return True
    if (lower.get("x-lail-token") or "") == token:
        return True
    return False
