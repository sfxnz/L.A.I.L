"""Host bind + token policy for serve-engine."""
from __future__ import annotations

import re
from typing import Mapping

_DEFAULT_CORS_ORIGINS = [
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:8787",
    "http://localhost:8787",
]

_JOB_LOGS = re.compile(r"^/api/jobs/[^/]+/logs$")


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


def cors_origins(extra: str | None = None) -> list[str]:
    """Loopback defaults plus comma-separated extras. Never replace the defaults."""
    extras = [x.strip() for x in (extra or "").split(",") if x.strip()]
    out: list[str] = []
    for origin in _DEFAULT_CORS_ORIGINS + extras:
        if origin not in out:
            out.append(origin)
    return out


def allow_query_token(path: str) -> bool:
    """Query-string tokens are only for EventSource job logs."""
    return bool(_JOB_LOGS.match(path or ""))
