"""Host bind + token policy for serve-engine."""
from __future__ import annotations

import re
from typing import Mapping
from urllib.parse import urlsplit

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


def is_cross_site_write(method: str, headers: Mapping[str, str], allow: list[str]) -> bool:
    """Cross-site write guard, independent of LAIL_TOKEN (same rule as the controller).

    A browser page on another origin can still *reach* a handler with a CORS "simple"
    request (no body, text/plain or a form) although it cannot read the reply — enough
    for POST /api/serve/stop. Such a write is refused unless it is JSON, which forces a
    preflight only the configured origins pass. The controller's proxy, curl and other
    non-browser clients send no Origin and are unaffected.
    """
    if method.upper() in {"GET", "HEAD", "OPTIONS"}:
        return False
    lower = {str(k).lower(): str(v) for k, v in headers.items()}
    origin = lower.get("origin")
    if not origin:
        return False
    if lower.get("sec-fetch-site") == "same-origin" or origin in allow:
        return False
    ct = (lower.get("content-type") or "").split(";")[0].strip().lower()
    return ct != "application/json"


_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def _hostname(value: str) -> str:
    h = value.strip().lower()
    if h.startswith("["):
        end = h.find("]")
        return h[1:end] if end > 0 else h[1:]
    return re.sub(r":\d+$", "", h)


def is_untrusted_host(headers: Mapping[str, str], allow: list[str]) -> bool:
    """DNS-rebinding guard for the token-less setup (same rule as the controller).

    Allowed: IP literals, localhost / *.localhost, single-label names (docker service
    names, `spark1` — no public DNS name rebinds to them) and the configured origins'
    hosts. Any other dotted name is a page that rebound its own domain to loopback.
    """
    lower = {str(k).lower(): str(v) for k, v in headers.items()}
    trusted = {(urlsplit(o).hostname or "").lower() for o in allow}
    for key in ("host", "x-forwarded-host"):
        for part in (lower.get(key) or "").split(","):
            name = _hostname(part)
            if not name or name in trusted or name == "localhost" or name.endswith(".localhost"):
                continue
            if "." not in name or ":" in name or _IPV4.match(name):
                continue
            return True
    return False
