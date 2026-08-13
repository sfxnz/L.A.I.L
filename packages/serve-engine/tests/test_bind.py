"""Bind / token policy for serve-engine (off-loopback requires LAIL_TOKEN)."""
from __future__ import annotations

import pytest

from app.bind import (
    BindPolicyError,
    allow_query_token,
    assert_safe_bind,
    cors_origins,
    is_loopback_host,
    token_from_headers,
)


def test_loopback_hosts():
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("localhost")
    assert is_loopback_host("::1")
    assert not is_loopback_host("0.0.0.0")
    assert not is_loopback_host("::")
    assert not is_loopback_host("10.0.0.5")


def test_loopback_allows_empty_token():
    assert_safe_bind("127.0.0.1", token="")
    assert_safe_bind("localhost", token="")


def test_off_loopback_requires_token():
    with pytest.raises(BindPolicyError, match="LAIL_TOKEN"):
        assert_safe_bind("0.0.0.0", token="")
    with pytest.raises(BindPolicyError, match="LAIL_TOKEN"):
        assert_safe_bind("10.0.0.5", token="")
    assert_safe_bind("0.0.0.0", token="secret")


def test_insecure_bind_escape_hatch():
    # Container-internal 0.0.0.0 with host publish pinned to loopback.
    assert_safe_bind("0.0.0.0", token="", allow_insecure=True)


def test_token_from_headers():
    assert token_from_headers({"authorization": "Bearer abc"}, "abc") is True
    assert token_from_headers({"x-lail-token": "abc"}, "abc") is True
    assert token_from_headers({"authorization": "Bearer no"}, "abc") is False
    assert token_from_headers({}, "abc") is False
    assert token_from_headers({}, "") is True


def test_cors_origins_appends_extras():
    out = cors_origins("http://lab.example:3000")
    assert "http://127.0.0.1:3000" in out
    assert "http://localhost:3000" in out
    assert "http://lab.example:3000" in out


def test_cors_origins_empty_keeps_defaults():
    out = cors_origins("")
    assert "http://127.0.0.1:3000" in out
    assert "http://lab.example:3000" not in out


def test_query_token_only_on_job_logs():
    assert allow_query_token("/api/jobs/abc/logs") is True
    assert allow_query_token("/api/serve/start") is False
    assert allow_query_token("/api/status") is False
