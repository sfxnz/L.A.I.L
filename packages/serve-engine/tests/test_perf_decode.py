"""Decode-bench kinds, sequential waves, and tok/s math — shipped perf.py."""
from __future__ import annotations

import threading
import time

from app.services import perf


def test_completion_body_forces_full_decode_length():
    body = perf.completion_body(model="m", user_content="x", max_tokens=256)
    assert body["max_tokens"] == 256
    assert body["min_tokens"] == 256
    assert body["ignore_eos"] is True
    assert body["stream"] is True


def test_each_workload_kind_has_a_distinct_prompt_family():
    families = {k: perf.jobs_for_kind(k) for k in perf.WORKLOAD_KINDS}
    assert set(families) == {"structured", "prose", "code", "json"}
    prompts = {k: families[k][0][1] for k in families}
    labels = {k: families[k][0][0] for k in families}
    assert len(set(prompts.values())) == 4
    assert len(set(labels.values())) == 4
    assert "JSON" in prompts["json"] or "json" in prompts["json"].lower()
    assert "function" in prompts["code"].lower() or "Python" in prompts["code"]
    assert "essay" in prompts["prose"].lower() or "prose" in prompts["prose"].lower()
    assert "field" in prompts["structured"].lower() or "Fill" in prompts["structured"]


def test_unknown_kind_is_rejected():
    try:
        perf.jobs_for_kind("mixed")
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "mixed" in str(e)


def test_normalize_concurrencies_clamps_to_1_32_and_sorts():
    assert perf.normalize_concurrencies([16, 1, 16, 4, 99, 0, -1]) == [1, 4, 16]
    assert perf.normalize_concurrencies(None) == [1]
    assert perf.normalize_concurrencies([]) == [1]


def test_expand_wave_jobs_opens_exactly_n_streams():
    jobs = perf.jobs_for_kind("code")
    work = perf.expand_wave_jobs(jobs, 7)
    assert len(work) == 7
    assert all(w[0] == jobs[0][0] for w in work)
    assert all(w[1] == jobs[0][1] for w in work)


def test_run_concurrency_levels_runs_in_order():
    seen: list[int] = []

    def run_level(c: int) -> dict:
        seen.append(c)
        return {"concurrency": c}

    out = perf.run_concurrency_levels([1, 4, 16], run_level)
    assert seen == [1, 4, 16]
    assert [a["concurrency"] for a in out] == [1, 4, 16]


def test_summarize_decode_and_prefill_from_request_timing():
    r = perf.ReqResult(
        ok=True,
        wall_s=2.0,
        ttft_s=0.5,
        prompt_tokens=100,
        completion_tokens=150,
        label="prose_essay",
    )
    s = perf.summarize([r], concurrency=1)
    assert s["decode_tok_per_s_median"] == 100.0
    assert s["prefill_tok_per_s_median"] == 200.0


def test_decode_tok_s_uses_model_usage_over_first_to_last_token():
    r = perf.ReqResult(
        ok=True,
        wall_s=3.0,
        ttft_s=0.5,
        prompt_tokens=80,
        completion_tokens=150,
        label="structured_fields",
        last_s=2.0,
    )
    assert perf.decode_tok_per_s(r) == 100.0
    assert perf.prefill_tok_per_s(r) == 160.0
    s = perf.summarize([r], concurrency=1)
    assert s["decode_tok_per_s_median"] == 100.0


def test_decode_tok_s_absent_without_usage_counts():
    r = perf.ReqResult(
        ok=True,
        wall_s=2.0,
        ttft_s=0.5,
        prompt_tokens=None,
        completion_tokens=None,
        label="x",
        last_s=1.5,
    )
    assert perf.decode_tok_per_s(r) is None
    assert perf.prefill_tok_per_s(r) is None


def test_output_piece_counts_reasoning_as_model_output():
    assert perf.output_piece({"content": "hi"}) == "hi"
    assert perf.output_piece({"reasoning_content": "think"}) == "think"
    assert perf.output_piece({}) == ""


def test_summarize_does_not_invent_zero_rates_when_timing_is_missing():
    r = perf.ReqResult(
        ok=True,
        wall_s=1.0,
        ttft_s=None,
        prompt_tokens=None,
        completion_tokens=None,
        label="x",
    )
    s = perf.summarize([r], concurrency=2)
    assert s["decode_tok_per_s_median"] is None
    assert s["prefill_tok_per_s_median"] is None


def test_run_wave_uses_n_parallel_streams():
    lock = threading.Lock()
    in_flight = {"n": 0, "peak": 0}

    def fake_stream(base, model, user_content, max_tokens, label):
        with lock:
            in_flight["n"] += 1
            in_flight["peak"] = max(in_flight["peak"], in_flight["n"])
        time.sleep(0.08)
        with lock:
            in_flight["n"] -= 1
        return perf.ReqResult(
            ok=True,
            wall_s=0.2,
            ttft_s=0.05,
            prompt_tokens=10,
            completion_tokens=20,
            label=label,
        )

    summary = perf.run_wave(
        "http://example.invalid",
        "m",
        4,
        kind="json",
        stream_fn=fake_stream,
    )
    assert summary["concurrency"] == 4
    assert summary["requests"] == 4
    assert in_flight["peak"] == 4
    assert summary["decode_tok_per_s_median"] is not None
    assert summary["prefill_tok_per_s_median"] is not None
