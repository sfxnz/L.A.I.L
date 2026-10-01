"""Live endpoint rates from vLLM counter deltas: busy-time decode, aggregate, prefill,
spec-decode acceptance, the last burst, counter resets, labelled-series summing."""
from __future__ import annotations

import pytest

from app.services import metadata


@pytest.fixture(autouse=True)
def fresh_state():
    metadata.reset_live_rate_state()
    yield
    metadata.reset_live_rate_state()


def counters(**kw) -> dict:
    base = {
        "generation_tokens_total": 4829.0,
        "ttft_sum": 6.68,
        "ttft_count": 36.0,
        "itl_sum": 67.40,
        "prefill_time_s_sum": 5.36,
        "prefill_time_s_count": 36.0,
        "prefill_tokens_sum": 1397.0,
        "requests_running": 0.0,
    }
    base.update(kw)
    return base


def step(now: float, **kw) -> dict:
    return metadata.live_token_rates(counters(**kw), now=now)


def test_first_scrape_has_every_rate_key_and_no_value():
    out = step(10.0)
    for key in metadata.LIVE_RATE_KEYS:
        assert key in out and out[key] is None, key
    assert out["last_burst"] is None and out["last_prefill"] is None


def test_short_burst_is_not_diluted_by_the_window():
    """A 64-token reply decoded in 0.9 s inside a 2 s window: per-stream decode is the real
    speed (busy time from Σ inter-token latency); the aggregate is the wall-clock average."""
    step(10.0)
    out = step(12.0, generation_tokens_total=4829.0 + 64, ttft_count=37.0, ttft_sum=6.68 + 0.08, itl_sum=67.40 + 0.9)
    assert out["decode_tok_per_s"] == 70.0  # (64 − 1 first token) / 0.9 s
    assert out["throughput_tok_per_s"] == 32.0  # 64 / 2 s
    assert out["ttft_s"] == 0.08
    assert out["rate_window_s"] == 2.0


def test_busy_with_no_token_movement_is_zero_never_a_lifetime_average():
    """Long prefill / stall: requests running, counters flat. The old code showed the lifetime
    gen_sum/decode_s (≈71 tok/s here) as if it were live."""
    step(10.0, requests_running=1.0)
    out = step(12.0, requests_running=1.0)
    assert out["decode_tok_per_s"] == 0.0
    assert out["throughput_tok_per_s"] == 0.0
    assert out["prefill_tok_per_s"] is None


def test_idle_is_none_for_decode_and_zero_throughput():
    step(10.0)
    out = step(12.0)
    assert out["decode_tok_per_s"] is None
    assert out["throughput_tok_per_s"] == 0.0


def test_concurrent_streams_decode_is_per_stream_throughput_is_total():
    step(10.0, requests_running=4.0)
    # 4 streams × 50 tok/s for the whole 2 s window: 400 tokens, Σ ITL = 4 × 2 s
    out = step(12.0, requests_running=4.0, generation_tokens_total=4829.0 + 400, itl_sum=67.40 + 8.0)
    assert out["decode_tok_per_s"] == 50.0
    assert out["throughput_tok_per_s"] == 200.0


def test_prefill_rate_only_from_requests_that_finished_in_the_window():
    step(10.0, requests_running=1.0)
    running = step(11.0, requests_running=1.0, generation_tokens_total=4829.0 + 30, itl_sum=67.40 + 0.4)
    assert running["prefill_tok_per_s"] is None
    done = step(
        12.0,
        generation_tokens_total=4829.0 + 60,
        itl_sum=67.40 + 0.8,
        prefill_time_s_count=37.0,
        prefill_time_s_sum=5.36 + 0.5,
        prefill_tokens_sum=1397.0 + 8000,  # computed tokens; cache hits are not in this counter
    )
    assert done["prefill_tok_per_s"] == 16000.0
    # held, labelled with its time, until the next request finishes
    after = step(13.0, generation_tokens_total=4829.0 + 60, itl_sum=67.40 + 0.8, prefill_time_s_count=37.0,
                 prefill_time_s_sum=5.36 + 0.5, prefill_tokens_sum=1397.0 + 8000)
    assert after["prefill_tok_per_s"] is None
    assert after["last_prefill"]["tok_per_s"] == 16000.0 and isinstance(after["last_prefill"]["at"], int)


def test_counter_reset_rebaselines_and_never_shows_a_stale_rate():
    step(10.0, requests_running=1.0)
    step(12.0, requests_running=1.0, generation_tokens_total=4829.0 + 140, itl_sum=67.40 + 2.0)
    # vLLM restarted: counters drop to a fresh engine's values while a request runs
    restarted = counters(generation_tokens_total=50.0, ttft_count=1.0, ttft_sum=0.1, itl_sum=0.7,
                         prefill_time_s_count=0.0, prefill_time_s_sum=0.0, prefill_tokens_sum=0.0,
                         requests_running=1.0)
    out = metadata.live_token_rates(restarted, now=14.0)
    assert out["decode_tok_per_s"] is None and out["throughput_tok_per_s"] is None
    assert out["last_burst"] is None and out["last_prefill"] is None
    after = metadata.live_token_rates({**restarted, "generation_tokens_total": 120.0, "itl_sum": 1.7}, now=16.0)
    assert after["decode_tok_per_s"] == 70.0 and after["throughput_tok_per_s"] == 35.0


def test_last_burst_is_kept_after_traffic_stops_and_labelled_by_time():
    step(10.0)
    step(11.0, requests_running=1.0, generation_tokens_total=4829.0 + 1, ttft_count=37.0, ttft_sum=6.68 + 0.2)
    step(12.0, requests_running=1.0, generation_tokens_total=4829.0 + 71, ttft_count=37.0, ttft_sum=6.88, itl_sum=67.40 + 1.0)
    finish = step(
        13.0, generation_tokens_total=4829.0 + 141, ttft_count=37.0, ttft_sum=6.88, itl_sum=67.40 + 2.0,
        prefill_time_s_count=37.0, prefill_time_s_sum=5.36 + 0.1, prefill_tokens_sum=1397.0 + 500,
    )
    assert finish["last_burst"] is None  # still in the burst
    idle = step(14.0, generation_tokens_total=4829.0 + 141, ttft_count=37.0, ttft_sum=6.88, itl_sum=69.40,
                prefill_time_s_count=37.0, prefill_time_s_sum=5.46, prefill_tokens_sum=1897.0)
    burst = idle["last_burst"]
    assert idle["decode_tok_per_s"] is None
    assert burst["decode_tok_per_s"] == 70.0  # 140 decode tokens over 2.0 s of steps
    assert burst["tokens"] == 141
    assert idle["last_prefill"]["tok_per_s"] == 5000.0
    assert isinstance(burst["ended_at"], int)
    later = step(16.0, generation_tokens_total=4829.0 + 141, ttft_count=37.0, ttft_sum=6.88, itl_sum=69.40,
                 prefill_time_s_count=37.0, prefill_time_s_sum=5.46, prefill_tokens_sum=1897.0)
    assert later["last_burst"] == burst


def test_spec_decode_acceptance_windowed_and_lifetime():
    base = counters(spec_drafts=1352.0, spec_draft_tokens=4056.0, spec_accepted=3492.0)
    first = metadata.live_token_rates(base, now=10.0)
    assert first["spec_accept_rate"] is None
    assert first["spec_accept_rate_lifetime"] == 0.8609
    out = metadata.live_token_rates(
        {**base, "spec_drafts": 1368.0, "spec_draft_tokens": 4104.0, "spec_accepted": 3533.0,
         "generation_tokens_total": 4829.0 + 57, "itl_sum": 67.40 + 0.8, "requests_running": 1.0},
        now=11.0,
    )
    assert out["spec_accept_rate"] == 0.8542  # 41 / 48 drafted
    assert out["spec_tokens_per_step"] == 3.56  # 1 + 41 / 16 drafts
    plain = metadata.live_token_rates(counters(), now=12.0)
    assert plain["spec_accept_rate"] is None and plain["spec_accept_rate_lifetime"] is None


def test_reset_live_rate_state_forgets_everything():
    step(10.0)
    metadata.reset_live_rate_state()
    out = step(12.0, generation_tokens_total=9999.0)
    assert out["throughput_tok_per_s"] is None


# ─── parse_prometheus ─────────────────────────────────────────────────────────

_TWO_ENGINES = """\
# HELP vllm:generation_tokens_total Number of generation tokens processed.
vllm:generation_tokens_total{engine="0",model_name="m"} 100.0
vllm:generation_tokens_total{engine="1",model_name="m"} 50.0
vllm:generation_tokens_created{engine="0",model_name="m"} 1.79e+09
vllm:num_requests_running{engine="0",model_name="m"} 2.0
vllm:num_requests_running{engine="1",model_name="m"} 1.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.25
vllm:kv_cache_usage_perc{engine="1",model_name="m"} 0.5
vllm:inter_token_latency_seconds_sum{engine="0",model_name="m"} 1.5
vllm:inter_token_latency_seconds_sum{engine="1",model_name="m"} 0.5
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 10.0
vllm:prefix_cache_hits{engine="0",model_name="m"} 999.0
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 40.0
vllm:engine_sleep_state{engine="0",model_name="m",sleep_state="awake"} 1.0
vllm:engine_sleep_state{engine="0",model_name="m",sleep_state="weights_offloaded"} 0.0
vllm:num_requests_waiting{engine="0",model_name="m"} NaN
"""


def test_parse_prometheus_sums_labelled_series_and_maxes_kv():
    m = metadata.parse_prometheus(_TWO_ENGINES, now=1.0)
    assert m["generation_tokens_total"] == 150.0  # was 50 (last series wins)
    assert m["requests_running"] == 3.0
    assert m["itl_sum"] == 2.0
    assert m["gpu_kv_cache_usage"] == 0.5  # fullest engine, never 0.75
    assert m["prefix_cache_hits"] == 10.0  # the `_total` spelling wins over the legacy name
    assert m["prefix_cache_hit_rate"] == 0.25
    assert m["sleep_state"] == "awake"
    assert "requests_waiting" not in m  # NaN dropped
    assert "gen_tok_per_s" not in m and "prompt_tok_per_s" not in m


def test_engine_kv_usage_pct_is_a_percent_with_two_decimals():
    eng = metadata.build_engine({"metrics": {"gpu_kv_cache_usage": 0.00836}}, None)
    assert eng["kv_usage_pct"] == 0.84  # 0.84 %, which the web must not read as 84 %
    assert metadata.build_engine({"metrics": {"sleep_state": "awake"}}, None)["sleep_state"] == "awake"
