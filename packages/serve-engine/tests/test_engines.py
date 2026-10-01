"""Engine adapters: argv per engine (single node + TP ranks), detection, /metrics → contract.

Fixtures `{sglang,llamacpp,tensorfold}_metrics_*.prom` and `tensorfold_health.json` are
synthesised from the exact metric names, labels and update points in upstream source
(sgl-project/sglang @ b51d4a04, ggml-org/llama.cpp @ 32dd62ee, ashhart/TensorFold v0.6.0
@ c4646171) — none of these engines could be launched on the live lab while it serves.
"""
from __future__ import annotations

import json
import shlex
import threading
from pathlib import Path

import httpx
import pytest

from app.services import autoconfig as ac
from app.services import engines, metadata, node_probe, serve
from app.services.engines import ENGINES, Rank, ServeSpec
from app.services.jobs import JobCancelled

FIX = Path(__file__).parent / "fixtures"
HEAD = {"id": "spark1", "qsfp_ip": "10.100.8.1", "qsfp_if": "enp1s0f1np1", "local": True}
WORKER = {"id": "spark2", "qsfp_ip": "10.100.8.2", "ssh_host": "spark2"}


def _fix(name: str) -> str:
    return (FIX / name).read_text()


def _launch(name: str, **spec_kw):
    eng = engines.get(name)
    spec = ServeSpec(model="org/Model-NVFP4", port=eng.default_port, tensor_parallel_size=2, **spec_kw)
    return eng, serve.build_multi_node_launch(
        engine=eng, spec=spec, image=eng.image(), env_list=["NCCL_SOCKET_IFNAME=enp1s0f1np1"],
        head=HEAD, workers=[WORKER], hf_token="hf_realtoken1234567890",
    )


# ─── registry / detection ─────────────────────────────────────────────────────


def test_registry_has_four_engines_with_distinct_ports_all_probed_by_discovery():
    assert set(ENGINES) == {"vllm", "sglang", "llamacpp", "tensorfold"}
    ports = {e.default_port for e in ENGINES.values()}
    assert len(ports) == 4  # TensorFold moved off its upstream 8080 (llama.cpp's)
    assert ports <= set(node_probe.DEFAULT_SERVE_PORTS)
    assert engines.get("llama.cpp") is ENGINES["llamacpp"] and engines.get(None) is engines.VLLM
    with pytest.raises(ValueError):
        engines.get("ollama")


@pytest.mark.parametrize(
    "models, text, hint, want",
    [
        ([{"id": "m", "owned_by": "sglang"}], None, None, "sglang"),
        ([{"id": "m", "owned_by": "llamacpp"}], None, "vllm", "llamacpp"),
        ([{"id": "m", "owned_by": "tensorfold"}], None, None, "tensorfold"),
        ([{"id": "m", "owned_by": "vllm"}], None, None, "vllm"),
        ([], "# HELP x\ntensorfold:requests_running 0\n", None, "tensorfold"),
        ([], "llamacpp:requests_processing 0\n", None, "llamacpp"),
        ([], None, "sglang", "sglang"),
        ([], "foo 1\n", None, None),
    ],
)
def test_detect_engine_from_owned_by_then_metrics_prefix_then_hint(models, text, hint, want):
    assert engines.detect(models, text, hint) == want


# ─── argv: single node ────────────────────────────────────────────────────────


def test_single_node_commands_bind_loopback_label_and_enable_metrics():
    spec = ServeSpec(model="org/Model-NVFP4", port=0, util=0.7, max_model_len=65536, kv_cache_dtype="fp8",
                     trust_remote_code=True, tool_call_parser="qwen3_coder", max_num_seqs=4,
                     extra_flags="--port 1 --enable-metrics --chunked-prefill-size 4096")
    sgl = engines.get("sglang")
    cmd = serve.build_single_node_docker_cmd(
        engine=sgl, spec=ServeSpec(**{**spec.__dict__, "port": 30000}), image="img:sgl",
        env_list=[], container=sgl.container, hf_token="hf_x")
    j = shlex.join(cmd)
    assert "127.0.0.1:30000:30000" in cmd and "lail.engine=sglang" in cmd
    assert cmd[cmd.index("--entrypoint") + 1] == "python3" and cmd[cmd.index("img:sgl") + 1:][:2] == ["-m", "sglang.launch_server"]
    for frag in ("--model-path org/Model-NVFP4", "--tp-size 1", "--mem-fraction-static 0.7", "--context-length 65536",
                 "--kv-cache-dtype fp8_e4m3", "--max-running-requests 4", "--tool-call-parser qwen3_coder",
                 "--trust-remote-code", "--chunked-prefill-size 4096"):
        assert frag in j, frag
    assert j.count("--enable-metrics") == 1 and j.count("--port") == 1  # extras never duplicate owned flags
    # The token is passed by name (docker reads it from its environment), never on argv.
    assert "hf_x" not in j and "-e HF_TOKEN -e HUGGING_FACE_HUB_TOKEN" in j

    lc = engines.get("llamacpp")
    cmd = serve.build_single_node_docker_cmd(
        engine=lc, spec=ServeSpec(model="unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL", port=8080, max_model_len=32768,
                                  extra_flags="-ngl 99 --flash-attn on"),
        image=lc.image(), env_list=[], container=lc.container)
    assert "--entrypoint" not in cmd  # the image's ENTRYPOINT is llama-server
    tail = cmd[cmd.index(lc.image()) + 1:]
    assert tail[:2] == ["-hf", "unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL"]
    assert "--metrics" in tail and tail[tail.index("-c") + 1] == "32768" and "-ngl" in tail
    assert "127.0.0.1:8080:8080" in cmd

    tf = engines.get("tensorfold")
    cmd = serve.build_single_node_docker_cmd(
        engine=tf, spec=ServeSpec(model="zai-org/GLM-5.3-Flash", port=8090), image=tf.image(),
        env_list=[], container=tf.container)
    assert "--network" in cmd and cmd[cmd.index("--network") + 1] == "host" and "-p" not in cmd
    assert cmd[cmd.index("--ipc") + 1] == "host"
    assert f"{engines.tensorfold.CACHE_DIR}:/root/.cache" in cmd  # pip + kernel builds survive Stop
    i = cmd.index(tf.image())
    assert cmd[i + 1] == "-lc" and "pip install" in cmd[i + 2] and "command -v tensorfold" in cmd[i + 2]
    assert "env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN pip install" in cmd[i + 2]  # the install never sees the token
    assert "TensorFold.git@c4646171139ee8a3c38103eaa1699dad226ec12b" in cmd[i + 2]  # the verified commit
    argv = cmd[cmd.index("--") + 1:]
    assert argv[:3] == ["tensorfold", "serve", "zai-org/GLM-5.3-Flash"]
    assert argv[argv.index("--host") + 1] == "127.0.0.1" and "--no-update-check" in argv
    assert "--context" not in argv  # unset: TensorFold sizes the affordable native window


def test_llamacpp_refuses_tensor_parallel():
    with pytest.raises(ValueError):
        engines.get("llamacpp").argv(ServeSpec(model="m", port=8080), Rank(0, 2, "10.0.0.1", 0))


# ─── argv: TP ranks ───────────────────────────────────────────────────────────


def test_sglang_ranks_use_nnodes_node_rank_and_dist_init_addr():
    eng, launch = _launch("sglang", util=0.8)
    head = shlex.join(launch["head"]["cmd"])
    worker = shlex.join(launch["workers"][0]["cmd"])
    for rank, cmd in ((0, head), (1, worker)):
        assert f"--nnodes 2 --node-rank {rank} --dist-init-addr 10.100.8.1:{eng.master_port}" in cmd
        assert "--tp-size 2" in cmd and "--enable-metrics" in cmd
    assert "--host 127.0.0.1" in head and "0.0.0.0" not in head
    assert launch["head"]["container"] == "lail-sglang-n0" and launch["workers"][0]["container"] == "lail-sglang-n1"


def test_tensorfold_ranks_tp2_rank_master_and_rdma_devices():
    eng, launch = _launch("tensorfold", max_model_len=32768)
    for rank, entry in ((0, launch["head"]), (1, launch["workers"][0])):
        cmd = entry["cmd"]
        argv = cmd[cmd.index("--") + 1:]
        assert argv[argv.index("--tp") + 1] == "2" and argv[argv.index("--rank") + 1] == str(rank)
        assert argv[argv.index("--master") + 1] == "10.100.8.1" and argv[argv.index("--master-port") + 1] == "29551"
        assert argv[argv.index("--context") + 1] == "32768"  # both ranks must agree
        assert "/dev/infiniband" in cmd and "IPC_LOCK" in cmd and "memlock=-1:-1" in cmd
        assert f"{engines.tensorfold.CACHE_DIR}:/root/.cache" in cmd  # engine docker_opts reach every rank
        assert cmd.count("--ipc") == 1
    assert engines.get("tensorfold").max_tp == 2


def test_every_rank_gets_the_real_hf_token_never_on_a_command_line():
    """ENG-8: the head got no token on TP=2 (gated models failed). Every rank now asks
    docker for HF_TOKEN by name; the value never appears in any argv (`ps` shows argv)."""
    for name in ("vllm", "sglang", "tensorfold"):
        _eng, launch = _launch(name)
        for cmd in (launch["head"]["cmd"], launch["workers"][0]["cmd"]):
            j = shlex.join(cmd)
            assert "-e HF_TOKEN -e HUGGING_FACE_HUB_TOKEN" in j and "hf_realtoken" not in j, name


def test_launch_hands_the_token_over_stdin_and_env_only(monkeypatch):
    eng, launch = _launch("sglang")
    seen: list[tuple[list[str], dict]] = []
    monkeypatch.setattr(serve.subprocess, "run", lambda cmd, **kw: seen.append((list(cmd), kw)) or _Proc())
    monkeypatch.setattr(serve, "wait_ready", lambda *a, **k: None)
    serve._launch_multi_node(launch, engine=eng, hf_token="hf_realtoken1234567890")
    worker = next((c, kw) for c, kw in seen if c[0] == "ssh" and "docker run" in c[-1])
    head = next((c, kw) for c, kw in seen if c[:2] == ["docker", "run"])
    for cmd, _kw in (worker, head):
        assert not any("hf_realtoken" in a for a in cmd)
    assert worker[1]["input"] == "hf_realtoken1234567890\n"
    assert worker[0][-1].startswith("IFS= read -r HF_TOKEN && export HF_TOKEN")
    assert head[1]["env"]["HF_TOKEN"] == head[1]["env"]["HUGGING_FACE_HUB_TOKEN"] == "hf_realtoken1234567890"


def test_preview_shows_the_process_command_per_rank():
    eng = engines.get("sglang")
    ranks = [Rank(r, 2, "10.0.0.1", eng.master_port) for r in range(2)]
    procs = engines.preview(eng, ServeSpec(model="m", port=30000, tensor_parallel_size=2), ranks)
    assert [p["rank"] for p in procs] == [0, 1]
    assert procs[1]["argv"].startswith("python3 -m sglang.launch_server") and "--node-rank 1" in procs[1]["argv"]


# ─── /metrics (+ /health) → the shared contract ───────────────────────────────


def test_sglang_metrics_live_decode_from_itl_histogram_kv_fraction_and_spec_gauges():
    metadata.reset_live_rate_state()
    t0 = metadata.parse_prometheus(_fix("sglang_metrics_t0.prom"), 10.0, engine="sglang")
    assert t0["requests_running"] == 2 and t0["requests_waiting"] == 1
    assert t0["gpu_kv_cache_usage"] == 0.12  # a 0-1 fraction already
    assert t0["kv_cache_size_tokens"] == 100000
    assert t0["prompt_tokens_total"] == 5000  # summed across is_streaming
    t1 = metadata.parse_prometheus(_fix("sglang_metrics_t1.prom"), 11.0, engine="sglang")
    # 180 tokens after the first in 2.0 s of per-stream busy time → 90 tok/s per stream;
    # the finish-only generation_tokens_total (unchanged) never drives the live rate.
    assert t1["decode_tok_per_s"] == 90.0
    # Aggregate throughput is the scheduler's gen_throughput gauge (every request, streamed
    # or not), not the streamed-chunk count.
    assert t1["throughput_tok_per_s"] == 176.4
    assert t1["spec_accept_rate"] == 0.71 and t1["spec_tokens_per_step"] == 3.13
    eng = metadata.build_engine({"engine": "sglang", "metrics": t1, "models": [{"id": "m", "max_model_len": 65536}]}, None)
    assert eng["name"] == "sglang" and eng["kv_usage_pct"] == 12.0 and eng["kv_capacity_tokens"] == 100000


def test_window_latency_percentiles_from_histogram_buckets():
    """ENG-9: ITL p50/p95 and TTFT p50 over the window, from bucket deltas (never lifetime)."""
    inf = float("inf")
    # vLLM's live ITL edges: 2455 tokens ≤ 50 ms, 740 more ≤ 75 ms, 4 ≤ 100 ms (window delta).
    delta = {0.01: 0.0, 0.025: 0.0, 0.05: 2455.0, 0.075: 3195.0, 0.1: 3199.0, inf: 3199.0}
    assert metadata.bucket_quantile(0.5, delta) == 0.0413  # 0.025 + 0.025 × 1599.5 / 2455
    assert metadata.bucket_quantile(0.95, delta) == 0.0697  # 0.05 + 0.025 × 584.05 / 740
    assert metadata.bucket_quantile(0.5, {0.05: 0.0, inf: 0.0}) is None
    assert metadata.bucket_quantile(0.99, {0.05: 1.0, inf: 10.0}) == 0.05  # beyond the last edge

    metadata.reset_live_rate_state()
    t0 = metadata.parse_prometheus(_fix("sglang_metrics_t0.prom"), 10.0, engine="sglang")
    assert t0["itl_p50_s"] is None  # one scrape is no window
    t1 = metadata.parse_prometheus(_fix("sglang_metrics_t1.prom"), 11.0, engine="sglang")
    # 180 new tokens, all in the ≤ 20 ms bucket; no new first token in the window.
    assert t1["itl_p50_s"] == 0.01 and t1["itl_p95_s"] == 0.019 and t1["ttft_p50_s"] is None
    assert not any(isinstance(k, float) for k in t1)  # buckets stay internal


def test_sglang_non_streaming_reply_is_not_read_as_zero_decode():
    """A non-streaming reply reaches the tokenizer as one chunk: TTFT counts it once, the
    ITL histogram never moves. Upstream undercounts those tokens in the histograms, so the
    live numbers come from the scheduler gauge and per-stream decode reads as no-reading."""
    metadata.reset_live_rate_state()
    t1 = _fix("sglang_metrics_t1.prom")
    metadata.parse_prometheus(t1, 10.0, engine="sglang")
    one_chunk = t1.replace('is_streaming="true"} 10.0', 'is_streaming="true"} 11.0')
    out = metadata.parse_prometheus(one_chunk, 11.0, engine="sglang")
    assert out["requests_running"] == 2
    assert out["decode_tok_per_s"] is None  # not 0.0: tokens are moving, just not streamed
    assert out["throughput_tok_per_s"] == 176.4
    idle = metadata.parse_prometheus(one_chunk.replace("} 2.0\n", "} 0.0\n", 1), 12.0, engine="sglang")
    # The gauge holds for 30 s after decode stops upstream; never shown once idle.
    assert idle["requests_running"] == 0 and idle["throughput_tok_per_s"] == 0.0


def test_llamacpp_rates_only_when_requests_finish():
    metadata.reset_live_rate_state()
    t0 = _fix("llamacpp_metrics_t0.prom")
    metadata.parse_prometheus(t0, 10.0, engine="llamacpp")
    running = metadata.parse_prometheus(t0, 11.0, engine="llamacpp")
    # Decoding, but llama.cpp folds generation counters in at slot reset: no live number,
    # and never a fake 0.
    assert running["requests_running"] == 1 and running["decode_tok_per_s"] is None
    done = metadata.parse_prometheus(_fix("llamacpp_metrics_t1.prom"), 12.0, engine="llamacpp")
    assert done["decode_tok_per_s"] is None and done["throughput_tok_per_s"] is None
    assert done["prefill_tok_per_s"] == 1280.0  # 512 computed prompt tokens in 0.4 s
    assert done["spec_accept_rate"] == 0.75
    idle = metadata.parse_prometheus(_fix("llamacpp_metrics_t1.prom"), 13.0, engine="llamacpp")
    assert idle["last_burst"]["decode_tok_per_s"] == 80.0  # 128 tokens / 1.6 s of generation time
    assert idle["last_burst"]["tokens"] == 128
    assert metadata.build_engine({"metrics": idle}, None)["kv_usage_pct"] is None  # no KV gauge upstream


def test_tensorfold_live_decode_from_health_and_exact_last_burst():
    metadata.reset_live_rate_state()
    health = json.loads(_fix("tensorfold_health.json"))
    m0, m2 = _fix("tensorfold_metrics_t0.prom"), _fix("tensorfold_metrics_t2.prom")
    metadata.parse_prometheus(m0, 10.0, engine="tensorfold", health=health["t0"])
    live = metadata.parse_prometheus(m0, 11.0, engine="tensorfold", health=health["t1"])
    # /metrics counters do not move mid-stream; /health completion tokens do.
    assert live["decode_tok_per_s"] == 60.0 and live["throughput_tok_per_s"] == 60.0
    assert live.get("gpu_kv_cache_usage") is None  # kv_cache_usage_ratio is per-stream fill, not a pool
    done = metadata.parse_prometheus(m2, 12.0, engine="tensorfold", health=health["t2"])
    assert done["prefill_tok_per_s"] == 4000.0  # 800 uncached prompt tokens in 0.2 s
    assert done["spec_accept_rate"] == 0.75
    assert done["spec_tokens_per_step"] is None  # rounds_total's meaning is unverified
    assert done["ttft_s"] == 0.25
    idle = metadata.parse_prometheus(m2, 13.0, engine="tensorfold", health=health["t2"])
    # 100 tokens − 1 first token over the request's own 1.6 s of decode time
    assert idle["last_burst"]["decode_tok_per_s"] == 61.87 and idle["last_burst"]["tokens"] == 100
    eng = metadata.build_engine({"engine": "tensorfold", "metrics": idle, "models": [{"id": "m"}]}, None)
    assert eng["max_model_len"] == 131072 and eng["kv_usage_pct"] is None


def test_probe_endpoint_detects_sglang_and_reads_server_info_version(monkeypatch):
    metadata.reset_live_rate_state()

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "m", "owned_by": "sglang", "max_model_len": 65536}]})
        if path == "/health":
            return httpx.Response(200, text="")
        if path == "/server_info":
            return httpx.Response(200, json={"version": "0.5.20", "tp_size": 1})
        if path == "/metrics":
            return httpx.Response(200, text=_fix("sglang_metrics_t0.prom"))
        return httpx.Response(404)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await metadata.probe_endpoint("http://x:30000", client=c, engine="sglang")

    import asyncio

    out = asyncio.run(run())
    assert out["engine"] == "sglang" and out["version"] == {"version": "0.5.20"}
    assert out["metrics"]["requests_running"] == 2 and out["metrics"]["gpu_kv_cache_usage"] == 0.12


def test_envelope_records_the_engine_that_answered(monkeypatch):
    probe = {"engine": "tensorfold", "models": [{"id": "m"}], "metrics": {}, "containers": [], "hardware": {"ram_gib": 121.7}}
    env = metadata.build_envelope(model_id="m", probe=probe)
    assert env["engine"]["name"] == "tensorfold" and env["hardware"] == {"ram_gib": 121.7}


# ─── containers: discovery and stop predicate ─────────────────────────────────


def _ps(*rows: tuple[str, ...]) -> str:
    return "".join("\t".join(r) + "\n" for r in rows)


def test_serve_containers_are_found_by_label_command_or_image_not_by_model_words():
    text = _ps(
        ("lail-tensorfold-n1", "Up 3 minutes", "nvcr.io/nvidia/pytorch:26.07-py3", "a", "tensorfold", '"bash -lc …"'),
        ("custom", "Up 1 hour", "glm53-sm121-v13", "b", "", '"python3 -m sglang.launch_server --model-path x"'),
        ("my-llama", "Up 1 hour", "ghcr.io/ggml-org/llama.cpp:server-cuda13", "c", "", '"/app/llama-server -hf x"'),
        ("ollama", "Up 2 days", "ollama/ollama:latest", "d", "", '"/bin/ollama serve"'),
        ("open-webui-llama", "Up 2 days", "ghcr.io/open-webui/open-webui:main", "e", "", '"bash start.sh"'),
        ("xray", "Up 2 days", "teddysun/xray", "f", "", '"/usr/bin/xray"'),
        ("trainer", "Up 2 days", "nvcr.io/nvidia/pytorch:26.07-py3", "g", "", '"torchrun --node-rank 1 train.py"'),
    )
    found = {c["name"]: c for c in node_probe.parse_docker_ps(text)}
    assert set(found) == {"lail-tensorfold-n1", "custom", "my-llama"}
    assert found["lail-tensorfold-n1"]["engine"] == "tensorfold" and found["lail-tensorfold-n1"]["lail"] is True
    assert found["custom"]["engine"] == "sglang" and found["my-llama"]["engine"] == "llamacpp"


def test_headless_ranks_of_sglang_and_tensorfold_are_serving_workers():
    containers = [
        {"name": "lail-tensorfold-n1", "status": "Up 3 minutes", "engine": "tensorfold"},
        {"name": "lail-sglang-n1", "status": "Up 3 minutes", "engine": "sglang"},
    ]
    inspect = json.dumps([
        {"Name": "/lail-tensorfold-n1", "Config": {"Cmd": ["-lc", "x", "--", "tensorfold", "serve", "m", "--tp", "2", "--rank", "1"]}},
        {"Name": "/lail-sglang-n1", "Config": {"Cmd": ["-m", "sglang.launch_server", "--tp-size", "2", "--node-rank", "1"]}},
    ])
    node_probe.enrich_containers(containers, inspect)
    for c in containers:
        assert c["node_rank"] == 1 and c["headless"] is True and c["tensor_parallel_size"] == 2
    assert node_probe.rank_from_container_name("lail-sglang-n1") == 1


# ─── multi-node launch: order, stale ranks, teardown ──────────────────────────


class _Proc:
    def __init__(self, rc=0, out=""):
        self.returncode, self.stdout, self.stderr = rc, out, ""


def test_multinode_launch_records_state_first_clears_stale_ranks_and_tears_down_on_failure(monkeypatch, tmp_path):
    eng, launch = _launch("vllm")
    calls: list[str] = []
    state_when_worker_started: list[bool] = []

    def fake_run(cmd, **kw):
        line = " ".join(cmd)
        calls.append(line)
        if "docker run" in line and cmd[0] == "ssh":
            state_when_worker_started.append(serve._MULTINODE_STATE.exists())
            # the real token, on stdin — never "HF_TOKEN=***" and never on the command line
            assert kw.get("input") == "hf_realtoken1234567890\n" and "hf_realtoken" not in line
        return _Proc()

    monkeypatch.setattr(serve.subprocess, "run", fake_run)
    monkeypatch.setattr(serve, "_ensure_image_present", lambda *a, **k: None)

    def dies(*a, **k):
        raise RuntimeError("vLLM worker spark-vllm-n1 on spark2 exited (exited 1)")

    monkeypatch.setattr(serve, "wait_ready", dies)
    with pytest.raises(RuntimeError, match="exited"):
        serve._launch_multi_node({**launch, "image": ""}, engine=eng, hf_token="hf_realtoken1234567890")
    assert state_when_worker_started == [True]  # Stop can find the ranks while they load
    rm_before = [i for i, c in enumerate(calls) if "docker rm -f spark-vllm-n" in c]
    first_run = next(i for i, c in enumerate(calls) if "docker run" in c)
    assert len([i for i in rm_before if i < first_run]) == 2  # missed-2: stale head + worker removed first
    # teardown: both ranks removed again after the failure, state file gone
    assert sum(1 for c in calls[first_run:] if "docker rm -f spark-vllm-n" in c) == 2
    assert not serve._MULTINODE_STATE.exists()


def test_wait_ready_fails_fast_on_a_dead_remote_rank(monkeypatch):
    seen: list[str] = []

    def fake_run(cmd, **kw):
        line = " ".join(cmd)
        seen.append(line)
        if "docker inspect" in line:
            return _Proc(0, "exited 1" if cmd[0] == "ssh" else "running 0")
        return _Proc()

    monkeypatch.setattr(serve.subprocess, "run", fake_run)
    monkeypatch.setattr(serve, "_tail_logs", lambda *a, **k: "NCCL error: unhandled system error")
    monkeypatch.setattr(serve, "_answers", lambda url: False)
    with pytest.raises(RuntimeError, match="worker spark-vllm-n1 on spark2 exited"):
        serve.wait_ready(engines.VLLM, 8000, container="spark-vllm-n0", remote=[("spark2", "spark-vllm-n1")], timeout_s=60)


def test_wait_ready_honours_cancel_and_reports_progress(monkeypatch):
    cancel = threading.Event()
    progress: list[tuple[float, str]] = []
    monkeypatch.setattr(serve, "_container_dead", lambda name, host=None: None)
    monkeypatch.setattr(serve, "_answers", lambda url: False)
    monkeypatch.setattr(serve, "_tail_logs", lambda *a, **k: "Loading safetensors checkpoint shards:  42%")
    monkeypatch.setattr(serve, "READY_POLL_S", 0.01)

    def report(p, msg):
        progress.append((p, msg))
        cancel.set()

    with pytest.raises(JobCancelled):
        serve.wait_ready(engines.get("sglang"), 30000, container="lail-sglang", timeout_s=60,
                         progress=report, cancel=cancel)
    assert progress and "42%" in progress[0][1] and 0.35 <= progress[0][0] < 0.95


@pytest.mark.parametrize(
    "rc,out,err,want",
    [
        (0, "running 0", "", None),
        (0, "exited 1", "", "exited 1"),
        (1, "", "Error: No such object: lail-sglang", "missing"),
        (1, "", "permission denied while trying to connect to the Docker daemon socket", None),
        (255, "", "ssh: connect to host spark2 port 22: Connection timed out", None),
    ],
)
def test_container_dead_is_a_verdict_only_when_docker_says_so(monkeypatch, rc, out, err, want):
    """A daemon hiccup or ssh failure must not abort (and tear down) a 30-minute load."""
    def fake_run(cmd, **kw):
        p = _Proc(rc, out)
        p.stderr = err
        return p

    monkeypatch.setattr(serve.subprocess, "run", fake_run)
    assert serve._container_dead("lail-sglang") == want
    assert serve._container_dead("lail-sglang-n1", "spark2") == want


def test_cancel_during_the_image_pull_never_touches_the_running_serve(monkeypatch):
    """Cancel while a 20-30 GB image pulls: the serve that is up stays up, nothing is run."""
    cancel = threading.Event()
    calls: list[str] = []
    stopped: list[bool] = []
    monkeypatch.setattr(serve.subprocess, "run", lambda cmd, **kw: calls.append(" ".join(cmd)) or _Proc(1))
    monkeypatch.setattr(serve, "_resolve_hf_token_for_container", lambda: "")
    monkeypatch.setattr(serve, "stop_all", lambda **k: stopped.append(True) or {"ok": True})
    monkeypatch.setattr(serve, "stop_multi_node", lambda **k: stopped.append(True) or {"ok": True})
    # Nodes carry ram_gib: without it the fit gate reads this host's MemTotal (fails on a 16 GiB runner).
    monkeypatch.setattr(ac, "_cluster_topology", _two_spark_topo)
    monkeypatch.setattr(ac, "estimate_weights_gib", lambda *a, **k: 20.0)
    monkeypatch.setattr(ac, "load_local_fallback", lambda m: {"config": None})

    def pull(argv, *, what, cancel=None, **kw):
        cancel.set()  # the operator presses Cancel mid-pull
        raise JobCancelled(f"cancelled during {what}")

    monkeypatch.setattr(serve, "_run_watched", pull)
    for tp in (1, 2):
        with pytest.raises(JobCancelled):
            serve.serve_model(model="org/m", engine="sglang", tensor_parallel_size=tp, cancel=cancel)
        cancel.clear()
    assert not stopped and not any("docker run" in c for c in calls)

    # Cancel that lands after the pull finished (the download step, the token check) is
    # also honoured before Stop.
    monkeypatch.setattr(serve, "_ensure_image_present", lambda *a, **k: cancel.set())
    with pytest.raises(JobCancelled):
        serve.serve_model(model="org/m", engine="sglang", cancel=cancel)
    assert not stopped


def test_run_watched_kills_the_pull_on_cancel_and_on_timeout():
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    with pytest.raises(JobCancelled):
        serve._run_watched(["sleep", "30"], what="docker pull x", cancel=cancel)
    with pytest.raises(TimeoutError):
        serve._run_watched(["sleep", "30"], what="docker pull x", timeout_s=0.5)
    assert serve._run_watched(["sh", "-c", "echo pulled; exit 3"], what="x") == (3, "pulled\n")


def test_serve_cancel_removes_the_loading_container(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(serve.subprocess, "run", lambda cmd, **kw: calls.append(" ".join(cmd)) or _Proc())
    monkeypatch.setattr(serve, "_ensure_image_present", lambda *a, **k: None)
    monkeypatch.setattr(serve, "_resolve_hf_token_for_container", lambda: "")
    monkeypatch.setattr(serve, "stop_all", lambda **k: {"ok": True})
    head = dict(HEAD, ram_gib=121.7)  # explicit RAM: the fit gate must not read this host's MemTotal
    monkeypatch.setattr(ac, "_cluster_topology", lambda: {"nodes": 1, "node_list": [head], "head": head, "workers": []})
    monkeypatch.setattr(ac, "estimate_weights_gib", lambda *a, **k: 20.0)
    monkeypatch.setattr(ac, "load_local_fallback", lambda m: {"config": None})

    def cancelled(*a, **k):
        raise JobCancelled("cancelled while the model was loading")

    monkeypatch.setattr(serve, "wait_ready", cancelled)
    with pytest.raises(JobCancelled):
        serve.serve_model(model="org/m", engine="sglang")
    run = next(c for c in calls if c.startswith("docker run"))
    assert "--name lail-sglang" in run and "127.0.0.1:30000:30000" in run
    assert calls[-1] == "docker rm -f lail-sglang"


def test_serve_reports_fields_an_engine_cannot_translate():
    tf = engines.get("tensorfold")
    spec = ServeSpec(model="m", port=8090, util=0.8, moe_backend="marlin", mtp=True, max_model_len=8192)
    assert serve.ignored_fields(tf, spec) == ["util", "moe_backend", "mtp"]
    assert serve.ignored_fields(engines.VLLM, spec) == []


def test_serve_refuses_tp_beyond_what_the_engine_runs():
    with pytest.raises(RuntimeError, match="at most TP=1"):
        serve.serve_model(model="org/m", engine="llamacpp", tensor_parallel_size=2)
    with pytest.raises(RuntimeError, match="at most TP=2"):
        serve.serve_model(model="org/m", engine="tensorfold", tensor_parallel_size=3)


# ─── recommend: placement → engine form + argv ────────────────────────────────


def _two_spark_topo():
    nodes = [dict(HEAD, ram_gib=121.7, online=True), dict(WORKER, ram_gib=121.7, online=True, local=False)]
    return {"nodes": 2, "node_list": nodes, "head": nodes[0], "workers": nodes[1:], "fabric_ok": True, "available": True}


def test_recommend_tensorfold_glm_needs_two_ranks_and_previews_both(monkeypatch):
    monkeypatch.setattr(ac, "fetch_hf_card", lambda *a, **k: {"config": None})
    monkeypatch.setattr(ac, "load_local_fallback", lambda m: {"config": None})
    # 60 GiB fits one Spark, but GLM-5.3-Flash runs on 2 ranks on CUDA.
    monkeypatch.setattr(ac, "estimate_weights_gib", lambda *a, **k: 60.0)
    monkeypatch.setattr(ac, "_cluster_topology", _two_spark_topo)
    monkeypatch.setattr(ac, "_ib_hca_for_iface", lambda iface: "rocep1s0f1")
    rec = ac.recommend("zai-org/GLM-5.3-Flash-NVFP4", backend="tensorfold")
    assert rec["engine"] == "tensorfold" and rec["serve_blocked"] is False
    cfg = rec["config"]
    assert cfg["tensor_parallel_size"] == 2 and cfg["port"] == 8090
    assert cfg.get("max_model_len") is None  # TensorFold sizes the window
    assert "NCCL_SOCKET_IFNAME=enp1s0f1np1" in cfg["docker_env"]
    assert [(p["rank"], p.get("node")) for p in rec["processes"]] == [(0, "spark1"), (1, "spark2")]
    assert "--rank 1 --master 10.100.8.1 --master-port 29551" in rec["processes"][1]["argv"]


def test_recommend_sglang_two_sparks_emits_a_command_per_rank(monkeypatch):
    """ENG-4: TP=2 across one-GPU Sparks needs --nnodes/--node-rank/--dist-init-addr per rank."""
    monkeypatch.setattr(ac, "fetch_hf_card", lambda *a, **k: {"readme": None, "config": None})
    monkeypatch.setattr(ac, "load_local_fallback", lambda m: {"config": None, "readme": None})
    monkeypatch.setattr(ac, "fetch_cookbook_text", lambda *a, **k: (None, "offline"))
    monkeypatch.setattr(ac, "estimate_weights_gib", lambda *a, **k: 160.0)
    monkeypatch.setattr(ac, "_cluster_topology", _two_spark_topo)
    monkeypatch.setattr(ac, "_ib_hca_for_iface", lambda iface: None)
    rec = ac.recommend("org/Big-Model-NVFP4", backend="sglang")
    assert rec["config"]["tensor_parallel_size"] == 2
    assert 0 < rec["config"]["util"] < 1  # --mem-fraction-static from placement
    ranks = rec["processes"]
    assert [p["rank"] for p in ranks] == [0, 1] and ranks[1]["node"] == "spark2"
    assert "--nnodes 2 --node-rank 1 --dist-init-addr 10.100.8.1:" in ranks[1]["argv"]
    assert "--enable-metrics" in ranks[0]["argv"]


def test_overlay_engine_key_scopes_a_recipe_to_its_engine(monkeypatch, tmp_path):
    import app.config as cfgmod

    (tmp_path / "serve_overlays.json").write_text(json.dumps([{
        "match": {"all": ["futuremodel"]}, "family_key": "future_sgl", "label": "Future on SGLang",
        "engine": "sglang", "config": {"extra_flags": "--attention-backend triton", "util": 0.6}, "rationale": [],
    }]))
    monkeypatch.setattr(cfgmod, "DATA_DIR", tmp_path)
    assert ac._family_overlay("org/FutureModel", {}) is None  # not a vLLM recipe
    assert ac._family_overlay("org/FutureModel", {}, engine="sglang")["family_key"] == "future_sgl"
