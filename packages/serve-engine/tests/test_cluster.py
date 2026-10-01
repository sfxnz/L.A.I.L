"""Default cluster is one local node; remotes are opt-in and offline when unreachable."""
from __future__ import annotations

import json

import pytest

from app.services import autoconfig as ac
from app.services import cluster, metadata, node_probe
from app.services.autoconfig import plan_placement


def test_glm_flash_container_counts_as_a_serve():
    assert metadata.is_serve_container("glm53-flash-nvfp4", "glm53-sm121-v11")
    assert metadata.is_serve_container("spark-vllm-n1", "vllm/vllm-openai:latest")
    assert not metadata.is_serve_container("conduit", "matrixconduit/matrix-conduit:latest")
    # one container filter for local and remote (the remote runs node_probe itself)
    assert metadata.is_serve_container is node_probe.is_serve_container
    assert cluster._container_serve_family("glm53-flash-nvfp4") == "glm53-flash-nvfp4"
    head = {
        "tensor_parallel_size": 2,
        "containers": [{"name": "glm53-flash-nvfp4", "status": "Up 15 hours"}],
    }
    worker = {
        "containers": [{"name": "glm53-flash-nvfp4", "status": "Up 15 hours"}],
    }
    assert cluster._worker_aligned_with_head(worker, head, rank=1) is True


_BANNED_DEFAULTS = (
    "spark1",
    "spark2",
    "sfxnz",
    "10.20.20.",
    "10.100.8.",
    "100.115.190.113",
    "100.101.109.7",
)


def _two_node_cfg() -> dict:
    return {
        "name": "lab",
        "nodes": [
            {
                "id": "head",
                "label": "head",
                "role": "head",
                "local": True,
                "vllm_url": "http://127.0.0.1:8000",
            },
            {
                "id": "worker",
                "label": "worker",
                "role": "worker",
                "local": False,
                "ssh_host": "worker.invalid",
                "vllm_url": "http://127.0.0.1:8000",
            },
        ],
    }


def _probed(node: dict, *, local: bool, online: bool, **extra) -> dict:
    out = {
        "id": node["id"],
        "label": node.get("label") or node["id"],
        "role": node.get("role") or "node",
        "local": local,
        "online": online,
        "probe_error": None if online else "ssh failed",
        "hostname": "testhost" if local else ("workerhost" if online else None),
        "lan_ip": node.get("lan_ip"),
        "tailscale_ip": node.get("tailscale_ip"),
        "qsfp_ip": node.get("qsfp_ip"),
        "gpu_sku": "NVIDIA GB10" if online else None,
        "ram_gib": 121.7 if online else None,
        "available_gib": 80.0 if online else None,
        "endpoint_healthy": False,
        "model_id": None,
        "models": [],
        "containers": [],
        "tensor_parallel_size": None,
        "ray_hint": False,
        "qsfp_if": node.get("qsfp_if"),
        "qsfp_carrier": None,
        "qsfp_speed_mbps": None,
        "roce_up_ifs": [],
        "vllm_url": node.get("vllm_url") or "http://127.0.0.1:8000",
        "ssh_host": node.get("ssh_host"),
    }
    out.update(extra)
    return out


@pytest.fixture
def isolated_cluster(monkeypatch, tmp_path):
    monkeypatch.delenv("LAIL_CLUSTER_JSON", raising=False)
    monkeypatch.setattr("app.config.DATA_DIR", tmp_path)
    monkeypatch.setattr(cluster.platform, "node", lambda: "testhost")
    monkeypatch.setattr(cluster, "_detect_local_net", lambda: {})
    monkeypatch.setattr(cluster, "_discover_fabric_peers", lambda local: [])
    return tmp_path


def _install_probes(monkeypatch, *, remote_online: bool = False):
    remote_calls: list[dict] = []

    def fake_local(node, base_url=None):
        return _probed(node, local=True, online=True)

    def fake_remote(node, ping_targets=None):
        remote_calls.append(node)
        return _probed(node, local=False, online=remote_online)

    monkeypatch.setattr(cluster, "_probe_local", fake_local)
    monkeypatch.setattr(cluster, "_probe_remote_ssh", fake_remote)
    monkeypatch.setattr(cluster, "_ping_ok", lambda ip, timeout_s=1.0: {"ok": False, "error": "no_ping"})
    return remote_calls


def test_fallback_constant_is_single_local():
    cfg = cluster._FALLBACK_CLUSTER
    assert len(cfg["nodes"]) == 1
    node = cfg["nodes"][0]
    assert node["id"] == "local"
    assert node["local"] is True
    assert node["vllm_url"] == "http://127.0.0.1:8000"
    assert not node.get("ssh_host")
    blob = json.dumps(cfg).lower()
    for bad in _BANNED_DEFAULTS:
        assert bad.lower() not in blob


def test_no_env_one_local_node_no_remote_ssh(isolated_cluster, monkeypatch):
    remote_calls = _install_probes(monkeypatch)
    ssh_cmds: list[list[str]] = []
    real_run = cluster.subprocess.run

    def spy_run(cmd, *a, **kw):
        if isinstance(cmd, (list, tuple)) and cmd and cmd[0] == "ssh":
            ssh_cmds.append(list(cmd))
            raise AssertionError(f"unexpected ssh: {cmd}")
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(cluster.subprocess, "run", spy_run)

    cfg = cluster._load_cluster_config()
    assert len(cfg["nodes"]) == 1
    assert cfg["nodes"][0]["id"] == "testhost"
    assert cfg["nodes"][0]["local"] is True

    data = cluster.collect_cluster()
    assert len(data["nodes"]) == 1
    node = data["nodes"][0]
    assert node["id"] == "testhost"
    assert node["local"] is True
    assert node["vllm_url"] == "http://127.0.0.1:8000"
    assert node.get("state") != "offline"
    assert remote_calls == []
    assert ssh_cmds == []
    assert "spark2" not in json.dumps(data).lower()
    note = (data.get("fabric") or {}).get("note") or ""
    assert "no multi-node fabric" in note.lower()
    assert not (data.get("fabric") or {}).get("links")
    summary = data["summary"]
    assert summary["nodes_total"] == 1
    assert summary["nodes_online"] == 1
    assert summary["healthy"] is True

    topo = ac._cluster_topology()
    assert topo["nodes"] == 1
    assert plan_placement(21.0, topo, mode="lab_safe", overlay=None)["nodes_available"] == 1


def test_two_nodes_remote_ssh_fail_offline_placement_one(isolated_cluster, monkeypatch):
    monkeypatch.setenv("LAIL_CLUSTER_JSON", json.dumps(_two_node_cfg()))
    remote_calls = _install_probes(monkeypatch, remote_online=False)

    data = cluster.collect_cluster()
    assert len(data["nodes"]) == 2
    remote = next(n for n in data["nodes"] if n["id"] == "worker")
    assert remote["online"] is False
    assert remote["state"] == "offline"
    assert remote["local"] is False
    assert len(remote_calls) == 1
    assert remote_calls[0]["ssh_host"] == "worker.invalid"

    topo = ac._cluster_topology()
    assert topo["nodes"] == 1
    assert topo["workers"] == []
    assert plan_placement(21.0, topo, mode="lab_safe", overlay=None)["nodes_available"] == 1


def test_two_nodes_both_online_topology_two(isolated_cluster, monkeypatch):
    monkeypatch.setenv("LAIL_CLUSTER_JSON", json.dumps(_two_node_cfg()))
    _install_probes(monkeypatch, remote_online=True)

    data = cluster.collect_cluster()
    online = [n for n in data["nodes"] if n.get("state") != "offline"]
    assert len(online) == 2
    assert all(n.get("online") for n in data["nodes"])

    topo = ac._cluster_topology()
    assert topo["nodes"] == 2
    assert len(topo["workers"]) == 1
    assert plan_placement(21.0, topo, mode="lab_safe", overlay=None)["nodes_available"] == 2


def test_serving_worker_detection_unchanged():
    head = {
        "id": "head",
        "local": True,
        "online": True,
        "endpoint_healthy": True,
        "model_id": "org/model",
        "containers": [{"name": "spark-vllm-n0", "status": "Up 3 minutes"}],
        "tensor_parallel_size": 2,
    }
    worker = {
        "id": "worker",
        "local": False,
        "online": True,
        "endpoint_healthy": False,
        "model_id": None,
        "containers": [{"name": "spark-vllm-n1", "status": "Up 3 minutes"}],
    }
    assert cluster._node_state(worker) == "serving_worker"
    summary = cluster._summarize([head, worker], {"ok": True, "links": []})
    assert worker["state"] == "serving_worker"
    assert worker.get("headless_worker") is True
    assert worker.get("model_id") == "org/model"
    assert summary["nodes_serving"] == 2
    assert summary["multi"]["mode"] == "multi_aligned"
    assert summary["multi"]["tensor_parallel_hint"] == 2


def test_dspark_named_worker_is_serving_worker_with_head_model():
    """Community/Anemll/Mia layout: …-vllm-dspark-1 on spark2 must not stay LOADING."""
    head = {
        "id": "spark1",
        "local": True,
        "online": True,
        "endpoint_healthy": True,
        "model_id": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "containers": [
            {
                "name": "deepseek-v4-flash-vllm-dspark-0",
                "status": "Up 10 minutes",
                "ports": [8888],
            }
        ],
        "tensor_parallel_size": 2,
        "vllm_url": "http://127.0.0.1:8888",
    }
    worker = {
        "id": "spark2",
        "local": False,
        "online": True,
        "endpoint_healthy": False,
        "model_id": None,
        "containers": [
            {
                "name": "deepseek-v4-flash-vllm-dspark-1",
                "status": "Up 10 minutes",
            }
        ],
    }
    assert cluster._rank_from_container_name("deepseek-v4-flash-vllm-dspark-1") == 1
    assert cluster._multinode_worker_rank(worker) == 1
    assert cluster._node_state(worker) == "serving_worker"
    summary = cluster._summarize([head, worker], {"ok": True, "links": []})
    assert worker["state"] == "serving_worker"
    assert worker.get("model_id") == "deepseek-ai/DeepSeek-V4-Flash-0731"
    assert summary["nodes_serving"] == 2
    assert summary["multi"]["mode"] == "multi_aligned"
    assert summary["multi"]["model_id"] == "deepseek-ai/DeepSeek-V4-Flash-0731"


def test_dspark_name_requires_vllm_token():
    assert cluster._rank_from_container_name("deepseek-v4-flash-vllm-dspark-1") == 1
    assert cluster._rank_from_container_name("vllm-dspark-2") == 2
    assert cluster._rank_from_container_name("backup-dspark-1") is None
    assert cluster._rank_from_container_name("my-dspark-3") is None
    assert cluster._rank_from_container_name("dspark-1") is None


def test_headless_with_explicit_rank_is_worker():
    worker = {
        "id": "worker",
        "local": False,
        "online": True,
        "endpoint_healthy": False,
        "model_id": None,
        "containers": [
            {
                "name": "custom-vllm-worker",
                "status": "Up 1 minute",
                "headless": True,
                "node_rank": 1,
                "cmd_blob": "vllm serve m --headless --node-rank 1",
            }
        ],
    }
    assert cluster._multinode_worker_rank(worker) == 1
    assert cluster._node_state(worker) == "serving_worker"


def test_headless_rank_none_or_zero_not_forced_to_one():
    for rank in (None, 0):
        node = {
            "id": "worker",
            "online": True,
            "endpoint_healthy": False,
            "containers": [
                {
                    "name": "custom-vllm-worker",
                    "status": "Up 1 minute",
                    "headless": True,
                    "node_rank": rank,
                    "cmd_blob": "vllm serve m --headless"
                    + ("" if rank is None else f" --node-rank {rank}"),
                }
            ],
        }
        assert cluster._multinode_worker_rank(node) is None
        assert cluster._node_state(node) == "loading"


def test_unrelated_up_container_is_not_serving_worker():
    head = {
        "id": "spark1",
        "local": True,
        "online": True,
        "endpoint_healthy": True,
        "model_id": "org/live",
        "containers": [{"name": "spark-vllm-n0", "status": "Up"}],
        "tensor_parallel_size": 2,
    }
    leftover = {
        "id": "spark2",
        "local": False,
        "online": True,
        "endpoint_healthy": False,
        "model_id": None,
        "containers": [{"name": "backup-dspark-1", "status": "Up 2 days"}],
    }
    assert cluster._multinode_worker_rank(leftover) is None
    # Up for days with no endpoint and no TP role: a stray, not "loading" forever
    assert cluster._node_state(leftover) == "stray"
    summary = cluster._summarize([head, leftover], {"ok": True})
    assert leftover["state"] == "stray"
    assert leftover.get("model_id") is None
    assert summary["nodes_serving"] == 1
    assert summary["multi"]["mode"] == "multi_partial" or summary["multi"]["mode"] == "single"


def test_leftover_higher_dspark_rank_not_attributed():
    """…-vllm-dspark-2 must not paint as TP worker when head TP=2."""
    head = {
        "id": "spark1",
        "online": True,
        "endpoint_healthy": True,
        "model_id": "org/live",
        "containers": [{"name": "deepseek-v4-flash-vllm-dspark-0", "status": "Up"}],
        "tensor_parallel_size": 2,
    }
    leftover = {
        "id": "spark2",
        "online": True,
        "endpoint_healthy": False,
        "model_id": None,
        "containers": [{"name": "deepseek-v4-flash-vllm-dspark-2", "status": "Up"}],
    }
    assert cluster._multinode_worker_rank(leftover) == 2
    summary = cluster._summarize([head, leftover], {"ok": True})
    assert leftover["state"] == "loading"
    assert leftover.get("model_id") is None
    assert summary["nodes_serving"] == 1


def test_candidate_vllm_ports_prefers_discovered_over_configured_8000():
    ports = node_probe.candidate_ports(
        "http://127.0.0.1:8000",
        [{"ports": [8888]}],
    )
    assert ports[0] == 8888
    assert 8000 in ports
    # Official path: discovered 8000 stays first.
    ports_official = node_probe.candidate_ports(
        "http://127.0.0.1:8000",
        [{"ports": [8000]}],
    )
    assert ports_official[0] == 8000
    ports2 = node_probe.candidate_ports(None, [{"ports": [8888]}])
    assert ports2[0] == 8888
    assert 8000 in ports2


class _FakeResp:
    def __init__(self, body: dict):
        self._body = json.dumps(body).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_urlopen(monkeypatch, served: dict[int, str], calls: list[str] | None = None):
    import urllib.error
    import urllib.request

    def urlopen(url, timeout=2.5):
        if calls is not None:
            calls.append(str(url))
        for port, model in served.items():
            if str(url).startswith(f"http://127.0.0.1:{port}/"):
                return _FakeResp({"data": [{"id": model}]})
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)


def test_live_8888_beats_stale_healthy_8000(monkeypatch):
    """Discovered :8888 must win even when a leftover :8000 also returns /v1/models."""
    calls: list[str] = []
    _fake_urlopen(monkeypatch, {8888: "org/live-on-8888", 8000: "org/stale-on-8000"}, calls)
    ports = node_probe.candidate_ports(
        "http://127.0.0.1:8000",
        [{"name": "deepseek-v4-flash-vllm-dspark-0", "ports": [8888]}],
    )
    assert ports[0] == 8888
    out = node_probe.probe_models(ports, fallback_url="http://127.0.0.1:8000")
    assert out["healthy"] is True
    assert out["model_id"] == "org/live-on-8888"
    assert out["vllm_url"] == "http://127.0.0.1:8888"
    assert calls[0].endswith(":8888/v1/models")


def test_official_spark_vllm_8000_still_selected(monkeypatch):
    _fake_urlopen(monkeypatch, {8000: "org/official"})
    ports = node_probe.candidate_ports(
        "http://127.0.0.1:8000",
        [{"name": "spark-vllm-n0", "ports": [8000]}],
    )
    out = node_probe.probe_models(ports)
    assert out["model_id"] == "org/official"
    assert out["vllm_url"] == "http://127.0.0.1:8000"


def test_official_and_dspark_status_summaries_both_serving():
    """Required layouts: spark-vllm-nN:8000 and dspark-style + non-8000 both SERVING."""
    official_head = {
        "id": "spark1",
        "local": True,
        "online": True,
        "endpoint_healthy": True,
        "model_id": "org/official",
        "containers": [{"name": "spark-vllm-n0", "status": "Up", "ports": [8000]}],
        "vllm_url": "http://127.0.0.1:8000",
        "tensor_parallel_size": 2,
    }
    official_worker = {
        "id": "spark2",
        "local": False,
        "online": True,
        "endpoint_healthy": False,
        "model_id": None,
        "containers": [{"name": "spark-vllm-n1", "status": "Up"}],
    }
    s1 = cluster._summarize([official_head, official_worker], {"ok": True})
    assert s1["multi"]["mode"] == "multi_aligned"
    assert official_worker["state"] == "serving_worker"
    assert official_worker["model_id"] == "org/official"

    dspark_head = {
        "id": "spark1",
        "local": True,
        "online": True,
        "endpoint_healthy": True,
        "model_id": "org/dspark",
        "containers": [
            {
                "name": "deepseek-v4-flash-vllm-dspark-0",
                "status": "Up",
                "ports": [8888],
            }
        ],
        "vllm_url": "http://127.0.0.1:8888",
        "tensor_parallel_size": 2,
    }
    dspark_worker = {
        "id": "spark2",
        "local": False,
        "online": True,
        "endpoint_healthy": False,
        "model_id": None,
        "containers": [
            {
                "name": "deepseek-v4-flash-vllm-dspark-1",
                "status": "Up",
            }
        ],
    }
    s2 = cluster._summarize([dspark_head, dspark_worker], {"ok": True})
    assert s2["multi"]["mode"] == "multi_aligned"
    assert dspark_worker["state"] == "serving_worker"
    assert dspark_worker["model_id"] == "org/dspark"
    assert s2["multi"]["model_id"] == "org/dspark"


_INSPECT_DSPARK = json.dumps(
    [
        {
            "Name": "/deepseek-v4-flash-vllm-dspark-0",
            "Config": {
                "Cmd": ["vllm", "serve", "m", "--port", "8888", "--host", "0.0.0.0", "--tensor-parallel-size", "2"],
                "Env": ["NODE_RANK=0", "HF_TOKEN=hf_secret"],
            },
            "Args": [],
            "State": {"StartedAt": "2026-09-30T23:10:33.062542994Z"},
            "NetworkSettings": {"Ports": {"8888/tcp": [{"HostIp": "0.0.0.0", "HostPort": "8888"}]}},
            "HostConfig": {"PortBindings": {}, "NetworkMode": "bridge"},
        }
    ]
)


def test_inventory_prefers_discovered_8888_and_ships_a_lean_container(monkeypatch):
    """The same inventory runs locally and (piped over ssh) on every remote node."""
    calls: list[list[str]] = []

    def fake_run(cmd, timeout=8):
        calls.append(cmd)
        if cmd[:2] == ["docker", "ps"]:
            return (
                "deepseek-v4-flash-vllm-dspark-0\tUp 5 minutes\tghcr.io/anemll/dspark-vllm-gx10:0.1.1\tabc\n"
                "conduit\tUp 11 days\tmatrixconduit/matrix-conduit:latest\tdef\n"
            )
        if cmd[:2] == ["docker", "inspect"]:
            return _INSPECT_DSPARK
        if cmd[0] == "ip":
            return (
                "2: eno1    inet 192.168.10.4/24 brd 192.168.10.255 scope global eno1\n"
                "4: roce0    inet 192.0.2.1/24 brd 192.0.2.255 scope global roce0\n"
                "6: roce1    inet 198.51.100.1/24 brd 198.51.100.255 scope global roce1\n"
            )
        if cmd[0] == "ibdev2netdev":
            return "r0 port 1 ==> roce0 (Up)\nr1 port 1 ==> roce1 (Up)\nr2 port 1 ==> roce2 (Down)\n"
        if cmd[0] == "lscpu":
            return "Model name: Cortex-X925\n"
        return None

    probed: list[str] = []
    _fake_urlopen(monkeypatch, {8888: "org/live-8888", 8000: "org/stale-8000"}, probed)
    monkeypatch.setattr(node_probe, "run", fake_run)
    monkeypatch.setattr(node_probe, "_sys_int", lambda iface, key: {"carrier": 1, "speed": 200000}[key])
    monkeypatch.setattr(node_probe, "ping", lambda ip, timeout_s=1.0: {"ok": ip == "192.0.2.8", "rtt_ms": 0.4, "error": None})

    node_probe.cpu_model.cache_clear()
    data = node_probe.inventory("", "http://127.0.0.1:8000", ["192.0.2.8", "192.0.2.9"])
    assert data["endpoint_healthy"] is True
    assert data["model_id"] == "org/live-8888"
    assert data["vllm_url"] == "http://127.0.0.1:8888"
    assert probed[0].startswith("http://127.0.0.1:8888/")
    assert [c["name"] for c in data["containers"]] == ["deepseek-v4-flash-vllm-dspark-0"]
    c = data["containers"][0]
    assert c["ports"] == [8888] and c["node_rank"] == 0 and c["headless"] is False
    assert c["tensor_parallel_size"] == 2 and data["tensor_parallel_size"] == 2
    assert "cmd_blob" not in c and "hf_secret" not in json.dumps(data)
    # ONE docker inspect for all serve containers, not one per container
    assert sum(1 for cmd in calls if cmd[:2] == ["docker", "inspect"]) == 1
    # both RoCE rails, with addresses; the LAN is the non-rail interface
    assert [(r["if"], r["ip"]) for r in data["rails"]] == [("roce0", "192.0.2.1"), ("roce1", "198.51.100.1")]
    assert data["qsfp_if"] == "roce0" and data["qsfp_ip"] == "192.0.2.1"
    assert data["lan_ip"] == "192.168.10.4"
    assert data["pings"]["192.0.2.8"]["ok"] is True and data["pings"]["192.0.2.9"]["ok"] is False
    assert data["cpu"] == "Cortex-X925"
    node_probe.cpu_model.cache_clear()


def test_invalid_cluster_json_falls_back_to_local(isolated_cluster, monkeypatch):
    monkeypatch.setenv("LAIL_CLUSTER_JSON", "{not-json")
    cfg = cluster._load_cluster_config()
    assert len(cfg["nodes"]) == 1
    assert cfg["nodes"][0]["id"] == "testhost"

    monkeypatch.setenv("LAIL_CLUSTER_JSON", json.dumps({"name": "only"}))
    cfg = cluster._load_cluster_config()
    assert len(cfg["nodes"]) == 1
    assert cfg["nodes"][0]["id"] == "testhost"


def test_cluster_file_used_when_env_unset(isolated_cluster, monkeypatch):
    payload = _two_node_cfg()
    payload["name"] = "from-file"
    (isolated_cluster / "cluster.json").write_text(json.dumps(payload))
    cfg = cluster._load_cluster_config()
    assert cfg["name"] == "from-file"
    assert len(cfg["nodes"]) == 2

    monkeypatch.setenv("LAIL_CLUSTER_JSON", json.dumps({"name": "from-env", "nodes": payload["nodes"][:1]}))
    cfg = cluster._load_cluster_config()
    assert cfg["name"] == "from-env"
    assert len(cfg["nodes"]) == 1


def test_stop_all_default_does_not_ssh(isolated_cluster, monkeypatch):
    from app.services import serve

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd) if isinstance(cmd, (list, tuple)) else [str(cmd)])

        class R:
            returncode = 0
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr(serve.subprocess, "run", fake_run)
    monkeypatch.setattr(serve, "_MULTINODE_STATE", type(serve._MULTINODE_STATE)("/no/such/multinode_serve.json"))
    monkeypatch.setattr(serve, "SPARK_LAB", type(serve.SPARK_LAB)("/no/such/spark_lab.sh"))
    monkeypatch.setattr(serve, "list_vllm_containers", lambda: [])
    result = serve.stop_all()
    assert result["ok"] is True
    assert not any(c and c[0] == "ssh" for c in calls)


def test_hostname_spark2_does_not_steal_worker(isolated_cluster, monkeypatch):
    monkeypatch.setattr(cluster.platform, "node", lambda: "spark2.home")
    payload = {
        "name": "lab",
        "nodes": [
            {"id": "spark1", "local": True, "vllm_url": "http://127.0.0.1:8000"},
            {"id": "spark2", "local": False, "ssh_host": "spark2", "vllm_url": "http://127.0.0.1:8000"},
        ],
    }
    monkeypatch.setenv("LAIL_CLUSTER_JSON", json.dumps(payload))
    remote_calls = _install_probes(monkeypatch, remote_online=False)
    data = cluster.collect_cluster()
    worker = next(n for n in data["nodes"] if n["id"] == "spark2")
    assert worker["local"] is False
    assert worker["state"] == "offline"
    assert [c["id"] for c in remote_calls] == ["spark2"]


_ADDR_SAMPLE = """\
1: lo    inet 127.0.0.1/8 scope host lo
2: eno1    inet 192.168.10.4/24 brd 192.168.10.255 scope global eno1
4: roce0    inet 192.0.2.1/24 brd 192.0.2.255 scope global roce0
8: tailscale0    inet 100.64.0.5/32 scope global tailscale0
9: docker0    inet 172.17.0.1/16 brd 172.17.255.255 scope global docker0
"""

_IB_SAMPLE = """\
roce0 port 1 ==> roce0 (Up)
roce1 port 1 ==> roce1 (Down)
"""

_NEIGH_SAMPLE = """\
192.0.2.8 lladdr aa:bb:cc:dd:ee:ff STALE
192.0.2.9 FAILED
"""


def test_parse_live_net_from_proc_text():
    addrs = node_probe.parse_ip_cidrs(_ADDR_SAMPLE)
    assert ("eno1", "192.168.10.4", 24) in addrs
    assert ("roce0", "192.0.2.1", 24) in addrs
    assert ("lo", "127.0.0.1", 8) in addrs
    assert node_probe.parse_roce_up(_IB_SAMPLE) == ["roce0"]
    assert cluster._parse_neigh(_NEIGH_SAMPLE) == ["192.0.2.8"]


def test_detect_local_net_uses_roce_and_lan(monkeypatch):
    def fake_run(cmd, timeout=8):
        if cmd[0] == "ip" and "-o" in cmd:
            return _ADDR_SAMPLE
        if cmd[0] == "ibdev2netdev":
            return _IB_SAMPLE
        if cmd[0] == "tailscale":
            return "100.64.0.5\n"
        return None

    monkeypatch.setattr(node_probe, "run", fake_run)
    monkeypatch.setattr(node_probe, "_sys_int", lambda iface, key: None)
    net = cluster._detect_local_net()
    assert net["lan_ip"] == "192.168.10.4"
    assert net["qsfp_if"] == "roce0"
    assert net["qsfp_ip"] == "192.0.2.1"
    assert net["tailscale_ip"] == "100.64.0.5"
    blob = json.dumps(net)
    for bad in _BANNED_DEFAULTS:
        assert bad.lower() not in blob.lower()


def test_parse_ssh_config_maps_ip_to_alias():
    text = "Host workerbox\n  Hostname 192.0.2.8\nHost github.com-lab\n  Hostname github.com\n"
    m = cluster._parse_ssh_config(text)
    assert m.get("192.0.2.8") == "workerbox"
    assert m.get("github.com") == "github.com-lab"


def test_discover_fabric_peers_from_neigh(monkeypatch):
    cluster._clear_peer_cache()
    local = {"id": "testhost", "qsfp_if": "roce0", "qsfp_ip": "192.0.2.1"}

    def fake_run(cmd, timeout=12):
        if cmd[0] == "ip" and "neigh" in cmd:
            return 0, _NEIGH_SAMPLE, ""
        if cmd[0] == "getent":
            return 0, "192.0.2.8 workerbox\n", ""
        return 1, "", ""

    monkeypatch.setattr(cluster, "_run", fake_run)
    monkeypatch.setattr(cluster, "_ping_ok", lambda ip, timeout_s=1.0: {"ok": True, "rtt_ms": 0.2})
    peers = cluster._discover_fabric_peers(local)
    assert len(peers) == 1
    assert peers[0]["id"] == "workerbox"
    assert peers[0]["local"] is False
    assert peers[0]["qsfp_ip"] == "192.0.2.8"
    assert peers[0]["ssh_host"] == "workerbox"


def test_discover_fabric_peers_scans_subnet_when_neigh_empty(monkeypatch):
    """After a reboot the ARP table is empty; still find a pingable QSFP peer."""
    cluster._clear_peer_cache()
    local = {"id": "testhost", "qsfp_if": "roce0", "qsfp_ip": "192.0.2.1"}
    pinged: list[str] = []

    def fake_run(cmd, timeout=12):
        if cmd[0] == "ip" and "neigh" in cmd:
            return 0, "", ""
        if cmd[0] == "ip" and "-o" in cmd:
            return 0, "4: roce0    inet 192.0.2.1/24 brd 192.0.2.255 scope global roce0\n", ""
        if cmd[0] == "getent" and cmd[-1] == "192.0.2.8":
            return 0, "192.0.2.8 workerbox\n", ""
        return 1, "", ""

    def fake_ping(ip, timeout_s=1.0):
        pinged.append(ip)
        return {"ok": ip == "192.0.2.8", "rtt_ms": 0.2}

    monkeypatch.setattr(cluster, "_run", fake_run)
    monkeypatch.setattr(cluster, "_ping_ok", fake_ping)
    peers = cluster._discover_fabric_peers(local)
    assert "192.0.2.8" in pinged
    assert len(peers) == 1
    assert peers[0]["id"] == "workerbox"
    assert peers[0]["qsfp_ip"] == "192.0.2.8"
    assert peers[0]["local"] is False


def test_subnet_hosts_skips_wide_prefixes():
    assert cluster._subnet_hosts("10.0.0.1", 16) == []
    hosts = cluster._subnet_hosts("192.0.2.1", 24)
    assert "192.0.2.1" not in hosts
    assert "192.0.2.8" in hosts
    assert "192.0.2.0" not in hosts
    assert "192.0.2.255" not in hosts


def test_default_cluster_is_probed_host_plus_roce_peer(monkeypatch, tmp_path):
    monkeypatch.delenv("LAIL_CLUSTER_JSON", raising=False)
    monkeypatch.setattr("app.config.DATA_DIR", tmp_path)
    monkeypatch.setattr(cluster.platform, "node", lambda: "headbox.lan")
    monkeypatch.setattr(
        cluster,
        "_detect_local_net",
        lambda: {"lan_ip": "192.168.10.4", "qsfp_if": "roce0", "qsfp_ip": "192.0.2.1"},
    )
    monkeypatch.setattr(
        cluster,
        "_discover_fabric_peers",
        lambda local: [
            {
                "id": "workerbox",
                "label": "workerbox",
                "role": "worker",
                "local": False,
                "ssh_host": "workerbox",
                "qsfp_ip": "192.0.2.8",
                "qsfp_if": "roce0",
                "vllm_url": "http://127.0.0.1:8000",
            }
        ],
    )
    cfg = cluster._load_cluster_config()
    assert [n["id"] for n in cfg["nodes"]] == ["headbox", "workerbox"]
    assert cfg["nodes"][0]["local"] is True
    assert cfg["nodes"][1]["local"] is False
    blob = json.dumps(cfg).lower()
    for bad in _BANNED_DEFAULTS:
        assert bad.lower() not in blob


# ─── states: offline vs unreachable, loading vs stray, TP role ────────────────


def test_down_host_is_offline_but_a_pinging_host_with_broken_ssh_is_unreachable():
    down = {"id": "w", "online": False, "ping": {"via": "qsfp_ip", "ip": "192.0.2.8", "ok": False}}
    broken_ssh = {"id": "w", "online": False, "ping": {"via": "qsfp_ip", "ip": "192.0.2.8", "ok": True}}
    assert cluster._node_state(down) == "offline"
    assert cluster._node_state(broken_ssh) == "unreachable"
    assert cluster._node_state({"id": "w", "online": False}) == "offline"
    head = {"id": "h", "local": True, "online": True}
    summary = cluster._summarize([head, broken_ssh], {"ok": True, "links": []})
    assert summary["nodes_online"] == 1 and summary["cluster_reachable"] is False


@pytest.mark.parametrize(
    "status,age",
    [
        ("Up 5 seconds", 5),
        ("Up Less than a second", 0.5),
        ("Up About a minute", 60),
        ("Up 12 minutes", 720),
        ("Up About an hour", 3600),
        ("Up 9 hours (healthy)", 32400),
        ("Up 2 days", 172800),
        ("Up", None),
        ("Exited (137) 8 days ago", None),
    ],
)
def test_container_age_from_docker_status(status, age):
    assert cluster._container_age_s(status) == age


def test_young_container_is_loading_old_one_is_stray():
    young = {"id": "w", "online": True, "containers": [{"name": "spark-vllm", "status": "Up 3 minutes"}]}
    old = {"id": "w", "online": True, "containers": [{"name": "spark-vllm", "status": "Up 2 hours"}]}
    unknown = {"id": "w", "online": True, "containers": [{"name": "spark-vllm", "status": "Up"}]}
    assert cluster._node_state(young) == "loading"
    assert cluster._node_state(old) == "stray"
    assert cluster._node_state(unknown) == "loading"


def test_tp_role_replaces_the_copied_endpoint_rate():
    head = {
        "id": "spark1", "local": True, "online": True, "endpoint_healthy": True, "model_id": "org/m",
        "containers": [{"name": "spark-vllm-n0", "status": "Up 3 minutes"}], "tensor_parallel_size": 2,
    }
    worker = {"id": "spark2", "online": True, "containers": [{"name": "spark-vllm-n1", "status": "Up 3 minutes"}]}
    idle = {"id": "spark3", "online": True, "containers": []}
    cluster._summarize([head, worker, idle], {"ok": True})
    assert (head["tp_rank"], worker["tp_rank"], idle["tp_rank"]) == (0, 1, None)
    for n in (head, worker, idle):
        assert "gen_tok_per_s" not in n


# ─── fabric: every rail, reachability ping reused, remote ↔ remote ────────────


def _rails(*ips: str) -> list[dict]:
    return [
        {"if": f"roce{i}", "ip": ip, "prefix": 24, "carrier": 1, "speed_mbps": 200000}
        for i, ip in enumerate(ips)
    ]


def test_fabric_has_one_link_per_rail_and_reuses_the_reachability_ping(monkeypatch):
    pinged: list[str] = []

    def fake_ping(ip, timeout_s=1.0):
        pinged.append(ip)
        return {"ok": ip != "198.51.100.2", "rtt_ms": 0.3, "error": None if ip != "198.51.100.2" else "down"}

    monkeypatch.setattr(cluster, "_ping_ok", fake_ping)
    head = {"id": "h", "local": True, "rails": _rails("192.0.2.1", "198.51.100.1")}
    worker = {
        "id": "w", "local": False, "qsfp_ip": "192.0.2.2", "rails": _rails("192.0.2.2", "198.51.100.2"),
        "ping": {"via": "qsfp_ip", "ip": "192.0.2.2", "ok": True, "rtt_ms": 0.8, "error": None},
    }
    links = cluster._fabric_links([head, worker])
    assert [(lnk["iface"], lnk["target_ip"], lnk["ok"]) for lnk in links] == [
        ("roce0", "192.0.2.2", True),
        ("roce1", "198.51.100.2", False),
    ]
    assert pinged == ["198.51.100.2"]  # rail 1's IP was already pinged for reachability
    assert links[0]["rtt_ms"] == 0.8 and links[1]["from_speed_mbps"] == 200000


def test_fabric_falls_back_to_the_configured_qsfp_ip_without_rail_data(monkeypatch):
    monkeypatch.setattr(cluster, "_ping_ok", lambda ip, timeout_s=1.0: {"ok": True, "rtt_ms": 0.2})
    links = cluster._fabric_links([
        {"id": "h", "local": True, "qsfp_if": "roce0", "qsfp_carrier": 1},
        {"id": "w", "local": False, "qsfp_ip": "192.0.2.2", "online": False},
    ])
    assert [(lnk["iface"], lnk["target_ip"], lnk["ok"]) for lnk in links] == [("roce0", "192.0.2.2", True)]


def test_remote_to_remote_links_come_from_the_remote_pings(monkeypatch):
    monkeypatch.setattr(cluster, "_ping_ok", lambda ip, timeout_s=1.0: {"ok": True, "rtt_ms": 0.2})
    head = {"id": "h", "local": True, "rails": _rails("192.0.2.1")}
    w1 = {"id": "w1", "local": False, "rails": _rails("192.0.2.2"),
          "pings": {"192.0.2.3": {"ok": False, "rtt_ms": None, "error": "100% loss"}}}
    w2 = {"id": "w2", "local": False, "rails": _rails("192.0.2.3"), "pings": {}}
    links = cluster._fabric_links([head, w1, w2])
    pair = [lnk for lnk in links if lnk["from"] == "w1"]
    assert [(lnk["to"], lnk["ok"]) for lnk in pair] == [("w2", False)]


# ─── collect_cluster: parallel, lean payload, peers persist ───────────────────


def test_collect_cluster_probes_nodes_in_parallel_and_ships_a_lean_payload(isolated_cluster, monkeypatch):
    import time as _time

    cfg = _two_node_cfg()
    cfg["nodes"].append({**cfg["nodes"][1], "id": "worker2", "ssh_host": "worker2.invalid"})
    monkeypatch.setenv("LAIL_CLUSTER_JSON", json.dumps(cfg))
    seen_targets: dict[str, list] = {}

    def slow_remote(node, ping_targets=None):
        _time.sleep(0.3)
        seen_targets[node["id"]] = ping_targets
        return _probed(node, local=False, online=True, pings={}, rails=[])

    monkeypatch.setattr(cluster, "_probe_local", lambda node, tel=None: _probed(node, local=True, online=True, net={"x": 1}))
    monkeypatch.setattr(cluster, "_probe_remote_ssh", slow_remote)
    monkeypatch.setattr(cluster, "_ping_ok", lambda ip, timeout_s=1.0: {"ok": True})
    t0 = _time.monotonic()
    data = cluster.collect_cluster()
    assert _time.monotonic() - t0 < 0.55  # two 0.3 s remotes, probed concurrently
    for n in data["nodes"]:
        assert "pings" not in n and "net" not in n
    assert seen_targets == {"worker": [], "worker2": []}  # no qsfp_ip configured → nothing to ping


def test_discovered_peer_that_goes_down_stays_listed(monkeypatch):
    cluster._clear_peer_cache()
    local = {"id": "testhost", "qsfp_if": "roce0", "qsfp_ip": "192.0.2.1"}
    alive = {"ok": True}

    def fake_run(cmd, timeout=12):
        if cmd[0] == "ip" and "neigh" in cmd:
            return 0, "192.0.2.8 lladdr aa:bb:cc:dd:ee:ff REACHABLE\n", ""
        if cmd[0] == "getent":
            return 0, "192.0.2.8 workerbox\n", ""
        return 1, "", ""

    monkeypatch.setattr(cluster, "_run", fake_run)
    monkeypatch.setattr(cluster, "_ping_ok", lambda ip, timeout_s=1.0: dict(alive))
    assert [p["id"] for p in cluster._discover_fabric_peers(local)] == ["workerbox"]
    alive["ok"] = False
    cluster._PEER_CACHE.clear()  # cache expired; the peer no longer answers
    assert [p["id"] for p in cluster._discover_fabric_peers(local)] == ["workerbox"]
    cluster._clear_peer_cache()


def test_cluster_topology_uses_the_sampler_cache_without_reprobing(monkeypatch):
    from app.services import status_sampler

    s = status_sampler.StatusSampler()
    s._cluster = s._published_cluster = {
        "nodes": [
            {"id": "h", "local": True, "online": True, "state": "serving", "ram_gib": 121.7},
            {"id": "w", "local": False, "online": True, "state": "serving_worker", "ram_gib": 121.7},
        ],
        "fabric": {"ok": True},
    }
    import time as _time

    s._cluster_mono = _time.monotonic()
    monkeypatch.setattr(status_sampler, "SAMPLER", s)
    monkeypatch.setattr(cluster, "collect_cluster", lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-probed")))
    topo = ac._cluster_topology()
    assert topo["nodes"] == 2 and topo["fabric_ok"] is True
    assert plan_placement(21.0, topo, mode="lab_safe", overlay=None)["nodes_available"] == 2


# ─── ssh plumbing ─────────────────────────────────────────────────────────────


def test_ssh_cmd_multiplexes_and_quotes_the_request():
    cmd = cluster._ssh_cmd("spark2", {"mode": "inventory", "vllm_url": "http://127.0.0.1:8000"})
    assert cmd[0] == "ssh" and "BatchMode=yes" in cmd and "ServerAliveInterval=5" in cmd
    assert any(o.startswith("ControlPath=") for o in cmd) and "ControlMaster=auto" in cmd
    assert cmd[-5:-1] == ["spark2", "python3", "-u", "-"]
    # the remote shell sees one quoted JSON word
    assert json.loads(__import__("shlex").split(cmd[-1])[0])["mode"] == "inventory"


def test_peer_stream_runs_the_probe_and_reads_lines(monkeypatch):
    """Run the stream end to end with the remote side replaced by a local python3."""
    import sys
    import time as _time

    monkeypatch.setattr(
        cluster, "_ssh_cmd", lambda host, req: [sys.executable, "-u", "-", json.dumps(req)]
    )
    s = cluster.PeerStream("fake", interval_s=0.2)
    s.start()
    try:
        deadline = _time.monotonic() + 10
        first = None
        while _time.monotonic() < deadline:
            first = first or s.reading()
            r = s.reading()
            if first and r and r["sampled_at"] > first["sampled_at"]:
                break
            _time.sleep(0.05)
    finally:
        s.stop()
    assert r and r["sampled_at"] > first["sampled_at"], s.error
    assert set(cluster.TELEMETRY_FIELDS) <= set(r)
    assert s.error is None and not s._thread.is_alive()


def test_peer_stream_reports_why_it_is_down(monkeypatch):
    import sys
    import time as _time

    monkeypatch.setattr(
        cluster, "_ssh_cmd", lambda host, req: [sys.executable, "-c", "print('ssh: connect to host w: No route to host'); raise SystemExit(255)"]
    )
    s = cluster.PeerStream("w", interval_s=0.2)
    s.start()
    deadline = _time.monotonic() + 5
    while s.error is None and _time.monotonic() < deadline:
        _time.sleep(0.05)
    s.stop()
    assert "No route to host" in (s.error or "")
    assert s.reading() is None
