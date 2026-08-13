"""Default cluster is one local node; remotes are opt-in and offline when unreachable."""
from __future__ import annotations

import json

import pytest

from app.services import autoconfig as ac
from app.services import cluster
from app.services.autoconfig import plan_placement


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

    def fake_remote(node):
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
    addrs = cluster._parse_ip_addrs(_ADDR_SAMPLE)
    assert ("eno1", "192.168.10.4") in addrs
    assert ("roce0", "192.0.2.1") in addrs
    assert ("lo", "127.0.0.1") in addrs
    assert cluster._parse_roce_up(_IB_SAMPLE) == ["roce0"]
    assert cluster._parse_neigh(_NEIGH_SAMPLE) == ["192.0.2.8"]


def test_detect_local_net_uses_roce_and_lan(monkeypatch):
    def fake_run(cmd, timeout=12):
        if cmd[:3] == ["ip", "-4", "-o"] or (len(cmd) >= 3 and cmd[0] == "ip" and "-o" in cmd):
            return 0, _ADDR_SAMPLE, ""
        if cmd[0] == "ibdev2netdev":
            return 0, _IB_SAMPLE, ""
        if cmd[0] == "tailscale":
            return 0, "100.64.0.5\n", ""
        return 1, "", "no"

    monkeypatch.setattr(cluster, "_run", fake_run)
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
