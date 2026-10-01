"""GPU probe parse and Spark node payload — shipped helpers only."""
from __future__ import annotations

from app.services import cluster, node_probe


def test_parse_gpu_telemetry_reads_temp_usage_power():
    tel = node_probe.parse_gpu_telemetry(
        "NVIDIA GB10, 47, 83, 32.1, [N/A], [N/A]\n"
    )
    assert tel["gpu_sku"] == "NVIDIA GB10"
    assert tel["temperature_c"] == 47
    assert tel["gpu_util_pct"] == 83
    assert tel["power_w"] == 32.1
    assert tel["memory_used_mib"] is None
    assert tel["memory_total_mib"] is None


def test_parse_gpu_telemetry_blank_is_nil_not_zero():
    tel = node_probe.parse_gpu_telemetry("")
    assert tel["gpu_sku"] is None
    assert tel["temperature_c"] is None
    assert tel["gpu_util_pct"] is None
    assert tel["power_w"] is None
    assert tel["memory_used_mib"] is None


def test_parse_gpu_telemetry_na_fields_stay_none():
    tel = node_probe.parse_gpu_telemetry("NVIDIA GB10, [N/A], [N/A], [N/A], [N/A], [N/A]")
    assert tel["gpu_sku"] == "NVIDIA GB10"
    assert tel["temperature_c"] is None
    assert tel["gpu_util_pct"] is None
    assert tel["power_w"] is None


def test_node_payload_includes_temperature_and_usage_from_smi():
    tel = node_probe.parse_gpu_telemetry("NVIDIA GB10, 41.0, 7, 18.5, [N/A], [N/A]")
    node = cluster.apply_gpu_telemetry({}, tel)
    assert node["temperature_c"] == 41.0
    assert node["gpu_util_pct"] == 7
    assert node["power_w"] == 18.5
    assert node["memory_used_mib"] is None
    # GB10 unified memory: nvidia-smi says [N/A] → null, never a fabricated 0
    assert node["gpu_mem_used_gib"] is None and node["gpu_mem_total_gib"] is None


def test_node_payload_gpu_memory_in_gib_when_smi_reports_it():
    tel = node_probe.parse_gpu_telemetry("NVIDIA RTX 6000, 50, 30, 120.0, 20480, 49140")
    node = cluster.apply_gpu_telemetry({}, tel)
    assert node["memory_used_mib"] == 20480
    assert node["gpu_mem_used_gib"] == 20.0
    assert node["gpu_mem_total_gib"] == 47.99


def test_node_payload_nil_when_smi_returns_nothing():
    node = cluster.apply_gpu_telemetry({"id": "spark1"}, node_probe.parse_gpu_telemetry(""))
    assert node["temperature_c"] is None
    assert node["gpu_util_pct"] is None
    assert node["power_w"] is None
    assert node["gpu_mem_used_gib"] is None and node["gpu_mem_total_gib"] is None
    assert 0 not in (node["temperature_c"], node["gpu_util_pct"], node["power_w"])


def test_apply_telemetry_copies_the_whole_live_set_including_nulls():
    node = {"id": "spark2", "power_w": 7.0, "available_gib": 18.0, "sampled_at": 1}
    cluster.apply_telemetry(node, {"power_w": 13.3, "soc_temp_c": 44.7, "sampled_at": 2})
    assert node["power_w"] == 13.3 and node["soc_temp_c"] == 44.7 and node["sampled_at"] == 2
    # a field the new sample lacks is cleared, not left over from an older reading
    assert node["available_gib"] is None
    assert set(cluster.TELEMETRY_FIELDS) <= set(node)


# ─── node_probe: the one probe that runs locally and on every remote ──────────

_MEMINFO = """\
MemTotal:       127600752 kB
MemFree:         6291456 kB
MemAvailable:   13945856 kB
SwapTotal:      16777212 kB
SwapFree:        8290872 kB
"""


def test_meminfo_is_kb_precise_not_floored_like_free_g():
    m = node_probe.parse_meminfo(_MEMINFO)
    # `free -g` said 13 for this box; MemAvailable is 13.30 GiB
    assert m["available_gib"] == 13.3
    assert m["ram_gib"] == 121.7  # one decimal: placement input, unchanged
    assert m["swap_total_gib"] == 16.0
    assert m["swap_used_gib"] == 8.09
    assert node_probe.parse_meminfo("") == {
        "ram_gib": None, "available_gib": None, "swap_total_gib": None, "swap_used_gib": None,
    }


def test_engine_reservation_from_compute_apps():
    # GB10: memory.used is [N/A] but the vLLM process reservation is reported per process
    assert node_probe.parse_compute_apps("1090685, 94103\n") == 91.9
    assert node_probe.parse_compute_apps("1, 1024\n2, 2048\n") == 3.0
    assert node_probe.parse_compute_apps("") == 0.0  # GPU present, nothing running
    assert node_probe.parse_compute_apps(None) is None  # nvidia-smi failed


def test_cpu_util_from_proc_stat_deltas():
    a = node_probe.parse_cpu_times("cpu  100 0 100 700 100 0 0 0 0 0\ncpu0 1 2 3 4\n")
    b = node_probe.parse_cpu_times("cpu  200 0 200 800 100 0 0 0 0 0\n")
    assert a == (200.0, 1000.0)
    assert node_probe.cpu_util_pct(a, b) == 66.7  # 200 busy of 300 jiffies
    assert node_probe.cpu_util_pct(None, b) is None
    assert node_probe.cpu_util_pct(b, b) is None


def test_psi_full_avg10():
    text = "some avg10=0.40 avg60=0.04 avg300=0.03 total=1\nfull avg10=0.25 avg60=0.04 avg300=0.03 total=1\n"
    assert node_probe.parse_psi_full_avg10(text) == 0.25
    assert node_probe.parse_psi_full_avg10(None) is None


def test_hwmon_temps_soc_nvme_nic(tmp_path):
    def mk(name, temps, labels=None):
        d = tmp_path / f"hwmon{len(list(tmp_path.iterdir()))}"
        d.mkdir()
        (d / "name").write_text(name + "\n")
        for i, t in enumerate(temps, 1):
            (d / f"temp{i}_input").write_text(f"{t}\n")
            if labels:
                (d / f"temp{i}_label").write_text(labels[i - 1] + "\n")

    mk("acpitz", [46800, 44200, 47800])
    mk("nvme", [43850, 60850], ["Composite", "Sensor 1"])
    mk("mlx5", [51000])
    mk("mlx5", [52000])
    mk("mt7925_phy0", [43000])
    assert node_probe.read_temps(str(tmp_path)) == {"soc_temp_c": 47.8, "nvme_temp_c": 43.9, "nic_temp_c": 52.0}
    assert node_probe.read_temps(str(tmp_path / "missing")) == {
        "soc_temp_c": None, "nvme_temp_c": None, "nic_temp_c": None,
    }


def test_cpu_model_from_lscpu_big_little():
    text = (
        "Architecture: aarch64\nModel name: Cortex-X925\nCore(s) per socket: 10\nSocket(s): 1\n"
        "Model name: Cortex-A725\nCore(s) per socket: 10\nSocket(s): 1\n"
    )
    assert node_probe.parse_lscpu(text) == "10× Cortex-X925 + 10× Cortex-A725"
    assert node_probe.parse_lscpu("Model name: AMD EPYC 9654\nCore(s) per socket: 96\n") == "AMD EPYC 9654"
    assert node_probe.parse_lscpu("") is None


def test_telemetry_sample_shape(monkeypatch):
    outputs = {
        "--query-gpu": "NVIDIA GB10, 43, 7, 9.08, [N/A], [N/A]\n",
        "--query-compute-apps": "1090685, 94103\n",
    }
    monkeypatch.setattr(node_probe, "run", lambda cmd, timeout=8: next(v for k, v in outputs.items() if cmd[1].startswith(k)))
    files = {"/proc/meminfo": _MEMINFO, "/proc/stat": "cpu  1 0 1 8 0 0 0 0\n", "/proc/pressure/memory": "full avg10=0.00 x\n"}
    monkeypatch.setattr(node_probe, "_read", lambda p: files.get(p))
    monkeypatch.setattr(node_probe, "read_temps", lambda: {"soc_temp_c": 46.8, "nvme_temp_c": 43.9, "nic_temp_c": 51.0})
    tel = node_probe.Telemetry()
    first = tel.sample()
    files["/proc/stat"] = "cpu  6 0 1 13 0 0 0 0\n"
    second = tel.sample()
    assert first["cpu_util_pct"] is None and second["cpu_util_pct"] == 50.0
    assert second["gpu_sku"] == "NVIDIA GB10" and second["power_w"] == 9.08 and second["memory_used_mib"] is None
    assert second["engine_reserved_gib"] == 91.9
    assert second["available_gib"] == 13.3 and second["swap_used_gib"] == 8.09
    assert second["soc_temp_c"] == 46.8 and isinstance(second["sampled_at"], int)
    assert set(cluster.TELEMETRY_FIELDS) <= set(second)
