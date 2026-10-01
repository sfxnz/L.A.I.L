"""POST /api/runs/import — external bench runs become Run Envelopes in the run index."""
from __future__ import annotations

import json
import re

import pytest
from fastapi.testclient import TestClient

import app.main as main_mod
from app.services import metadata, status_sampler

PROBE = {
    "base_url": "http://127.0.0.1:8000",
    "healthy": True,
    "models": [{"id": "org/m", "max_model_len": 262144}],
    "version": {"version": "0.28.1"},
    "metrics": {"gen_tok_per_s": 40.0},
    "error": None,
}

BODY = {
    "kind": "decode",
    "model": "org/m",
    "workload": {"pack": "prose", "levels": [1, 2], "max_tokens": 512, "serve_fingerprint": "8488b677"},
    "metrics": {
        "arms": [{"concurrency": 1, "aggregate_tok_per_s": 33.1}],
        "headline": {"decode_tok_per_s_median_c1": 33.1, "aggregate_peak_tok_per_s": 96.0},
    },
    "summary": {"decode_tok_per_s_median_c1": 33.1, "aggregate_peak_tok_per_s": 96.0},
    "source": "controller-streams",
}


@pytest.fixture
def client(isolated_data, monkeypatch):
    monkeypatch.setattr(main_mod, "_LAIL_TOKEN", "")
    monkeypatch.setattr(metadata, "collect_hardware", lambda: {"gpu_sku": "NVIDIA GB10", "ram_gib": 121.7})
    monkeypatch.setattr(metadata, "list_vllm_containers", lambda: [])
    monkeypatch.setattr(status_sampler.SAMPLER, "probe", lambda: PROBE)
    return TestClient(main_mod.app)


def test_import_persists_envelope_and_indexes_the_run(client, isolated_data):
    r = client.post("/api/runs/import", json=BODY)
    assert r.status_code == 200, r.text
    assert set(r.json()) == {"run_id"}
    run_id = r.json()["run_id"]
    assert re.match(r"^\d{8}T\d{6}Z_[0-9a-f]{6}$", run_id)

    path = isolated_data / "runs" / f"{run_id}.json"
    env = json.loads(path.read_text())
    assert env["run_id"] == run_id and env["kind"] == "decode" and env["intent"] == "attach"
    assert env["source"] == "controller-streams"
    assert env["model"]["id"] == "org/m" and env["model"]["max_model_len"] == 262144
    assert env["hardware"]["gpu_sku"] == "NVIDIA GB10"
    assert env["engine"]["version"] == "0.28.1"
    # The serve config the run measured; no post-run cumulative counters.
    assert env["engine"]["flags_fingerprint"] == "8488b677"
    assert "metrics_snapshot" not in env["engine"]
    assert env["endpoint"]["base_url"] == "http://127.0.0.1:8000"
    assert env["workload"] == BODY["workload"] and env["metrics"] == BODY["metrics"]

    rows = client.get("/api/runs").json()
    assert [x["run_id"] for x in rows] == [run_id]
    assert rows[0]["kind"] == "decode" and rows[0]["model_id"] == "org/m"
    assert rows[0]["summary"] == BODY["summary"] and rows[0]["path"] == str(path)
    assert client.get("/api/runs", params={"kind": "decode"}).json()[0]["run_id"] == run_id

    detail = client.get(f"/api/runs/{run_id}").json()
    assert detail["index"]["run_id"] == run_id and detail["envelope"]["source"] == "controller-streams"


def test_import_without_model_falls_back_to_the_served_model(client):
    body = {**BODY, "model": None, "kind": "streams", "summary": {}}
    run_id = client.post("/api/runs/import", json=body).json()["run_id"]
    row = client.get(f"/api/runs/{run_id}").json()["index"]
    assert row["model_id"] == "org/m" and row["kind"] == "streams" and row["summary"] == {}


def test_import_validates_kind_and_shapes(client):
    assert client.post("/api/runs/import", json={**BODY, "kind": "agentic"}).status_code == 422
    assert client.post("/api/runs/import", json={"model": "m"}).status_code == 422
    assert client.post("/api/runs/import", json={**BODY, "metrics": [1, 2]}).status_code == 422
    minimal = client.post("/api/runs/import", json={"kind": "prefill"})
    assert minimal.status_code == 200
    assert client.get("/api/runs").json()[0]["kind"] == "prefill"


def test_import_requires_the_same_token_as_sibling_routes(client, monkeypatch):
    monkeypatch.setattr(main_mod, "_LAIL_TOKEN", "s3cret")
    denied = client.post("/api/runs/import", json=BODY)
    assert denied.status_code == 401 and "LAIL_TOKEN required" in denied.text
    assert client.post("/api/runs/import", json=BODY, headers={"X-Lail-Token": "s3cret"}).status_code == 200
    assert client.post("/api/jobs/x/cancel").status_code == 401


# ─── run index: filters before LIMIT ──────────────────────────────────────────


def _row(run_id: str, created: str, kind: str, summary: dict | None = None, path: str = "/nope.json") -> None:
    from app import db

    db.insert_run(run_id=run_id, created_at=created, kind=kind, intent="attach", model_id="org/m", summary=summary or {}, path=path)


def test_kind_filter_runs_before_the_limit(client):
    _row("old-prefill", "2026-09-01T00:00:00", "prefill")
    _row("old-tool", "2026-09-02T00:00:00", "agentic_tool_eval", {"final_score": 86})
    for i in range(150):  # a newer decode flood, as on the live box
        _row(f"d{i:03d}", f"2026-09-20T00:{i // 60:02d}:{i % 60:02d}", "decode")
    assert [r["run_id"] for r in client.get("/api/runs", params={"kind": "prefill", "limit": 12}).json()] == ["old-prefill"]
    assert [r["run_id"] for r in client.get("/api/runs", params={"kind": "agentic_tool_eval"}).json()] == ["old-tool"]
    assert len(client.get("/api/runs", params={"kind": "decode", "limit": 12}).json()) == 12
    assert len(client.get("/api/runs").json()) == 50
    assert client.get("/api/runs/count").json() == {"count": 152}
    assert client.get("/api/runs/count", params={"kind": "agentic_tool_eval"}).json() == {"count": 1}
    board = client.get("/api/runs/tool-eval/board").json()
    assert board["count"] == 1 and board["runs"][0]["run_id"] == "old-tool" and board["runs"][0]["final_score"] == 86


def test_tool_eval_compare_loads_runs_by_id(client):
    for i, score in enumerate((86, 65)):
        _row(f"t{i}", f"2026-09-0{i + 1}T00:00:00", "agentic_tool_eval", {"final_score": score})
    for i in range(100):
        _row(f"d{i:03d}", f"2026-09-20T00:00:{i:02d}", "decode")
    r = client.get("/api/runs/tool-eval/compare", params={"ids": "t0,t1"})
    assert r.status_code == 200, r.text
    assert r.json()["winner_run_id"] == "t0"
    assert client.get("/api/runs/tool-eval/compare", params={"ids": "t0,d001"}).status_code == 404


def test_init_db_rekinds_retired_perf_rows_once(isolated_data):
    from app import db

    _row("perfpy", "2026-09-20T00:00:00", "decode", {"workload": "prose", "decode_tok_per_s_median_c1": 29.1})
    _row("ctrl", "2026-09-21T00:00:00", "decode", {"aggregate_peak_tok_per_s": 96.0})
    _row("wf", "2026-08-01T00:00:00", "perf_workflow")
    db.init_db()
    db.init_db()  # idempotent
    kinds = {r["run_id"]: r["kind"] for r in db.list_runs(10)}
    assert kinds == {"perfpy": "legacy_decode", "ctrl": "decode", "wf": "legacy_perf_workflow"}
