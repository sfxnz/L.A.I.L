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
    "workload": {"pack": "prose", "levels": [1, 2], "max_tokens": 512},
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
    assert env["engine"]["metrics_snapshot"] == {"gen_tok_per_s": 40.0}
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
