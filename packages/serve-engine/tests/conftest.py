from __future__ import annotations

import pytest

from app import db
from app.api import routes
from app.services import agentic
from app.services import jobs


@pytest.fixture(autouse=True)
def isolated_data(monkeypatch, tmp_path):
    """Fresh sqlite + runs/logs under tmp_path — never the repo's data/.

    Autouse: the app lifespan (init_db's re-kind, fail_orphaned_jobs) writes the DB, and a
    test run in the live checkout must never flip the running serve-engine's jobs to failed.
    """
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "lab.sqlite")
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(routes, "RUNS_DIR", runs)
    monkeypatch.setattr(agentic, "RUNS_DIR", runs)
    monkeypatch.setattr(jobs, "DATA_DIR", tmp_path)
    db.init_db()
    return tmp_path
