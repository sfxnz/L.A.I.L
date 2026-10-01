from __future__ import annotations

import pytest

from app import db
from app.api import routes
from app.services import agentic
from app.services import jobs


@pytest.fixture
def isolated_data(monkeypatch, tmp_path):
    """Fresh sqlite + runs/logs under tmp_path — never the repo's data/."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "lab.sqlite")
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(routes, "RUNS_DIR", runs)
    monkeypatch.setattr(agentic, "RUNS_DIR", runs)
    monkeypatch.setattr(jobs, "DATA_DIR", tmp_path)
    db.init_db()
    return tmp_path
