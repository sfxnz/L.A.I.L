"""Local AI Lab — FastAPI entrypoint."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Ensure backend package root on path
BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.api.routes import router  # noqa: E402
from app import db  # noqa: E402
from app.config import APP_ROOT  # noqa: E402

app = FastAPI(
    title="Local AI Lab",
    description="Serve, benchmark, and evaluate local LLMs on DGX Spark",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.on_event("startup")
def _startup() -> None:
    db.init_db()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Serve frontend build if present
FRONTEND_DIST = APP_ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def spa(full_path: str):
        index = FRONTEND_DIST / "index.html"
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.exists() and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index)


def run() -> None:
    import uvicorn

    host = os.environ.get("LAB_HOST", "127.0.0.1")
    port = int(os.environ.get("LAB_UI_PORT", "5173"))
    # Dev: use 8765 for API when frontend vite proxies
    port = int(os.environ.get("LAB_API_PORT", "8765"))
    uvicorn.run("app.main:app", host=host, port=port, reload=False, app_dir=str(BACKEND))


if __name__ == "__main__":
    run()
