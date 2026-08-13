"""Local AI Lab — FastAPI entrypoint."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure backend package root on path
BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.api.routes import router  # noqa: E402
from app import db  # noqa: E402
from app.bind import assert_safe_bind, token_from_headers  # noqa: E402
from app.config import APP_ROOT  # noqa: E402

_LAIL_TOKEN = (os.environ.get("LAIL_TOKEN") or "").strip()
_CORS_ORIGINS = [
    x.strip()
    for x in (os.environ.get("LAIL_CORS_ORIGINS") or "").split(",")
    if x.strip()
] or [
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:8787",
    "http://localhost:8787",
]

app = FastAPI(
    title="Local AI Lab",
    description="Serve, benchmark, and evaluate local LLMs on DGX Spark",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _token_guard(request: Request, call_next):
    if not _LAIL_TOKEN:
        return await call_next(request)
    if request.method == "OPTIONS":
        return await call_next(request)
    path = request.url.path
    if path in ("/api/health", "/health"):
        return await call_next(request)
    if token_from_headers(request.headers, _LAIL_TOKEN):
        return await call_next(request)
    q = request.query_params.get("token") or ""
    if q == _LAIL_TOKEN:
        return await call_next(request)
    return JSONResponse(
        {"error": "unauthorized", "message": "LAIL_TOKEN required"},
        status_code=401,
    )


app.include_router(router, prefix="/api")


@app.on_event("startup")
def _startup() -> None:
    host = os.environ.get("LAB_HOST") or os.environ.get("LAIL_HOST") or "127.0.0.1"
    allow_insecure = (os.environ.get("LAIL_INSECURE_BIND") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    # Same policy as run() and the controller. Docker/compose set LAIL_HOST=0.0.0.0
    # plus LAIL_INSECURE_BIND=1 because host ports stay on 127.0.0.1.
    assert_safe_bind(host, os.environ.get("LAIL_TOKEN") or "", allow_insecure=allow_insecure)
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

    host = os.environ.get("LAB_HOST") or os.environ.get("LAIL_HOST") or "127.0.0.1"
    allow_insecure = (os.environ.get("LAIL_INSECURE_BIND") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    assert_safe_bind(host, os.environ.get("LAIL_TOKEN") or "", allow_insecure=allow_insecure)
    port = int(os.environ.get("LAB_API_PORT", "8765"))
    uvicorn.run("app.main:app", host=host, port=port, reload=False, app_dir=str(BACKEND))


if __name__ == "__main__":
    run()
