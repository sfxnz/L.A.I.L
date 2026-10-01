"""Local AI Lab — FastAPI entrypoint."""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
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
from app.bind import (  # noqa: E402
    allow_query_token,
    assert_safe_bind,
    cors_origins,
    is_cross_site_write,
    is_untrusted_host,
    token_from_headers,
)
from app.config import APP_ROOT  # noqa: E402
from app.services import status_sampler  # noqa: E402

_LAIL_TOKEN = (os.environ.get("LAIL_TOKEN") or "").strip()
_CORS_ORIGINS = cors_origins(os.environ.get("LAIL_CORS_ORIGINS"))


@asynccontextmanager
async def _lifespan(_app: FastAPI):
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
    db.fail_orphaned_jobs()  # no runner survives a restart; their rows must not stay "running"
    await status_sampler.SAMPLER.start()
    try:
        yield
    finally:
        await status_sampler.SAMPLER.stop()


app = FastAPI(
    title="Local AI Lab",
    description="Serve, benchmark, and evaluate local LLMs on DGX Spark",
    version="0.1.0",
    lifespan=_lifespan,
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
    if allow_query_token(path) and q == _LAIL_TOKEN:
        return await call_next(request)
    return JSONResponse(
        {"error": "unauthorized", "message": "LAIL_TOKEN required"},
        status_code=401,
    )


# Registered after the token guard, so it runs first: the serve-engine listens on loopback
# without a token in the default setup, and any page the operator opens can POST to it.
@app.middleware("http")
async def _browser_guard(request: Request, call_next):
    if not _LAIL_TOKEN and is_untrusted_host(request.headers, _CORS_ORIGINS):
        return JSONResponse({"error": "untrusted_host", "message": "Host not allowed"}, status_code=403)
    if is_cross_site_write(request.method, request.headers, _CORS_ORIGINS):
        return JSONResponse(
            {"error": "cross_site_write", "message": "Cross-site write refused (send Content-Type: application/json)"},
            status_code=403,
        )
    return await call_next(request)


app.include_router(router, prefix="/api")


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
