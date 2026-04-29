"""Model Price Backend — FastAPI application (v2 only)."""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api_v2 import router as api_v2_router
from config import settings
from services import RefreshScheduler
from services.entity_store import get_store

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format=settings.log_format,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    try:
        get_store().load_from_disk_or_fixture()
        v2_stats = get_store().stats()
        logger.info(
            "v2 cached: %s entities, %s offerings (fixture=%s)",
            v2_stats.total_entities,
            v2_stats.total_offerings,
            v2_stats.fixture,
        )
    except Exception as exc:  # pragma: no cover - startup must not crash
        logger.exception("v2 EntityStore load failed: %s", exc)

    scheduler: Optional[RefreshScheduler] = None
    if settings.auto_refresh_enabled:
        scheduler = RefreshScheduler(
            interval_seconds=settings.auto_refresh_interval_seconds,
        )
        scheduler.start()

    yield

    if scheduler:
        await scheduler.stop()

    logger.info("Shutting down...")


app = FastAPI(
    title="Model Price API",
    description="API for AI model pricing comparison",
    version=settings.api_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_v2_router)


@app.middleware("http")
async def no_store_api_responses(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
    return response


@app.get("/")
async def root():
    return {
        "message": "Welcome to Model Price API",
        "version": settings.api_version,
        "docs": "/docs",
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint.

    Pinged every 10 minutes by .github/workflows/keepalive.yml to keep
    the Render free-tier backend warm. Do not remove or rename without
    updating the workflow.
    """
    stats = get_store().stats()
    return {
        "status": "healthy",
        "entities_count": stats.total_entities,
        "offerings_count": stats.total_offerings,
        "last_refresh": stats.last_refresh.isoformat() if stats.last_refresh else None,
    }


@app.post("/api/refresh")
async def refresh(
    provider: Optional[str] = Query(
        None,
        description="Deprecated — per-provider refresh is not supported in v2.",
    ),
) -> JSONResponse:
    """Compatibility alias for POST /api/v2/refresh.

    Kept for external callers documented in CLAUDE.md. Internally runs
    the full v2 pipeline; the `provider` query parameter is accepted
    but ignored (v2 always refreshes the whole store atomically).
    """
    if provider:
        logger.info("POST /api/refresh called with provider=%s — ignored in v2", provider)
    try:
        report = await get_store().refresh_from_pipeline(force_network=True)
    except Exception as exc:
        logger.exception("refresh failed")
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": {"code": "internal_error", "message": str(exc)}},
        )
    return JSONResponse(
        content={
            "ok": True,
            "counts": report.counts.model_dump(),
            "generated_at": report.generated_at.isoformat(),
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
    )
