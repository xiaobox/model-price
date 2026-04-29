"""v2 API router — consumes EntityStore.

During Phase 0 this module served fixture data. In Phase 1 it delegates
to services.entity_store.EntityStore, which is populated either from
backend/data/v2/{entities,offerings}.json (real pipeline output) or,
as a last-resort cold start, from backend/data/v2/fixtures/sample.json.

The endpoint shapes never change — docs/plans/v2-api-contract.md is
the source of truth and must stay byte-for-byte aligned with the
Pydantic response models in models/v2.py.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from models.v2 import (
    CompareResultV2,
    DriftReportV2,
    EntityDetailV2,
    EntityListItemV2,
    SearchResultV2,
    StatsV2,
)
from services.entity_store import MAX_COMPARE_IDS, get_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["v2"])

SHARED_SNAPSHOT_CACHE_CONTROL = (
    "public, max-age=0, s-maxage=3600, "
    "stale-while-revalidate=86400, stale-if-error=86400"
)


@router.get("/entities", response_model=List[EntityListItemV2])
def list_entities(
    q: str | None = Query(None, description="Substring search on name/canonical_id"),
    family: str | None = Query(None),
    maker: str | None = Query(None),
    capability: str | None = Query(None),
    min_context: int | None = Query(None),
    max_input_price: float | None = Query(None),
    sort: str = Query("name", pattern="^(name|input|output|context)$"),
    order: str = Query("asc", pattern="^(asc|desc)$"),
) -> List[EntityListItemV2]:
    store = get_store()
    return store.list_filtered(
        q=q,
        family=family,
        maker=maker,
        capability=capability,
        min_context=min_context,
        max_input_price=max_input_price,
        sort=sort,
        order=order,
    )


@router.get("/entities/{slug}", response_model=EntityDetailV2)
def get_entity(slug: str) -> EntityDetailV2:
    store = get_store()
    detail = store.detail(slug)
    if detail is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "not_found",
                "message": f"Entity '{slug}' not found",
            },
        )
    return detail


@router.get("/search", response_model=List[SearchResultV2])
def search(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
) -> List[SearchResultV2]:
    store = get_store()
    return store.search(q=q, limit=limit)


@router.get("/compare", response_model=CompareResultV2)
def compare(
    ids: str = Query(..., description="Comma-separated slugs, max 4"),
) -> CompareResultV2:
    cleaned = [s.strip() for s in ids.split(",") if s.strip()]
    if not cleaned:
        raise HTTPException(
            status_code=400,
            detail={"code": "bad_request", "message": "ids must not be empty"},
        )
    if len(cleaned) > MAX_COMPARE_IDS:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "too_many_ids",
                "message": f"compare supports up to {MAX_COMPARE_IDS} entities",
            },
        )
    store = get_store()
    return store.compare(cleaned)


@router.get("/stats", response_model=StatsV2)
def stats() -> StatsV2:
    return get_store().stats()


@router.get("/snapshot")
def snapshot() -> JSONResponse:
    store = get_store()
    stats = store.stats()
    entities = store.all_entities()
    offerings_by_entity = {
        entity.slug: store.offerings_for(entity.slug) for entity in entities
    }
    last_refresh = stats.last_refresh
    payload = {
        "version": "v2-live-snapshot.1",
        "generated_at": last_refresh,
        "entity_count": len(entities),
        "source_last_refresh": last_refresh,
        "entities": entities,
        "offerings_by_entity": offerings_by_entity,
        "alternatives_by_entity": {},
    }
    return JSONResponse(
        content=jsonable_encoder(payload),
        headers={"Cache-Control": SHARED_SNAPSHOT_CACHE_CONTROL},
    )


@router.get("/drift", response_model=DriftReportV2 | Dict[str, Any])
def drift() -> Any:
    report = get_store().drift_report()
    if report is None:
        return {
            "generated_at": None,
            "counts": None,
            "note": "No drift report yet — run POST /api/v2/refresh first.",
        }
    return report


@router.post("/refresh")
async def refresh(force_network: bool = Query(True)) -> JSONResponse:
    store = get_store()
    try:
        report = await store.refresh_from_pipeline(force_network=force_network)
    except Exception as exc:
        logger.exception("v2 refresh failed")
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": {"code": "internal_error", "message": str(exc)},
            },
        )
    return JSONResponse(
        content={
            "ok": True,
            "counts": report.counts.model_dump(),
            "generated_at": report.generated_at.isoformat(),
        }
    )
