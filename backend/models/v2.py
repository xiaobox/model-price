"""Pydantic models for v2 API.

Mirrors frontend/src/types/v2.ts byte-for-byte.
All prices are per 1,000,000 tokens; the merger converts LiteLLM's
per-token values before constructing these models.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


OfferingSource = Literal[
    "provider_api",       # Directly scraped / fetched from the provider's own API
    "provider_scrape",    # Playwright scrape of the provider's pricing page
    "via_litellm",        # Mirror of provider's first-party pricing via LiteLLM
    "litellm_fallback",   # Placeholder for entities with no real provider attached
]


class PricingV2(BaseModel):
    input: Optional[float] = None
    output: Optional[float] = None
    cache_read: Optional[float] = None
    cache_write: Optional[float] = None
    image_input: Optional[float] = None
    audio_input: Optional[float] = None
    audio_output: Optional[float] = None
    embedding: Optional[float] = None


class BatchPricingV2(BaseModel):
    input: Optional[float] = None
    output: Optional[float] = None


class OfferingV2(BaseModel):
    provider: str
    provider_model_id: str
    pricing: PricingV2
    batch_pricing: Optional[BatchPricingV2] = None
    availability: str = "ga"
    region: Optional[str] = None
    notes: Optional[str] = None
    last_updated: datetime
    source: OfferingSource = "provider_api"


class EntityCoreV2(BaseModel):
    canonical_id: str
    slug: str
    name: str
    family: str
    maker: str
    context_length: Optional[int] = None
    max_output_tokens: Optional[int] = None
    capabilities: List[str] = Field(default_factory=list)
    input_modalities: List[str] = Field(default_factory=list)
    output_modalities: List[str] = Field(default_factory=list)
    mode: str = "chat"
    is_open_source: Optional[bool] = None
    primary_offering_provider: str
    sources: List[str] = Field(default_factory=list)
    last_refreshed: datetime


class EntityListItemV2(EntityCoreV2):
    primary_offering: Optional[OfferingV2] = None


class AlternativeV2(BaseModel):
    canonical_id: str
    name: str
    delta_input_pct: float
    delta_output_pct: float
    capability_overlap: float


class EntityDetailV2(BaseModel):
    entity: EntityCoreV2
    offerings: List[OfferingV2]
    alternatives: List[AlternativeV2]


class SearchResultV2(BaseModel):
    canonical_id: str
    slug: str
    name: str
    family: Optional[str]
    maker: Optional[str]
    primary_input_price: Optional[float]
    primary_output_price: Optional[float]


class CompareResultV2(BaseModel):
    entities: List[EntityDetailV2]
    common_capabilities: List[str]
    requested_ids: List[str]
    missing_ids: List[str]


class StatsV2(BaseModel):
    total_entities: int
    total_offerings: int
    makers: int
    families: int
    last_refresh: Optional[datetime]
    fixture: bool = False


class DriftCountsV2(BaseModel):
    entities_total: int
    entities_new: int
    entities_removed: int
    offerings_total: int
    unmatched_provider_models: int
    orphan_entities: int
    price_drift_items: int


class UnmatchedProviderModel(BaseModel):
    provider: str
    model_id: str
    tried_aliases: List[str] = Field(default_factory=list)


class PriceDriftItem(BaseModel):
    entity: str
    provider: str
    field: str
    provider_value: float
    litellm_value: float
    delta_pct: float


class DriftReportV2(BaseModel):
    generated_at: datetime
    counts: DriftCountsV2
    unmatched_provider_models: List[UnmatchedProviderModel] = Field(default_factory=list)
    price_drift: List[PriceDriftItem] = Field(default_factory=list)
    new_entities: List[str] = Field(default_factory=list)
    removed_entities: List[str] = Field(default_factory=list)
    orphan_entities: List[str] = Field(default_factory=list)
    note: Optional[str] = None


class EntityStoreSnapshot(BaseModel):
    """On-disk storage shape. Single file holds the full v2 state."""

    version: str = "v2.0"
    generated_at: datetime
    entities: List[EntityCoreV2]
    offerings: List[OfferingV2]

    # Map from entity.slug → list[offering index] so the API layer
    # can look up without scanning. Built at load time.
    model_config = {"from_attributes": True}
