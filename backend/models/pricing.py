"""Pydantic models for pricing data."""

from datetime import datetime
from pydantic import BaseModel


class Pricing(BaseModel):
    """Price info (USD per million tokens/units)."""

    input: float | None = None
    output: float | None = None
    cached_input: float | None = None
    cached_write: float | None = None
    reasoning: float | None = None
    image_input: float | None = None
    audio_input: float | None = None
    audio_output: float | None = None
    embedding: float | None = None


class BatchPricing(BaseModel):
    """Batch processing discounted prices."""

    input: float | None = None
    output: float | None = None


class ModelPricing(BaseModel):
    """Complete pricing info for a single model."""

    id: str  # Unique: "{provider}:{model_id}"
    provider: str  # aws_bedrock, openai, azure, etc.
    model_id: str  # Original model ID
    model_name: str  # Display name
    pricing: Pricing
    batch_pricing: BatchPricing | None = None
    context_length: int | None = None
    max_output_tokens: int | None = None
    capabilities: list[str] = []  # ["text", "vision", "audio", "embedding"]
    last_updated: datetime


class PricingDatabase(BaseModel):
    """JSON file root structure."""

    version: str = "1.0"
    last_refresh: datetime
    models: list[ModelPricing] = []


class ProviderInfo(BaseModel):
    """Provider metadata for API response."""

    name: str
    display_name: str
    model_count: int
    last_updated: datetime | None = None
