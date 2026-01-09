# Multi-Provider Pricing System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the application to fetch pricing data from multiple AI providers starting with AWS Bedrock.

**Architecture:** Plugin-based provider system with abstract base class. JSON file storage. Startup + manual refresh. Dual view mode (table/card) frontend.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, httpx (async HTTP), React 19, TypeScript

---

## Task 1: Add Dependencies

**Files:**
- Modify: `backend/pyproject.toml`

**Step 1: Add httpx and pyyaml dependencies**

Edit `backend/pyproject.toml`:
```toml
[project]
name = "model-price-backend"
version = "0.1.0"
description = "AI Model Pricing API"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.128.0",
    "uvicorn[standard]>=0.40.0",
    "httpx>=0.28.0",
    "pyyaml>=6.0",
]
```

**Step 2: Install dependencies**

Run: `cd /Users/drake/Desktop/model-price/backend && uv sync`
Expected: Dependencies installed successfully

**Step 3: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock
git commit -m "chore: add httpx and pyyaml dependencies"
```

---

## Task 2: Create Data Models

**Files:**
- Create: `backend/models/__init__.py`
- Create: `backend/models/pricing.py`

**Step 1: Create models directory and __init__.py**

Create `backend/models/__init__.py`:
```python
from .pricing import (
    Pricing,
    BatchPricing,
    ModelPricing,
    PricingDatabase,
    ProviderInfo,
)

__all__ = [
    "Pricing",
    "BatchPricing",
    "ModelPricing",
    "PricingDatabase",
    "ProviderInfo",
]
```

**Step 2: Create pricing models**

Create `backend/models/pricing.py`:
```python
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
```

**Step 3: Verify models import correctly**

Run: `cd /Users/drake/Desktop/model-price/backend && uv run python -c "from models import ModelPricing, PricingDatabase; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add backend/models/
git commit -m "feat: add Pydantic models for pricing data"
```

---

## Task 3: Create Provider Base Class and Registry

**Files:**
- Create: `backend/providers/__init__.py`
- Create: `backend/providers/base.py`
- Create: `backend/providers/registry.py`

**Step 1: Create providers directory and __init__.py**

Create `backend/providers/__init__.py`:
```python
from .base import BaseProvider
from .registry import ProviderRegistry

__all__ = ["BaseProvider", "ProviderRegistry"]
```

**Step 2: Create abstract base provider**

Create `backend/providers/base.py`:
```python
"""Abstract base class for pricing providers."""

from abc import ABC, abstractmethod

from models import ModelPricing


class BaseProvider(ABC):
    """Base class for all pricing data providers."""

    name: str  # e.g., "aws_bedrock"
    display_name: str  # e.g., "AWS Bedrock"

    @abstractmethod
    async def fetch(self) -> list[ModelPricing]:
        """Fetch all model prices from this provider.

        Returns:
            List of ModelPricing objects.

        Raises:
            Exception: If fetching fails.
        """
        pass
```

**Step 3: Create provider registry**

Create `backend/providers/registry.py`:
```python
"""Provider registry for managing multiple pricing sources."""

import asyncio
import logging
from datetime import datetime

from .base import BaseProvider
from models import ModelPricing

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Registry for all pricing providers."""

    _providers: dict[str, BaseProvider] = {}

    @classmethod
    def register(cls, provider: BaseProvider) -> None:
        """Register a provider instance."""
        cls._providers[provider.name] = provider
        logger.info(f"Registered provider: {provider.name}")

    @classmethod
    def get(cls, name: str) -> BaseProvider | None:
        """Get a provider by name."""
        return cls._providers.get(name)

    @classmethod
    def all(cls) -> list[BaseProvider]:
        """Get all registered providers."""
        return list(cls._providers.values())

    @classmethod
    async def fetch_all(cls) -> list[ModelPricing]:
        """Fetch from all providers concurrently.

        Failed providers are logged but don't stop other fetches.
        """
        if not cls._providers:
            logger.warning("No providers registered")
            return []

        tasks = [p.fetch() for p in cls._providers.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_models: list[ModelPricing] = []
        for provider, result in zip(cls._providers.values(), results):
            if isinstance(result, Exception):
                logger.error(f"Provider {provider.name} failed: {result}")
            else:
                logger.info(f"Provider {provider.name}: fetched {len(result)} models")
                all_models.extend(result)

        return all_models

    @classmethod
    async def fetch_provider(cls, name: str) -> list[ModelPricing]:
        """Fetch from a single provider."""
        provider = cls.get(name)
        if not provider:
            raise ValueError(f"Unknown provider: {name}")
        return await provider.fetch()
```

**Step 4: Verify imports**

Run: `cd /Users/drake/Desktop/model-price/backend && uv run python -c "from providers import BaseProvider, ProviderRegistry; print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add backend/providers/
git commit -m "feat: add provider base class and registry"
```

---

## Task 4: Create Pricing Service

**Files:**
- Create: `backend/services/__init__.py`
- Create: `backend/services/pricing.py`

**Step 1: Create services directory and __init__.py**

Create `backend/services/__init__.py`:
```python
from .pricing import PricingService

__all__ = ["PricingService"]
```

**Step 2: Create pricing service**

Create `backend/services/pricing.py`:
```python
"""Pricing data service for CRUD and query operations."""

import json
import logging
from datetime import datetime
from pathlib import Path

from models import ModelPricing, PricingDatabase, ProviderInfo

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_FILE = DATA_DIR / "pricing.json"


class PricingService:
    """Service for managing pricing data."""

    @classmethod
    def _ensure_data_dir(cls) -> None:
        """Ensure data directory exists."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _load_database(cls) -> PricingDatabase:
        """Load database from JSON file."""
        cls._ensure_data_dir()
        if not DATA_FILE.exists():
            return PricingDatabase(last_refresh=datetime.now(), models=[])

        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return PricingDatabase.model_validate(data)
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            return PricingDatabase(last_refresh=datetime.now(), models=[])

    @classmethod
    def _save_database(cls, db: PricingDatabase) -> None:
        """Save database to JSON file atomically."""
        cls._ensure_data_dir()
        temp_file = DATA_FILE.with_suffix(".tmp")

        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(db.model_dump_json(indent=2))

        temp_file.replace(DATA_FILE)
        logger.info(f"Saved {len(db.models)} models to {DATA_FILE}")

    @classmethod
    def get_all(
        cls,
        provider: str | None = None,
        capability: str | None = None,
        search: str | None = None,
        sort_by: str = "model_name",
        sort_order: str = "asc",
    ) -> list[ModelPricing]:
        """Get all models with optional filters and sorting."""
        db = cls._load_database()
        models = db.models

        # Filter by provider
        if provider:
            models = [m for m in models if m.provider == provider]

        # Filter by capability
        if capability:
            models = [m for m in models if capability in m.capabilities]

        # Search by model name
        if search:
            search_lower = search.lower()
            models = [m for m in models if search_lower in m.model_name.lower()]

        # Sort
        reverse = sort_order == "desc"
        if sort_by == "model_name":
            models.sort(key=lambda m: m.model_name.lower(), reverse=reverse)
        elif sort_by == "input":
            models.sort(key=lambda m: m.pricing.input or 0, reverse=reverse)
        elif sort_by == "output":
            models.sort(key=lambda m: m.pricing.output or 0, reverse=reverse)
        elif sort_by == "context_length":
            models.sort(key=lambda m: m.context_length or 0, reverse=reverse)

        return models

    @classmethod
    def get_by_id(cls, model_id: str) -> ModelPricing | None:
        """Get a single model by ID."""
        db = cls._load_database()
        for model in db.models:
            if model.id == model_id:
                return model
        return None

    @classmethod
    def get_providers(cls) -> list[ProviderInfo]:
        """Get list of all providers with stats."""
        db = cls._load_database()
        provider_stats: dict[str, dict] = {}

        for model in db.models:
            if model.provider not in provider_stats:
                provider_stats[model.provider] = {
                    "count": 0,
                    "last_updated": model.last_updated,
                }
            provider_stats[model.provider]["count"] += 1
            if model.last_updated > provider_stats[model.provider]["last_updated"]:
                provider_stats[model.provider]["last_updated"] = model.last_updated

        # Map provider names to display names
        display_names = {
            "aws_bedrock": "AWS Bedrock",
            "openai": "OpenAI",
            "azure": "Azure OpenAI",
            "google": "Google",
            "openrouter": "OpenRouter",
            "anthropic": "Anthropic",
            "xai": "xAI",
        }

        return [
            ProviderInfo(
                name=name,
                display_name=display_names.get(name, name),
                model_count=stats["count"],
                last_updated=stats["last_updated"],
            )
            for name, stats in sorted(provider_stats.items())
        ]

    @classmethod
    def save_models(cls, models: list[ModelPricing]) -> None:
        """Save models to database (full replace)."""
        db = PricingDatabase(
            last_refresh=datetime.now(),
            models=models,
        )
        cls._save_database(db)

    @classmethod
    def update_provider(cls, provider_name: str, models: list[ModelPricing]) -> None:
        """Update models for a single provider (keep others)."""
        db = cls._load_database()

        # Remove old models from this provider
        other_models = [m for m in db.models if m.provider != provider_name]

        # Add new models
        db.models = other_models + models
        db.last_refresh = datetime.now()

        cls._save_database(db)

    @classmethod
    def get_stats(cls) -> dict:
        """Get overall statistics."""
        db = cls._load_database()
        models = db.models

        if not models:
            return {
                "total_models": 0,
                "providers": 0,
                "avg_input_price": 0,
                "avg_output_price": 0,
                "last_refresh": db.last_refresh.isoformat(),
            }

        input_prices = [m.pricing.input for m in models if m.pricing.input is not None]
        output_prices = [m.pricing.output for m in models if m.pricing.output is not None]

        return {
            "total_models": len(models),
            "providers": len(set(m.provider for m in models)),
            "avg_input_price": sum(input_prices) / len(input_prices) if input_prices else 0,
            "avg_output_price": sum(output_prices) / len(output_prices) if output_prices else 0,
            "last_refresh": db.last_refresh.isoformat(),
        }
```

**Step 3: Verify service imports**

Run: `cd /Users/drake/Desktop/model-price/backend && uv run python -c "from services import PricingService; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add backend/services/
git commit -m "feat: add pricing service for data persistence"
```

---

## Task 5: Implement AWS Bedrock Provider

**Files:**
- Create: `backend/providers/aws_bedrock.py`
- Modify: `backend/providers/__init__.py`

**Step 1: Create AWS Bedrock provider**

Create `backend/providers/aws_bedrock.py`:
```python
"""AWS Bedrock pricing provider."""

import logging
import re
from datetime import datetime

import httpx

from models import ModelPricing, Pricing, BatchPricing
from .base import BaseProvider
from .registry import ProviderRegistry

logger = logging.getLogger(__name__)

# AWS Pricing API endpoints (public, no auth required)
BEDROCK_URL = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/current/us-east-1/index.json"
BEDROCK_FM_URL = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrockFoundationModels/current/us-east-1/index.json"


class AWSBedrockProvider(BaseProvider):
    """Provider for AWS Bedrock pricing data."""

    name = "aws_bedrock"
    display_name = "AWS Bedrock"

    async def fetch(self) -> list[ModelPricing]:
        """Fetch pricing from both Bedrock sources."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Fetch both sources concurrently
            bedrock_resp, fm_resp = await asyncio.gather(
                client.get(BEDROCK_URL),
                client.get(BEDROCK_FM_URL),
            )
            bedrock_resp.raise_for_status()
            fm_resp.raise_for_status()

            bedrock_data = bedrock_resp.json()
            fm_data = fm_resp.json()

        models: dict[str, ModelPricing] = {}

        # Parse AmazonBedrock data
        self._parse_bedrock_data(bedrock_data, models)

        # Parse AmazonBedrockFoundationModels data
        self._parse_fm_data(fm_data, models)

        return list(models.values())

    def _parse_bedrock_data(
        self, data: dict, models: dict[str, ModelPricing]
    ) -> None:
        """Parse AmazonBedrock pricing data."""
        products = data.get("products", {})
        terms = data.get("terms", {}).get("OnDemand", {})

        for sku, product in products.items():
            attrs = product.get("attributes", {})
            model_name = attrs.get("model", "")
            if not model_name:
                continue

            # Skip non-model products (Guardrails, etc.)
            usage_type = attrs.get("usagetype", "")
            if "Guardrail" in usage_type or "CustomModel" in usage_type:
                continue

            # Get price
            term_data = terms.get(sku)
            if not term_data:
                continue

            term = list(term_data.values())[0]
            price_dim = list(term["priceDimensions"].values())[0]
            price_usd = float(price_dim["pricePerUnit"].get("USD", "0"))
            description = price_dim.get("description", "")

            # Determine price type from description/usagetype
            is_input = "input" in usage_type.lower() or "input" in description.lower()
            is_output = "output" in usage_type.lower() or "output" in description.lower()
            is_batch = "batch" in usage_type.lower() or "batch" in description.lower()

            # Create or update model
            model_id = self._normalize_model_id(model_name)
            full_id = f"{self.name}:{model_id}"

            if full_id not in models:
                models[full_id] = ModelPricing(
                    id=full_id,
                    provider=self.name,
                    model_id=model_id,
                    model_name=model_name,
                    pricing=Pricing(),
                    batch_pricing=None,
                    capabilities=["text"],
                    last_updated=datetime.now(),
                )

            model = models[full_id]

            # Update prices
            if is_batch:
                if model.batch_pricing is None:
                    model.batch_pricing = BatchPricing()
                if is_input:
                    model.batch_pricing.input = price_usd
                elif is_output:
                    model.batch_pricing.output = price_usd
            else:
                if is_input:
                    model.pricing.input = price_usd
                elif is_output:
                    model.pricing.output = price_usd

    def _parse_fm_data(
        self, data: dict, models: dict[str, ModelPricing]
    ) -> None:
        """Parse AmazonBedrockFoundationModels pricing data."""
        products = data.get("products", {})
        terms = data.get("terms", {}).get("OnDemand", {})

        for sku, product in products.items():
            attrs = product.get("attributes", {})
            service_name = attrs.get("servicename", "")
            if not service_name:
                continue

            # Extract model name from service name
            # e.g., "Claude 3.5 Sonnet (Amazon Bedrock Edition)" -> "Claude 3.5 Sonnet"
            model_name = re.sub(r"\s*\(Amazon Bedrock Edition\)\s*$", "", service_name)

            usage_type = attrs.get("usagetype", "")

            # Get price
            term_data = terms.get(sku)
            if not term_data:
                continue

            term = list(term_data.values())[0]
            price_dim = list(term["priceDimensions"].values())[0]
            price_usd = float(price_dim["pricePerUnit"].get("USD", "0"))
            description = price_dim.get("description", "")

            # Determine price type
            is_input = "Input" in usage_type
            is_output = "Output" in usage_type or "Response" in description
            is_batch = "batch" in usage_type.lower() or "Batch" in description
            is_cache_read = "CacheRead" in usage_type or "Cache Read" in description
            is_cache_write = "CacheWrite" in usage_type or "Cache Write" in description

            # Skip provisioned throughput and other non-token pricing
            if "ProvisionedThroughput" in usage_type:
                continue

            # Create or update model
            model_id = self._normalize_model_id(model_name)
            full_id = f"{self.name}:{model_id}"

            if full_id not in models:
                # Detect capabilities from model name
                capabilities = ["text"]
                name_lower = model_name.lower()
                if any(x in name_lower for x in ["vision", "vl", "image", "stable"]):
                    capabilities.append("vision")
                if any(x in name_lower for x in ["audio", "sonic", "voxtral"]):
                    capabilities.append("audio")
                if "embed" in name_lower:
                    capabilities = ["embedding"]

                models[full_id] = ModelPricing(
                    id=full_id,
                    provider=self.name,
                    model_id=model_id,
                    model_name=model_name,
                    pricing=Pricing(),
                    batch_pricing=None,
                    capabilities=capabilities,
                    last_updated=datetime.now(),
                )

            model = models[full_id]

            # Update prices based on type
            if is_batch:
                if model.batch_pricing is None:
                    model.batch_pricing = BatchPricing()
                if is_input:
                    model.batch_pricing.input = price_usd
                elif is_output:
                    model.batch_pricing.output = price_usd
            elif is_cache_read:
                model.pricing.cached_input = price_usd
            elif is_cache_write:
                model.pricing.cached_write = price_usd
            elif is_input and not is_cache_read and not is_cache_write:
                # Only set if not already set (prefer standard over latency optimized)
                if model.pricing.input is None:
                    model.pricing.input = price_usd
            elif is_output:
                if model.pricing.output is None:
                    model.pricing.output = price_usd

    def _normalize_model_id(self, name: str) -> str:
        """Normalize model name to ID format."""
        # Lowercase, replace spaces with hyphens, remove special chars
        model_id = name.lower()
        model_id = re.sub(r"[^a-z0-9\s\-\.]", "", model_id)
        model_id = re.sub(r"\s+", "-", model_id)
        return model_id


# Need to import asyncio for gather
import asyncio

# Register provider
ProviderRegistry.register(AWSBedrockProvider())
```

**Step 2: Update providers __init__.py**

Update `backend/providers/__init__.py`:
```python
from .base import BaseProvider
from .registry import ProviderRegistry

# Import providers to trigger registration
from . import aws_bedrock

__all__ = ["BaseProvider", "ProviderRegistry"]
```

**Step 3: Test provider fetch**

Run: `cd /Users/drake/Desktop/model-price/backend && uv run python -c "
import asyncio
from providers import ProviderRegistry

async def test():
    models = await ProviderRegistry.fetch_all()
    print(f'Fetched {len(models)} models')
    for m in models[:3]:
        print(f'  {m.model_name}: input={m.pricing.input}, output={m.pricing.output}')

asyncio.run(test())
"`
Expected: Shows model count and sample prices

**Step 4: Commit**

```bash
git add backend/providers/
git commit -m "feat: add AWS Bedrock provider"
```

---

## Task 6: Create Fetcher Service

**Files:**
- Create: `backend/services/fetcher.py`
- Modify: `backend/services/__init__.py`

**Step 1: Create fetcher service**

Create `backend/services/fetcher.py`:
```python
"""Data fetch orchestrator."""

import logging
from datetime import datetime

from providers import ProviderRegistry
from .pricing import PricingService

logger = logging.getLogger(__name__)


class Fetcher:
    """Orchestrates fetching pricing data from all providers."""

    @classmethod
    async def refresh_all(cls) -> dict:
        """Refresh data from all providers."""
        logger.info("Starting full refresh...")
        start = datetime.now()

        models = await ProviderRegistry.fetch_all()
        PricingService.save_models(models)

        elapsed = (datetime.now() - start).total_seconds()
        logger.info(f"Refresh complete: {len(models)} models in {elapsed:.2f}s")

        return {
            "status": "ok",
            "models_count": len(models),
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
        }

    @classmethod
    async def refresh_provider(cls, provider_name: str) -> dict:
        """Refresh data from a single provider."""
        logger.info(f"Refreshing provider: {provider_name}")
        start = datetime.now()

        models = await ProviderRegistry.fetch_provider(provider_name)
        PricingService.update_provider(provider_name, models)

        elapsed = (datetime.now() - start).total_seconds()
        logger.info(f"Provider {provider_name}: {len(models)} models in {elapsed:.2f}s")

        return {
            "status": "ok",
            "provider": provider_name,
            "models_count": len(models),
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
        }
```

**Step 2: Update services __init__.py**

Update `backend/services/__init__.py`:
```python
from .pricing import PricingService
from .fetcher import Fetcher

__all__ = ["PricingService", "Fetcher"]
```

**Step 3: Commit**

```bash
git add backend/services/
git commit -m "feat: add fetcher service for refresh orchestration"
```

---

## Task 7: Refactor main.py with New API Endpoints

**Files:**
- Modify: `backend/main.py`
- Create: `backend/data/.gitkeep`

**Step 1: Create data directory**

```bash
mkdir -p /Users/drake/Desktop/model-price/backend/data
touch /Users/drake/Desktop/model-price/backend/data/.gitkeep
```

**Step 2: Rewrite main.py**

Replace `backend/main.py` with:
```python
"""Model Price Backend - FastAPI Application."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from models import ModelPricing, ProviderInfo
from services import PricingService, Fetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: refresh data from all providers
    logger.info("Starting up - refreshing pricing data...")
    try:
        result = await Fetcher.refresh_all()
        logger.info(f"Startup refresh complete: {result['models_count']} models")
    except Exception as e:
        logger.error(f"Startup refresh failed: {e}")
    yield
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="Model Price API",
    description="API for AI model pricing comparison",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """API root."""
    return {
        "message": "Welcome to Model Price API",
        "version": "0.2.0",
        "docs": "/docs",
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    stats = PricingService.get_stats()
    return {
        "status": "healthy",
        "models_count": stats["total_models"],
        "last_refresh": stats["last_refresh"],
    }


@app.get("/api/models", response_model=list[ModelPricing])
async def list_models(
    provider: str | None = Query(None, description="Filter by provider"),
    capability: str | None = Query(None, description="Filter by capability"),
    search: str | None = Query(None, description="Search model name"),
    sort_by: str = Query("model_name", description="Sort field"),
    sort_order: str = Query("asc", description="Sort order: asc or desc"),
):
    """List all models with optional filters and sorting."""
    return PricingService.get_all(
        provider=provider,
        capability=capability,
        search=search,
        sort_by=sort_by,
        sort_order=sort_order,
    )


@app.get("/api/models/{model_id:path}", response_model=ModelPricing)
async def get_model(model_id: str):
    """Get a single model by ID."""
    model = PricingService.get_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@app.get("/api/providers", response_model=list[ProviderInfo])
async def list_providers():
    """List all providers with stats."""
    return PricingService.get_providers()


@app.get("/api/stats")
async def get_stats():
    """Get overall statistics."""
    return PricingService.get_stats()


@app.post("/api/refresh")
async def refresh(provider: str | None = Query(None, description="Provider to refresh")):
    """Manually refresh pricing data."""
    try:
        if provider:
            result = await Fetcher.refresh_provider(provider)
        else:
            result = await Fetcher.refresh_all()
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Refresh failed: {e}")
        raise HTTPException(status_code=500, detail="Refresh failed")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

**Step 3: Test API startup**

Run: `cd /Users/drake/Desktop/model-price/backend && timeout 15 uv run main.py || true`
Expected: Server starts, fetches data, shows model count

**Step 4: Commit**

```bash
git add backend/main.py backend/data/.gitkeep
git commit -m "feat: refactor API with provider-based architecture"
```

---

## Task 8: Create TypeScript Types

**Files:**
- Create: `frontend/src/types/pricing.ts`

**Step 1: Create types directory and pricing types**

Create `frontend/src/types/pricing.ts`:
```typescript
export interface Pricing {
  input: number | null;
  output: number | null;
  cached_input: number | null;
  cached_write: number | null;
  reasoning: number | null;
  image_input: number | null;
  audio_input: number | null;
  audio_output: number | null;
  embedding: number | null;
}

export interface BatchPricing {
  input: number | null;
  output: number | null;
}

export interface ModelPricing {
  id: string;
  provider: string;
  model_id: string;
  model_name: string;
  pricing: Pricing;
  batch_pricing: BatchPricing | null;
  context_length: number | null;
  max_output_tokens: number | null;
  capabilities: string[];
  last_updated: string;
}

export interface ProviderInfo {
  name: string;
  display_name: string;
  model_count: number;
  last_updated: string | null;
}

export interface Stats {
  total_models: number;
  providers: number;
  avg_input_price: number;
  avg_output_price: number;
  last_refresh: string;
}

export type ViewMode = 'table' | 'card';

export type SortField = 'model_name' | 'input' | 'output' | 'context_length';

export interface SortConfig {
  field: SortField;
  order: 'asc' | 'desc';
}

export interface Filters {
  provider: string | null;
  capability: string | null;
  search: string;
}
```

**Step 2: Commit**

```bash
git add frontend/src/types/
git commit -m "feat: add TypeScript types for pricing data"
```

---

## Task 9: Create useModels Hook

**Files:**
- Create: `frontend/src/hooks/useModels.ts`

**Step 1: Create hooks directory and useModels**

Create `frontend/src/hooks/useModels.ts`:
```typescript
import { useState, useEffect, useCallback } from 'react';
import type { ModelPricing, ProviderInfo, Stats, Filters, SortConfig, ViewMode } from '../types/pricing';

const API_BASE = '/api';

export function useModels() {
  const [models, setModels] = useState<ModelPricing[]>([]);
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [view, setView] = useState<ViewMode>('card');
  const [filters, setFilters] = useState<Filters>({
    provider: null,
    capability: null,
    search: '',
  });
  const [sortConfig, setSortConfig] = useState<SortConfig>({
    field: 'model_name',
    order: 'asc',
  });

  const buildQueryString = useCallback(() => {
    const params = new URLSearchParams();
    if (filters.provider) params.set('provider', filters.provider);
    if (filters.capability) params.set('capability', filters.capability);
    if (filters.search) params.set('search', filters.search);
    params.set('sort_by', sortConfig.field);
    params.set('sort_order', sortConfig.order);
    return params.toString();
  }, [filters, sortConfig]);

  const fetchModels = useCallback(async () => {
    try {
      const queryString = buildQueryString();
      const response = await fetch(`${API_BASE}/models?${queryString}`);
      if (!response.ok) throw new Error('Failed to fetch models');
      const data = await response.json();
      setModels(data);
      setError(null);
    } catch (err) {
      setError('无法连接到后端服务，请确保后端已启动 (port 8000)');
      console.error(err);
    }
  }, [buildQueryString]);

  const fetchProviders = async () => {
    try {
      const response = await fetch(`${API_BASE}/providers`);
      if (!response.ok) throw new Error('Failed to fetch providers');
      const data = await response.json();
      setProviders(data);
    } catch (err) {
      console.error('Failed to fetch providers:', err);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_BASE}/stats`);
      if (!response.ok) throw new Error('Failed to fetch stats');
      const data = await response.json();
      setStats(data);
    } catch (err) {
      console.error('Failed to fetch stats:', err);
    }
  };

  const refresh = async (provider?: string) => {
    setRefreshing(true);
    try {
      const url = provider
        ? `${API_BASE}/refresh?provider=${provider}`
        : `${API_BASE}/refresh`;
      const response = await fetch(url, { method: 'POST' });
      if (!response.ok) throw new Error('Failed to refresh');

      // Reload data after refresh
      await Promise.all([fetchModels(), fetchProviders(), fetchStats()]);
    } catch (err) {
      setError('刷新失败');
      console.error(err);
    } finally {
      setRefreshing(false);
    }
  };

  const handleSort = (field: SortConfig['field']) => {
    setSortConfig(prev => ({
      field,
      order: prev.field === field && prev.order === 'asc' ? 'desc' : 'asc',
    }));
  };

  // Initial load
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([fetchModels(), fetchProviders(), fetchStats()]);
      setLoading(false);
    };
    loadData();
  }, []);

  // Reload when filters or sort change
  useEffect(() => {
    if (!loading) {
      fetchModels();
    }
  }, [filters, sortConfig, fetchModels, loading]);

  return {
    models,
    providers,
    stats,
    loading,
    refreshing,
    error,
    view,
    setView,
    filters,
    setFilters,
    sortConfig,
    handleSort,
    refresh,
    retry: fetchModels,
  };
}
```

**Step 2: Commit**

```bash
git add frontend/src/hooks/
git commit -m "feat: add useModels hook for data management"
```

---

## Task 10: Create ViewToggle Component

**Files:**
- Create: `frontend/src/components/ViewToggle.tsx`

**Step 1: Create components directory and ViewToggle**

Create `frontend/src/components/ViewToggle.tsx`:
```typescript
import type { ViewMode } from '../types/pricing';

interface ViewToggleProps {
  view: ViewMode;
  onViewChange: (view: ViewMode) => void;
}

export function ViewToggle({ view, onViewChange }: ViewToggleProps) {
  return (
    <div className="view-toggle">
      <button
        className={`view-btn ${view === 'card' ? 'active' : ''}`}
        onClick={() => onViewChange('card')}
        title="卡片视图"
      >
        ▦
      </button>
      <button
        className={`view-btn ${view === 'table' ? 'active' : ''}`}
        onClick={() => onViewChange('table')}
        title="表格视图"
      >
        ☰
      </button>
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add frontend/src/components/ViewToggle.tsx
git commit -m "feat: add ViewToggle component"
```

---

## Task 11: Create FilterBar Component

**Files:**
- Create: `frontend/src/components/FilterBar.tsx`

**Step 1: Create FilterBar component**

Create `frontend/src/components/FilterBar.tsx`:
```typescript
import type { Filters, ProviderInfo } from '../types/pricing';

interface FilterBarProps {
  filters: Filters;
  onFiltersChange: (filters: Filters) => void;
  providers: ProviderInfo[];
}

const CAPABILITIES = [
  { value: 'text', label: '文本' },
  { value: 'vision', label: '视觉' },
  { value: 'audio', label: '音频' },
  { value: 'embedding', label: '嵌入' },
];

export function FilterBar({ filters, onFiltersChange, providers }: FilterBarProps) {
  return (
    <div className="filter-bar">
      <div className="filter-group">
        <label className="filter-label">提供商</label>
        <select
          className="filter-select"
          value={filters.provider || ''}
          onChange={(e) =>
            onFiltersChange({ ...filters, provider: e.target.value || null })
          }
        >
          <option value="">全部</option>
          {providers.map((p) => (
            <option key={p.name} value={p.name}>
              {p.display_name} ({p.model_count})
            </option>
          ))}
        </select>
      </div>

      <div className="filter-group">
        <label className="filter-label">能力</label>
        <select
          className="filter-select"
          value={filters.capability || ''}
          onChange={(e) =>
            onFiltersChange({ ...filters, capability: e.target.value || null })
          }
        >
          <option value="">全部</option>
          {CAPABILITIES.map((c) => (
            <option key={c.value} value={c.value}>
              {c.label}
            </option>
          ))}
        </select>
      </div>

      <div className="filter-group search-group">
        <label className="filter-label">搜索</label>
        <input
          type="text"
          className="filter-input"
          placeholder="模型名称..."
          value={filters.search}
          onChange={(e) =>
            onFiltersChange({ ...filters, search: e.target.value })
          }
        />
      </div>
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add frontend/src/components/FilterBar.tsx
git commit -m "feat: add FilterBar component"
```

---

## Task 12: Create ModelCard Component

**Files:**
- Create: `frontend/src/components/ModelCard.tsx`

**Step 1: Create ModelCard component**

Create `frontend/src/components/ModelCard.tsx`:
```typescript
import { useState } from 'react';
import type { ModelPricing } from '../types/pricing';

interface ModelCardProps {
  model: ModelPricing;
  index: number;
}

const providerColors: Record<string, string> = {
  aws_bedrock: 'var(--aws-color)',
  openai: 'var(--openai-color)',
  anthropic: 'var(--anthropic-color)',
  google: 'var(--google-color)',
  azure: 'var(--azure-color)',
  openrouter: 'var(--openrouter-color)',
};

const providerDisplayNames: Record<string, string> = {
  aws_bedrock: 'AWS Bedrock',
  openai: 'OpenAI',
  anthropic: 'Anthropic',
  google: 'Google',
  azure: 'Azure',
  openrouter: 'OpenRouter',
};

const capabilityIcons: Record<string, string> = {
  text: '📝',
  vision: '🖼️',
  audio: '🎧',
  embedding: '📊',
};

function formatPrice(price: number | null): string {
  if (price === null) return '-';
  if (price === 0) return 'Free';
  return '$' + price.toFixed(2);
}

function formatNumber(num: number | null): string {
  if (num === null) return '-';
  if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
  if (num >= 1000) return (num / 1000).toFixed(0) + 'K';
  return num.toString();
}

export function ModelCard({ model, index }: ModelCardProps) {
  const [expanded, setExpanded] = useState(false);
  const hasExtendedPricing =
    model.pricing.cached_input !== null ||
    model.pricing.cached_write !== null ||
    model.pricing.reasoning !== null ||
    model.batch_pricing !== null;

  return (
    <article
      className="model-card"
      style={{
        animationDelay: `${index * 0.05}s`,
        '--provider-color': providerColors[model.provider] || 'var(--accent-cyan)',
      } as React.CSSProperties}
    >
      <div className="card-header">
        <div className="provider-badge">
          <span>{providerDisplayNames[model.provider] || model.provider}</span>
        </div>
        <div className="card-badges">
          <span className="capability-icons">
            {model.capabilities.map((cap) => (
              <span key={cap} title={cap}>
                {capabilityIcons[cap] || ''}
              </span>
            ))}
          </span>
          {model.context_length && (
            <span className="context-badge mono">
              {formatNumber(model.context_length)} ctx
            </span>
          )}
        </div>
      </div>

      <h2 className="model-name">{model.model_name}</h2>

      <div className="pricing">
        <div className="price-item">
          <span className="price-label">输入</span>
          <span className="price-value mono">
            {formatPrice(model.pricing.input)}
            {model.pricing.input !== null && model.pricing.input > 0 && (
              <span className="price-unit">/M</span>
            )}
          </span>
        </div>
        <div className="price-divider"></div>
        <div className="price-item">
          <span className="price-label">输出</span>
          <span className="price-value mono">
            {formatPrice(model.pricing.output)}
            {model.pricing.output !== null && model.pricing.output > 0 && (
              <span className="price-unit">/M</span>
            )}
          </span>
        </div>
      </div>

      {/* Extended pricing (expandable) */}
      {hasExtendedPricing && (
        <>
          <button
            className="expand-btn"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? '收起详情 ▲' : '更多价格 ▼'}
          </button>

          {expanded && (
            <div className="extended-pricing">
              {model.pricing.cached_input !== null && (
                <div className="ext-price-row">
                  <span>缓存读取</span>
                  <span className="mono">{formatPrice(model.pricing.cached_input)}/M</span>
                </div>
              )}
              {model.pricing.cached_write !== null && (
                <div className="ext-price-row">
                  <span>缓存写入</span>
                  <span className="mono">{formatPrice(model.pricing.cached_write)}/M</span>
                </div>
              )}
              {model.pricing.reasoning !== null && (
                <div className="ext-price-row">
                  <span>推理</span>
                  <span className="mono">{formatPrice(model.pricing.reasoning)}/M</span>
                </div>
              )}
              {model.batch_pricing && (
                <>
                  <div className="ext-price-row batch">
                    <span>批量输入</span>
                    <span className="mono">{formatPrice(model.batch_pricing.input)}/M</span>
                  </div>
                  <div className="ext-price-row batch">
                    <span>批量输出</span>
                    <span className="mono">{formatPrice(model.batch_pricing.output)}/M</span>
                  </div>
                </>
              )}
            </div>
          )}
        </>
      )}

      {/* Price bar visualization */}
      <div className="price-bar-container">
        <div
          className="price-bar input-bar"
          style={{
            width: `${Math.min(((model.pricing.input || 0) / 20) * 100, 100)}%`,
          }}
        ></div>
        <div
          className="price-bar output-bar"
          style={{
            width: `${Math.min(((model.pricing.output || 0) / 80) * 100, 100)}%`,
          }}
        ></div>
      </div>
    </article>
  );
}
```

**Step 2: Commit**

```bash
git add frontend/src/components/ModelCard.tsx
git commit -m "feat: add ModelCard component with extended pricing"
```

---

## Task 13: Create ModelTable Component

**Files:**
- Create: `frontend/src/components/ModelTable.tsx`

**Step 1: Create ModelTable component**

Create `frontend/src/components/ModelTable.tsx`:
```typescript
import type { ModelPricing, SortConfig } from '../types/pricing';

interface ModelTableProps {
  models: ModelPricing[];
  sortConfig: SortConfig;
  onSort: (field: SortConfig['field']) => void;
}

const providerDisplayNames: Record<string, string> = {
  aws_bedrock: 'AWS Bedrock',
  openai: 'OpenAI',
  anthropic: 'Anthropic',
  google: 'Google',
  azure: 'Azure',
  openrouter: 'OpenRouter',
};

const capabilityIcons: Record<string, string> = {
  text: '📝',
  vision: '🖼️',
  audio: '🎧',
  embedding: '📊',
};

function formatPrice(price: number | null): string {
  if (price === null) return '-';
  if (price === 0) return 'Free';
  return '$' + price.toFixed(2);
}

function formatNumber(num: number | null): string {
  if (num === null) return '-';
  if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
  if (num >= 1000) return (num / 1000).toFixed(0) + 'K';
  return num.toString();
}

export function ModelTable({ models, sortConfig, onSort }: ModelTableProps) {
  const renderSortIndicator = (field: SortConfig['field']) => {
    if (sortConfig.field !== field) return null;
    return <span className="sort-indicator">{sortConfig.order === 'asc' ? '↑' : '↓'}</span>;
  };

  return (
    <div className="table-container">
      <table className="model-table">
        <thead>
          <tr>
            <th>提供商</th>
            <th
              className="sortable"
              onClick={() => onSort('model_name')}
            >
              模型 {renderSortIndicator('model_name')}
            </th>
            <th
              className="sortable numeric"
              onClick={() => onSort('input')}
            >
              输入 {renderSortIndicator('input')}
            </th>
            <th
              className="sortable numeric"
              onClick={() => onSort('output')}
            >
              输出 {renderSortIndicator('output')}
            </th>
            <th className="numeric">缓存</th>
            <th
              className="sortable numeric"
              onClick={() => onSort('context_length')}
            >
              上下文 {renderSortIndicator('context_length')}
            </th>
            <th>能力</th>
          </tr>
        </thead>
        <tbody>
          {models.map((model) => (
            <tr key={model.id}>
              <td className="provider-cell">
                {providerDisplayNames[model.provider] || model.provider}
              </td>
              <td className="model-name-cell">{model.model_name}</td>
              <td className="mono numeric">{formatPrice(model.pricing.input)}</td>
              <td className="mono numeric">{formatPrice(model.pricing.output)}</td>
              <td className="mono numeric secondary">
                {formatPrice(model.pricing.cached_input)}
              </td>
              <td className="mono numeric">{formatNumber(model.context_length)}</td>
              <td className="capabilities-cell">
                {model.capabilities.map((cap) => (
                  <span key={cap} title={cap}>
                    {capabilityIcons[cap] || ''}
                  </span>
                ))}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add frontend/src/components/ModelTable.tsx
git commit -m "feat: add ModelTable component"
```

---

## Task 14: Create RefreshButton Component

**Files:**
- Create: `frontend/src/components/RefreshButton.tsx`

**Step 1: Create RefreshButton component**

Create `frontend/src/components/RefreshButton.tsx`:
```typescript
interface RefreshButtonProps {
  refreshing: boolean;
  onRefresh: () => void;
}

export function RefreshButton({ refreshing, onRefresh }: RefreshButtonProps) {
  return (
    <button
      className={`refresh-btn ${refreshing ? 'refreshing' : ''}`}
      onClick={onRefresh}
      disabled={refreshing}
      title="刷新数据"
    >
      <span className="refresh-icon">⟳</span>
      {refreshing ? '刷新中...' : '刷新'}
    </button>
  );
}
```

**Step 2: Commit**

```bash
git add frontend/src/components/RefreshButton.tsx
git commit -m "feat: add RefreshButton component"
```

---

## Task 15: Create Components Index

**Files:**
- Create: `frontend/src/components/index.ts`

**Step 1: Create components index**

Create `frontend/src/components/index.ts`:
```typescript
export { ViewToggle } from './ViewToggle';
export { FilterBar } from './FilterBar';
export { ModelCard } from './ModelCard';
export { ModelTable } from './ModelTable';
export { RefreshButton } from './RefreshButton';
```

**Step 2: Commit**

```bash
git add frontend/src/components/index.ts
git commit -m "feat: add components index"
```

---

## Task 16: Update App.tsx to Use New Components

**Files:**
- Modify: `frontend/src/App.tsx`

**Step 1: Rewrite App.tsx**

Replace `frontend/src/App.tsx` with:
```typescript
import './App.css';
import { useModels } from './hooks/useModels';
import {
  ViewToggle,
  FilterBar,
  ModelCard,
  ModelTable,
  RefreshButton,
} from './components';

function formatPrice(price: number): string {
  if (price === 0) return 'Free';
  return '$' + price.toFixed(2);
}

function App() {
  const {
    models,
    providers,
    stats,
    loading,
    refreshing,
    error,
    view,
    setView,
    filters,
    setFilters,
    sortConfig,
    handleSort,
    refresh,
    retry,
  } = useModels();

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <span className="logo-icon">⬡</span>
            <h1>Model Price</h1>
            <span className="version mono">v0.2.0</span>
          </div>
          <p className="tagline">AI 模型定价一览表</p>
        </div>
        <div className="header-glow"></div>
      </header>

      {/* Stats Bar */}
      <section className="stats-bar">
        <div className="stat-item">
          <span className="stat-label">模型数量</span>
          <span className="stat-value mono">{stats?.total_models || 0}</span>
        </div>
        <div className="stat-divider"></div>
        <div className="stat-item">
          <span className="stat-label">提供商</span>
          <span className="stat-value mono">{stats?.providers || 0}</span>
        </div>
        <div className="stat-divider"></div>
        <div className="stat-item">
          <span className="stat-label">平均输入价格</span>
          <span className="stat-value mono">
            {formatPrice(stats?.avg_input_price || 0)}
          </span>
        </div>
        <div className="stat-divider"></div>
        <div className="stat-item">
          <span className="stat-label">平均输出价格</span>
          <span className="stat-value mono">
            {formatPrice(stats?.avg_output_price || 0)}
          </span>
        </div>
      </section>

      {/* Main Content */}
      <main className="main-content">
        {loading ? (
          <div className="loading">
            <div className="loading-spinner"></div>
            <p>正在加载数据...</p>
          </div>
        ) : error ? (
          <div className="error-card">
            <span className="error-icon">⚠</span>
            <p>{error}</p>
            <button onClick={retry} className="retry-btn">
              重试连接
            </button>
          </div>
        ) : (
          <>
            {/* Controls */}
            <div className="controls">
              <FilterBar
                filters={filters}
                onFiltersChange={setFilters}
                providers={providers}
              />
              <div className="controls-right">
                <RefreshButton refreshing={refreshing} onRefresh={() => refresh()} />
                <ViewToggle view={view} onViewChange={setView} />
              </div>
            </div>

            {/* Model Display */}
            {view === 'card' ? (
              <div className="model-grid">
                {models.map((model, index) => (
                  <ModelCard key={model.id} model={model} index={index} />
                ))}
              </div>
            ) : (
              <ModelTable
                models={models}
                sortConfig={sortConfig}
                onSort={handleSort}
              />
            )}

            {models.length === 0 && (
              <div className="no-results">
                <p>没有找到匹配的模型</p>
              </div>
            )}
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>
          <span className="mono">{'<'}</span>
          Built with FastAPI + React
          <span className="mono">{'>'}</span>
        </p>
        <p className="footer-note">价格单位：$/百万 tokens</p>
      </footer>
    </div>
  );
}

export default App;
```

**Step 2: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat: integrate new components into App"
```

---

## Task 17: Update CSS for New Components

**Files:**
- Modify: `frontend/src/App.css`

**Step 1: Add new CSS styles**

Append to `frontend/src/App.css`:
```css
/* === Provider Colors === */
:root {
  --aws-color: #ff9900;
  --openai-color: #10a37f;
  --anthropic-color: #d4a27f;
  --google-color: #4285f4;
  --azure-color: #0078d4;
  --openrouter-color: #6366f1;
}

/* === View Toggle === */
.view-toggle {
  display: flex;
  gap: 0.25rem;
  background: var(--card-bg);
  padding: 0.25rem;
  border-radius: 0.5rem;
  border: 1px solid var(--border-color);
}

.view-btn {
  padding: 0.5rem 0.75rem;
  background: transparent;
  border: none;
  border-radius: 0.375rem;
  cursor: pointer;
  font-size: 1rem;
  color: var(--text-muted);
  transition: all 0.2s;
}

.view-btn:hover {
  color: var(--text-primary);
  background: var(--hover-bg);
}

.view-btn.active {
  background: var(--accent-color);
  color: white;
}

/* === Filter Bar === */
.filter-bar {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  align-items: flex-end;
}

.filter-group {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.filter-label {
  font-size: 0.75rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.filter-select,
.filter-input {
  padding: 0.5rem 0.75rem;
  background: var(--card-bg);
  border: 1px solid var(--border-color);
  border-radius: 0.5rem;
  font-size: 0.875rem;
  color: var(--text-primary);
  min-width: 140px;
}

.filter-select:focus,
.filter-input:focus {
  outline: none;
  border-color: var(--accent-color);
}

.search-group {
  flex: 1;
  min-width: 200px;
}

.search-group .filter-input {
  width: 100%;
}

/* === Controls Layout === */
.controls {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 1rem;
  flex-wrap: wrap;
  margin-bottom: 1.5rem;
}

.controls-right {
  display: flex;
  gap: 0.75rem;
  align-items: center;
}

/* === Refresh Button === */
.refresh-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: var(--card-bg);
  border: 1px solid var(--border-color);
  border-radius: 0.5rem;
  font-size: 0.875rem;
  color: var(--text-primary);
  cursor: pointer;
  transition: all 0.2s;
}

.refresh-btn:hover:not(:disabled) {
  border-color: var(--accent-color);
  color: var(--accent-color);
}

.refresh-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.refresh-icon {
  font-size: 1rem;
  transition: transform 0.3s;
}

.refresh-btn.refreshing .refresh-icon {
  animation: spin 1s linear infinite;
}

/* === Model Card Extensions === */
.card-badges {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.capability-icons {
  display: flex;
  gap: 0.125rem;
  font-size: 0.875rem;
}

.expand-btn {
  width: 100%;
  padding: 0.5rem;
  margin-top: 0.75rem;
  background: transparent;
  border: 1px dashed var(--border-color);
  border-radius: 0.375rem;
  font-size: 0.75rem;
  color: var(--text-muted);
  cursor: pointer;
  transition: all 0.2s;
}

.expand-btn:hover {
  border-color: var(--accent-color);
  color: var(--accent-color);
}

.extended-pricing {
  margin-top: 0.75rem;
  padding-top: 0.75rem;
  border-top: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  gap: 0.375rem;
}

.ext-price-row {
  display: flex;
  justify-content: space-between;
  font-size: 0.8rem;
  color: var(--text-muted);
}

.ext-price-row.batch {
  color: var(--text-secondary);
}

/* === Model Table === */
.table-container {
  overflow-x: auto;
  background: var(--card-bg);
  border-radius: 0.75rem;
  border: 1px solid var(--border-color);
}

.model-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.875rem;
}

.model-table th,
.model-table td {
  padding: 0.75rem 1rem;
  text-align: left;
  border-bottom: 1px solid var(--border-color);
}

.model-table th {
  background: var(--bg-secondary);
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  font-size: 0.75rem;
  letter-spacing: 0.05em;
}

.model-table th.sortable {
  cursor: pointer;
  user-select: none;
  transition: color 0.2s;
}

.model-table th.sortable:hover {
  color: var(--accent-color);
}

.model-table th .sort-indicator {
  margin-left: 0.25rem;
  color: var(--accent-color);
}

.model-table th.numeric,
.model-table td.numeric {
  text-align: right;
}

.model-table td.secondary {
  color: var(--text-muted);
}

.model-table tbody tr:hover {
  background: var(--hover-bg);
}

.model-table tbody tr:last-child td {
  border-bottom: none;
}

.provider-cell {
  color: var(--text-secondary);
  font-weight: 500;
}

.model-name-cell {
  font-weight: 500;
  color: var(--text-primary);
}

.capabilities-cell {
  font-size: 1rem;
}

/* === No Results === */
.no-results {
  text-align: center;
  padding: 3rem;
  color: var(--text-muted);
}
```

**Step 2: Commit**

```bash
git add frontend/src/App.css
git commit -m "feat: add CSS styles for new components"
```

---

## Task 18: Test Full Integration

**Step 1: Start backend**

Run in terminal 1:
```bash
cd /Users/drake/Desktop/model-price/backend && uv run main.py
```
Expected: Server starts, fetches AWS Bedrock data, logs model count

**Step 2: Start frontend**

Run in terminal 2:
```bash
cd /Users/drake/Desktop/model-price/frontend && npm run dev
```
Expected: Vite dev server starts on port 5173

**Step 3: Manual testing checklist**

Open http://localhost:5173 and verify:
- [ ] Models load from API (should see AWS Bedrock models)
- [ ] Stats bar shows correct counts
- [ ] Filter by provider works
- [ ] Search works
- [ ] Sort by clicking table headers works
- [ ] Toggle between table and card view works
- [ ] Refresh button fetches new data
- [ ] Card expand/collapse shows extended pricing

**Step 4: Commit final state**

```bash
git add -A
git commit -m "feat: complete multi-provider pricing system v0.2.0"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add dependencies | pyproject.toml |
| 2 | Create data models | models/pricing.py |
| 3 | Create provider base/registry | providers/base.py, registry.py |
| 4 | Create pricing service | services/pricing.py |
| 5 | Implement AWS Bedrock provider | providers/aws_bedrock.py |
| 6 | Create fetcher service | services/fetcher.py |
| 7 | Refactor main.py | main.py |
| 8 | Create TypeScript types | types/pricing.ts |
| 9 | Create useModels hook | hooks/useModels.ts |
| 10-14 | Create React components | components/*.tsx |
| 15 | Create components index | components/index.ts |
| 16 | Update App.tsx | App.tsx |
| 17 | Update CSS | App.css |
| 18 | Integration test | - |

Total: 18 tasks, ~20 commits
