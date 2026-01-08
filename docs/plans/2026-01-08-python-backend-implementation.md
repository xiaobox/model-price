# Python 后端重构实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 TypeScript/Express 后端重构为 Python/FastAPI 后端，保持前端 React 应用基本不变。

**Architecture:** 分层架构 (routes → services → providers)，使用 Pydantic 做数据校验，httpx 做异步 HTTP 请求。

**Tech Stack:** Python 3.11+, FastAPI, Pydantic, httpx, uvicorn

---

## Task 1: 创建 Python 项目结构

**Files:**
- Create: `backend/app/__init__.py`
- Create: `backend/app/main.py`
- Create: `backend/app/config.py`
- Create: `backend/requirements.txt`
- Create: `backend/app/models/__init__.py`
- Create: `backend/app/services/__init__.py`
- Create: `backend/app/providers/__init__.py`
- Create: `backend/app/routes/__init__.py`

**Step 1: 创建目录结构**

Run: `mkdir -p backend/app/{models,services,providers,routes}`

**Step 2: 创建 requirements.txt**

```txt
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
httpx>=0.27.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
```

**Step 3: 创建 __init__.py 文件**

创建空的 `__init__.py` 文件：
- `backend/app/__init__.py`
- `backend/app/models/__init__.py`
- `backend/app/services/__init__.py`
- `backend/app/providers/__init__.py`
- `backend/app/routes/__init__.py`

**Step 4: 创建 config.py**

```python
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    data_dir: Path = Path(__file__).parent.parent.parent.parent / "data"
    host: str = "0.0.0.0"
    port: int = 3001
    cors_origins: list[str] = ["http://localhost:5173"]

    model_config = SettingsConfigDict(env_prefix="APP_")


settings = Settings()
```

**Step 5: 创建基础 main.py**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

app = FastAPI(title="Model Price API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
```

**Step 6: 验证 FastAPI 启动**

Run: `cd backend && python -m uvicorn app.main:app --port 3001`
Expected: Server starts, `http://localhost:3001/api/health` returns `{"status": "ok"}`

**Step 7: Commit**

```bash
git add backend/
git commit -m "feat: initialize Python backend structure with FastAPI"
```

---

## Task 2: 定义 Pydantic 数据模型

**Files:**
- Create: `backend/app/models/pricing.py`

**Step 1: 创建 pricing.py**

```python
from datetime import datetime
from typing import Literal
from pydantic import BaseModel


class Pricing(BaseModel):
    input_tokens: float | None = None
    output_tokens: float | None = None
    cached_input_tokens: float | None = None
    reasoning_tokens: float | None = None
    image_input: float | None = None
    embedding: float | None = None


class Capabilities(BaseModel):
    text: bool | None = None
    vision: bool | None = None
    audio: bool | None = None
    embedding: bool | None = None
    image_generation: bool | None = None


ProviderType = Literal[
    "openrouter", "azure_openai", "aws_bedrock", "openai", "xai", "google_vertex"
]


class ModelPricing(BaseModel):
    id: str
    provider: ProviderType
    model_id: str
    model_name: str
    pricing: Pricing
    billing_mode: Literal["per_token", "per_image", "per_request"] = "per_token"
    currency: Literal["USD"] = "USD"
    capabilities: Capabilities | None = None
    context_length: int | None = None
    max_output_tokens: int | None = None
    source: Literal["api", "manual"]
    source_url: str | None = None
    last_updated: str
    last_verified: str | None = None
    notes: str | None = None


class UnifiedPricingDatabase(BaseModel):
    version: str
    last_sync: str
    models: list[ModelPricing]


class DisplayModel(BaseModel):
    provider_id: str
    provider_name: str
    model_id: str
    model_name: str
    cost_input: float | None = None
    cost_output: float | None = None
    cost_cache_read: float | None = None
    cost_reasoning: float | None = None
    context_limit: int | None = None
    output_limit: int | None = None
    input_modalities: list[str] | None = None
    output_modalities: list[str] | None = None
    last_updated: str | None = None
    source: Literal["api", "manual"] | None = None


class ProviderInfo(BaseModel):
    id: str
    name: str
    pricing_url: str | None = None
    api_url: str | None = None
    source: Literal["api", "manual"]


class ProviderWithModels(BaseModel):
    id: str
    name: str
    doc: str | None = None
    api: str | None = None
    model_count: int
    models: list[dict]


class SyncResult(BaseModel):
    provider: str
    success: bool
    models_count: int
    error: str | None = None


class ManualModel(BaseModel):
    model_id: str
    model_name: str
    pricing: Pricing
    billing_mode: Literal["per_token", "per_image", "per_request"] | None = None
    capabilities: Capabilities | None = None
    context_length: int | None = None
    notes: str | None = None


class ManualDataFile(BaseModel):
    provider: str
    source_url: str
    last_verified: str
    models: list[ManualModel]
```

**Step 2: 更新 models/__init__.py**

```python
from app.models.pricing import (
    Pricing,
    Capabilities,
    ProviderType,
    ModelPricing,
    UnifiedPricingDatabase,
    DisplayModel,
    ProviderInfo,
    ProviderWithModels,
    SyncResult,
    ManualModel,
    ManualDataFile,
)

__all__ = [
    "Pricing",
    "Capabilities",
    "ProviderType",
    "ModelPricing",
    "UnifiedPricingDatabase",
    "DisplayModel",
    "ProviderInfo",
    "ProviderWithModels",
    "SyncResult",
    "ManualModel",
    "ManualDataFile",
]
```

**Step 3: Commit**

```bash
git add backend/app/models/
git commit -m "feat: add Pydantic data models for pricing"
```

---

## Task 3: 实现 PricingService

**Files:**
- Create: `backend/app/services/pricing.py`

**Step 1: 创建 pricing.py service**

```python
import json
from pathlib import Path

from app.config import settings
from app.models import (
    ModelPricing,
    UnifiedPricingDatabase,
    DisplayModel,
    ProviderInfo,
    ProviderWithModels,
    ProviderType,
)


PROVIDER_NAMES: dict[str, str] = {
    "openrouter": "OpenRouter",
    "azure_openai": "Azure OpenAI",
    "aws_bedrock": "AWS Bedrock",
    "openai": "OpenAI",
    "xai": "xAI",
    "google_vertex": "Google Vertex AI",
}

PROVIDER_DOCS: dict[str, dict[str, str]] = {
    "openrouter": {
        "pricing": "https://openrouter.ai/models",
        "api": "https://openrouter.ai/api/v1/models",
    },
    "azure_openai": {
        "pricing": "https://azure.microsoft.com/pricing/details/cognitive-services/openai-service/",
        "api": "https://prices.azure.com/api/retail/prices",
    },
    "aws_bedrock": {
        "pricing": "https://aws.amazon.com/bedrock/pricing/",
        "api": "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/current/region_index.json",
    },
    "openai": {
        "pricing": "https://openai.com/api/pricing/",
    },
    "xai": {
        "pricing": "https://x.ai/api",
    },
    "google_vertex": {
        "pricing": "https://cloud.google.com/vertex-ai/generative-ai/pricing",
        "api": "https://cloud.google.com/billing/docs/how-to/get-pricing-information-api",
    },
}


class PricingService:
    def __init__(self, data_path: Path | None = None):
        self.data_path = data_path or settings.data_dir / "pricing.json"

    def _load_data(self) -> UnifiedPricingDatabase:
        content = self.data_path.read_text(encoding="utf-8")
        return UnifiedPricingDatabase.model_validate_json(content)

    def _to_display_model(self, model: ModelPricing) -> DisplayModel:
        return DisplayModel(
            provider_id=model.provider,
            provider_name=PROVIDER_NAMES.get(model.provider, model.provider),
            model_id=model.model_id,
            model_name=model.model_name,
            cost_input=model.pricing.input_tokens,
            cost_output=model.pricing.output_tokens,
            cost_cache_read=model.pricing.cached_input_tokens,
            cost_reasoning=model.pricing.reasoning_tokens,
            context_limit=model.context_length,
            output_limit=model.max_output_tokens,
            input_modalities=(
                ["text", "image"]
                if model.capabilities and model.capabilities.vision
                else ["text"]
            ),
            output_modalities=["text"],
            last_updated=model.last_updated,
            source=model.source,
        )

    def get_all(
        self,
        provider: str | None = None,
        capability: str | None = None,
        q: str | None = None,
    ) -> dict:
        data = self._load_data()
        models = data.models

        # Apply filters
        if provider:
            models = [m for m in models if m.provider == provider]

        if capability:
            if capability == "vision":
                models = [
                    m for m in models if m.capabilities and m.capabilities.vision
                ]
            elif capability == "audio":
                models = [m for m in models if m.capabilities and m.capabilities.audio]
            elif capability == "embedding":
                models = [
                    m for m in models if m.capabilities and m.capabilities.embedding
                ]

        if q:
            q_lower = q.lower()
            models = [
                m
                for m in models
                if q_lower in m.model_name.lower()
                or q_lower in m.model_id.lower()
                or q_lower in m.provider.lower()
            ]

        display_models = [self._to_display_model(m) for m in models]
        providers = set(m.provider for m in data.models)

        return {
            "version": data.version,
            "source": "official",
            "last_sync": data.last_sync,
            "total_providers": len(providers),
            "total_models": len(display_models),
            "models": [m.model_dump() for m in display_models],
        }

    def get_by_id(self, model_id: str) -> ModelPricing | None:
        data = self._load_data()
        for model in data.models:
            if model.id == model_id:
                return model
        return None

    def get_by_provider(self) -> dict:
        data = self._load_data()

        provider_map: dict[str, list[ModelPricing]] = {}
        for model in data.models:
            if model.provider not in provider_map:
                provider_map[model.provider] = []
            provider_map[model.provider].append(model)

        provider_list = []
        for provider_id, models in provider_map.items():
            provider_docs = PROVIDER_DOCS.get(provider_id, {})
            model_dicts = sorted(
                [
                    {
                        "model_id": m.model_id,
                        "model_name": m.model_name,
                        "cost_input": m.pricing.input_tokens,
                        "cost_output": m.pricing.output_tokens,
                        "cost_cache_read": m.pricing.cached_input_tokens,
                        "cost_reasoning": m.pricing.reasoning_tokens,
                        "context_limit": m.context_length,
                        "output_limit": m.max_output_tokens,
                        "reasoning": m.capabilities.text if m.capabilities else None,
                        "vision": m.capabilities.vision if m.capabilities else None,
                        "source": m.source,
                        "deprecated": False,
                    }
                    for m in models
                ],
                key=lambda x: x.get("cost_input") or 0,
            )

            provider_list.append(
                {
                    "id": provider_id,
                    "name": PROVIDER_NAMES.get(provider_id, provider_id),
                    "doc": provider_docs.get("pricing"),
                    "api": provider_docs.get("api"),
                    "model_count": len(models),
                    "models": model_dicts,
                }
            )

        provider_list.sort(key=lambda x: x["model_count"], reverse=True)

        return {
            "version": data.version,
            "last_sync": data.last_sync,
            "total_providers": len(provider_list),
            "providers": provider_list,
        }

    def get_by_family(self) -> dict:
        data = self._load_data()
        display_models = [self._to_display_model(m) for m in data.models]

        family_groups: dict[str, dict] = {}
        for model in display_models:
            family = self._extract_model_family(model.model_name)
            if family not in family_groups:
                family_groups[family] = {"family": family, "models": []}
            family_groups[family]["models"].append(
                {
                    "provider_id": model.provider_id,
                    "provider_name": model.provider_name,
                    "model_id": model.model_id,
                    "model_name": model.model_name,
                    "cost_input": model.cost_input,
                    "cost_output": model.cost_output,
                    "context_limit": model.context_limit,
                    "source": model.source,
                }
            )

        # Only return families with multiple providers
        multi_provider_families = [
            group
            for group in family_groups.values()
            if len(set(m["provider_id"] for m in group["models"])) > 1
        ]
        multi_provider_families.sort(key=lambda x: x["family"])

        return {
            "version": data.version,
            "last_sync": data.last_sync,
            "families": multi_provider_families,
        }

    def _extract_model_family(self, model_name: str) -> str:
        lower_name = model_name.lower()

        patterns = [
            ("gpt-4o-mini", "GPT-4o mini"),
            ("gpt4o-mini", "GPT-4o mini"),
            ("gpt-4o", "GPT-4o"),
            ("gpt4o", "GPT-4o"),
            ("gpt-4-turbo", "GPT-4 Turbo"),
            ("gpt-4", "GPT-4"),
            ("gpt-3.5", "GPT-3.5"),
            ("o3-mini", "o3-mini"),
            ("o1-pro", "o1-pro"),
            ("o1-mini", "o1-mini"),
            ("o1", "o1"),
            ("claude-3.5-sonnet", "Claude 3.5 Sonnet"),
            ("claude 3.5 sonnet", "Claude 3.5 Sonnet"),
            ("claude-3-5-sonnet", "Claude 3.5 Sonnet"),
            ("claude-3.5-haiku", "Claude 3.5 Haiku"),
            ("claude 3.5 haiku", "Claude 3.5 Haiku"),
            ("claude-3-5-haiku", "Claude 3.5 Haiku"),
            ("claude-3-opus", "Claude 3 Opus"),
            ("claude 3 opus", "Claude 3 Opus"),
            ("claude-3-sonnet", "Claude 3 Sonnet"),
            ("claude 3 sonnet", "Claude 3 Sonnet"),
            ("claude-3-haiku", "Claude 3 Haiku"),
            ("claude 3 haiku", "Claude 3 Haiku"),
            ("gemini-2.0-flash", "Gemini 2.0 Flash"),
            ("gemini 2.0 flash", "Gemini 2.0 Flash"),
            ("gemini-1.5-pro", "Gemini 1.5 Pro"),
            ("gemini 1.5 pro", "Gemini 1.5 Pro"),
            ("gemini-1.5-flash-8b", "Gemini 1.5 Flash-8B"),
            ("gemini-1.5-flash", "Gemini 1.5 Flash"),
            ("gemini 1.5 flash", "Gemini 1.5 Flash"),
            ("gemini-1.0-pro", "Gemini 1.0 Pro"),
            ("grok-2-vision", "Grok 2 Vision"),
            ("grok-2", "Grok 2"),
            ("grok", "Grok"),
            ("llama-3.2", "Llama 3.2"),
            ("llama 3.2", "Llama 3.2"),
            ("llama-3.1", "Llama 3.1"),
            ("llama 3.1", "Llama 3.1"),
            ("llama-3", "Llama 3"),
            ("llama 3", "Llama 3"),
            ("mistral-large", "Mistral Large"),
            ("mistral-small", "Mistral Small"),
            ("mixtral", "Mixtral"),
            ("mistral", "Mistral"),
            ("nova-pro", "Amazon Nova Pro"),
            ("nova-lite", "Amazon Nova Lite"),
            ("nova-micro", "Amazon Nova Micro"),
            ("titan", "Amazon Titan"),
        ]

        for pattern, family in patterns:
            if pattern in lower_name:
                return family

        return model_name

    def get_providers(self) -> list[ProviderInfo]:
        api_providers = {"openrouter", "azure_openai", "aws_bedrock"}
        result = []

        for provider_id, name in PROVIDER_NAMES.items():
            docs = PROVIDER_DOCS.get(provider_id, {})
            result.append(
                ProviderInfo(
                    id=provider_id,
                    name=name,
                    pricing_url=docs.get("pricing"),
                    api_url=docs.get("api"),
                    source="api" if provider_id in api_providers else "manual",
                )
            )

        return result

    def get_stats(self) -> dict:
        data = self._load_data()
        display_models = [self._to_display_model(m) for m in data.models]

        with_vision = len(
            [m for m in display_models if m.input_modalities and "image" in m.input_modalities]
        )

        priced = [m for m in display_models if m.cost_input is not None]
        input_prices = sorted([m.cost_input for m in priced if m.cost_input is not None])

        providers = set(m.provider for m in data.models)
        api_models = len([m for m in data.models if m.source == "api"])
        manual_models = len([m for m in data.models if m.source == "manual"])

        return {
            "total_providers": len(providers),
            "total_models": len(display_models),
            "sources": {
                "api": api_models,
                "manual": manual_models,
            },
            "capabilities": {
                "vision": with_vision,
            },
            "pricing": {
                "models_with_price": len(priced),
                "min_input": input_prices[0] if input_prices else None,
                "max_input": input_prices[-1] if input_prices else None,
                "median_input": (
                    input_prices[len(input_prices) // 2] if input_prices else None
                ),
            },
            "last_sync": data.last_sync,
        }

    def save(self, models: list[ModelPricing]) -> None:
        from datetime import datetime

        db = UnifiedPricingDatabase(
            version="3.0.0",
            last_sync=datetime.now().isoformat(),
            models=models,
        )

        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self.data_path.write_text(
            db.model_dump_json(indent=2),
            encoding="utf-8",
        )
```

**Step 2: 更新 services/__init__.py**

```python
from app.services.pricing import PricingService

__all__ = ["PricingService"]
```

**Step 3: Commit**

```bash
git add backend/app/services/
git commit -m "feat: add PricingService for data access"
```

---

## Task 4: 实现 Provider 基类和 ManualProvider

**Files:**
- Create: `backend/app/providers/base.py`
- Create: `backend/app/providers/manual.py`

**Step 1: 创建 base.py**

```python
from abc import ABC, abstractmethod

from app.models import ModelPricing, SyncResult, ProviderType


class BaseProvider(ABC):
    id: ProviderType
    name: str
    source: str  # "api" or "manual"

    @abstractmethod
    async def fetch_models(self) -> list[ModelPricing]:
        """Fetch models from data source."""
        pass

    async def fetch(self) -> SyncResult:
        """Fetch with error handling."""
        try:
            models = await self.fetch_models()
            return SyncResult(
                provider=self.id,
                success=True,
                models_count=len(models),
            )
        except Exception as e:
            return SyncResult(
                provider=self.id,
                success=False,
                models_count=0,
                error=str(e),
            )
```

**Step 2: 创建 manual.py**

```python
import json
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.models import ModelPricing, ManualDataFile, ProviderType
from app.providers.base import BaseProvider


class ManualProvider(BaseProvider):
    source = "manual"

    def __init__(self, provider_id: ProviderType):
        self.id = provider_id
        self.name = provider_id
        self.file_path = settings.data_dir / "manual" / f"{provider_id}.json"

    async def fetch_models(self) -> list[ModelPricing]:
        content = self.file_path.read_text(encoding="utf-8")
        raw_data = json.loads(content)
        data = ManualDataFile.model_validate(raw_data)
        now = datetime.now().isoformat()

        return [
            ModelPricing(
                id=f"{self.id}:{model.model_id}",
                provider=self.id,
                model_id=model.model_id,
                model_name=model.model_name,
                pricing=model.pricing,
                billing_mode=model.billing_mode or "per_token",
                currency="USD",
                capabilities=model.capabilities,
                context_length=model.context_length,
                source="manual",
                source_url=data.source_url,
                last_updated=now,
                last_verified=data.last_verified,
                notes=model.notes,
            )
            for model in data.models
        ]


def get_manual_providers() -> list[ProviderType]:
    return ["openai", "xai", "google_vertex"]
```

**Step 3: 更新 providers/__init__.py**

```python
from app.providers.base import BaseProvider
from app.providers.manual import ManualProvider, get_manual_providers

__all__ = ["BaseProvider", "ManualProvider", "get_manual_providers"]
```

**Step 4: Commit**

```bash
git add backend/app/providers/
git commit -m "feat: add BaseProvider and ManualProvider"
```

---

## Task 5: 实现 OpenRouterProvider

**Files:**
- Create: `backend/app/providers/openrouter.py`

**Step 1: 创建 openrouter.py**

```python
from datetime import datetime

import httpx

from app.models import ModelPricing, Pricing, Capabilities, ProviderType
from app.providers.base import BaseProvider


class OpenRouterProvider(BaseProvider):
    id: ProviderType = "openrouter"
    name = "OpenRouter"
    source = "api"

    async def fetch_models(self) -> list[ModelPricing]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/models",
                timeout=30.0,
            )
            response.raise_for_status()

        data = response.json()
        now = datetime.now().isoformat()

        # Filter out router models (negative or special prices)
        valid_models = [
            model
            for model in data.get("data", [])
            if float(model.get("pricing", {}).get("prompt", "0")) >= 0
            and float(model.get("pricing", {}).get("completion", "0")) >= 0
            and not model.get("id", "").startswith("openrouter/")
        ]

        return [self._transform_model(model, now) for model in valid_models]

    def _transform_model(self, model: dict, now: str) -> ModelPricing:
        pricing = model.get("pricing", {})
        # OpenRouter prices are per-token, convert to per-million
        prompt_price = float(pricing.get("prompt", "0")) * 1_000_000
        completion_price = float(pricing.get("completion", "0")) * 1_000_000
        image_price = (
            float(pricing.get("image", "0")) if pricing.get("image") else None
        )

        modality = model.get("architecture", {}).get("modality", "")

        return ModelPricing(
            id=f"openrouter:{model['id']}",
            provider="openrouter",
            model_id=model["id"],
            model_name=model.get("name", model["id"]),
            pricing=Pricing(
                input_tokens=prompt_price,
                output_tokens=completion_price,
                image_input=image_price,
            ),
            billing_mode="per_token",
            currency="USD",
            capabilities=self._parse_capabilities(modality),
            context_length=model.get("context_length"),
            max_output_tokens=model.get("top_provider", {}).get("max_completion_tokens"),
            source="api",
            source_url="https://openrouter.ai/models",
            last_updated=now,
            last_verified=now,
        )

    def _parse_capabilities(self, modality: str) -> Capabilities:
        modality_lower = modality.lower() if modality else ""
        return Capabilities(
            text=True,
            vision="image" in modality_lower or "vision" in modality_lower,
            audio="audio" in modality_lower,
            embedding="embedding" in modality_lower,
            image_generation="image" in modality_lower and "generation" in modality_lower,
        )
```

**Step 2: 更新 providers/__init__.py**

```python
from app.providers.base import BaseProvider
from app.providers.manual import ManualProvider, get_manual_providers
from app.providers.openrouter import OpenRouterProvider

__all__ = [
    "BaseProvider",
    "ManualProvider",
    "get_manual_providers",
    "OpenRouterProvider",
]
```

**Step 3: Commit**

```bash
git add backend/app/providers/
git commit -m "feat: add OpenRouterProvider"
```

---

## Task 6: 实现 AzureProvider

**Files:**
- Create: `backend/app/providers/azure.py`

**Step 1: 创建 azure.py**

```python
import re
from datetime import datetime

import httpx

from app.models import ModelPricing, Pricing, Capabilities, ProviderType
from app.providers.base import BaseProvider


class AzureProvider(BaseProvider):
    id: ProviderType = "azure_openai"
    name = "Azure OpenAI"
    source = "api"

    async def fetch_models(self) -> list[ModelPricing]:
        base_url = "https://prices.azure.com/api/retail/prices"
        filter_query = (
            "(productName eq 'Azure OpenAI' or productName eq 'Azure OpenAI Service' "
            "or serviceName eq 'Azure OpenAI Service') and priceType eq 'Consumption'"
        )

        all_items = []
        next_page = f"{base_url}?api-version=2023-01-01-preview&$filter={filter_query}"
        page_count = 0
        max_pages = 30

        async with httpx.AsyncClient() as client:
            while next_page and page_count < max_pages:
                response = await client.get(next_page, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                all_items.extend(data.get("Items", []))
                next_page = data.get("NextPageLink")
                page_count += 1

        return self._transform_pricing(all_items)

    def _transform_pricing(self, items: list[dict]) -> list[ModelPricing]:
        model_map: dict[str, dict] = {}
        now = datetime.now().isoformat()

        # Filter token-related prices
        token_items = [
            item
            for item in items
            if self._is_token_item(item)
        ]

        for item in token_items:
            parsed = self._parse_meter_name(item["meterName"], item["skuName"])
            if not parsed:
                continue

            model_id = parsed["model_id"]
            if model_id not in model_map:
                model_map[model_id] = {
                    "model_id": model_id,
                    "model_name": parsed["model_name"],
                    "regions": set(),
                }

            model = model_map[model_id]
            price_per_million = self._calculate_price_per_million(item)

            if price_per_million is None or price_per_million <= 0 or price_per_million > 10000:
                continue

            if parsed["is_cached"]:
                if not model.get("cached_input_price") or price_per_million < model["cached_input_price"]:
                    model["cached_input_price"] = price_per_million
            elif parsed["is_input"]:
                if not model.get("input_price") or price_per_million < model["input_price"]:
                    model["input_price"] = price_per_million
            elif parsed["is_output"]:
                if not model.get("output_price") or price_per_million < model["output_price"]:
                    model["output_price"] = price_per_million

            model["regions"].add(item.get("armRegionName", ""))

        return [
            ModelPricing(
                id=f"azure_openai:{m['model_id']}",
                provider="azure_openai",
                model_id=m["model_id"],
                model_name=m["model_name"],
                pricing=Pricing(
                    input_tokens=m.get("input_price"),
                    output_tokens=m.get("output_price"),
                    cached_input_tokens=m.get("cached_input_price"),
                ),
                billing_mode="per_token",
                currency="USD",
                capabilities=self._infer_capabilities(m["model_id"]),
                source="api",
                source_url="https://azure.microsoft.com/pricing/details/cognitive-services/openai-service/",
                last_updated=now,
                last_verified=now,
                notes=f"Available in {len(m['regions'])} regions",
            )
            for m in model_map.values()
            if m.get("input_price") is not None or m.get("output_price") is not None
        ]

    def _is_token_item(self, item: dict) -> bool:
        unit = item.get("unitOfMeasure", "").lower()
        is_token_unit = "token" in unit or unit == "1k" or unit == "1m"
        is_consumption = item.get("type") == "Consumption"

        sku_name = item.get("skuName", "")
        is_not_provisioned = not any(
            x in sku_name for x in ["Provisioned", "Commitment", "PTU"]
        )

        meter_name = item.get("meterName", "").lower()
        has_token_indicator = any(
            x in meter_name
            for x in ["token", "in ", "out ", "input", "output", "cached"]
        )

        return is_token_unit and is_consumption and is_not_provisioned and has_token_indicator

    def _parse_meter_name(self, meter_name: str, sku_name: str) -> dict | None:
        meter_lower = meter_name.lower()

        is_input = "input" in meter_lower or (" in " in meter_lower and "cached" not in meter_lower)
        is_output = "output" in meter_lower or " out " in meter_lower
        is_cached = "cached" in meter_lower

        if not is_input and not is_output and not is_cached:
            return None

        # Extract model name
        model_name = re.sub(r"\s*(Input|Output|In|Out|Cached)\s*Tokens?", "", meter_name, flags=re.IGNORECASE)
        model_name = re.sub(r"\s*(Input|Output|In|Out)\s*", "", model_name, flags=re.IGNORECASE)
        model_name = re.sub(r"\s*\d+[KM]?\s*Tokens?", "", model_name, flags=re.IGNORECASE)
        model_name = re.sub(r"\s*DZ\s*$", "", model_name, flags=re.IGNORECASE)
        model_name = re.sub(r"\s*1M\s*$", "", model_name, flags=re.IGNORECASE)
        model_name = model_name.strip()

        if not model_name or len(model_name) < 2:
            model_name = re.sub(r"\s*(Standard|Premium|Basic)\s*", "", sku_name, flags=re.IGNORECASE).strip()

        if not model_name or len(model_name) < 2:
            return None

        model_id = re.sub(r"[^a-z0-9]+", "-", model_name.lower())
        model_id = re.sub(r"^-+|-+$", "", model_id)
        model_id = re.sub(r"--+", "-", model_id)

        display_name = self._normalize_model_name(model_name)

        return {
            "model_name": display_name,
            "model_id": model_id,
            "is_input": is_input and not is_output,
            "is_output": is_output,
            "is_cached": is_cached,
        }

    def _normalize_model_name(self, name: str) -> str:
        normalizations = [
            (r"gpt-?4o-?mini", "GPT-4o mini"),
            (r"gpt-?4o", "GPT-4o"),
            (r"gpt-?4-?turbo", "GPT-4 Turbo"),
            (r"gpt-?4-?32k", "GPT-4 32K"),
            (r"gpt-?4", "GPT-4"),
            (r"gpt-?3\.?5-?turbo", "GPT-3.5 Turbo"),
            (r"o1-?mini", "o1-mini"),
            (r"o1-?preview", "o1-preview"),
            (r"o3-?mini", "o3-mini"),
            (r"text-?embedding-?3-?large", "Text Embedding 3 Large"),
            (r"text-?embedding-?3-?small", "Text Embedding 3 Small"),
            (r"text-?embedding-?ada", "Text Embedding Ada"),
            (r"dall-?e-?3", "DALL-E 3"),
            (r"dall-?e-?2", "DALL-E 2"),
            (r"whisper", "Whisper"),
            (r"tts-?1-?hd", "TTS-1 HD"),
            (r"tts-?1", "TTS-1"),
        ]

        for pattern, replacement in normalizations:
            if re.search(pattern, name, re.IGNORECASE):
                return replacement

        return " ".join(word.capitalize() for word in name.split())

    def _calculate_price_per_million(self, item: dict) -> float | None:
        unit_lower = item.get("unitOfMeasure", "").lower()
        price = item.get("retailPrice", 0)

        if "1m" in unit_lower:
            return price
        elif "1k" in unit_lower or "1000" in unit_lower:
            return price * 1000
        elif "token" in unit_lower:
            return price * 1_000_000
        else:
            return price * 1000  # Default assume 1K

    def _infer_capabilities(self, model_id: str) -> Capabilities:
        id_lower = model_id.lower()
        return Capabilities(
            text=not any(x in id_lower for x in ["embedding", "dall", "whisper", "tts"]),
            vision="4o" in id_lower or "gpt-4-turbo" in id_lower or "vision" in id_lower,
            audio="whisper" in id_lower or "tts" in id_lower or "4o" in id_lower,
            embedding="embedding" in id_lower,
            image_generation="dall" in id_lower,
        )
```

**Step 2: 更新 providers/__init__.py 添加 AzureProvider**

```python
from app.providers.base import BaseProvider
from app.providers.manual import ManualProvider, get_manual_providers
from app.providers.openrouter import OpenRouterProvider
from app.providers.azure import AzureProvider

__all__ = [
    "BaseProvider",
    "ManualProvider",
    "get_manual_providers",
    "OpenRouterProvider",
    "AzureProvider",
]
```

**Step 3: Commit**

```bash
git add backend/app/providers/
git commit -m "feat: add AzureProvider"
```

---

## Task 7: 实现 AWSBedrockProvider

**Files:**
- Create: `backend/app/providers/aws_bedrock.py`

**Step 1: 创建 aws_bedrock.py**

```python
import re
from datetime import datetime

import httpx

from app.models import ModelPricing, Pricing, Capabilities, ProviderType
from app.providers.base import BaseProvider


AWS_PRICE_BASE_URL = "https://pricing.us-east-1.amazonaws.com"
BEDROCK_SERVICE_CODE = "AmazonBedrock"


class AWSBedrockProvider(BaseProvider):
    id: ProviderType = "aws_bedrock"
    name = "AWS Bedrock"
    source = "api"

    async def fetch_models(self) -> list[ModelPricing]:
        price_list_url = f"{AWS_PRICE_BASE_URL}/offers/v1.0/aws/{BEDROCK_SERVICE_CODE}/current/us-east-1/index.json"

        async with httpx.AsyncClient() as client:
            response = await client.get(price_list_url, timeout=60.0)
            response.raise_for_status()

        data = response.json()
        return self._transform_pricing(data)

    def _transform_pricing(self, data: dict) -> list[ModelPricing]:
        model_map: dict[str, dict] = {}
        now = datetime.now().isoformat()
        products = data.get("products", {})
        terms = data.get("terms", {}).get("OnDemand", {})

        for sku, product in products.items():
            usagetype = product.get("attributes", {}).get("usagetype", "")

            if not usagetype or "Guardrail" in usagetype or "CustomModel" in usagetype:
                continue

            model_info = self._parse_usage_type(usagetype)
            if not model_info:
                continue

            model_id = model_info["model_id"]
            if model_id not in model_map:
                model_map[model_id] = {
                    "model_id": model_id,
                    "model_name": model_info["model_name"],
                    "vision": self._has_vision_capability(model_id),
                    "context_length": self._get_context_length(model_id),
                }

            model = model_map[model_id]
            term_data = terms.get(sku, {})

            for offer in term_data.values():
                for dim in offer.get("priceDimensions", {}).values():
                    price_usd = float(dim.get("pricePerUnit", {}).get("USD", "0"))
                    if price_usd <= 0:
                        continue

                    unit = dim.get("unit", "").lower()
                    if "request" in unit:
                        continue

                    price_per_million = price_usd * 1000  # 1K -> 1M

                    if model_info["is_cached"]:
                        if not model.get("cached_input_per_million") or price_per_million < model["cached_input_per_million"]:
                            model["cached_input_per_million"] = price_per_million
                    elif model_info["is_input"]:
                        if not model.get("input_per_million") or price_per_million < model["input_per_million"]:
                            model["input_per_million"] = price_per_million
                    elif model_info["is_output"]:
                        if not model.get("output_per_million") or price_per_million < model["output_per_million"]:
                            model["output_per_million"] = price_per_million

        return [
            ModelPricing(
                id=f"aws_bedrock:{m['model_id']}",
                provider="aws_bedrock",
                model_id=m["model_id"],
                model_name=m["model_name"],
                pricing=Pricing(
                    input_tokens=m.get("input_per_million"),
                    output_tokens=m.get("output_per_million"),
                    cached_input_tokens=m.get("cached_input_per_million"),
                ),
                billing_mode="per_token",
                currency="USD",
                capabilities=self._infer_capabilities(m["model_id"], m.get("vision")),
                context_length=m.get("context_length"),
                source="api",
                source_url="https://aws.amazon.com/bedrock/pricing/",
                last_updated=now,
                last_verified=now,
            )
            for m in model_map.values()
            if m.get("input_per_million") is not None or m.get("output_per_million") is not None
        ]

    def _parse_usage_type(self, usagetype: str) -> dict | None:
        # Remove region prefix
        cleaned = re.sub(r"^[A-Z]{2,4}\d?-", "", usagetype)
        lower = cleaned.lower()

        is_input = "-input" in lower or "input-" in lower
        is_output = "-output" in lower or "output-" in lower
        is_cached = "cached" in lower or "cache" in lower

        if not any(x in lower for x in ["token", "-input", "-output"]):
            return None

        if any(x in lower for x in ["provisioned", "batch", "commitment", "grounding", "video", "audio"]):
            return None

        model_info = self._extract_model_info(cleaned)
        if not model_info:
            return None

        return {
            **model_info,
            "is_input": is_input and not is_output,
            "is_output": is_output,
            "is_cached": is_cached,
        }

    def _extract_model_info(self, usagetype: str) -> dict | None:
        model_patterns = [
            # Amazon Nova
            (r"nova2\.?0pro|nova-?pro", "amazon.nova-pro-v1:0", "Amazon Nova Pro"),
            (r"nova2\.?0lite|nova-?lite", "amazon.nova-lite-v1:0", "Amazon Nova Lite"),
            (r"nova2\.?0micro|nova-?micro", "amazon.nova-micro-v1:0", "Amazon Nova Micro"),
            (r"nova2\.?0omni|nova-?omni", "amazon.nova-omni-v1:0", "Amazon Nova Omni"),
            # Anthropic Claude 3.5
            (r"claude-?3-?5-?sonnet-?v2|claude3\.5sonnetv2", "anthropic.claude-3-5-sonnet-v2", "Claude 3.5 Sonnet v2"),
            (r"claude-?3-?5-?sonnet|claude3\.5sonnet", "anthropic.claude-3-5-sonnet", "Claude 3.5 Sonnet"),
            (r"claude-?3-?5-?haiku|claude3\.5haiku", "anthropic.claude-3-5-haiku", "Claude 3.5 Haiku"),
            # Anthropic Claude 3
            (r"claude-?3-?opus|claude3opus", "anthropic.claude-3-opus", "Claude 3 Opus"),
            (r"claude-?3-?sonnet|claude3sonnet", "anthropic.claude-3-sonnet", "Claude 3 Sonnet"),
            (r"claude-?3-?haiku|claude3haiku", "anthropic.claude-3-haiku", "Claude 3 Haiku"),
            # Anthropic Claude 2/Instant
            (r"claude-?instant", "anthropic.claude-instant", "Claude Instant"),
            (r"claude-?2", "anthropic.claude-2", "Claude 2"),
            # Meta Llama
            (r"llama-?3-?3-?70b|llama3\.3-70b", "meta.llama-3-3-70b-instruct", "Llama 3.3 70B Instruct"),
            (r"llama-?3-?2-?90b|llama3\.2-90b", "meta.llama-3-2-90b-instruct", "Llama 3.2 90B Instruct"),
            (r"llama-?3-?2-?11b|llama3\.2-11b", "meta.llama-3-2-11b-instruct", "Llama 3.2 11B Instruct"),
            (r"llama-?3-?2-?3b|llama3\.2-3b", "meta.llama-3-2-3b-instruct", "Llama 3.2 3B Instruct"),
            (r"llama-?3-?2-?1b|llama3\.2-1b", "meta.llama-3-2-1b-instruct", "Llama 3.2 1B Instruct"),
            (r"llama-?3-?1-?405b|llama3\.1-405b", "meta.llama-3-1-405b-instruct", "Llama 3.1 405B Instruct"),
            (r"llama-?3-?1-?70b|llama3\.1-70b", "meta.llama-3-1-70b-instruct", "Llama 3.1 70B Instruct"),
            (r"llama-?3-?1-?8b|llama3\.1-8b", "meta.llama-3-1-8b-instruct", "Llama 3.1 8B Instruct"),
            (r"llama-?3-?70b|llama3-70b", "meta.llama-3-70b-instruct", "Llama 3 70B Instruct"),
            (r"llama-?3-?8b|llama3-8b", "meta.llama-3-8b-instruct", "Llama 3 8B Instruct"),
            # Mistral
            (r"mistral-?large-?2", "mistral.mistral-large-2", "Mistral Large 2"),
            (r"mistral-?large", "mistral.mistral-large", "Mistral Large"),
            (r"mistral-?small", "mistral.mistral-small", "Mistral Small"),
            (r"mixtral-?8x7b", "mistral.mixtral-8x7b-instruct", "Mixtral 8x7B Instruct"),
            (r"mistral-?7b", "mistral.mistral-7b-instruct", "Mistral 7B Instruct"),
            # Cohere
            (r"command-?r-?plus", "cohere.command-r-plus", "Cohere Command R+"),
            (r"command-?r(?!-?plus)", "cohere.command-r", "Cohere Command R"),
            (r"cohere-?embed", "cohere.embed", "Cohere Embed"),
            # AI21
            (r"jamba-?1-?5-?large", "ai21.jamba-1-5-large", "AI21 Jamba 1.5 Large"),
            (r"jamba-?1-?5-?mini", "ai21.jamba-1-5-mini", "AI21 Jamba 1.5 Mini"),
            (r"jamba-?instruct", "ai21.jamba-instruct", "AI21 Jamba Instruct"),
            (r"jurassic", "ai21.jurassic", "AI21 Jurassic"),
            # Amazon Titan
            (r"titan-?text-?premier", "amazon.titan-text-premier", "Amazon Titan Text Premier"),
            (r"titan-?text-?express", "amazon.titan-text-express", "Amazon Titan Text Express"),
            (r"titan-?text-?lite", "amazon.titan-text-lite", "Amazon Titan Text Lite"),
            (r"titan-?embed", "amazon.titan-embed", "Amazon Titan Embed"),
            # DeepSeek
            (r"deepseek-?r1", "deepseek.deepseek-r1", "DeepSeek R1"),
        ]

        for pattern, model_id, model_name in model_patterns:
            if re.search(pattern, usagetype, re.IGNORECASE):
                return {"model_id": model_id, "model_name": model_name}

        return None

    def _has_vision_capability(self, model_id: str) -> bool:
        id_lower = model_id.lower()
        return any(x in id_lower for x in ["claude-3", "nova", "llama-3-2-90b", "llama-3-2-11b"])

    def _get_context_length(self, model_id: str) -> int | None:
        context_lengths = {
            "amazon.nova-pro-v1:0": 300000,
            "amazon.nova-lite-v1:0": 300000,
            "amazon.nova-micro-v1:0": 128000,
            "anthropic.claude-3-5-sonnet-v2": 200000,
            "anthropic.claude-3-5-sonnet": 200000,
            "anthropic.claude-3-5-haiku": 200000,
            "anthropic.claude-3-opus": 200000,
            "anthropic.claude-3-sonnet": 200000,
            "anthropic.claude-3-haiku": 200000,
            "meta.llama-3-3-70b-instruct": 128000,
            "meta.llama-3-2-90b-instruct": 128000,
            "meta.llama-3-2-11b-instruct": 128000,
            "meta.llama-3-1-405b-instruct": 128000,
            "meta.llama-3-1-70b-instruct": 128000,
            "meta.llama-3-1-8b-instruct": 128000,
            "mistral.mistral-large-2": 128000,
            "mistral.mistral-large": 128000,
            "mistral.mixtral-8x7b-instruct": 32000,
        }
        return context_lengths.get(model_id)

    def _infer_capabilities(self, model_id: str, has_vision: bool | None) -> Capabilities:
        id_lower = model_id.lower()
        return Capabilities(
            text=True,
            vision=has_vision or False,
            audio="nova-omni" in id_lower,
            embedding="embed" in id_lower or "titan-embed" in id_lower,
            image_generation="titan-image" in id_lower or "stable-diffusion" in id_lower,
        )
```

**Step 2: 更新 providers/__init__.py**

```python
from app.providers.base import BaseProvider
from app.providers.manual import ManualProvider, get_manual_providers
from app.providers.openrouter import OpenRouterProvider
from app.providers.azure import AzureProvider
from app.providers.aws_bedrock import AWSBedrockProvider

__all__ = [
    "BaseProvider",
    "ManualProvider",
    "get_manual_providers",
    "OpenRouterProvider",
    "AzureProvider",
    "AWSBedrockProvider",
]
```

**Step 3: Commit**

```bash
git add backend/app/providers/
git commit -m "feat: add AWSBedrockProvider"
```

---

## Task 8: 实现 SyncService

**Files:**
- Create: `backend/app/services/sync.py`

**Step 1: 创建 sync.py**

```python
import asyncio

from app.models import ModelPricing, SyncResult, ProviderType
from app.providers import (
    BaseProvider,
    ManualProvider,
    OpenRouterProvider,
    AzureProvider,
    AWSBedrockProvider,
    get_manual_providers,
)
from app.services.pricing import PricingService


class SyncService:
    def __init__(self, pricing_service: PricingService | None = None):
        self.pricing_service = pricing_service or PricingService()

    def _create_all_providers(self) -> list[BaseProvider]:
        providers: list[BaseProvider] = [
            OpenRouterProvider(),
            AzureProvider(),
            AWSBedrockProvider(),
        ]
        providers.extend(ManualProvider(p) for p in get_manual_providers())
        return providers

    def _get_provider_by_name(self, name: str) -> BaseProvider | None:
        providers: dict[str, type[BaseProvider] | tuple] = {
            "openrouter": OpenRouterProvider,
            "azure_openai": AzureProvider,
            "aws_bedrock": AWSBedrockProvider,
            "google_vertex": (ManualProvider, "google_vertex"),
            "openai": (ManualProvider, "openai"),
            "xai": (ManualProvider, "xai"),
        }

        provider_info = providers.get(name)
        if provider_info is None:
            return None

        if isinstance(provider_info, tuple):
            cls, provider_id = provider_info
            return cls(provider_id)
        return provider_info()

    async def sync_all(self) -> dict:
        providers = self._create_all_providers()
        all_models: list[ModelPricing] = []
        provider_stats: list[dict] = []

        # Fetch all providers in parallel
        tasks = [self._fetch_provider(p) for p in providers]
        results = await asyncio.gather(*tasks)

        for models, stat in results:
            all_models.extend(models)
            provider_stats.append(stat)

        # Save data
        self.pricing_service.save(all_models)

        successful_providers = len([s for s in provider_stats if s["success"]])
        api_providers = len([s for s in provider_stats if s["source"] == "api"])
        manual_providers = len([s for s in provider_stats if s["source"] == "manual"])

        return {
            "success": True,
            "totalModels": len(all_models),
            "totalProviders": len(provider_stats),
            "successfulProviders": successful_providers,
            "apiProviders": api_providers,
            "manualProviders": manual_providers,
            "providers": provider_stats,
        }

    async def sync_provider(self, provider_name: str) -> dict:
        provider = self._get_provider_by_name(provider_name)
        if not provider:
            return {"success": False, "error": f"Unknown provider: {provider_name}"}

        models, stat = await self._fetch_provider(provider)

        # Load existing data and merge
        try:
            existing = self.pricing_service._load_data()
            other_models = [m for m in existing.models if m.provider != provider_name]
            all_models = other_models + models
        except Exception:
            all_models = models

        self.pricing_service.save(all_models)

        return {
            "success": stat["success"],
            "totalModels": len(all_models),
            "totalProviders": 1,
            "successfulProviders": 1 if stat["success"] else 0,
            "providers": [stat],
        }

    async def _fetch_provider(self, provider: BaseProvider) -> tuple[list[ModelPricing], dict]:
        try:
            models = await provider.fetch_models()
            source = getattr(provider, "source", "unknown")
            return models, {
                "provider": provider.id,
                "count": len(models),
                "success": True,
                "source": source,
            }
        except Exception as e:
            return [], {
                "provider": provider.id,
                "count": 0,
                "success": False,
                "error": str(e),
                "source": getattr(provider, "source", "unknown"),
            }
```

**Step 2: 更新 services/__init__.py**

```python
from app.services.pricing import PricingService
from app.services.sync import SyncService

__all__ = ["PricingService", "SyncService"]
```

**Step 3: Commit**

```bash
git add backend/app/services/
git commit -m "feat: add SyncService for data synchronization"
```

---

## Task 9: 实现 API 路由

**Files:**
- Create: `backend/app/routes/pricing.py`
- Modify: `backend/app/main.py`

**Step 1: 创建 routes/pricing.py**

```python
from fastapi import APIRouter, Query, HTTPException

from app.services import PricingService, SyncService

router = APIRouter(prefix="/api", tags=["pricing"])

pricing_service = PricingService()
sync_service = SyncService(pricing_service)


@router.get("/models")
async def get_models(
    provider: str | None = Query(None, description="Filter by provider"),
    capability: str | None = Query(None, description="Filter by capability"),
    q: str | None = Query(None, description="Search query"),
):
    try:
        return pricing_service.get_all(provider=provider, capability=capability, q=q)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="No pricing data. Run sync first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/by-provider")
async def get_models_by_provider():
    try:
        return pricing_service.get_by_provider()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="No pricing data. Run sync first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/by-family")
async def get_models_by_family():
    try:
        return pricing_service.get_by_family()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="No pricing data. Run sync first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id:path}")
async def get_model(model_id: str):
    model = pricing_service.get_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model.model_dump()


@router.get("/providers")
async def get_providers():
    providers = pricing_service.get_providers()
    return {"providers": [p.model_dump() for p in providers]}


@router.get("/stats")
async def get_stats():
    try:
        return pricing_service.get_stats()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="No pricing data. Run sync first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync")
async def sync_data(provider: str | None = Query(None, description="Sync specific provider")):
    if provider:
        return await sync_service.sync_provider(provider)
    return await sync_service.sync_all()


# Legacy endpoints for backward compatibility with frontend
@router.get("/pricing")
async def get_pricing():
    """Legacy endpoint - redirects to /models"""
    return await get_models()


@router.get("/pricing/by-provider")
async def get_pricing_by_provider():
    """Legacy endpoint - redirects to /models/by-provider"""
    return await get_models_by_provider()


@router.get("/pricing/by-model")
async def get_pricing_by_model():
    """Legacy endpoint - redirects to /models/by-family"""
    return await get_models_by_family()


@router.get("/search")
async def search_models(q: str = Query("", description="Search query")):
    """Legacy endpoint - redirects to /models with query"""
    return await get_models(q=q)
```

**Step 2: 更新 routes/__init__.py**

```python
from app.routes.pricing import router as pricing_router

__all__ = ["pricing_router"]
```

**Step 3: 更新 main.py 引入路由**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import pricing_router

app = FastAPI(title="Model Price API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pricing_router)


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
```

**Step 4: 验证 API 启动**

Run: `cd backend && python -m uvicorn app.main:app --reload --port 3001`
Expected: Server starts, all endpoints available

**Step 5: Commit**

```bash
git add backend/app/
git commit -m "feat: add API routes for pricing endpoints"
```

---

## Task 10: 更新前端 API 调用

**Files:**
- Create: `web/src/api.ts`
- Modify: `web/src/App.tsx`
- Modify: `web/src/components/ProviderView.tsx`
- Modify: `web/src/components/ModelView.tsx`
- Modify: `web/src/components/SearchView.tsx`

**Step 1: 创建 api.ts**

```typescript
const API_BASE = '/api';

export const api = {
  getModels: async (params?: { provider?: string; q?: string }) => {
    const searchParams = new URLSearchParams();
    if (params?.provider) searchParams.set('provider', params.provider);
    if (params?.q) searchParams.set('q', params.q);
    const query = searchParams.toString();
    return fetch(`${API_BASE}/models${query ? `?${query}` : ''}`);
  },

  getModelsByProvider: () => fetch(`${API_BASE}/models/by-provider`),

  getModelsByFamily: () => fetch(`${API_BASE}/models/by-family`),

  getProviders: () => fetch(`${API_BASE}/providers`),

  getStats: () => fetch(`${API_BASE}/stats`),

  sync: (provider?: string) =>
    fetch(`${API_BASE}/sync${provider ? `?provider=${provider}` : ''}`, {
      method: 'POST',
    }),

  // Legacy aliases
  getPricing: () => fetch(`${API_BASE}/pricing`),
  getPricingByProvider: () => fetch(`${API_BASE}/pricing/by-provider`),
  getPricingByModel: () => fetch(`${API_BASE}/pricing/by-model`),
  search: (q: string) => fetch(`${API_BASE}/search?q=${encodeURIComponent(q)}`),
};
```

**Step 2: 更新 App.tsx**

Replace `fetch("/api/pricing")` with `fetch("/api/models")` in loadLastSync:

```typescript
const loadLastSync = useCallback(async () => {
  try {
    const res = await fetch("/api/models");
    const data = await res.json();
    setLastSync(data.last_sync);
  } catch {
    // ignore
  }
}, []);
```

Remove the validate button and handler (validate endpoint removed).

**Step 3: Commit**

```bash
git add web/src/
git commit -m "feat: add API wrapper and update frontend"
```

---

## Task 11: 更新 package.json 脚本

**Files:**
- Modify: `package.json`

**Step 1: 更新 scripts**

```json
{
  "scripts": {
    "dev": "concurrently \"npm run server\" \"npm run web\"",
    "server": "cd backend && python -m uvicorn app.main:app --reload --port 3001",
    "web": "vite",
    "build:web": "vite build"
  }
}
```

Remove TypeScript-specific scripts (build, sync:*, validate, list).

**Step 2: 清理依赖**

Remove server dependencies no longer needed:
- `commander`
- `cors`
- `express`
- `zod`
- `@types/cors`
- `@types/express`
- `tsx`

Keep:
- `concurrently`
- React and Vite dependencies

**Step 3: Commit**

```bash
git add package.json
git commit -m "chore: update package.json for Python backend"
```

---

## Task 12: 删除 TypeScript 后端代码

**Files:**
- Delete: `src/` directory
- Delete: `tsconfig.json`

**Step 1: 删除 src 目录**

Run: `rm -rf src/`

**Step 2: 删除 tsconfig.json**

Run: `rm tsconfig.json`

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove TypeScript backend code"
```

---

## Task 13: 安装 Python 依赖并测试

**Step 1: 创建虚拟环境**

Run: `cd backend && python -m venv venv`

**Step 2: 安装依赖**

Run: `cd backend && source venv/bin/activate && pip install -r requirements.txt`

**Step 3: 启动后端**

Run: `cd backend && source venv/bin/activate && uvicorn app.main:app --reload --port 3001`

**Step 4: 测试同步**

Run (in another terminal): `curl -X POST http://localhost:3001/api/sync`
Expected: Returns sync result with models

**Step 5: 测试前端**

Run: `npm run web`
Open: `http://localhost:5173`
Expected: Frontend loads and shows synced data

**Step 6: Commit**

```bash
git add -A
git commit -m "chore: complete Python backend migration"
```

---

## Summary

完成后的项目结构:

```
model-price/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── pricing.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── pricing.py
│   │   │   └── sync.py
│   │   ├── providers/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── openrouter.py
│   │   │   ├── azure.py
│   │   │   ├── aws_bedrock.py
│   │   │   └── manual.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       └── pricing.py
│   ├── requirements.txt
│   └── venv/
├── web/
├── data/
├── docs/plans/
├── package.json
├── vite.config.ts
└── README.md
```
