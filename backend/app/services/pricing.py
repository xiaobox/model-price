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
