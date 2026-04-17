"""Verify that offerings from LiteLLM-mirrored providers carry the
``source="via_litellm"`` tag so the UI can label their pricing row
as a two-hop mirror rather than a one-hop first-party fetch.

Providers whose data chain runs through LiteLLM's community JSON:
anthropic / openai / xai / deepseek / google_vertex_ai. Everything
else — aws_bedrock / azure_ai / openrouter / google_gemini — must
stay ``source="provider_api"`` because those pull directly from the
vendor's own API.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from models import ModelPricing, Pricing
from services.offering_merger import (
    LITELLM_SOURCED_PROVIDERS,
    OfferingMerger,
)


def _model(provider: str) -> ModelPricing:
    return ModelPricing(
        id=f"{provider}:test",
        provider=provider,
        model_id="test-model",
        model_name="Test",
        pricing=Pricing(input=1.0, output=2.0),
        last_updated=datetime.utcnow(),
    )


def _merger() -> OfferingMerger:
    return OfferingMerger(MagicMock(), MagicMock())


class TestViaLitellmSourceTag:
    def test_constant_lists_expected_providers(self) -> None:
        """If someone adds or removes a LiteLLM-mirrored provider they
        must also add/remove it here, so this test is a tripwire."""
        assert LITELLM_SOURCED_PROVIDERS == frozenset({
            "anthropic",
            "openai",
            "xai",
            "deepseek",
            "google_vertex_ai",
            "mistral",
            "moonshot",
            "cohere",
            "ai21",
            "alibaba_qwen",
            "zai",
            "minimax",
            "volcengine",
            "gigachat",
            "meta_llama",
        })

    @pytest.mark.parametrize(
        "provider",
        sorted(LITELLM_SOURCED_PROVIDERS),
    )
    def test_litellm_mirrored_providers_tagged_via_litellm(
        self, provider: str
    ) -> None:
        offering = _merger()._offering_from_v1(
            _model(provider), provider, datetime.utcnow()
        )
        assert offering.source == "via_litellm"

    @pytest.mark.parametrize(
        "provider",
        ["aws_bedrock", "azure_ai", "openrouter", "google_gemini"],
    )
    def test_direct_providers_tagged_provider_api(self, provider: str) -> None:
        offering = _merger()._offering_from_v1(
            _model(provider), provider, datetime.utcnow()
        )
        assert offering.source == "provider_api"
