"""Tests for _dedupe_offerings_per_provider.

OpenRouter publishes every pinned-date snapshot of a model
(openai/gpt-4o-2024-05-13, openai/gpt-4o-2024-08-06, openai/gpt-4o)
as a separate entry. Canonical resolution maps them all to the same
entity, so the detail page would otherwise show four "OpenRouter" rows
for gpt-4o — with one of them carrying the stale 2024-05 price.

These tests pin the collapse rules so a future refactor can't
regress the UX.
"""

from datetime import datetime
from typing import Optional

from models.v2 import OfferingV2, PricingV2
from services.offering_merger import (
    _dedupe_offerings_per_provider,
    _looks_date_pinned,
)


def _off(
    provider: str,
    model_id: str,
    *,
    input_price: Optional[float] = 2.5,
    output_price: Optional[float] = 10.0,
    source: str = "provider_api",
) -> OfferingV2:
    return OfferingV2(
        provider=provider,
        provider_model_id=model_id,
        pricing=PricingV2(input=input_price, output=output_price),
        batch_pricing=None,
        availability="ga",
        region=None,
        notes=None,
        last_updated=datetime.utcnow(),
        source=source,  # type: ignore[arg-type]
    )


class TestLooksDatePinned:
    def test_dashed_date_suffix_matches(self) -> None:
        assert _looks_date_pinned("openai/gpt-4o-2024-11-20") is True
        assert _looks_date_pinned("openai/gpt-4o-2024-08-06") is True
        assert _looks_date_pinned("openai/gpt-4o-2024-05-13") is True

    def test_compact_8_digit_suffix_matches(self) -> None:
        assert _looks_date_pinned("anthropic/claude-sonnet-4-5-20250929") is True
        assert _looks_date_pinned("qwen-plus-20250728") is True

    def test_undated_id_does_not_match(self) -> None:
        assert _looks_date_pinned("openai/gpt-4o") is False
        assert _looks_date_pinned("anthropic/claude-opus-4-7") is False
        assert _looks_date_pinned("qwen/qwen-plus") is False

    def test_version_suffix_not_mistaken_for_date(self) -> None:
        """Version numbers like v1.0 / 4-5 must not trip the date regex."""
        assert _looks_date_pinned("anthropic/claude-opus-4-1") is False
        assert _looks_date_pinned("bedrock/claude-opus-4-6-v1") is False


class TestDedupeOfferingsPerProvider:
    def test_keeps_single_offering_untouched(self) -> None:
        offerings = [_off("openrouter", "openai/gpt-4o")]
        assert _dedupe_offerings_per_provider(offerings) == offerings

    def test_collapses_gpt4o_style_date_pinned_duplicates(self) -> None:
        """The exact production bug: 4 OpenRouter rows for gpt-4o."""
        offerings = [
            _off("openrouter", "openai/gpt-4o-2024-11-20"),
            _off("openrouter", "openai/gpt-4o-2024-08-06"),
            _off("openrouter", "openai/gpt-4o-2024-05-13",
                 input_price=5.0, output_price=15.0),  # stale price
            _off("openrouter", "openai/gpt-4o"),  # un-dated alias
        ]
        result = _dedupe_offerings_per_provider(offerings)
        assert len(result) == 1
        # Un-dated alias wins because it's complete AND un-dated.
        assert result[0].provider_model_id == "openai/gpt-4o"
        # Stale 2024-05 pricing is not surfaced.
        assert result[0].pricing.input == 2.5
        assert result[0].pricing.output == 10.0

    def test_prefers_complete_pricing_over_incomplete(self) -> None:
        """Even if an un-dated alias exists, an offering with null
        output should not win over a dated offering that has both
        fields populated."""
        offerings = [
            _off(
                "openrouter",
                "openai/gpt-4o",
                input_price=2.5,
                output_price=None,
            ),
            _off(
                "openrouter",
                "openai/gpt-4o-2024-11-20",
                input_price=2.5,
                output_price=10.0,
            ),
        ]
        result = _dedupe_offerings_per_provider(offerings)
        assert len(result) == 1
        assert result[0].provider_model_id == "openai/gpt-4o-2024-11-20"

    def test_keeps_different_providers_separate(self) -> None:
        """Dedupe only collapses within the same provider."""
        offerings = [
            _off("openai", "openai/gpt-4o"),
            _off("openrouter", "openai/gpt-4o"),
            _off("azure_ai", "gpt-4o"),
        ]
        result = _dedupe_offerings_per_provider(offerings)
        assert {o.provider for o in result} == {
            "openai",
            "openrouter",
            "azure_ai",
        }
        assert len(result) == 3

    def test_preserves_insertion_order_across_providers(self) -> None:
        offerings = [
            _off("anthropic", "claude-opus-4-7"),
            _off("aws_bedrock", "claude-opus-4.7"),
            _off("openrouter", "anthropic/claude-opus-4.7"),
        ]
        result = _dedupe_offerings_per_provider(offerings)
        assert [o.provider for o in result] == [
            "anthropic",
            "aws_bedrock",
            "openrouter",
        ]

    def test_all_dated_picks_first(self) -> None:
        """Stable tie-break: if every variant is dated and complete,
        keep the first one seen so results are deterministic."""
        offerings = [
            _off("openrouter", "qwen/qwen-plus-2025-07-28"),
            _off("openrouter", "qwen/qwen-plus-2024-12-01"),
        ]
        result = _dedupe_offerings_per_provider(offerings)
        assert len(result) == 1
        assert result[0].provider_model_id == "qwen/qwen-plus-2025-07-28"

    def test_empty_input_returns_empty(self) -> None:
        assert _dedupe_offerings_per_provider([]) == []
