"""Tests for OfferingMerger._choose_primary and its completeness gate.

The gate's point is: never surface a provider whose headline price
would render as "—" in the list row when there's a better candidate.
The canonical real-world case is Bedrock publishing a freshly-released
Claude with ``input=null`` on day one while the Anthropic first-party
offering has the full numbers. We want authority to bend to data
quality there.
"""

from datetime import datetime
from typing import List, Optional
from unittest.mock import MagicMock

from models.v2 import EntityCoreV2, OfferingV2, PricingV2
from services.offering_merger import OfferingMerger, _has_complete_headline_pricing


def _entity(maker: str, mode: str = "chat") -> EntityCoreV2:
    now = datetime.utcnow()
    return EntityCoreV2(
        canonical_id="test-model",
        slug="test-model",
        name="Test Model",
        family="Test",
        maker=maker,
        context_length=100_000,
        max_output_tokens=8_000,
        capabilities=["text"],
        input_modalities=["text"],
        output_modalities=["text"],
        mode=mode,
        is_open_source=False,
        primary_offering_provider="",
        sources=[],
        last_refreshed=now,
    )


def _offering(
    provider: str,
    *,
    input_price: Optional[float] = 1.0,
    output_price: Optional[float] = 2.0,
    source: str = "provider_api",
    image_input: Optional[float] = None,
    embedding_price: Optional[float] = None,
    audio_input: Optional[float] = None,
    audio_output: Optional[float] = None,
) -> OfferingV2:
    return OfferingV2(
        provider=provider,
        provider_model_id=f"{provider}:test",
        pricing=PricingV2(
            input=input_price,
            output=output_price,
            image_input=image_input,
            embedding=embedding_price,
            audio_input=audio_input,
            audio_output=audio_output,
        ),
        batch_pricing=None,
        availability="ga",
        region=None,
        notes=None,
        last_updated=datetime.utcnow(),
        source=source,  # type: ignore[arg-type]
    )


def _merger() -> OfferingMerger:
    # _choose_primary does not touch registry / resolver state, so
    # lightweight MagicMocks keep the test hermetic and fast.
    return OfferingMerger(MagicMock(), MagicMock())


# ─── _has_complete_headline_pricing ─────────────────────────────


class TestHasCompleteHeadlinePricing:
    def test_chat_needs_both_input_and_output(self) -> None:
        assert _has_complete_headline_pricing(_offering("x"), "chat") is True
        assert (
            _has_complete_headline_pricing(
                _offering("x", input_price=None), "chat"
            )
            is False
        )
        assert (
            _has_complete_headline_pricing(
                _offering("x", output_price=None), "chat"
            )
            is False
        )

    def test_completion_mode_treated_like_chat(self) -> None:
        assert _has_complete_headline_pricing(_offering("x"), "completion") is True
        assert (
            _has_complete_headline_pricing(
                _offering("x", input_price=None), "completion"
            )
            is False
        )

    def test_empty_mode_treated_like_chat(self) -> None:
        assert _has_complete_headline_pricing(_offering("x"), "") is True
        assert (
            _has_complete_headline_pricing(
                _offering("x", output_price=None), ""
            )
            is False
        )

    def test_embedding_accepts_input_or_embedding_field(self) -> None:
        only_input = _offering(
            "x", input_price=0.1, output_price=None, embedding_price=None
        )
        only_embedding = _offering(
            "x", input_price=None, output_price=None, embedding_price=0.1
        )
        neither = _offering(
            "x", input_price=None, output_price=None, embedding_price=None
        )
        assert _has_complete_headline_pricing(only_input, "embedding") is True
        assert _has_complete_headline_pricing(only_embedding, "embedding") is True
        assert _has_complete_headline_pricing(neither, "embedding") is False

    def test_image_generation_accepts_image_input_or_output(self) -> None:
        only_image_in = _offering(
            "x",
            input_price=None,
            output_price=None,
            image_input=0.04,
        )
        only_output = _offering("x", input_price=None, output_price=0.04)
        neither = _offering("x", input_price=None, output_price=None)
        assert (
            _has_complete_headline_pricing(only_image_in, "image_generation") is True
        )
        assert (
            _has_complete_headline_pricing(only_output, "image_generation") is True
        )
        assert _has_complete_headline_pricing(neither, "image_generation") is False

    def test_audio_transcription_accepts_audio_input_or_input(self) -> None:
        only_audio = _offering(
            "x",
            input_price=None,
            output_price=None,
            audio_input=0.006,
        )
        neither = _offering("x", input_price=None, output_price=None)
        assert (
            _has_complete_headline_pricing(only_audio, "audio_transcription") is True
        )
        assert (
            _has_complete_headline_pricing(neither, "audio_transcription") is False
        )

    def test_audio_speech_accepts_audio_output_or_output(self) -> None:
        only_audio_out = _offering(
            "x",
            input_price=None,
            output_price=None,
            audio_output=0.015,
        )
        neither = _offering("x", input_price=None, output_price=None)
        assert (
            _has_complete_headline_pricing(only_audio_out, "audio_speech") is True
        )
        assert _has_complete_headline_pricing(neither, "audio_speech") is False

    def test_unknown_mode_defaults_permissive(self) -> None:
        # Rerank / moderation / anything we haven't mapped: don't
        # reject, since we'd otherwise demote a perfectly real offering
        # for an unfamiliar mode.
        stripped = _offering("x", input_price=None, output_price=None)
        assert _has_complete_headline_pricing(stripped, "rerank") is True
        assert _has_complete_headline_pricing(stripped, "moderation") is True


# ─── OfferingMerger._choose_primary ─────────────────────────────


class TestChoosePrimary:
    def test_empty_offerings_falls_back_to_litellm(self) -> None:
        assert _merger()._choose_primary(_entity("Anthropic"), []) == "litellm"

    def test_authority_order_picks_first_complete_match(self) -> None:
        """Anthropic authority is anthropic → aws_bedrock → azure → openrouter.
        All complete → anthropic wins."""
        offerings: List[OfferingV2] = [
            _offering("aws_bedrock"),
            _offering("openrouter"),
            _offering("anthropic"),
        ]
        assert (
            _merger()._choose_primary(_entity("Anthropic"), offerings) == "anthropic"
        )

    def test_authority_head_incomplete_demotes_to_next_complete(self) -> None:
        """Bedrock shipping input=null for Opus-4.7-on-release bug:
        anthropic missing, aws_bedrock has input=null → demote to
        openrouter, not to incomplete Bedrock."""
        offerings = [
            _offering("aws_bedrock", input_price=None, output_price=27.5),
            _offering("openrouter", input_price=5.0, output_price=25.0),
        ]
        assert (
            _merger()._choose_primary(_entity("Anthropic"), offerings) == "openrouter"
        )

    def test_anthropic_first_party_wins_over_bedrock_with_null_input(self) -> None:
        """End-to-end reproduction of the Opus 4.7 regression: all three
        authorities present but Bedrock has bad data — anthropic must
        be picked not because it's first in the list but because it's
        first AND complete."""
        offerings = [
            _offering("aws_bedrock", input_price=None, output_price=27.5),
            _offering("openrouter", input_price=5.0, output_price=25.0),
            _offering("anthropic", input_price=5.0, output_price=25.0),
        ]
        assert (
            _merger()._choose_primary(_entity("Anthropic"), offerings) == "anthropic"
        )

    def test_all_authorities_incomplete_still_picks_from_authority(self) -> None:
        """If every authority is incomplete, we still want a stable
        primary from the authority list, not a random non-authority
        provider sneaking in."""
        offerings = [
            _offering("openrouter", input_price=None, output_price=None),
            _offering("aws_bedrock", input_price=None, output_price=None),
            _offering("anthropic", input_price=None, output_price=None),
        ]
        # anthropic is first in AUTHORITY_BY_MAKER for Anthropic maker.
        assert (
            _merger()._choose_primary(_entity("Anthropic"), offerings) == "anthropic"
        )

    def test_no_authority_match_picks_complete_non_fallback(self) -> None:
        """Unknown maker: no AUTHORITY_BY_MAKER entry → fall to Pass 3
        (first complete non-fallback)."""
        offerings = [
            _offering("vendor_x", input_price=None, output_price=None),
            _offering("vendor_y", input_price=0.5, output_price=1.0),
        ]
        entity = _entity("ObscureLab")
        assert _merger()._choose_primary(entity, offerings) == "vendor_y"

    def test_no_authority_all_incomplete_picks_first_non_fallback(self) -> None:
        offerings = [
            _offering("vendor_x", input_price=None, output_price=None),
            _offering("vendor_y", input_price=None, output_price=None),
        ]
        entity = _entity("ObscureLab")
        # First non-fallback offering is vendor_x.
        assert _merger()._choose_primary(entity, offerings) == "vendor_x"

    def test_only_fallback_offerings_picks_first(self) -> None:
        offerings = [
            _offering(
                "litellm", input_price=None, output_price=None, source="litellm_fallback"
            ),
            _offering(
                "litellm", input_price=1.0, output_price=2.0, source="litellm_fallback"
            ),
        ]
        entity = _entity("ObscureLab")
        # All fallback + no complete non-fallback → final fallback path.
        assert _merger()._choose_primary(entity, offerings) == "litellm"

    def test_embedding_mode_uses_embedding_completeness(self) -> None:
        """Cohere embed on Bedrock: output price is conceptually N/A
        for embeddings, but input must be non-null to show a price."""
        offerings = [
            _offering(
                "aws_bedrock",
                input_price=None,
                output_price=None,
                embedding_price=None,
            ),
            _offering(
                "openrouter",
                input_price=0.1,
                output_price=None,
                embedding_price=None,
            ),
        ]
        entity = _entity("Cohere", mode="embedding")
        assert _merger()._choose_primary(entity, offerings) == "openrouter"
