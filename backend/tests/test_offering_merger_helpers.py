"""Tests for the small pure helpers inside offering_merger.

These don't need network or a registry — they exercise the name
cleanup, cluster keying, and author prefix mapping that drive
synthetic entity creation.
"""

from datetime import datetime

from models import ModelPricing, Pricing
from models.v2 import OfferingV2, PricingV2
from services.offering_merger import (
    _is_embedding_price_outlier,
    _is_stub_offering_set,
    _maker_from_model_id,
    _round_price,
    _unmatched_cluster_key,
)


def _fallback_offering(input_price: float, output_price: float) -> OfferingV2:
    return OfferingV2(
        provider="litellm",
        provider_model_id="whatever",
        pricing=PricingV2(input=input_price, output=output_price),
        batch_pricing=None,
        availability="ga",
        region=None,
        notes=None,
        last_updated=datetime.utcnow(),
        source="litellm_fallback",
    )


def _api_offering(
    provider: str, input_price: float, output_price: float
) -> OfferingV2:
    return OfferingV2(
        provider=provider,
        provider_model_id="whatever",
        pricing=PricingV2(input=input_price, output=output_price),
        batch_pricing=None,
        availability="ga",
        region=None,
        notes=None,
        last_updated=datetime.utcnow(),
        source="provider_api",
    )


def _make_model(model_id: str, model_name: str = "") -> ModelPricing:
    return ModelPricing(
        id=f"openrouter:{model_id}",
        provider="openrouter",
        model_id=model_id,
        model_name=model_name or model_id,
        pricing=Pricing(input=1.0, output=2.0),
        capabilities=["text"],
        input_modalities=["text"],
        output_modalities=["text"],
        last_updated=datetime.utcnow(),
    )


class TestRoundPrice:
    def test_rounds_to_four_decimals(self):
        assert _round_price(0.19999999999999998) == 0.2

    def test_passes_through_clean_values(self):
        assert _round_price(3.0) == 3.0
        assert _round_price(0.0) == 0.0

    def test_handles_none(self):
        assert _round_price(None) is None

    def test_handles_invalid(self):
        assert _round_price("not a number") is None  # type: ignore[arg-type]


class TestUnmatchedClusterKey:
    def test_prefers_model_id_over_name(self):
        """model_id is more stable across providers and carries version
        information in the raw form we want to cluster by."""
        model = _make_model("moonshotai/kimi-k2.5", "MoonshotAI: Kimi K2.5")
        assert _unmatched_cluster_key(model) == "kimi-k2-5"

    def test_different_versions_stay_separate(self):
        """K2.5 and K2 Thinking must land in different clusters even
        though the model_name prefix is identical."""
        k25 = _make_model("moonshotai/kimi-k2.5", "MoonshotAI: Kimi K2.5")
        k2_thinking = _make_model("moonshotai/kimi-k2-thinking", "MoonshotAI: Kimi K2 Thinking")
        assert _unmatched_cluster_key(k25) != _unmatched_cluster_key(k2_thinking)

    def test_falls_back_to_name_when_id_is_too_short(self):
        model = _make_model("chat", "MoonshotAI: Kimi Chat")
        # "chat" is too short → name wins
        key = _unmatched_cluster_key(model)
        assert "kimi" in key or "moonshot" in key

    def test_strips_date_suffix_before_clustering(self):
        """Regression: Vertex's ``claude-3-5-haiku@20241022`` used to
        cluster as ``claude-3-5-haiku-20241022`` and live as its own
        synthetic next to ``claude-3-5-haiku``. strip_version_suffix
        now runs on the cluster key so both collapse.

        Bedrock's ``anthropic.claude-...-v1:0`` form keeps its
        ``anthropic-`` prefix after slugify (we don't strip maker
        prefixes at cluster-key time — that would collapse
        ``qwen-turbo`` → ``turbo``). Bedrock rows merge via the
        dangling-target path in ``run_refresh_pipeline`` instead."""
        vertex_dated = _make_model("vertex_ai/claude-3-5-haiku@20241022")
        undated = _make_model("claude-3-5-haiku")
        assert (
            _unmatched_cluster_key(vertex_dated)
            == _unmatched_cluster_key(undated)
            == "claude-3-5-haiku"
        )

    def test_strips_default_suffix_before_clustering(self):
        """Vertex's ``@default`` variant must collapse into the bare
        canonical id, not live as its own synthetic."""
        at_default = _make_model("vertex_ai/claude-opus-4-7@default")
        undated = _make_model("claude-opus-4-7")
        assert _unmatched_cluster_key(at_default) == _unmatched_cluster_key(undated)


class TestMakerFromModelId:
    def test_known_authors(self):
        assert _maker_from_model_id("anthropic/claude-3.5-sonnet") == "Anthropic"
        assert _maker_from_model_id("allenai/olmo-3-32b") == "AllenAI"
        assert _maker_from_model_id("meta-llama/llama-4-maverick") == "Meta"
        assert _maker_from_model_id("moonshotai/kimi-k2.5") == "Moonshot AI"

    def test_unknown_falls_back_to_titled_prefix(self):
        # No entry in AUTHOR_PREFIX_TO_MAKER → title-case the prefix
        assert _maker_from_model_id("somelab/weird-model") == "Somelab"

    def test_returns_none_for_nameless(self):
        assert _maker_from_model_id("") is None
        assert _maker_from_model_id(None) is None

    def test_returns_none_when_no_separator(self):
        assert _maker_from_model_id("justamodel") is None


class TestIsStubOfferingSet:
    """Regression coverage for the data bug that let phantom entries
    like kimi-k2-thinking-251104 appear with $0 pricing."""

    def test_empty_list_not_a_stub(self):
        assert _is_stub_offering_set([]) is False

    def test_fallback_zero_zero_is_stub(self):
        """The exact shape kimi-k2-thinking-251104 had."""
        assert _is_stub_offering_set([_fallback_offering(0.0, 0.0)]) is True

    def test_fallback_with_none_prices_is_stub(self):
        off = OfferingV2(
            provider="litellm",
            provider_model_id="x",
            pricing=PricingV2(input=None, output=None),
            batch_pricing=None,
            availability="ga",
            region=None,
            notes=None,
            last_updated=datetime.utcnow(),
            source="litellm_fallback",
        )
        assert _is_stub_offering_set([off]) is True

    def test_fallback_with_real_price_not_stub(self):
        # e.g. kimi-latest with input=2.0 — real data, keep it
        assert _is_stub_offering_set([_fallback_offering(2.0, 5.0)]) is False

    def test_fallback_with_partial_price_not_stub(self):
        # Non-zero input OR output keeps the entity
        assert _is_stub_offering_set([_fallback_offering(0.0, 5.0)]) is False
        assert _is_stub_offering_set([_fallback_offering(1.0, 0.0)]) is False

    def test_real_provider_zero_zero_not_stub(self):
        """OpenRouter's *-free variants are legitimately free.
        They come through as provider_api, not litellm_fallback,
        so they must be preserved."""
        assert (
            _is_stub_offering_set([_api_offering("openrouter", 0.0, 0.0)]) is False
        )

    def test_mixed_fallback_and_real_not_stub(self):
        """A single non-fallback offering saves the whole entity."""
        offs = [
            _fallback_offering(0.0, 0.0),
            _api_offering("openrouter", 0.1, 0.3),
        ]
        assert _is_stub_offering_set(offs) is False

    def test_via_litellm_with_null_prices_is_stub(self):
        """Regression: Volcengine's RMB-only upstream rows (e.g.
        deepseek-v3-2-251201) land as via_litellm offerings with
        input=null / output=null. They used to escape the stub filter
        and surfaced as a single "Volcengine" row with no numbers."""
        off = OfferingV2(
            provider="volcengine",
            provider_model_id="deepseek-v3-2-251201",
            pricing=PricingV2(input=None, output=None),
            batch_pricing=None,
            availability="ga",
            region=None,
            notes=None,
            last_updated=datetime.utcnow(),
            source="via_litellm",
        )
        assert _is_stub_offering_set([off]) is True

    def test_via_litellm_with_real_price_not_stub(self):
        """Anthropic / OpenAI / ... via_litellm rows with real prices
        must survive — they are first-party pricing mirrored through
        LiteLLM and are the headline data for those makers."""
        off = OfferingV2(
            provider="anthropic",
            provider_model_id="claude-opus-4-7",
            pricing=PricingV2(input=5.0, output=25.0),
            batch_pricing=None,
            availability="ga",
            region=None,
            notes=None,
            last_updated=datetime.utcnow(),
            source="via_litellm",
        )
        assert _is_stub_offering_set([off]) is False

    def test_mixed_via_litellm_and_real_not_stub(self):
        """via_litellm with null prices gets rescued if any real
        provider_api row has actual pricing — same logic as fallback."""
        via = OfferingV2(
            provider="volcengine",
            provider_model_id="deepseek-v3-2-251201",
            pricing=PricingV2(input=None, output=None),
            batch_pricing=None,
            availability="ga",
            region=None,
            notes=None,
            last_updated=datetime.utcnow(),
            source="via_litellm",
        )
        offs = [via, _api_offering("openrouter", 0.3, 0.5)]
        assert _is_stub_offering_set(offs) is False

    def test_first_party_null_price_rescued_by_maker(self):
        """Regression: Doubao / GigaChat have LiteLLM canonical rows
        but null USD prices (RMB-only upstream). Without the
        first-party rescue, the stub filter pruned every Doubao entity
        and Doubao vanished from the site. A ByteDance entity whose
        only offering is Volcengine must stay — it is a first-party
        pairing, price column is allowed to show '—'."""
        off = OfferingV2(
            provider="volcengine",
            provider_model_id="doubao-seed-2-0-pro",
            pricing=PricingV2(input=None, output=None),
            batch_pricing=None,
            availability="ga",
            region=None,
            notes=None,
            last_updated=datetime.utcnow(),
            source="via_litellm",
        )
        assert _is_stub_offering_set([off], entity_maker="ByteDance") is False

    def test_third_party_null_price_still_pruned(self):
        """Counter-case: Volcengine hosting DeepSeek with null prices is
        NOT a first-party pairing (Volcengine belongs to ByteDance, not
        DeepSeek), so it should still be pruned. This protects against
        the Volcengine-redistributes-everything inflation path."""
        off = OfferingV2(
            provider="volcengine",
            provider_model_id="deepseek-v3-2-251201",
            pricing=PricingV2(input=None, output=None),
            batch_pricing=None,
            availability="ga",
            region=None,
            notes=None,
            last_updated=datetime.utcnow(),
            source="via_litellm",
        )
        assert _is_stub_offering_set([off], entity_maker="DeepSeek") is True

    def test_rescue_requires_entity_maker_argument(self):
        """Legacy call sites that don't pass entity_maker stay on the
        strict all-null rule; this preserves the invariant that the
        public helper is safe to call without the new context."""
        off = OfferingV2(
            provider="volcengine",
            provider_model_id="doubao-seed-2-0-pro",
            pricing=PricingV2(input=None, output=None),
            batch_pricing=None,
            availability="ga",
            region=None,
            notes=None,
            last_updated=datetime.utcnow(),
            source="via_litellm",
        )
        # No entity_maker → rescue skipped → classified as stub.
        assert _is_stub_offering_set([off]) is True


class TestIsEmbeddingPriceOutlier:
    """Regression: scraper unit bugs and stale LiteLLM entries used to
    poison the embedding alternatives list with prices 1000x off the
    real market (TwelveLabs Marengo at $0.0001/M from an AWS parser
    bug, Cohere embed-multilingual-light at $100/M from an unchecked
    LiteLLM update)."""

    def test_normal_embedding_price_not_outlier(self):
        # Cohere embed-v4 at $0.12 / M — in range.
        off = _api_offering("aws_bedrock", 0.12, 0.0)
        assert _is_embedding_price_outlier(off, "embedding") is False

    def test_cheap_embedding_outlier(self):
        # AWS scraper returning $0.0001 / M — clearly a unit bug.
        off = _api_offering("aws_bedrock", 0.0001, 0.0)
        assert _is_embedding_price_outlier(off, "embedding") is True

    def test_expensive_embedding_outlier(self):
        # LiteLLM stale $100 / M entry for embed-multilingual-light.
        off = _api_offering("litellm", 100.0, 0.0)
        assert _is_embedding_price_outlier(off, "embedding") is True

    def test_non_embedding_mode_not_checked(self):
        # Chat models are free to be $0.0001 / M (they aren't today,
        # but we don't want to flag legitimately cheap chat offerings).
        off = _api_offering("openai", 0.0001, 0.0)
        assert _is_embedding_price_outlier(off, "chat") is False

    def test_null_input_price_not_outlier(self):
        off = OfferingV2(
            provider="x",
            provider_model_id="x",
            pricing=PricingV2(input=None, output=None),
            batch_pricing=None,
            availability="ga",
            region=None,
            notes=None,
            last_updated=datetime.utcnow(),
            source="litellm_fallback",
        )
        assert _is_embedding_price_outlier(off, "embedding") is False
