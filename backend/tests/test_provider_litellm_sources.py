"""Tests for LiteLLM-backed providers: xai / openai / deepseek / google_vertex_ai.

(anthropic has its own file for historical reasons.)

All providers here share the same shape: filter the LiteLLM registry
by ``litellm_provider`` and emit ``ModelPricing`` records with
complete field mapping. These tests stub the registry so no network
I/O and no registry parsing happens.
"""

from __future__ import annotations

import asyncio
from typing import Iterable, List
from unittest.mock import patch

from providers.deepseek import DeepSeekProvider
from providers.google_vertex_ai import GoogleVertexAIProvider
from providers.openai import OpenAIProvider
from providers.xai import XAIProvider
from services.litellm_registry import LiteLLMEntry


def _entry(
    *,
    raw_key: str,
    canonical_id: str,
    litellm_provider: str,
    input_price: float | None = 1.0,
    output_price: float | None = 2.0,
    cache_read: float | None = 0.1,
    cache_write: float | None = None,
    batch_input: float | None = None,
    batch_output: float | None = None,
    maker: str = "Unknown",
    family: str = "Unknown",
) -> LiteLLMEntry:
    return LiteLLMEntry(
        raw_key=raw_key,
        canonical_id=canonical_id,
        slug=canonical_id,
        name=raw_key.split("/")[-1],
        family=family,
        maker=maker,
        litellm_provider=litellm_provider,
        is_canonical=True,
        input_price=input_price,
        output_price=output_price,
        cache_read_price=cache_read,
        cache_write_price=cache_write,
        image_input_price=None,
        audio_input_price=None,
        audio_output_price=None,
        embedding_price=None,
        batch_input_price=batch_input,
        batch_output_price=batch_output,
        context_length=128_000,
        max_output_tokens=8_000,
        capabilities=["text"],
        input_modalities=["text"],
        output_modalities=["text"],
        mode="chat",
    )


class _FakeRegistry:
    """Stubs the canonical + aggregator iteration surfaces."""

    def __init__(
        self,
        canonical: Iterable[LiteLLMEntry] = (),
        aggregator: dict[str, LiteLLMEntry] | None = None,
    ) -> None:
        self._canonical = list(canonical)
        self._aggregator_entries = dict(aggregator or {})

    def iter_canonical(self):
        return iter(self._canonical)


def _run(coro):
    return asyncio.run(coro)


async def _async(value):
    return value


# ─── XAIProvider ────────────────────────────────────────────────


class TestXAIProvider:
    def test_filters_to_xai_entries_only(self) -> None:
        registry = _FakeRegistry(
            canonical=[
                _entry(
                    raw_key="xai/grok-4",
                    canonical_id="grok-4",
                    litellm_provider="xai",
                    maker="xAI",
                    family="Grok",
                ),
                _entry(
                    raw_key="openai/gpt-5",
                    canonical_id="gpt-5",
                    litellm_provider="openai",
                    maker="OpenAI",
                ),
                _entry(
                    raw_key="xai/grok-4-fast-reasoning",
                    canonical_id="grok-4-fast-reasoning",
                    litellm_provider="xai",
                    maker="xAI",
                    family="Grok",
                ),
                _entry(
                    raw_key="anthropic/claude-opus-4-7",
                    canonical_id="claude-opus-4-7",
                    litellm_provider="anthropic",
                    maker="Anthropic",
                ),
            ]
        )
        with patch(
            "providers._litellm_first_party.get_registry",
            new=lambda force_network=False: _async(registry),
        ):
            models = _run(XAIProvider().fetch())

        slugs = {m.model_id for m in models}
        assert slugs == {"xai/grok-4", "xai/grok-4-fast-reasoning"}
        assert all(m.provider == "xai" for m in models)
        assert all(m.id.startswith("xai:") for m in models)

    def test_pricing_fields_roundtrip(self) -> None:
        registry = _FakeRegistry(
            canonical=[
                _entry(
                    raw_key="xai/grok-4",
                    canonical_id="grok-4",
                    litellm_provider="xai",
                    input_price=3.0,
                    output_price=15.0,
                    cache_read=0.75,
                    batch_input=1.5,
                    batch_output=7.5,
                )
            ]
        )
        with patch(
            "providers._litellm_first_party.get_registry",
            new=lambda force_network=False: _async(registry),
        ):
            [grok] = _run(XAIProvider().fetch())

        assert grok.pricing.input == 3.0
        assert grok.pricing.output == 15.0
        assert grok.pricing.cached_input == 0.75
        assert grok.batch_pricing is not None
        assert grok.batch_pricing.input == 1.5
        assert grok.batch_pricing.output == 7.5

    def test_registry_unavailable_returns_empty_list(self) -> None:
        async def _raise(force_network: bool = False):
            raise RuntimeError("no cache, no network")

        with patch("providers._litellm_first_party.get_registry", new=_raise):
            assert _run(XAIProvider().fetch()) == []

    def test_empty_registry_returns_empty_list(self) -> None:
        registry = _FakeRegistry()
        with patch(
            "providers._litellm_first_party.get_registry",
            new=lambda force_network=False: _async(registry),
        ):
            assert _run(XAIProvider().fetch()) == []


# ─── OpenAIProvider ─────────────────────────────────────────────


class TestOpenAIProvider:
    def test_filters_to_openai_and_text_completion_openai(self) -> None:
        """OpenAI ships canonical entries under two LiteLLM tags —
        ``openai`` (chat / responses) and ``text-completion-openai``
        (legacy instruct endpoints). Both must be picked up."""
        registry = _FakeRegistry(
            canonical=[
                _entry(
                    raw_key="openai/gpt-5",
                    canonical_id="gpt-5",
                    litellm_provider="openai",
                    maker="OpenAI",
                    family="GPT",
                ),
                _entry(
                    raw_key="openai/gpt-3.5-turbo-instruct",
                    canonical_id="gpt-3-5-turbo-instruct",
                    litellm_provider="text-completion-openai",
                    maker="OpenAI",
                    family="GPT",
                ),
                _entry(
                    raw_key="anthropic/claude-opus-4-7",
                    canonical_id="claude-opus-4-7",
                    litellm_provider="anthropic",
                    maker="Anthropic",
                ),
                _entry(
                    raw_key="azure_ai/gpt-5",
                    canonical_id="gpt-5",
                    litellm_provider="azure_ai",
                    maker="OpenAI",
                ),
            ]
        )
        with patch(
            "providers._litellm_first_party.get_registry",
            new=lambda force_network=False: _async(registry),
        ):
            models = _run(OpenAIProvider().fetch())

        ids = {m.model_id for m in models}
        assert ids == {"openai/gpt-5", "openai/gpt-3.5-turbo-instruct"}
        assert all(m.provider == "openai" for m in models)
        assert all(m.id.startswith("openai:") for m in models)

    def test_pricing_fields_roundtrip(self) -> None:
        registry = _FakeRegistry(
            canonical=[
                _entry(
                    raw_key="openai/gpt-5",
                    canonical_id="gpt-5",
                    litellm_provider="openai",
                    input_price=1.25,
                    output_price=10.0,
                    cache_read=0.125,
                    batch_input=0.625,
                    batch_output=5.0,
                )
            ]
        )
        with patch(
            "providers._litellm_first_party.get_registry",
            new=lambda force_network=False: _async(registry),
        ):
            [gpt5] = _run(OpenAIProvider().fetch())

        assert gpt5.pricing.input == 1.25
        assert gpt5.pricing.output == 10.0
        assert gpt5.pricing.cached_input == 0.125
        assert gpt5.batch_pricing is not None
        assert gpt5.batch_pricing.input == 0.625
        assert gpt5.batch_pricing.output == 5.0

    def test_registry_unavailable_returns_empty_list(self) -> None:
        async def _raise(force_network: bool = False):
            raise RuntimeError("no cache, no network")

        with patch("providers._litellm_first_party.get_registry", new=_raise):
            assert _run(OpenAIProvider().fetch()) == []

    def test_empty_registry_returns_empty_list(self) -> None:
        registry = _FakeRegistry()
        with patch(
            "providers._litellm_first_party.get_registry",
            new=lambda force_network=False: _async(registry),
        ):
            assert _run(OpenAIProvider().fetch()) == []


# ─── DeepSeekProvider ───────────────────────────────────────────


class TestDeepSeekProvider:
    def test_filters_to_deepseek_entries_only(self) -> None:
        registry = _FakeRegistry(
            canonical=[
                _entry(
                    raw_key="deepseek/deepseek-v3",
                    canonical_id="deepseek-v3",
                    litellm_provider="deepseek",
                    maker="DeepSeek",
                ),
                _entry(
                    raw_key="openai/gpt-5",
                    canonical_id="gpt-5",
                    litellm_provider="openai",
                    maker="OpenAI",
                ),
                _entry(
                    raw_key="deepseek/deepseek-reasoner",
                    canonical_id="deepseek-reasoner",
                    litellm_provider="deepseek",
                    maker="DeepSeek",
                ),
            ]
        )
        with patch(
            "providers._litellm_first_party.get_registry",
            new=lambda force_network=False: _async(registry),
        ):
            models = _run(DeepSeekProvider().fetch())

        ids = {m.model_id for m in models}
        assert ids == {"deepseek/deepseek-v3", "deepseek/deepseek-reasoner"}
        assert all(m.provider == "deepseek" for m in models)
        assert all(m.is_open_source is True for m in models)

    def test_registry_unavailable_returns_empty_list(self) -> None:
        async def _raise(force_network: bool = False):
            raise RuntimeError("registry unavailable")

        with patch("providers._litellm_first_party.get_registry", new=_raise):
            assert _run(DeepSeekProvider().fetch()) == []


# ─── GoogleVertexAIProvider ─────────────────────────────────────


class TestGoogleVertexAIProvider:
    def test_emits_third_party_vertex_entries_from_aggregator_bucket(self) -> None:
        """Claude / Mistral / Llama on Vertex live in LiteLLM's
        aggregator bucket (``vertex_ai-*_models`` are declared as
        aggregator providers in the registry). The provider must walk
        aggregator entries, not only canonical, to surface them."""
        registry = _FakeRegistry(
            aggregator={
                "vertex_ai/claude-opus-4-5": _entry(
                    raw_key="vertex_ai/claude-opus-4-5",
                    canonical_id="claude-opus-4-5",
                    litellm_provider="vertex_ai-anthropic_models",
                    maker="Anthropic",
                ),
                "vertex_ai/mistral-large": _entry(
                    raw_key="vertex_ai/mistral-large",
                    canonical_id="mistral-large",
                    litellm_provider="vertex_ai-mistral_models",
                    maker="Mistral",
                ),
                "vertex_ai/meta-llama-4": _entry(
                    raw_key="vertex_ai/meta-llama-4",
                    canonical_id="meta-llama-4",
                    litellm_provider="vertex_ai-llama_models",
                    maker="Meta",
                ),
                # Google-maker aggregator row (legacy vertex_ai / MedLM
                # fallback). Must still be skipped to avoid Gemini dup.
                "vertex_ai/medlm-medium": _entry(
                    raw_key="vertex_ai/medlm-medium",
                    canonical_id="medlm-medium",
                    litellm_provider="vertex_ai",
                    maker="Google",
                ),
                # Unrelated aggregator: Bedrock's Claude must not leak
                # into Vertex output.
                "bedrock/claude-opus-4-7": _entry(
                    raw_key="bedrock/claude-opus-4-7",
                    canonical_id="claude-opus-4-7",
                    litellm_provider="bedrock_converse",
                    maker="Anthropic",
                ),
            }
        )
        with patch(
            "providers._litellm_first_party.get_registry",
            new=lambda force_network=False: _async(registry),
        ):
            models = _run(GoogleVertexAIProvider().fetch())

        ids = {m.model_id for m in models}
        assert ids == {
            "vertex_ai/claude-opus-4-5",
            "vertex_ai/mistral-large",
            "vertex_ai/meta-llama-4",
        }
        assert all(m.provider == "google_vertex_ai" for m in models)

    def test_skips_google_maker_across_both_buckets(self) -> None:
        registry = _FakeRegistry(
            canonical=[
                _entry(
                    raw_key="vertex_ai/gemini-3-pro",
                    canonical_id="gemini-3-pro",
                    litellm_provider="vertex_ai-language-models",
                    maker="Google",
                ),
            ],
            aggregator={
                "vertex_ai/medlm-medium": _entry(
                    raw_key="vertex_ai/medlm-medium",
                    canonical_id="medlm-medium",
                    litellm_provider="vertex_ai",
                    maker="Google",
                ),
            },
        )
        with patch(
            "providers._litellm_first_party.get_registry",
            new=lambda force_network=False: _async(registry),
        ):
            assert _run(GoogleVertexAIProvider().fetch()) == []

    def test_non_vertex_prefix_not_matched(self) -> None:
        """Providers named like `vertex_something_else` or just `vertex`
        without the `_ai` token must NOT be picked up — only the
        Google Cloud Vertex AI family."""
        registry = _FakeRegistry(
            canonical=[
                _entry(
                    raw_key="vertex/unknown",
                    canonical_id="unknown",
                    litellm_provider="vertex",  # no `_ai`
                    maker="?",
                ),
            ]
        )
        with patch(
            "providers._litellm_first_party.get_registry",
            new=lambda force_network=False: _async(registry),
        ):
            assert _run(GoogleVertexAIProvider().fetch()) == []

    def test_registry_unavailable_returns_empty_list(self) -> None:
        async def _raise(force_network: bool = False):
            raise RuntimeError("registry unavailable")

        with patch("providers._litellm_first_party.get_registry", new=_raise):
            assert _run(GoogleVertexAIProvider().fetch()) == []
