"""Tests for providers.anthropic.AnthropicProvider.

The provider reads the shared LiteLLM registry and only emits entries
whose ``litellm_provider == "anthropic"``. These tests stub the
registry so the provider's filtering, field mapping, and failure
behavior can be verified without any network I/O.
"""

from __future__ import annotations

import asyncio
from typing import List
from unittest.mock import patch

import pytest

from providers.anthropic import AnthropicProvider
from services.litellm_registry import LiteLLMEntry


def _entry(
    *,
    canonical_id: str,
    litellm_provider: str,
    input_price: float | None = 3.0,
    output_price: float | None = 15.0,
    batch_input: float | None = None,
    batch_output: float | None = None,
) -> LiteLLMEntry:
    return LiteLLMEntry(
        raw_key=canonical_id,
        canonical_id=canonical_id,
        slug=canonical_id,
        name=canonical_id.replace("-", " ").title(),
        family="Claude" if litellm_provider == "anthropic" else "GPT",
        maker="Anthropic" if litellm_provider == "anthropic" else "OpenAI",
        litellm_provider=litellm_provider,
        is_canonical=True,
        input_price=input_price,
        output_price=output_price,
        cache_read_price=0.3,
        cache_write_price=3.75,
        image_input_price=None,
        audio_input_price=None,
        audio_output_price=None,
        embedding_price=None,
        batch_input_price=batch_input,
        batch_output_price=batch_output,
        context_length=200_000,
        max_output_tokens=64_000,
        capabilities=["text", "vision"],
        input_modalities=["text", "image"],
        output_modalities=["text"],
        mode="chat",
    )


class _FakeRegistry:
    def __init__(self, entries: List[LiteLLMEntry]) -> None:
        self._entries = entries

    def iter_canonical(self):
        return iter(self._entries)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def anthropic_only() -> List[LiteLLMEntry]:
    return [
        _entry(canonical_id="claude-opus-4-7", litellm_provider="anthropic"),
        _entry(canonical_id="claude-sonnet-4-6", litellm_provider="anthropic"),
    ]


@pytest.fixture
def mixed_entries() -> List[LiteLLMEntry]:
    return [
        _entry(canonical_id="claude-opus-4-7", litellm_provider="anthropic"),
        _entry(canonical_id="gpt-5", litellm_provider="openai"),
        _entry(canonical_id="claude-3-opus", litellm_provider="bedrock_converse"),
        _entry(canonical_id="claude-haiku-4-5", litellm_provider="anthropic"),
    ]


class TestAnthropicProvider:
    def test_filters_to_anthropic_entries_only(
        self, mixed_entries: List[LiteLLMEntry]
    ) -> None:
        registry = _FakeRegistry(mixed_entries)
        with patch(
            "providers._litellm_first_party.get_registry",
            new=lambda force_network=False: _async(registry),
        ):
            models = _run(AnthropicProvider().fetch())

        slugs = {m.model_id for m in models}
        assert slugs == {"claude-opus-4-7", "claude-haiku-4-5"}
        assert all(m.provider == "anthropic" for m in models)

    def test_maps_all_pricing_fields(
        self, anthropic_only: List[LiteLLMEntry]
    ) -> None:
        registry = _FakeRegistry(anthropic_only)
        with patch(
            "providers._litellm_first_party.get_registry",
            new=lambda force_network=False: _async(registry),
        ):
            models = _run(AnthropicProvider().fetch())

        opus = next(m for m in models if m.model_id == "claude-opus-4-7")
        assert opus.pricing.input == 3.0
        assert opus.pricing.output == 15.0
        assert opus.pricing.cached_input == 0.3
        assert opus.pricing.cached_write == 3.75
        assert opus.context_length == 200_000
        assert opus.max_output_tokens == 64_000
        assert "vision" in opus.capabilities
        assert "image" in opus.input_modalities

    def test_emits_batch_pricing_when_present(self) -> None:
        entries = [
            _entry(
                canonical_id="claude-with-batch",
                litellm_provider="anthropic",
                batch_input=1.5,
                batch_output=7.5,
            ),
            _entry(
                canonical_id="claude-without-batch",
                litellm_provider="anthropic",
            ),
        ]
        registry = _FakeRegistry(entries)
        with patch(
            "providers._litellm_first_party.get_registry",
            new=lambda force_network=False: _async(registry),
        ):
            models = _run(AnthropicProvider().fetch())

        with_batch = next(m for m in models if m.model_id == "claude-with-batch")
        without_batch = next(
            m for m in models if m.model_id == "claude-without-batch"
        )
        assert with_batch.batch_pricing is not None
        assert with_batch.batch_pricing.input == 1.5
        assert with_batch.batch_pricing.output == 7.5
        assert without_batch.batch_pricing is None

    def test_registry_unavailable_returns_empty_list(self) -> None:
        """If the registry can't load (no cache, no network), the provider
        must not take the whole refresh down."""

        async def _raise(force_network: bool = False):
            raise RuntimeError("LiteLLM registry unavailable")

        with patch("providers._litellm_first_party.get_registry", new=_raise):
            models = _run(AnthropicProvider().fetch())

        assert models == []

    def test_id_namespaced_with_provider_name(
        self, anthropic_only: List[LiteLLMEntry]
    ) -> None:
        registry = _FakeRegistry(anthropic_only)
        with patch(
            "providers._litellm_first_party.get_registry",
            new=lambda force_network=False: _async(registry),
        ):
            models = _run(AnthropicProvider().fetch())

        assert all(m.id.startswith("anthropic:") for m in models)

    def test_empty_registry_returns_empty_list(self) -> None:
        registry = _FakeRegistry([])
        with patch(
            "providers._litellm_first_party.get_registry",
            new=lambda force_network=False: _async(registry),
        ):
            models = _run(AnthropicProvider().fetch())
        assert models == []


async def _async(value):
    """Helper: make a sync value awaitable so patched get_registry is a
    drop-in for the real async function."""
    return value
