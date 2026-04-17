"""Tests for the shared LiteLLMFirstPartyProvider base class.

Individual providers (anthropic/openai/xai/...) are tiny declarative
subclasses. Each is covered by an integration-style test in
``test_provider_litellm_sources.py`` or ``test_anthropic_provider.py``.
This file asserts the base class's generic behavior and spot-checks
every subclass declares what it should.
"""

from __future__ import annotations

import asyncio
from typing import Iterable, List
from unittest.mock import patch

import pytest

from providers import (
    ai21,
    alibaba_qwen,
    anthropic,
    cohere,
    deepseek,
    gigachat,
    google_vertex_ai,
    meta_llama,
    minimax,
    mistral,
    moonshot,
    openai,
    volcengine,
    xai,
    zai,
)
from providers._litellm_first_party import LiteLLMFirstPartyProvider
from services.litellm_registry import LiteLLMEntry


# ─── Helpers ────────────────────────────────────────────────────


def _entry(
    *,
    raw_key: str,
    canonical_id: str,
    litellm_provider: str,
    maker: str = "Unknown",
) -> LiteLLMEntry:
    return LiteLLMEntry(
        raw_key=raw_key,
        canonical_id=canonical_id,
        slug=canonical_id,
        name=raw_key.split("/")[-1],
        family="Unknown",
        maker=maker,
        litellm_provider=litellm_provider,
        is_canonical=True,
        input_price=1.0,
        output_price=2.0,
        cache_read_price=0.1,
        cache_write_price=None,
        image_input_price=None,
        audio_input_price=None,
        audio_output_price=None,
        embedding_price=None,
        batch_input_price=None,
        batch_output_price=None,
        context_length=128_000,
        max_output_tokens=8_000,
        capabilities=["text"],
        input_modalities=["text"],
        output_modalities=["text"],
        mode="chat",
    )


class _FakeRegistry:
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


# ─── All 15 maker-operator subclasses ──────────────────────────


EXPECTED_SUBCLASSES = [
    # (provider_class, app_slug, expected_tags)
    (anthropic.AnthropicProvider, "anthropic", {"anthropic"}),
    (openai.OpenAIProvider, "openai", {"openai", "text-completion-openai"}),
    (xai.XAIProvider, "xai", {"xai"}),
    (deepseek.DeepSeekProvider, "deepseek", {"deepseek"}),
    (
        google_vertex_ai.GoogleVertexAIProvider,
        "google_vertex_ai",
        # Only assert a representative subset — the full list lives
        # in the provider module and is expected to grow.
        {"vertex_ai", "vertex_ai-anthropic_models", "vertex_ai-mistral_models"},
    ),
    (mistral.MistralProvider, "mistral", {"mistral", "codestral"}),
    (moonshot.MoonshotProvider, "moonshot", {"moonshot"}),
    (cohere.CohereProvider, "cohere", {"cohere", "cohere_chat"}),
    (ai21.AI21Provider, "ai21", {"ai21"}),
    (alibaba_qwen.AlibabaQwenProvider, "alibaba_qwen", {"dashscope"}),
    (zai.ZAIProvider, "zai", {"zai"}),
    (minimax.MiniMaxProvider, "minimax", {"minimax"}),
    (volcengine.VolcengineProvider, "volcengine", {"volcengine"}),
    (gigachat.GigaChatProvider, "gigachat", {"gigachat"}),
    (meta_llama.MetaLlamaProvider, "meta_llama", {"meta_llama", "meta"}),
]


@pytest.mark.parametrize(
    "cls,expected_slug,expected_tags",
    EXPECTED_SUBCLASSES,
    ids=[c[1] for c in EXPECTED_SUBCLASSES],
)
def test_subclass_declaration(
    cls: type[LiteLLMFirstPartyProvider],
    expected_slug: str,
    expected_tags: set[str],
) -> None:
    """Each subclass declares its slug and tags up front — no logic
    lives in __init__ that could drift away from what the comment says."""
    assert issubclass(cls, LiteLLMFirstPartyProvider)
    assert cls.name == expected_slug
    assert cls.display_name  # non-empty
    assert expected_tags.issubset(cls.litellm_tags)


# ─── Generic base-class behavior ────────────────────────────────


class TestBaseFetch:
    def test_filters_by_litellm_tags(self) -> None:
        """The base loop must drop entries whose litellm_provider is
        not in the subclass's tag set."""
        registry = _FakeRegistry(
            canonical=[
                _entry(
                    raw_key="moonshot/kimi-k2",
                    canonical_id="kimi-k2",
                    litellm_provider="moonshot",
                ),
                _entry(
                    raw_key="openai/gpt-5",
                    canonical_id="gpt-5",
                    litellm_provider="openai",
                ),
            ]
        )
        with patch(
            "providers._litellm_first_party.get_registry",
            new=lambda force_network=False: _async(registry),
        ):
            models = _run(moonshot.MoonshotProvider().fetch())

        assert [m.model_id for m in models] == ["moonshot/kimi-k2"]

    def test_registry_unavailable_returns_empty(self) -> None:
        async def _raise(force_network: bool = False):
            raise RuntimeError("no registry")

        with patch("providers._litellm_first_party.get_registry", new=_raise):
            assert _run(cohere.CohereProvider().fetch()) == []

    def test_is_open_source_flag_propagates(self) -> None:
        registry = _FakeRegistry(
            canonical=[
                _entry(
                    raw_key="deepseek/deepseek-v3",
                    canonical_id="deepseek-v3",
                    litellm_provider="deepseek",
                ),
            ]
        )
        with patch(
            "providers._litellm_first_party.get_registry",
            new=lambda force_network=False: _async(registry),
        ):
            [model] = _run(deepseek.DeepSeekProvider().fetch())
        assert model.is_open_source is True

    def test_aggregator_bucket_ignored_unless_opted_in(self) -> None:
        """Non-Vertex subclasses must not pick up aggregator rows even
        if the tag matches, otherwise Bedrock's Anthropic rows would
        leak into the Anthropic first-party provider."""
        registry = _FakeRegistry(
            aggregator={
                "anthropic/claude-opus": _entry(
                    raw_key="anthropic/claude-opus",
                    canonical_id="claude-opus",
                    litellm_provider="anthropic",
                    maker="Anthropic",
                )
            }
        )
        with patch(
            "providers._litellm_first_party.get_registry",
            new=lambda force_network=False: _async(registry),
        ):
            assert _run(anthropic.AnthropicProvider().fetch()) == []

    def test_vertex_aggregator_bucket_is_included(self) -> None:
        """Vertex opts in — aggregator rows are the main source of its
        third-party models (Claude / Mistral / Llama on Vertex)."""
        registry = _FakeRegistry(
            aggregator={
                "vertex_ai/claude-opus-4-5": _entry(
                    raw_key="vertex_ai/claude-opus-4-5",
                    canonical_id="claude-opus-4-5",
                    litellm_provider="vertex_ai-anthropic_models",
                    maker="Anthropic",
                )
            }
        )
        with patch(
            "providers._litellm_first_party.get_registry",
            new=lambda force_network=False: _async(registry),
        ):
            [model] = _run(google_vertex_ai.GoogleVertexAIProvider().fetch())
        assert model.model_id == "vertex_ai/claude-opus-4-5"
