"""Shared base class for providers whose first-party pricing is
mirrored through the LiteLLM community registry rather than pulled
directly from the vendor's own API.

Before this base existed, each such provider (anthropic / openai /
xai / deepseek / google_vertex_ai) had a ~100-line copy of the same
fetch loop. Adding a new vendor meant another copy. Subclassing this
base reduces each to a small declarative shell:

.. code-block:: python

    class AnthropicProvider(LiteLLMFirstPartyProvider):
        name = "anthropic"
        display_name = "Anthropic"
        litellm_tags = frozenset({"anthropic"})
        is_open_source = False

Offerings emitted here carry ``source="via_litellm"`` after the
merger runs (see ``LITELLM_SOURCED_PROVIDERS`` in
``services/offering_merger.py``), so the UI labels their pricing
row as a two-hop mirror, not a one-hop direct fetch.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import ClassVar, List, Optional

from models import BatchPricing, ModelPricing, Pricing
from services.litellm_registry import LiteLLMEntry, get_registry

from .base import BaseProvider

logger = logging.getLogger(__name__)


class LiteLLMFirstPartyProvider(BaseProvider):
    """Base class: filters the LiteLLM canonical bucket by tag."""

    # Subclasses override these.
    name: ClassVar[str] = ""
    display_name: ClassVar[str] = ""
    litellm_tags: ClassVar[frozenset[str]] = frozenset()
    is_open_source: ClassVar[Optional[bool]] = None
    # Vertex AI also needs to walk the aggregator bucket because its
    # Claude / Mistral / Llama entries are registered as aggregators
    # by the upstream litellm_registry parser.
    include_aggregator_bucket: ClassVar[bool] = False

    def maker_filter(self, entry: LiteLLMEntry) -> bool:
        """Hook to drop specific makers after tag filtering.

        Vertex overrides this to skip ``maker == "Google"`` so Gemini
        rows do not duplicate across google_gemini and
        google_vertex_ai at identical prices.
        """
        return True

    async def fetch(self) -> List[ModelPricing]:
        try:
            registry = await get_registry(force_network=False)
        except RuntimeError as exc:
            logger.warning(
                "%s provider: LiteLLM registry unavailable (%s); "
                "emitting no offerings this cycle",
                self.display_name,
                exc,
            )
            return []

        now = datetime.now()
        models: List[ModelPricing] = []

        def emit(entry: LiteLLMEntry, raw_key: str) -> None:
            if entry.litellm_provider not in self.litellm_tags:
                return
            if not self.maker_filter(entry):
                return
            pricing = Pricing(
                input=entry.input_price,
                output=entry.output_price,
                cached_input=entry.cache_read_price,
                cached_write=entry.cache_write_price,
                image_input=entry.image_input_price,
                audio_input=entry.audio_input_price,
                audio_output=entry.audio_output_price,
                embedding=entry.embedding_price,
            )
            batch = None
            if (
                entry.batch_input_price is not None
                or entry.batch_output_price is not None
            ):
                batch = BatchPricing(
                    input=entry.batch_input_price,
                    output=entry.batch_output_price,
                )
            models.append(
                ModelPricing(
                    id=f"{self.name}:{raw_key}",
                    provider=self.name,
                    model_id=raw_key,
                    model_name=entry.name,
                    pricing=pricing,
                    batch_pricing=batch,
                    context_length=entry.context_length,
                    max_output_tokens=entry.max_output_tokens,
                    is_open_source=self.is_open_source,
                    capabilities=entry.capabilities,
                    input_modalities=entry.input_modalities,
                    output_modalities=entry.output_modalities,
                    last_updated=now,
                )
            )

        for entry in registry.iter_canonical():
            emit(entry, entry.raw_key)

        if self.include_aggregator_bucket:
            for raw_key, entry in registry._aggregator_entries.items():  # noqa: SLF001
                emit(entry, raw_key)

        logger.info(
            "%s provider: emitted %s offerings", self.display_name, len(models)
        )
        return models
