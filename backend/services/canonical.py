"""Canonical resolver — maps any provider-specific model identifier
to a canonical_id that exists in the LiteLLM registry.

Resolution cascade (first hit wins):

1. Direct alias match via LiteLLMRegistry.resolve_alias(raw_id)
2. Strip common provider prefix (bedrock/, azure/, openrouter/, openai/,
   google/, anthropic/, x-ai/, deepseek/, mistralai/) then retry
3. Strip provider-dot-prefix form (anthropic.claude-sonnet-4-5-v1:0)
4. Strip version suffixes (-20250929, -v1:0, -latest, :beta) and
   check against the exact canonical slug set
5. None — caller logs to drift report; offering_merger Pass 2b
   promotes it into a synthetic entity from the raw data

The resolver never invents a canonical id and never accepts a
prefix/suffix boundary match ("kimi-k2" is NOT a match for
"kimi-k2-5"). Those heuristic matches routinely collapsed distinct
models (kimi-k2 vs kimi-k2.5, qwen3-coder vs qwen3-coder-plus,
veo-3 vs veo-3.1) into a single entity with mixed pricing. Anything
that needs fuzzy matching belongs in the LiteLLM registry's own
alias table, not here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from .litellm_registry import LiteLLMRegistry, slugify, strip_version_suffix

logger = logging.getLogger(__name__)

PROVIDER_PREFIXES_SLASH = (
    "openrouter/",
    "bedrock/",
    "bedrock_converse/",
    "azure/",
    "azure_openai/",
    "azure_ai/",
    "vertex_ai/",
    "google/",
    "openai/",
    "anthropic/",
    "x-ai/",
    "xai/",
    "deepseek/",
    "deepseek-ai/",
    "mistralai/",
    "mistral/",
    "meta/",
    "meta-llama/",
    "cohere/",
    "ai21/",
    "amazon/",
    "nvidia/",
    "perplexity/",
    "together_ai/",
    "fireworks_ai/",
    "groq/",
    "replicate/",
)

DOT_PREFIXES = (
    "anthropic.",
    "amazon.",
    "meta.",
    "mistral.",
    "cohere.",
    "ai21.",
    "stability.",
    "deepseek.",
)

# Explicit aliases where a provider's internal model id has no
# resemblance to the LiteLLM canonical slug. AWS Bedrock is the main
# offender: it publishes Cohere embeds as `cohere-embed-<N>-model[-variant]`
# while LiteLLM canonicalizes them as `embed[-variant]`. Without this
# map they go through Pass 2b as synthetic entities and we end up with
# two "Cohere Embed v4" cards (embed + cohere-embed-4-model). The map
# registers these as first-class aliases into the LiteLLM registry at
# resolver construction time so the merger sees them as one entity.
KNOWN_AGGREGATOR_ALIASES: dict[str, str] = {
    # AWS Bedrock Cohere embeds.
    "cohere-embed-4-model": "embed",
    "cohere-embed-3-model-english": "embed-english",
    "cohere-embed-3-model-multilingual": "embed-multilingual",
    # The aws api occasionally collapses dashes — accept both forms.
    "cohere-embed-model-3-multilingual": "embed-multilingual",
    "cohere-embed-model-3-english": "embed-english",
    # Vertex double-prefix: ``vertex_ai/mistralai/<model>``. Our slugify
    # only drops the first ``vertex_ai/`` so the ``mistralai-`` residue
    # is left behind, creating orphan entities alongside the real ones.
    # Hard-code the collapse so both Vertex forms resolve to the same
    # dangling target and cluster together.
    "mistralai-codestral-2": "codestral-2",
    "mistralai-codestral-2-001": "codestral-2-001",
    "mistralai-mistral-medium-3": "mistral-medium-3",
    "mistralai-mistral-medium-3-001": "mistral-medium-3-001",
    # LiteLLM upstream is inconsistent about Ministral 3 naming: some
    # canonical entries carry the family number ``-3-`` and some skip
    # it. Without aliasing the two sides would become parallel entity
    # pairs (ministral-14b-2512 vs ministral-3-14b-2512).
    "ministral-14b-2512": "ministral-3-14b-2512",
    "ministral-3b-2512": "ministral-3-3b-2512",
    "ministral-8b-2512": "ministral-3-8b-2512",
    # LiteLLM publishes Qwen3 Coder canonically as ``qwen3-coder`` but
    # some aggregator paths surface it as ``qwen3-coder-480b-a35b``
    # with the full parameter stamp.
    "qwen3-coder-480b-a35b": "qwen3-coder",
}


@dataclass
class Resolution:
    canonical_id: Optional[str]
    tried: List[str]
    strategy: str  # debugging aid: which step hit

    def matched(self) -> bool:
        return self.canonical_id is not None


class CanonicalResolver:
    """Stateful resolver bound to a LiteLLMRegistry instance."""

    def __init__(self, registry: LiteLLMRegistry) -> None:
        self.registry = registry
        self._canonical_slugs = {e.canonical_id for e in registry.iter_canonical()}

    # ─── Public API ──────────────────────────────────────────

    def resolve(self, provider: str, provider_model_id: str) -> Resolution:
        tried: List[str] = []
        raw = (provider_model_id or "").strip()
        if not raw:
            return Resolution(None, tried, "empty")

        # Step 1: direct alias hit (on raw and normalized forms)
        for candidate in self._candidates(raw):
            if candidate in tried:
                continue
            tried.append(candidate)
            hit = self.registry.resolve_alias(candidate)
            if hit:
                return Resolution(hit, tried, "alias")

        # Step 2: strip provider-specific prefixes then retry
        for candidate in self._strip_prefix_variants(raw):
            if candidate in tried:
                continue
            tried.append(candidate)
            hit = self.registry.resolve_alias(candidate)
            if hit:
                return Resolution(hit, tried, "prefix_strip")

        # Step 3: version-suffix-stripped form against canonical set
        stripped = strip_version_suffix(slugify(raw))
        if stripped and stripped not in tried:
            tried.append(stripped)
            if stripped in self._canonical_slugs:
                return Resolution(stripped, tried, "version_strip")
            hit = self.registry.resolve_alias(stripped)
            if hit:
                return Resolution(hit, tried, "version_strip_alias")

        return Resolution(None, tried, "miss")

    # ─── Internals ───────────────────────────────────────────

    def _candidates(self, raw: str) -> List[str]:
        """Variants that might directly match registry aliases."""
        out = [raw]
        lowered = raw.lower()
        if lowered != raw:
            out.append(lowered)
        slug = slugify(raw)
        if slug and slug != lowered:
            out.append(slug)
        stripped = strip_version_suffix(slug)
        if stripped and stripped != slug:
            out.append(stripped)
        return out

    def _strip_prefix_variants(self, raw: str) -> List[str]:
        """Try removing known provider prefixes; return cascading candidates."""
        variants: List[str] = []
        lowered = raw.lower()

        for prefix in PROVIDER_PREFIXES_SLASH:
            if lowered.startswith(prefix):
                rest = raw[len(prefix):]
                variants.extend(self._candidates(rest))

        for prefix in DOT_PREFIXES:
            if lowered.startswith(prefix):
                rest = raw[len(prefix):]
                variants.extend(self._candidates(rest))

        # Also try interpreting "a/b/c" by dropping only the first segment
        if "/" in raw:
            first_drop = raw.split("/", 1)[1]
            variants.extend(self._candidates(first_drop))
            # And dropping any leading "a/b/" pair if present
            if "/" in first_drop:
                variants.extend(self._candidates(first_drop.split("/", 1)[1]))

        # And the inverse for dot notation: anthropic.claude... → claude...
        if "." in raw:
            head, tail = raw.split(".", 1)
            if head.lower() in {"anthropic", "amazon", "meta", "mistral", "cohere", "ai21"}:
                variants.extend(self._candidates(tail))

        # Deduplicate while preserving order
        seen = set()
        ordered: List[str] = []
        for v in variants:
            if v and v not in seen:
                ordered.append(v)
                seen.add(v)
        return ordered

def build_resolver(registry: LiteLLMRegistry) -> CanonicalResolver:
    """Construct a resolver and register known aggregator aliases.

    The aliases live outside LiteLLM's published data but are stable
    enough to hard-code here. Registering them up front means the
    offering merger sees provider-native model ids (e.g. AWS Bedrock's
    `cohere-embed-4-model`) as members of the right canonical entity.
    """
    for alias, canonical_id in KNOWN_AGGREGATOR_ALIASES.items():
        # ``allow_dangling`` so aliases can point at synthetic-only
        # targets like ``codestral-2`` that exist as merged synthetic
        # entities but not as a LiteLLM canonical row.
        registry.register_alias(alias, canonical_id, allow_dangling=True)
    return CanonicalResolver(registry)
