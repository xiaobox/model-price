"""Offering merger — orchestrates v2 refresh pipeline.

Pipeline:
1. Load / refresh LiteLLM registry → canonical Entity skeletons
2. Run the existing v1 provider fetchers (unchanged) → List[ModelPricing]
3. For each v1 record, resolve to canonical_id via CanonicalResolver
4. Attach as Offering to the matching Entity; misses go to drift report
5. For any canonical Entity with zero provider offerings, synthesize
   a litellm_fallback Offering from the LiteLLM registry itself so the
   entity is still visible in the UI
6. Write entities.json + offerings.json + drift.json
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from models import ModelPricing
from models.v2 import (
    BatchPricingV2,
    EntityCoreV2,
    EntityStoreSnapshot,
    OfferingV2,
    PricingV2,
)
from providers.registry import ProviderRegistry

from .canonical import CanonicalResolver, build_resolver
from .drift_reporter import DriftReporter
from .litellm_registry import (
    DISPLAY_CAPABILITIES,
    LiteLLMEntry,
    LiteLLMRegistry,
    detect_family_maker,
    get_registry,
    slugify,
    strip_version_suffix,
)

logger = logging.getLogger(__name__)

V2_DATA_DIR = Path(__file__).parent.parent / "data" / "v2"
ENTITIES_PATH = V2_DATA_DIR / "entities.json"
OFFERINGS_PATH = V2_DATA_DIR / "offerings.json"
INDEX_PATH = V2_DATA_DIR / "index.json"


def _round_price(value: Optional[float]) -> Optional[float]:
    """Normalize float representation so JSON doesn't show 0.19999…"""
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


AUTHOR_PREFIX_TO_MAKER = {
    "aionlabs": "AionLabs",
    "allenai": "AllenAI",
    "alibaba": "Alibaba",
    "qwen": "Alibaba",
    "anthropic": "Anthropic",
    "arcee-ai": "Arcee AI",
    "arcee": "Arcee AI",
    "baidu": "Baidu",
    "bytedance": "ByteDance",
    "bytedance-research": "ByteDance",
    "cognitivecomputations": "Cognitive Computations",
    "cohere": "Cohere",
    "deepcogito": "Deep Cogito",
    "deepseek": "DeepSeek",
    "deepseek-ai": "DeepSeek",
    "deepseek-v3": "DeepSeek",
    "eleutherai": "EleutherAI",
    "fireworks": "Fireworks",
    "google": "Google",
    "inflection": "Inflection",
    "liquid": "Liquid AI",
    "meta": "Meta",
    "meta-llama": "Meta",
    "microsoft": "Microsoft",
    "minimax": "MiniMax",
    "mistralai": "Mistral",
    "mistral": "Mistral",
    "moonshotai": "Moonshot AI",
    "moonshot": "Moonshot AI",
    "neversleep": "NeverSleep",
    "nousresearch": "Nous Research",
    "nous": "Nous Research",
    "nvidia": "NVIDIA",
    "openai": "OpenAI",
    "openchat": "OpenChat",
    "opengvlab": "OpenGVLab",
    "perplexity": "Perplexity",
    "qwen": "Alibaba",
    "reka": "Reka",
    "sao10k": "Sao10k",
    "snowflake": "Snowflake",
    "stabilityai": "Stability AI",
    "stepfun-ai": "StepFun",
    "tencent": "Tencent",
    "thedrummer": "TheDrummer",
    "thudm": "THUDM",
    "together": "Together",
    "upstage": "Upstage",
    "venice": "Venice",
    "xai": "xAI",
    "x-ai": "xAI",
    "z-ai": "Z.AI",
    "zai": "Z.AI",
    "01-ai": "01.AI",
    "01.ai": "01.AI",
    "ai21": "AI21",
    "amazon": "Amazon",
}


def _maker_from_model_id(model_id: Optional[str]) -> Optional[str]:
    if not model_id:
        return None
    lowered = model_id.lower()
    for sep in ("/", "."):
        if sep in lowered:
            head = lowered.split(sep, 1)[0].strip()
            mapped = AUTHOR_PREFIX_TO_MAKER.get(head)
            if mapped:
                return mapped
            # Fall back to simple title case of the prefix itself
            if len(head) >= 2 and head.replace("-", "").replace("_", "").isalnum():
                return head.replace("-", " ").replace("_", " ").title()
    return None


def _family_from_model_name(name: Optional[str]) -> Optional[str]:
    """Deprecated — kept for reference. We now prefer maker-as-family
    in the synthetic path because this function produced too many
    ugly labels ("Aionlabs:", "Body", "Seed").
    """
    return None


# Double-prefix patterns that Vertex AI / Alibaba Model Studio use
# for publisher-qualified model ids:
# ``meta-llama-4-scout``, ``deepseek-ai-deepseek-v3``,
# ``qwen-qwen3-coder``, ``minimaxai-minimax-m2``,
# ``moonshotai-kimi-k2``, ``openai-gpt-oss-120b``, ``zai-org-glm-5``,
# ``mistralai-codestral-2``. Strip only when the prefix is one of
# these explicit vendor prefixes and only once, so we never chew into
# real model names (``mistral-large`` must stay ``mistral-large``,
# not become ``large``).
_MAKER_PUBLISHER_PREFIXES = (
    "meta-llama-",
    "deepseek-ai-",
    "mistralai-",
    "minimaxai-",
    "moonshotai-",
    "zai-org-",
    "openai-",
    "qwen-qwen",  # ``qwen-qwen3-*`` → ``qwen3-*`` (keep trailing no dash)
    "qwen-",
)


def _strip_publisher_prefix(slug: str) -> str:
    """Drop a single Vertex-style ``<publisher>-`` prefix.

    The prefix list is explicit rather than "anything before the first
    dash" — otherwise ``mistral-large`` would collapse to ``large`` and
    ``qwen-turbo`` to ``turbo``, which are legit product names."""
    for prefix in _MAKER_PUBLISHER_PREFIXES:
        if slug.startswith(prefix):
            tail = slug[len(prefix):]
            # Avoid producing an empty slug.
            if tail and tail != slug:
                return tail
    return slug


def _unmatched_cluster_key(model: ModelPricing) -> str:
    """Derive a stable cluster key from an unmatched v1 record so that
    the same underlying model from different providers ends up in one
    synthetic entity, while distinct versions (K2, K2.5, K2 Thinking)
    remain separate.

    Priority: model_id slug (most stable across providers, carries the
    version bits like ".5" or "-thinking") → fallback to model_name slug
    if the id isn't informative enough.

    The candidate runs through:
      1. ``strip_version_suffix`` — drops ``-maas`` / ``-20241022`` /
         ``-latest`` / ``-default`` / ``-free`` / ``-it`` etc. so
         ``vertex_ai/claude-3-5-haiku@20241022`` and
         ``anthropic.claude-3-5-haiku-20241022-v1:0`` cluster together.
      2. ``_strip_publisher_prefix`` — drops Vertex / dashscope style
         ``<publisher>-`` prefixes so ``qwen-qwen3-coder-*-maas`` and
         ``qwen3-coder`` cluster together.
    """
    id_candidate = _strip_publisher_prefix(
        strip_version_suffix(slugify(model.model_id or ""))
    )
    name_candidate = _strip_publisher_prefix(
        strip_version_suffix(slugify(model.model_name or ""))
    )
    # A good id_candidate has at least two dash-separated segments
    # ("kimi-k2-5" good, "chat" bad).
    if len(id_candidate) >= 4 and "-" in id_candidate:
        return id_candidate
    if len(name_candidate) >= 4:
        return name_candidate
    return id_candidate or name_candidate

# When multiple canonical providers offer the same entity, the UI needs
# one "primary" to show. We derive it from the entity's maker, falling
# back to whichever offering comes first.
AUTHORITY_BY_MAKER: Dict[str, List[str]] = {
    # First-party maker-operators go first; then major clouds in
    # typical availability order; then OpenRouter as the catch-all.
    "Anthropic": ["anthropic", "aws_bedrock", "google_vertex_ai", "azure_ai", "openrouter"],
    "OpenAI": ["openai", "azure_ai", "openrouter"],
    "Google": ["google_gemini", "google_vertex_ai", "openrouter"],
    "xAI": ["xai", "azure_ai", "openrouter"],
    "DeepSeek": ["deepseek", "aws_bedrock", "google_vertex_ai", "azure_ai", "openrouter"],
    "Meta": ["meta_llama", "aws_bedrock", "google_vertex_ai", "azure_ai", "openrouter"],
    "Mistral": ["mistral", "aws_bedrock", "google_vertex_ai", "azure_ai", "openrouter"],
    "Amazon": ["aws_bedrock"],
    "Cohere": ["cohere", "aws_bedrock", "azure_ai", "openrouter"],
    "AI21": ["ai21", "aws_bedrock", "openrouter"],
    "NVIDIA": ["openrouter"],
    "Microsoft": ["azure_ai", "openrouter"],
    "Black Forest Labs": ["azure_ai", "openrouter"],
    # First-party maker-operators we just added.
    "Moonshot AI": ["moonshot", "openrouter"],
    "Alibaba": ["alibaba_qwen", "google_vertex_ai", "openrouter"],
    "Z.AI": ["zai", "openrouter"],
    "MiniMax": ["minimax", "google_vertex_ai", "openrouter"],
    "ByteDance": ["volcengine", "openrouter"],
    "Sber": ["gigachat"],
}


_STUB_SOURCES = frozenset({"litellm_fallback", "via_litellm"})


# ─── Display-name styling ───────────────────────────────────────
#
# Until this lived in one place we had ``_pretty_model_name`` blindly
# uppercasing every short alphanum segment (so ``4o`` → ``4O``) on the
# canonical path, while ``_synthetic_entity_from_v1`` used whatever
# ``model_name`` the provider scraper reported (often with hyphens and
# mixed case: "GPT-4o Realtime"). The list view ended up showing the
# same family with "GPT 4O Realtime Preview" and "GPT-4o Realtime"
# side-by-side, which users read as duplicates.
#
# ``_polish_display_name`` is the single styling rule every entity
# name runs through on the way out.

_ALWAYS_UPPER_TOKENS = {
    # Real acronyms — letters standing for separate words. Everything
    # else that's just a short English word (Max, Air, Pro, Exp, Her)
    # should default to Title Case, not UPPER.
    "gpt", "ai", "api", "sdk", "llm", "mm", "mllm",
    "vl", "vlm", "pt", "dpo",
    "tts", "stt", "ocr", "oss", "hd", "ui", "ux",
    "cot", "moe", "rag", "kv",
    "ft",  # Fine-tuned; also see _polish_display_name for name rewrite
    # Model family acronyms that read as the whole product name.
    "glm", "qwq", "mlm", "lfm",
    # Version markers that read as acronyms upstream.
    "v1", "v2", "v3", "v4", "v5", "r1",
    # Vendor brand acronyms.
    "ibm", "nvidia", "nvda", "rwkv", "sdxl",
}

# Brand suffixes that are deliberately lower-case — the letter after
# the digit is a product tag, not an acronym. ``4o`` is OpenAI's
# "omni" naming; ``3n`` is Google's Gemma 3n variant.
_KEEP_LOWERCASE_TOKENS = {
    "4o", "3n", "4n",
}

# Compressed-word brand names that Python's ``.capitalize()`` would
# mangle ("chatgpt" → "Chatgpt" instead of "ChatGPT"). Match is done
# case-insensitively on the raw token; the value carries the exact
# mixed-case form we want to render.
_BRAND_CASE_TOKEN = {
    "chatgpt": "ChatGPT",
    "deepseek": "DeepSeek",
    "minimax": "MiniMax",
    "openrouter": "OpenRouter",
    "openai": "OpenAI",
    "bytedance": "ByteDance",
    "mistralai": "MistralAI",
    "moonshotai": "MoonshotAI",
    "aionlabs": "AionLabs",
    "allenai": "AllenAI",
    "nousresearch": "NousResearch",
    "thedrummer": "TheDrummer",
    "openchat": "OpenChat",
    "qwq": "QwQ",  # Qwen with Questions
}

# Parameter-size tokens: "70b" → "70B", "480m" → "480M", "8k" → "8K",
# "1.5t" → "1.5T". Anchored so random digit-letter combos don't match.
_PARAM_SIZE_RE = re.compile(r"^(\d+(?:\.\d+)?)([kmbt])$", re.IGNORECASE)

# MoE-style "active parameters" tokens like ``A22b`` / ``A3b`` /
# ``A47b`` / ``R7b``: a single letter, a digit run, a single unit
# letter. The unit letter should render uppercase so it reads
# consistently with plain parameter sizes (``235B A22B`` vs
# ``235B A22b``).
_ACTIVE_PARAM_RE = re.compile(r"^([a-zA-Z])(\d+)([kmbt])$", re.IGNORECASE)


def _style_name_token(token: str) -> str:
    """Make one token display-ready.

    Policy (previous short-word UPPER default is gone — it was turning
    every Max / Air / Pro / Exp into an all-caps pseudo-acronym):

    1. Compressed-word brand lookup first, so ``chatgpt`` becomes
       ``ChatGPT`` and ``deepseek`` becomes ``DeepSeek``.
    2. Brand suffixes that must stay lowercase (``4o`` / ``3n``).
    3. Explicit acronym whitelist (``GPT`` / ``TTS`` / ``OCR`` / ...).
    4. Parameter-size tokens (``70b`` → ``70B``, ``1.5t`` → ``1.5T``).
    5. Active-param MoE tokens (``A22b`` → ``A22B``).
    6. Pure digits pass through.
    7. Everything else (short or long) → Title Case.
    """
    if not token:
        return token
    lower = token.lower()
    if lower in _BRAND_CASE_TOKEN:
        return _BRAND_CASE_TOKEN[lower]
    if lower in _KEEP_LOWERCASE_TOKENS:
        return lower
    if lower in _ALWAYS_UPPER_TOKENS:
        return lower.upper()
    match = _PARAM_SIZE_RE.match(token)
    if match:
        return f"{match.group(1)}{match.group(2).upper()}"
    match = _ACTIVE_PARAM_RE.match(token)
    if match:
        return (
            match.group(1).upper()
            + match.group(2)
            + match.group(3).upper()
        )
    if token.isdigit():
        return token
    if token.isalpha():
        return token.capitalize()
    # Mixed alphanumeric: ``qwen3`` / ``glm4`` / ``llama2``. Capitalise
    # just the first character so the digit run stays intact.
    return token[:1].upper() + token[1:].lower()


def _polish_display_name(name: str) -> str:
    """Canonical styling pass for display names.

    Tokenises on whitespace / hyphens / underscores, styles each token
    via :func:`_style_name_token`, then stitches trailing digit runs
    back together with dots so ``["4", "5"]`` reads "4.5".

    Strips surrounding parentheses so OpenRouter-style tails like
    ``"Gemma 3n 2B (free)"`` or ``"Claude Opus 4.6 (fast)"`` turn into
    ``"Gemma 3n 2B Free"`` / ``"Claude Opus 4.6 Fast"`` — the
    parenthesised token is a product variant label, not annotation.
    """
    if not name:
        return name
    # Drop parentheses but keep their contents; they carry product
    # tags like "(free)" / "(thinking)" / "(extended)" that should
    # be tokenised alongside the rest of the name.
    cleaned = name.replace("(", " ").replace(")", " ")
    tokens = re.split(r"[\s\-_]+", cleaned.strip())
    styled = [_style_name_token(t) for t in tokens if t]
    # Reassemble trailing digit runs as version numbers: "4 5" →
    # "4.5", "M2 5" → "M2.5". Merge when the previous token ends in a
    # digit and the current token is pure digits. This covers both
    # "Claude Sonnet 4 5" (four-five version) and "MiniMax M2 5"
    # (M2.5 product line). Prior tokens ending in a letter (``70B``)
    # naturally don't merge.
    merged: List[str] = []
    for token in styled:
        if (
            token.isdigit()
            and merged
            and merged[-1][-1:].isdigit()
        ):
            merged[-1] = f"{merged[-1]}.{token}"
        else:
            merged.append(token)
    result = " ".join(merged)

    # OpenAI fine-tuning endpoints come in as ``FT GPT 4.1 Mini``; the
    # bare ``FT`` prefix isn't self-explanatory to anyone who hasn't
    # used the fine-tune API. Rewrite it as a trailing "(Fine-tuned)"
    # tag so the list view reads like a product variant.
    if result.startswith("FT "):
        result = f"{result[3:]} (Fine-tuned)"
    return result


# Reverse map: app provider slug → the maker that provider is the
# first-party API for. Used by the stub filter to rescue entities
# whose only surviving offering is a first-party LiteLLM mirror with
# upstream-null pricing (e.g. ByteDance's Doubao on Volcengine, which
# LiteLLM lists but leaves priced as null because Volcengine publishes
# RMB only — we still want the user to see the model exists).
#
# Aggregator-style providers are intentionally absent: aws_bedrock,
# azure_ai, google_vertex_ai, openrouter redistribute many makers and
# are nobody's "home". google_gemini is included because AI Studio is
# Google's own endpoint for Gemini.
FIRST_PARTY_PROVIDER_TO_MAKER: Dict[str, str] = {
    "ai21": "AI21",
    "alibaba_qwen": "Alibaba",
    "anthropic": "Anthropic",
    "cohere": "Cohere",
    "deepseek": "DeepSeek",
    "gigachat": "Sber",
    "google_gemini": "Google",
    "meta_llama": "Meta",
    "minimax": "MiniMax",
    "mistral": "Mistral",
    "moonshot": "Moonshot AI",
    "openai": "OpenAI",
    "volcengine": "ByteDance",
    "xai": "xAI",
    "zai": "Z.AI",
}


def _is_stub_offering_set(
    offerings: List[OfferingV2],
    *,
    entity_maker: Optional[str] = None,
) -> bool:
    """True if the entity has only LiteLLM-mirrored offerings with no
    usable price AND is not a first-party row we should preserve.

    A "stub" entity is one whose every offering is:

    - sourced from ``litellm_fallback`` or ``via_litellm``, AND
    - has both input and output price at 0 or missing.

    **First-party rescue (requires ``entity_maker``):** if any
    offering's provider is the entity maker's first-party API (e.g.
    ``volcengine`` for ``maker="ByteDance"``), the entity is kept
    even with all-null prices. Users expect Doubao / GigaChat /
    other-makers-with-RMB-pricing to show up as existing models with
    a blank price column rather than vanish silently. Without an
    ``entity_maker`` (legacy callers / unit tests), the rescue is
    skipped and the rule is strictly "all mirrored + all null".

    Third-party null mirrors (Volcengine's ``deepseek-v3-2-251201``
    before the YYMMDD normalization fix) still get pruned because
    ``volcengine`` is not DeepSeek's first-party — they fall through
    to the all-null rule.

    Real free OpenRouter ``*-free`` variants come through as
    ``provider_api``, pass this check, and are preserved.
    """
    if not offerings:
        return False

    # First-party rescue: any offering whose provider claims this
    # maker bypasses the stub rule, even with null prices.
    if entity_maker is not None:
        for offering in offerings:
            first_party_maker = FIRST_PARTY_PROVIDER_TO_MAKER.get(
                offering.provider
            )
            if first_party_maker == entity_maker:
                return False

    for offering in offerings:
        if offering.source not in _STUB_SOURCES:
            return False
        input_price = offering.pricing.input or 0
        output_price = offering.pricing.output or 0
        if input_price != 0 or output_price != 0:
            return False
    return True


# Sanity envelope for embedding input prices, in $/M tokens.
# Cheapest real embedding on the market is ~$0.01/M (Voyage Lite,
# text-embedding-3-small). $100/M is 100x the most expensive real
# embedding (Cohere embed-v4 at $0.12). Anything outside this range
# almost always traces to a provider scraper unit bug (AWS returning
# per-1k prices as per-token) or a stale LiteLLM entry that nobody
# noticed. We null the price rather than dropping the offering so the
# drift report keeps a record.
EMBEDDING_INPUT_PRICE_MIN = 0.001   # $0.001 / M
EMBEDDING_INPUT_PRICE_MAX = 10.0    # $10 / M


# Matches a pinned-date snapshot suffix on a provider_model_id. The
# regex deliberately allows either a dashed form (2024-11-20) or the
# 8-digit compact form (20241120); both show up in LiteLLM / Openrouter
# keys for OpenAI / Qwen / Anthropic dated releases.
_DATE_SUFFIX_RE = re.compile(r"-\d{4}-\d{2}-\d{2}$|-\d{8}$")


def _looks_date_pinned(provider_model_id: str) -> bool:
    return bool(_DATE_SUFFIX_RE.search(provider_model_id))


def _dedupe_offerings_per_provider(
    offerings: List[OfferingV2],
) -> List[OfferingV2]:
    """Collapse rows where the same provider publishes multiple variants
    that resolve to the same entity.

    Preference (lower score wins):
        1. Complete headline pricing (``input`` and ``output`` both set)
           beats incomplete. A stale 2024-05 row with different
           numbers is not informative.
        2. An un-dated ``provider_model_id`` beats a date-pinned one.
           Users expect the "current" alias.
        3. Stable tie-break by first occurrence.
    """
    if len(offerings) <= 1:
        return list(offerings)

    # Preserve insertion order for stable tie-breaks.
    groups: Dict[str, List[OfferingV2]] = {}
    order: List[str] = []
    for offering in offerings:
        if offering.provider not in groups:
            groups[offering.provider] = []
            order.append(offering.provider)
        groups[offering.provider].append(offering)

    out: List[OfferingV2] = []
    for provider in order:
        group = groups[provider]
        if len(group) == 1:
            out.append(group[0])
            continue

        def score(o: OfferingV2) -> tuple[int, int]:
            incomplete = int(
                o.pricing.input is None or o.pricing.output is None
            )
            dated = int(_looks_date_pinned(o.provider_model_id))
            return (incomplete, dated)

        best_idx, _ = min(
            enumerate(group), key=lambda pair: score(pair[1])
        )
        out.append(group[best_idx])
    return out


# Providers whose pricing reaches us via the LiteLLM community registry
# rather than a direct first-party API or scrape. Each of these is the
# official model vendor (Anthropic / OpenAI / xAI / DeepSeek) or a
# first-party distribution channel (Google Vertex AI) that does not
# publish a scrape-friendly price page. LiteLLM tracks their
# ``model_prices_and_context_window.json`` within hours of a release.
# Offerings produced by these providers carry ``source="via_litellm"``
# so the UI can label the pricing row transparently as a two-hop
# mirror rather than a one-hop first-party fetch.
LITELLM_SOURCED_PROVIDERS = frozenset({
    "anthropic",
    "openai",
    "xai",
    "deepseek",
    "google_vertex_ai",
    # Added in the "route A" symmetry pass — every maker that operates
    # its own first-party API goes here, same two-hop chain as the
    # five above.
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


_COMPLETE_CHAT_MODES = frozenset({"chat", "completion", "", None})


def _has_complete_headline_pricing(offering: OfferingV2, mode: Optional[str]) -> bool:
    """True if the offering has non-null values for the UI's headline
    price fields.

    Used by :meth:`OfferingMerger._choose_primary` to skip candidates
    whose public price would render as "—" in the list row. Kept
    permissive for modes we don't explicitly know about, so unknown
    modes never get silently demoted.
    """
    p = offering.pricing
    if mode in _COMPLETE_CHAT_MODES:
        return p.input is not None and p.output is not None
    if mode == "embedding":
        return p.input is not None or p.embedding is not None
    if mode == "image_generation":
        return p.image_input is not None or p.output is not None
    if mode == "audio_transcription":
        return p.audio_input is not None or p.input is not None
    if mode == "audio_speech":
        return p.audio_output is not None or p.output is not None
    # Rerank / moderation / unknown: pricing is per-request or unmapped
    # to a per-1M headline. Don't reject — any real offering is fine.
    return True


def _is_embedding_price_outlier(
    offering: OfferingV2, mode: str
) -> bool:
    """True if an embedding offering's input price is wildly out of range.

    Embedding input prices cluster in a narrow band ($0.01 – $1 per
    million tokens). Values far outside that band are almost always
    data bugs — TwelveLabs Marengo scraped as $0.0001/M (AWS parser
    unit error) and Cohere embed-multilingual-light stamped at $100/M
    (stale/bogus LiteLLM entry). Both used to surface as "cheap
    alternatives" for other embeddings, poisoning the alternatives list.
    """
    if mode != "embedding":
        return False
    inp = offering.pricing.input
    if inp is None:
        return False
    return inp < EMBEDDING_INPUT_PRICE_MIN or inp > EMBEDDING_INPUT_PRICE_MAX


class OfferingMerger:
    def __init__(
        self,
        registry: LiteLLMRegistry,
        resolver: CanonicalResolver,
    ) -> None:
        self.registry = registry
        self.resolver = resolver
        self.drift = DriftReporter()

    async def build_snapshot(
        self,
        v1_models_by_provider: Dict[str, List[ModelPricing]],
    ) -> Tuple[EntityStoreSnapshot, Dict[str, List[OfferingV2]]]:
        now = datetime.utcnow()
        entities: Dict[str, EntityCoreV2] = {}
        offerings_by_entity: Dict[str, List[OfferingV2]] = {}

        # ─── Pass 1: bootstrap entities from canonical LiteLLM entries
        for entry in self.registry.iter_canonical():
            slug = entry.canonical_id
            if slug in entities:
                continue
            entities[slug] = self._entity_from_litellm(entry, now)
            offerings_by_entity[slug] = []

        # ─── Pass 2: attach provider offerings
        attach_counts: Dict[str, int] = {}
        unmatched_buckets: Dict[str, List[Tuple[str, ModelPricing]]] = {}

        for provider_name, models in v1_models_by_provider.items():
            for model in models:
                resolution = self.resolver.resolve(provider_name, model.model_id)
                canonical_id = resolution.canonical_id if resolution.matched() else None

                # If the resolver matched an alias that points at a
                # canonical_id with no actual registry entry behind it
                # (a "dangling alias"), treat it as unmatched so the
                # record can still be promoted into a synthetic entity.
                # Remember the dangling target so Pass 2b can cluster
                # every aggregator row that resolves to the same alias
                # into one synthetic entity — without this, Vertex's
                # ``vertex_ai/claude-3-5-haiku@20241022`` and Bedrock's
                # ``anthropic.claude-3-5-haiku-20241022-v1:0`` end up
                # in two different synthetic buckets even though they
                # point at the same logical model.
                dangling_target: Optional[str] = None
                if canonical_id is not None and canonical_id not in entities:
                    entry = self.registry.get(canonical_id)
                    if entry is None:
                        dangling_target = canonical_id
                        canonical_id = None

                # Second chance: if the dangling target carries a
                # Vertex-style publisher prefix (``deepseek-ai-``,
                # ``qwen-``, ``meta-llama-`` etc.), try the stripped
                # form — it's often the real canonical id (``qwen-
                # qwen3-coder-*`` → ``qwen3-coder-*``). This closes
                # the gap where ``vertex_ai/publisher/model-maas`` rows
                # would otherwise create a parallel synthetic entity.
                if canonical_id is None and dangling_target is not None:
                    stripped = _strip_publisher_prefix(dangling_target)
                    if stripped != dangling_target:
                        entry = self.registry.get(stripped)
                        if entry is not None:
                            canonical_id = stripped
                            dangling_target = None
                        elif stripped in entities:
                            canonical_id = stripped
                            dangling_target = None
                        else:
                            dangling_target = stripped

                if canonical_id is None:
                    cluster_key = dangling_target or _unmatched_cluster_key(model)
                    unmatched_buckets.setdefault(cluster_key, []).append(
                        (provider_name, model)
                    )
                    self.drift.record_unmatched(
                        provider=provider_name,
                        model_id=model.model_id,
                        tried=resolution.tried,
                    )
                    continue

                if canonical_id not in entities:
                    entry = self.registry.get(canonical_id)
                    if entry is None:
                        continue
                    entities[canonical_id] = self._entity_from_litellm(entry, now)
                    offerings_by_entity.setdefault(canonical_id, [])

                offering = self._offering_from_v1(model, provider_name, now)
                offerings_by_entity[canonical_id].append(offering)
                self.registry.register_alias(model.model_id, canonical_id)
                self.registry.register_alias(
                    f"{provider_name}:{model.model_id}", canonical_id
                )
                attach_counts[provider_name] = attach_counts.get(provider_name, 0) + 1

        # ─── Pass 2b: promote unmatched clusters to synthetic entities
        synthetic_count = 0
        for cluster_key, bucket in unmatched_buckets.items():
            if not cluster_key:
                continue
            # Use the bare cluster_key as slug first so models like
            # "claude-3-5-sonnet" and "llama-4-maverick" — which LiteLLM
            # doesn't expose as first-party canonicals — still get the
            # clean URL users expect. Only prefix with "v1-" if a slug
            # collision would shadow a real canonical entry.
            slug = cluster_key
            if slug in entities:
                slug = f"v1-{cluster_key}"
            if slug in entities:
                slug = f"v1-{cluster_key}-{bucket[0][0]}"
            synthetic_entity = self._synthetic_entity_from_v1(slug, bucket, now)
            if synthetic_entity is None:
                continue
            entities[slug] = synthetic_entity
            offerings_by_entity[slug] = [
                self._offering_from_v1(m, p, now) for p, m in bucket
            ]
            synthetic_count += 1

        # ─── Pass 2c: drop embedding price outliers
        # Prices 1000x cheaper or 100x more expensive than the real
        # market are almost always scraper unit bugs. We run this BEFORE
        # Pass 3 so entities whose only provider offering was an outlier
        # get a chance to fall back to the LiteLLM reference price, and
        # the stub filter handles any that are left with nothing.
        outlier_dropped = 0
        outlier_log: List[tuple[str, str, float]] = []
        for slug, offs in list(offerings_by_entity.items()):
            entity = entities.get(slug)
            if entity is None:
                continue
            kept: List[OfferingV2] = []
            for offering in offs:
                if _is_embedding_price_outlier(offering, entity.mode):
                    outlier_dropped += 1
                    outlier_log.append((slug, offering.provider, offering.pricing.input or 0.0))
                    continue
                kept.append(offering)
            offerings_by_entity[slug] = kept
        if outlier_log:
            logger.warning(
                "OfferingMerger: dropped %s embedding price outliers: %s",
                outlier_dropped,
                outlier_log[:10],
            )

        # ─── Pass 3: synthesize LiteLLM-fallback offerings
        synthesized = 0
        for slug, entity in entities.items():
            if offerings_by_entity.get(slug):
                continue
            litellm_entry = self.registry.get(slug)
            if litellm_entry is None:
                continue
            fallback = self._offering_from_litellm(litellm_entry, now)
            if fallback is None:
                continue
            # Guard: don't synthesize a fallback whose LiteLLM price is
            # itself an outlier (embed-multilingual-light at $100/M).
            if (
                entity.mode == "embedding"
                and fallback.pricing.input is not None
                and (
                    fallback.pricing.input < EMBEDDING_INPUT_PRICE_MIN
                    or fallback.pricing.input > EMBEDDING_INPUT_PRICE_MAX
                )
            ):
                continue
            offerings_by_entity.setdefault(slug, []).append(fallback)
            synthesized += 1

        # ─── Pass 4: prune entities without any usable offering
        # Also drop "stubs" — entities whose only offerings are
        # litellm_fallback placeholders with no real price data
        # ($0 input AND $0 output). LiteLLM routinely publishes empty
        # entries for brand-new model releases before upstream pricing
        # is known, and for per-request APIs (moderation, rerank)
        # whose pricing model doesn't fit our per-token schema at all.
        # Keeping them surfaces misleading "free" entries to users
        # and creates phantom duplicates like kimi-k2-thinking-251104
        # alongside the real kimi-k2-thinking. Real free models
        # (OpenRouter's *-free variants) come through as provider_api
        # offerings and are preserved.
        final_slugs: set[str] = set()
        stub_count = 0
        for slug, offs in offerings_by_entity.items():
            if not offs:
                continue
            entity = entities.get(slug)
            maker = entity.maker if entity is not None else None
            if _is_stub_offering_set(offs, entity_maker=maker):
                stub_count += 1
                continue
            final_slugs.add(slug)
        pruned_entities = [entities[s] for s in sorted(final_slugs)]
        pruned_offerings: Dict[str, List[OfferingV2]] = {
            s: offerings_by_entity[s] for s in final_slugs
        }

        # ─── Pass 4.5: collapse per-provider date/snapshot duplicates
        # OpenRouter (and a few other aggregators) publish every pinned-
        # date snapshot of a model — openai/gpt-4o-2024-11-20,
        # openai/gpt-4o-2024-08-06, openai/gpt-4o-2024-05-13, and the
        # bare openai/gpt-4o — all as independent rows. Canonical
        # resolution maps all four to the same entity, so the detail
        # page ends up showing four "OpenRouter" lines, one of them
        # with the stale 2024-05 price that nobody should care about.
        # We keep the best one per (entity, provider), preferring
        # complete pricing and the un-dated alias.
        dedupe_dropped = 0
        for slug in list(pruned_offerings.keys()):
            deduped = _dedupe_offerings_per_provider(pruned_offerings[slug])
            dedupe_dropped += len(pruned_offerings[slug]) - len(deduped)
            pruned_offerings[slug] = deduped

        # ─── Pass 5: decide primary offering per entity and finalize sources
        for entity in pruned_entities:
            offs = pruned_offerings.get(entity.slug, [])
            primary = self._choose_primary(entity, offs)
            entity.primary_offering_provider = primary
            entity.sources = sorted({"litellm", *[o.provider for o in offs]})
            entity.last_refreshed = now

        if dedupe_dropped:
            logger.info(
                "OfferingMerger: collapsed %s per-provider duplicate offerings",
                dedupe_dropped,
            )

        flat_offerings: List[OfferingV2] = []
        for slug in sorted(pruned_offerings.keys()):
            flat_offerings.extend(pruned_offerings[slug])

        logger.info(
            "OfferingMerger: %s entities kept (of %s canonical); "
            "provider attach: %s; litellm_fallback: %s; v1_synthetic: %s; "
            "stubs pruned: %s",
            len(pruned_entities),
            len(entities),
            attach_counts,
            synthesized,
            synthetic_count,
            stub_count,
        )

        snapshot = EntityStoreSnapshot(
            version="v2.0",
            generated_at=now,
            entities=pruned_entities,
            offerings=flat_offerings,
        )
        return snapshot, pruned_offerings

    # ─── Helpers ─────────────────────────────────────────────

    def _entity_from_litellm(
        self, entry: LiteLLMEntry, now: datetime
    ) -> EntityCoreV2:
        return EntityCoreV2(
            canonical_id=entry.canonical_id,
            slug=entry.slug,
            name=self._pretty_model_name(entry),
            family=entry.family,
            maker=entry.maker,
            context_length=entry.context_length,
            max_output_tokens=entry.max_output_tokens,
            capabilities=entry.capabilities,
            input_modalities=entry.input_modalities,
            output_modalities=entry.output_modalities,
            mode=entry.mode,
            is_open_source=self._guess_open_source(entry.maker),
            primary_offering_provider="litellm",
            sources=["litellm"],
            last_refreshed=now,
        )

    def _offering_from_v1(
        self, model: ModelPricing, provider_name: str, now: datetime
    ) -> OfferingV2:
        # v1 Pricing maps cached_input → cache_read (best-effort guess)
        v1p = model.pricing
        pricing = PricingV2(
            input=_round_price(v1p.input),
            output=_round_price(v1p.output),
            cache_read=_round_price(v1p.cached_input),
            cache_write=_round_price(getattr(v1p, "cached_write", None)),
            image_input=_round_price(v1p.image_input),
            audio_input=_round_price(v1p.audio_input),
            audio_output=_round_price(v1p.audio_output),
            embedding=_round_price(v1p.embedding),
        )
        batch = None
        if model.batch_pricing is not None:
            batch = BatchPricingV2(
                input=_round_price(model.batch_pricing.input),
                output=_round_price(model.batch_pricing.output),
            )
        source = (
            "via_litellm"
            if provider_name in LITELLM_SOURCED_PROVIDERS
            else "provider_api"
        )
        return OfferingV2(
            provider=provider_name,
            provider_model_id=model.model_id,
            pricing=pricing,
            batch_pricing=batch,
            availability="ga",
            region=None,
            notes=None,
            last_updated=model.last_updated or now,
            source=source,  # type: ignore[arg-type]
        )

    def _synthetic_entity_from_v1(
        self,
        slug: str,
        bucket: List[Tuple[str, ModelPricing]],
        now: datetime,
    ) -> Optional[EntityCoreV2]:
        """Build an entity from v1 records that didn't match any LiteLLM
        canonical entry. Picks a representative record for the base fields
        and merges capabilities / modalities across the cluster.
        """
        if not bucket:
            return None

        # Pick the record with the richest metadata as the base
        base_provider, base = max(
            bucket,
            key=lambda pair: (
                (pair[1].context_length or 0),
                len(pair[1].capabilities or []),
                -len(pair[1].model_id),
            ),
        )

        display_name = (base.model_name or base.model_id).strip()
        # OpenRouter and similar aggregators prefix display names with the
        # maker ("AionLabs: Aion-1.0"). Strip it so the UI shows a clean
        # product name — the maker already renders alongside.
        if ": " in display_name:
            _prefix, _rest = display_name.split(": ", 1)
            if _prefix and _rest:
                display_name = _rest.strip()
        family, maker = detect_family_maker(slug, display_name)

        # When detect_family_maker can't place this model, fall back to the
        # author prefix from the v1 model_id (OpenRouter-style: "allenai/olmo-…").
        if maker == "Unknown":
            maker = _maker_from_model_id(base.model_id) or "Unknown"
        if family == "Other":
            # Prefer reusing maker as family when we can't detect a real
            # family name — avoids junk labels like "Aionlabs:" or "Body"
            # leaking into the dropdown.
            if maker != "Unknown":
                family = maker

        caps: set[str] = set()
        in_mods: set[str] = set()
        out_mods: set[str] = set()
        ctx = 0
        max_out = 0
        for _provider, model in bucket:
            for cap in model.capabilities or []:
                if cap in DISPLAY_CAPABILITIES:
                    caps.add(cap)
            for m in model.input_modalities or []:
                in_mods.add(m)
            for m in model.output_modalities or []:
                out_mods.add(m)
            if (model.context_length or 0) > ctx:
                ctx = model.context_length or 0
            if (model.max_output_tokens or 0) > max_out:
                max_out = model.max_output_tokens or 0

        if not caps:
            caps.add("text")
        if not in_mods:
            in_mods = {"text"}
        if not out_mods:
            out_mods = {"text"}

        # Primary offering follows authority order, or first bucket entry
        provider_order = AUTHORITY_BY_MAKER.get(maker, [])
        providers_present = {p for p, _ in bucket}
        primary_provider = next(
            (p for p in provider_order if p in providers_present),
            base_provider,
        )

        return EntityCoreV2(
            canonical_id=slug,
            slug=slug,
            # Normalize through the same styling rule canonical entities
            # use, so the list view never mixes "GPT-4o Realtime" and
            # "GPT 4O Realtime Preview" formatting across entities.
            name=_polish_display_name(display_name) or display_name,
            family=family,
            maker=maker,
            context_length=ctx or None,
            max_output_tokens=max_out or None,
            capabilities=sorted(caps),
            input_modalities=sorted(in_mods),
            output_modalities=sorted(out_mods),
            mode="chat",
            is_open_source=self._guess_open_source(maker),
            primary_offering_provider=primary_provider,
            sources=sorted({"v1_synthetic", *[p for p, _ in bucket]}),
            last_refreshed=now,
        )

    def _offering_from_litellm(
        self, entry: LiteLLMEntry, now: datetime
    ) -> Optional[OfferingV2]:
        if entry.input_price is None and entry.output_price is None:
            return None
        pricing = PricingV2(
            input=entry.input_price,
            output=entry.output_price,
            cache_read=entry.cache_read_price,
            cache_write=entry.cache_write_price,
            image_input=entry.image_input_price,
            audio_input=entry.audio_input_price,
            audio_output=entry.audio_output_price,
            embedding=entry.embedding_price,
        )
        batch = None
        if entry.batch_input_price is not None or entry.batch_output_price is not None:
            batch = BatchPricingV2(
                input=entry.batch_input_price,
                output=entry.batch_output_price,
            )
        return OfferingV2(
            provider="litellm",
            provider_model_id=entry.raw_key,
            pricing=pricing,
            batch_pricing=batch,
            availability="ga",
            region=None,
            notes="Price inherited from LiteLLM registry (no first-party fetch)",
            last_updated=now,
            source="litellm_fallback",
        )

    def _choose_primary(
        self, entity: EntityCoreV2, offerings: List[OfferingV2]
    ) -> str:
        """Pick the provider whose pricing heads the entity's list row.

        Precedence (most preferred first):

        1. Authority order with complete headline pricing. For chat
           models that means both ``input`` and ``output`` are non-null.
           Mode-specific headline rules live in
           :func:`_has_complete_headline_pricing`. This is the pass that
           saves us from showing "—" when, say, Bedrock publishes a
           newly-released Claude with ``input=null`` on day one while
           the Anthropic first-party offering has the full numbers.
        2. Authority order, pricing completeness ignored. Catches the
           case where every authority is incomplete — we still want a
           stable "primary" for UI purposes, just pick the most
           authoritative one we have.
        3. Any non-fallback provider with complete pricing (no authority
           match — e.g. unknown maker with a real scraper).
        4. First non-fallback if all real offerings are incomplete.
        5. First offering. Should never happen with a non-empty list
           but keeps the signature total.
        """
        if not offerings:
            return "litellm"

        by_provider: Dict[str, OfferingV2] = {}
        for offering in offerings:
            # If a provider appears more than once (it shouldn't in a
            # well-formed pipeline, but we guard), keep the first.
            by_provider.setdefault(offering.provider, offering)

        preference = AUTHORITY_BY_MAKER.get(entity.maker, [])

        # Pass 1: authority order, complete pricing only.
        for candidate in preference:
            offering = by_provider.get(candidate)
            if offering and _has_complete_headline_pricing(offering, entity.mode):
                return candidate

        # Pass 2: authority order, pricing ignored.
        for candidate in preference:
            if candidate in by_provider:
                return candidate

        # Pass 3 / 4: fall back outside the authority list.
        non_fallback = [o for o in offerings if o.source != "litellm_fallback"]
        for offering in non_fallback:
            if _has_complete_headline_pricing(offering, entity.mode):
                return offering.provider
        if non_fallback:
            return non_fallback[0].provider
        return offerings[0].provider

    def _pretty_model_name(self, entry: LiteLLMEntry) -> str:
        """Derive a display name from the canonical slug.

        Delegates styling to :func:`_polish_display_name` so the
        output lines up with the synthetic path (see
        :meth:`_synthetic_entity_from_v1`) — both produce consistent
        "GPT 4o Mini" / "Claude Sonnet 4.5" casing, no more "GPT 4O"
        vs "GPT-4o" mixed in the same list.
        """
        return _polish_display_name(entry.canonical_id)

    def _guess_open_source(self, maker: str) -> Optional[bool]:
        open_makers = {"Meta", "Mistral", "DeepSeek", "Alibaba", "NVIDIA", "Cohere"}
        if maker in open_makers:
            return True
        if maker in {"Anthropic", "OpenAI", "Google", "xAI", "Amazon", "AI21"}:
            return False
        return None


async def run_refresh_pipeline(
    *, force_network: bool = True
) -> Tuple[EntityStoreSnapshot, "DriftReportV2", Dict[str, List[OfferingV2]]]:  # noqa: F821
    """Single entry point that ties everything together.

    Returns the built snapshot, the drift report, and the offerings_by_entity
    map so callers can persist both to disk and surface the counts to API
    endpoints without rebuilding the reverse index themselves.
    """
    from models.v2 import DriftReportV2  # noqa: F401 - used in annotation

    registry = await get_registry(force_network=force_network)
    resolver = build_resolver(registry)
    merger = OfferingMerger(registry, resolver)

    # Run all v1 provider fetchers via the existing ProviderRegistry.
    # Falls back to per-provider fallback data on any network/parse error.
    logger.info("v2 pipeline: starting v1 provider fetch")
    try:
        v1_models_by_provider = await ProviderRegistry.fetch_all_grouped()
    except Exception as exc:
        logger.warning("v1 fetch_all_grouped failed (%s); using fallback data", exc)
        v1_models_by_provider = {}
        for provider in ProviderRegistry.all():
            try:
                v1_models_by_provider[provider.name] = provider.load_fallback_data()
            except Exception as inner:
                logger.error("fallback load failed for %s: %s", provider.name, inner)
                v1_models_by_provider[provider.name] = []

    total_v1 = sum(len(m) for m in v1_models_by_provider.values())
    logger.info(
        "v2 pipeline: v1 fetch done, %s models across %s providers",
        total_v1,
        len(v1_models_by_provider),
    )

    snapshot, offerings_by_entity = await merger.build_snapshot(v1_models_by_provider)

    previous_slugs = DriftReporter.load_previous_slugs()
    current_slugs = {e.slug for e in snapshot.entities}

    report = merger.drift.build_report(
        entities=snapshot.entities,
        offerings_by_entity=offerings_by_entity,
        previous_slugs=previous_slugs,
        registry=registry,
    )

    save_snapshot(snapshot, offerings_by_entity)
    DriftReporter.save_report(report, current_slugs)

    return snapshot, report, offerings_by_entity


def save_snapshot(
    snapshot: EntityStoreSnapshot,
    offerings_by_entity: Dict[str, List[OfferingV2]],
) -> None:
    V2_DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = snapshot.model_dump(mode="json")
    with ENTITIES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "version": snapshot.version,
                "generated_at": payload["generated_at"],
                "entities": payload["entities"],
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    # offerings.json is keyed by entity slug so load_snapshot() can rebuild
    # the reverse index without needing the LiteLLM registry or a resolver.
    by_slug_serialized: Dict[str, List[dict]] = {}
    for slug, offs in offerings_by_entity.items():
        by_slug_serialized[slug] = [o.model_dump(mode="json") for o in offs]
    with OFFERINGS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "version": snapshot.version,
                "generated_at": payload["generated_at"],
                "by_entity": by_slug_serialized,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    with INDEX_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "version": snapshot.version,
                "generated_at": payload["generated_at"],
                "entities_count": len(snapshot.entities),
                "offerings_count": len(snapshot.offerings),
            },
            handle,
            indent=2,
        )


def load_snapshot() -> Optional[
    Tuple[EntityStoreSnapshot, Dict[str, List[OfferingV2]]]
]:
    if not ENTITIES_PATH.exists() or not OFFERINGS_PATH.exists():
        return None
    try:
        with ENTITIES_PATH.open("r", encoding="utf-8") as handle:
            ent_data = json.load(handle)
        with OFFERINGS_PATH.open("r", encoding="utf-8") as handle:
            off_data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    entities = [EntityCoreV2.model_validate(e) for e in ent_data.get("entities", [])]
    by_entity_raw = off_data.get("by_entity", {})
    offerings_by_entity: Dict[str, List[OfferingV2]] = {}
    flat: List[OfferingV2] = []
    for slug, items in by_entity_raw.items():
        parsed = [OfferingV2.model_validate(item) for item in items]
        offerings_by_entity[slug] = parsed
        flat.extend(parsed)

    snapshot = EntityStoreSnapshot(
        version=ent_data.get("version", "v2.0"),
        generated_at=ent_data.get("generated_at") or datetime.utcnow().isoformat() + "Z",
        entities=entities,
        offerings=flat,
    )
    return snapshot, offerings_by_entity
