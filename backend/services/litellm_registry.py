"""LiteLLM registry — single source of truth for v2.

Fetches the community-maintained JSON at
https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json
and exposes it as:

- A dict of canonical entries keyed by canonical_id (normalized slug)
- A reverse alias table: any known provider-specific key or prefixed
  variant → canonical_id

Caches the raw JSON to backend/data/v2/cache/litellm_registry.json
so that cold-starts without network access still have usable data.

All pricing values are converted from per-token USD to per-1M-token USD
at parse time, so the rest of the backend never has to think about units.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx

logger = logging.getLogger(__name__)

LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

CACHE_PATH = (
    Path(__file__).parent.parent / "data" / "v2" / "cache" / "litellm_registry.json"
)

PER_TOKEN_TO_PER_MTOKEN = 1_000_000.0

# Providers whose entries we treat as "canonical" (first-party, original
# model vendor). Everything else is an aggregator / distribution offering
# that should merge into one of these canonical entities.
CANONICAL_LITELLM_PROVIDERS = {
    "anthropic",
    "openai",
    "chatgpt",
    "text-completion-openai",
    "google",
    "gemini",
    "palm",
    "vertex_ai-language-models",
    "xai",
    "deepseek",
    "mistral",
    "codestral",
    "meta",
    "meta_llama",
    "cohere",
    "cohere_chat",
    "ai21",
    "nvidia",
    "nvidia_nim",
    "voyage",
    "jina_ai",
    "assemblyai",
    "deepgram",
    "elevenlabs",
    # Chinese + other first-party model labs
    "moonshot",      # Moonshot AI — Kimi
    "zai",           # Z.AI — GLM
    "minimax",       # MiniMax — abab/m1
    "dashscope",     # Alibaba — Qwen official API
    "volcengine",    # ByteDance — Doubao
    "gigachat",      # Sber
    # Image / video / audio labs (first-party)
    "black_forest_labs",  # Flux
    "stability",     # Stable Diffusion / SDXL
    "runwayml",      # Gen / video
    "fal_ai",        # keep off — aggregator; stays out
    "aleph_alpha",   # Luminous
}

# Providers whose entries are explicitly aggregators / distribution channels.
# These do NOT become canonical entities; they become offerings that merge
# into canonical entities via alias resolution.
AGGREGATOR_LITELLM_PROVIDERS = {
    "bedrock",
    "bedrock_converse",
    "azure",
    "azure_text",
    "azure_ai",
    "vertex_ai-anthropic_models",
    "vertex_ai-mistral_models",
    "vertex_ai-llama_models",
    "vertex_ai-chat-models",
    "vertex_ai-code-chat-models",
    "openrouter",
    "databricks",
    "snowflake",
    "perplexity",
    "groq",
    "fireworks_ai",
    "together_ai",
    "replicate",
}

# Historical map of our provider slugs → LiteLLM tags. Never actually
# read by the pipeline — each provider in ``backend/providers/`` owns
# its own filter against ``LiteLLMEntry.litellm_provider``. Kept here
# removed to avoid giving the false impression that a slug has a
# backing provider just because it appears in this table.

# Only the 7 capabilities users actually care about when shopping for
# a model. Internal LiteLLM flags like prompt_caching / pdf / web_search
# are derived but NOT surfaced — they clutter list rows and don't drive
# decisions. If a user needs pdf or web search they'll find it on the
# provider page; pricing comparison isn't the place.
#
# Each flag maps to (capability, modality-hint). A modality hint of
# "in:image" means "if this flag is set, the model accepts image
# input". A value of None means the flag only contributes to the
# capability set. Keeping the two dimensions glued here is the only
# reliable way to prevent the two sides from drifting apart — the
# original bug was exactly that: caps derived from one field list,
# modalities from another, and embedding_image_input set vision
# without ever touching input_modalities.
CAPABILITY_FLAGS: List[tuple[str, Optional[str], Optional[str]]] = [
    ("supports_vision", "vision", "in:image"),
    ("supports_pdf_input", "vision", "in:image"),  # PDFs read as vision
    ("supports_embedding_image_input", "vision", "in:image"),
    ("supports_audio_input", "audio", "in:audio"),
    ("supports_audio_output", "audio", "out:audio"),
    ("supports_video_input", None, "in:video"),
    ("supports_function_calling", "function_calling", None),
    ("supports_tool_choice", "tool_use", None),
    ("supports_parallel_function_calling", "tool_use", None),
    ("supports_response_schema", "function_calling", None),
    ("supports_reasoning", "reasoning", None),
]

DISPLAY_CAPABILITIES = {
    "text",
    "vision",
    "audio",
    "tool_use",
    "reasoning",
    "function_calling",
    "image_generation",
    "embedding",
}

FAMILY_PATTERNS: List[tuple[str, str, List[str]]] = [
    # (family, maker, list of lowercase substrings in canonical id/name)
    # Order matters — more specific patterns first.
    ("Cogito", "Deep Cogito", ["cogito"]),  # Cogito models contain "llama", must check first
    ("Claude", "Anthropic", ["claude"]),
    ("Gemini", "Google", ["gemini"]),
    ("Gemma", "Google", ["gemma"]),
    ("Imagen", "Google", ["imagen"]),
    ("Veo", "Google", ["veo-"]),
    ("LearnLM", "Google", ["learnlm"]),
    ("MedLM", "Google", ["medlm"]),
    ("Google Deep Research", "Google", ["deep-research"]),
    # Codex must be checked BEFORE GPT because model names like
    # "gpt-5-codex" and "gpt-5.1-codex-max" contain the "gpt-" needle
    # and would otherwise collapse into the GPT family.
    ("Codex", "OpenAI", ["codex"]),
    ("OpenAI O-Series", "OpenAI", ["o1-", "o3-", "o4-", "o1 ", "o3 ", "o4 ", "o1,", "o3,", "o4,"]),
    ("GPT", "OpenAI", [
        "gpt-", "gpt4", "gpt3", "gpt5", "chatgpt",
        "babbage", "davinci",
        "ft:", "ft-",
    ]),
    ("Whisper", "OpenAI", ["whisper"]),
    ("DALL-E", "OpenAI", ["dall-e"]),
    ("OpenAI TTS", "OpenAI", ["tts-"]),
    ("OpenAI Embedding", "OpenAI", ["text-embedding"]),
    ("Sora", "OpenAI", ["sora"]),
    ("Llama", "Meta", ["llama", "llama-guard"]),
    ("Mistral", "Mistral", ["mistral", "mixtral", "codestral", "ministral", "pixtral", "devstral", "voxtral", "magistral"]),
    ("Nova", "Amazon", ["nova-"]),
    ("Titan", "Amazon", ["titan"]),
    ("Command", "Cohere", ["command"]),
    # `^embed-` only matches at the start of the target so TwelveLabs
    # Marengo ("twelvelabs-marengo-embed-2-7") no longer gets
    # misclassified as Cohere. The bare `cohere-embed` form picks up
    # AWS Bedrock's canonical ids like `cohere-embed-4-model`.
    ("Cohere Embed", "Cohere", ["^embed-", "cohere-embed", "cohere embed"]),
    # TwelveLabs must come after Cohere so "marengo-embed" doesn't
    # slip through, but still claims `twelvelabs-*` models.
    ("Marengo", "TwelveLabs", ["twelvelabs-marengo", "marengo-embed"]),
    ("Pegasus", "TwelveLabs", ["twelvelabs-pegasus", "pegasus-1"]),
    ("Rerank", "Cohere", ["cohere-rerank", "cohere rerank", "^rerank"]),
    ("Grok", "xAI", ["grok"]),
    ("DeepSeek", "DeepSeek", ["deepseek", "r1-"]),
    ("Qwen", "Alibaba", ["qwen", "qwq", "tongyi"]),
    ("Nemotron", "NVIDIA", ["nemotron"]),
    ("Jamba", "AI21", ["jamba"]),
    ("Jurassic", "AI21", ["jurassic", "j2-"]),
    ("Phi", "Microsoft", ["phi-"]),
    ("Stable Diffusion", "Stability AI", ["stable-diffusion", "sdxl", "sd3-", "sd-3"]),
    # Chinese labs
    ("Kimi", "Moonshot AI", ["kimi"]),
    ("Moonshot", "Moonshot AI", ["moonshot-v"]),
    ("GLM", "Z.AI", ["glm-", "glm4", "chatglm"]),
    ("MiniMax", "MiniMax", ["minimax", "abab", "m1-"]),
    ("Yi", "01.AI", ["yi-", "01-ai"]),
    ("Ernie", "Baidu", ["ernie"]),
    ("Doubao", "ByteDance", ["doubao"]),
    ("Hunyuan", "Tencent", ["hunyuan"]),
    ("Step", "StepFun", ["step-"]),
    # Image / video / audio labs
    ("Flux", "Black Forest Labs", ["flux-", "flux-1", "flux-pro"]),
    ("Runway", "Runway", ["runway-", "gen-4", "gen-3"]),
    ("Luminous", "Aleph Alpha", ["luminous"]),
    # Embedding / audio service canonicals
    ("Voyage", "Voyage AI", ["voyage-"]),
    ("Jina", "Jina AI", ["jina-"]),
    ("AssemblyAI", "AssemblyAI", ["assemblyai", "universal-"]),
    ("Deepgram", "Deepgram", ["deepgram", "nova-2-"]),
    ("ElevenLabs", "ElevenLabs", ["eleven-", "elevenlabs"]),
]


def slugify(raw: str) -> str:
    """Normalize a LiteLLM key to a stable slug.

    Examples:
        claude-sonnet-4-5 → claude-sonnet-4-5
        bedrock/anthropic.claude-sonnet-4-5-v1:0 → anthropic-claude-sonnet-4-5-v1-0
        openrouter/google/gemini-2.5-pro → google-gemini-2-5-pro
        azure/gpt-4o → gpt-4o
    """
    s = raw.lower().strip()
    # drop leading provider prefix segment, e.g. "bedrock/", "openrouter/",
    # "azure/", but preserve meaningful sub-paths
    s = re.sub(r"^[a-z0-9_-]+/", "", s)
    # flatten remaining slashes and dots to dashes
    s = s.replace("/", "-").replace(".", "-")
    # collapse multiple dashes, strip non-[a-z0-9-]
    s = re.sub(r"[^a-z0-9-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


_VARIANT_TAGS = (
    # Quantization / hardware packaging — same logical model, different
    # storage format. Safe to strip.
    "-rlhf",
    "-fp8",
    "-fp16",
    "-bf16",
    "-int8",
    "-int4",
    # MoE expert counts — same base model, different sparsity profile.
    "-128e",
    "-64e",
    "-32e",
    "-16e",
    "-8e",
    # Vertex AI's ``@default`` suffix means "this model's default
    # variant"; after slugify it lands as ``-default`` on an id that
    # already represents the same model (claude-opus-4-7@default →
    # claude-opus-4-7).
    "-default",
    # ``-latest`` is a floating alias that providers publish alongside
    # the pinned-date canonical. We used to preserve it as a distinct
    # product because the underlying target drifts over time, but from
    # the user's perspective it was pure noise — every Mistral /
    # Grok / Qwen / Devstral / Codex product showed up twice
    # ("mistral-large" vs "mistral-large-latest") with the same maker
    # and typically identical pricing. Strip it; rely on
    # _dedupe_offerings_per_provider to pick the offering with
    # complete pricing when the two variants differ.
    "-latest",
    # Gemma uses ``-it`` (instruction-tuned) to mark the main product
    # variant. The bare name (``gemma-3-12b``) and the ``-it`` form
    # (``gemma-3-12b-it``) are the same model in practice — Google
    # publishes Gemma only as instruction-tuned for API consumption.
    # Non-Gemma families do not use ``-it`` so this is safe globally.
    "-it",
    # OpenRouter ships ``*-free`` variants that are the same model on
    # the same infrastructure with a $0 price cap / rate limit. Merge
    # them so users see a single entity with both the regular and
    # the free-tier offering rows.
    "-free",
    # Vertex AI Model Garden / Alibaba Model Studio (dashscope)
    # expose their hosted-service offerings with a ``@maas`` suffix
    # ("Model as a Service"). After slugify it lands as ``-maas`` on
    # otherwise identical model ids, so it's pure packaging.
    "-maas",
)
# Tags that look like packaging but actually carry product identity
# and must NOT be stripped:
#
# -instruct : gpt-3.5-turbo-instruct is a separate OpenAI product
#             from gpt-3.5-turbo with different pricing.
# -preview  : gemini-2.5-pro-preview is a pinned snapshot that
#             diverges from gemini-2.5-pro over time.
# -exp / -experimental : DeepSeek v3.2-exp is a distinct product tier
#             from deepseek-v3.2 (confirmed by $0.27 vs $0.26 pricing).
# -chat / -base : legitimate product names (deepseek-chat, qwen-base,
#             gpt-5-chat is a distinct endpoint from gpt-5).


def strip_version_suffix(slug: str) -> str:
    """Strip provider/version/variant suffixes that do NOT carry identity.

    Conservative about version markers in model names: we don't strip
    bare "-v\\d+" (would break "deepseek-v3"), but we do strip Bedrock
    style "-v1:0" / "-v1-0", 8-digit date stamps, and a whitelist of
    variant / quantization tags that simply describe how the same
    logical model is packaged (instruct / fp8 / 128e experts / preview).

    Applied iteratively so chained suffixes collapse in one pass:
    llama-4-maverick-17b-128e-instruct-v1-0 → llama-4-maverick-17b
    """
    s = slug
    # Leading resolution prefix from LiteLLM image-model pricing rows:
    # ``1024-x-1024-gpt-image-1-5`` / ``1024-x-1536-gpt-image-1-5`` /
    # ``1536-x-1024-gpt-image-1-5`` / ``1792-x-1024-dall-e-3``. These
    # encode per-resolution pricing for the same logical model and
    # should collapse into one canonical entity. Match only digits +
    # ``-x-`` + digits at the start so a random ``-x-`` elsewhere
    # never gets shaved.
    s = re.sub(r"^\d+-x-\d+-", "", s)
    prev = None
    while s != prev:
        prev = s
        # Bedrock / Azure style: "-v1:0" or "-v1-0". Bedrock always
        # publishes the sub-revision as literal 0, so we anchor on that
        # — this keeps "deepseek-v3-2" (a real product name) intact.
        s = re.sub(r"-v\d+[-:]0$", "", s)
        # 8-digit date: -20250929
        s = re.sub(r"-\d{8}$", "", s)
        # YYYY-MM-DD
        s = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", s)
        # 6-digit compact date (YYMMDD), e.g. Volcengine's
        # deepseek-v3-2-251201 → deepseek-v3-2. Tightly anchored to
        # plausible 21st-century dates so arbitrary 6-digit numeric
        # tails in model names do not get shaved accidentally:
        #   - year 20-29 (``2\d``)
        #   - month 01-12
        #   - day 01-31
        s = re.sub(
            r"-2\d(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])$", "", s
        )
        # 4-digit compact date (MM-DD), e.g.
        # ``gemini-2-5-pro-preview-05-06`` / ``pixtral-large-25-02``.
        # Both month and day are two digits and zero-padded, so this
        # cannot accidentally match version numbers like
        # ``claude-sonnet-4-5`` (``-4-5`` has single digits without
        # leading zero).
        s = re.sub(
            r"-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$", "", s
        )
        # Variant tags
        for tag in _VARIANT_TAGS:
            if s.endswith(tag):
                s = s[: -len(tag)]
                break
    return s


def detect_family_maker(canonical_id: str, model_name: str) -> tuple[str, str]:
    """Classify a model by family and maker via substring match.

    Needles are matched against `"{canonical_id} {model_name}"`, lowered.
    A needle prefixed with `^` must appear at the start of the target
    (or right after the space separator between canonical_id and
    model_name). Use the anchor for short identifiers that otherwise
    collide across vendors — e.g. `"^embed-"` for Cohere Embed so it
    does NOT grab `twelvelabs-marengo-embed-2-7`.
    """
    target = f"{canonical_id} {model_name}".lower()
    name_offset = len(canonical_id) + 1
    for family, maker, needles in FAMILY_PATTERNS:
        for needle in needles:
            if needle.startswith("^"):
                bare = needle[1:]
                if target.startswith(bare) or target.startswith(bare, name_offset):
                    return family, maker
            elif needle in target:
                return family, maker
    return "Other", "Unknown"


_MODALITY_ORDER = ["text", "image", "audio", "video", "embedding"]


def _sorted_modalities(mods: set[str]) -> List[str]:
    ordered = [m for m in _MODALITY_ORDER if m in mods]
    # Preserve any custom modality outside the standard order.
    ordered.extend(sorted(m for m in mods if m not in _MODALITY_ORDER))
    return ordered


def derive_capability_modality(
    raw_entry: Dict[str, Any],
) -> tuple[List[str], List[str], List[str]]:
    """Derive capabilities, input modalities, and output modalities from
    a single signal pass.

    Caps and modalities used to live in two separate functions each
    reading its own subset of `supports_*` flags, which silently drifted
    (e.g. `supports_embedding_image_input` set `vision` but never
    `input_modalities=[image]`). Keeping them in one pass is the
    structural fix — any new flag must declare both sides here.
    """
    caps: set[str] = {"text"}
    in_mods: set[str] = {"text"}
    out_mods: set[str] = {"text"}

    for flag, cap, modality_hint in CAPABILITY_FLAGS:
        if not raw_entry.get(flag):
            continue
        if cap is not None:
            caps.add(cap)
        if modality_hint is None:
            continue
        direction, value = modality_hint.split(":", 1)
        if direction == "in":
            in_mods.add(value)
        elif direction == "out":
            out_mods.add(value)

    mode = raw_entry.get("mode") or ""
    if mode == "image_generation":
        caps.add("image_generation")
        caps.discard("text")
        in_mods.add("image")
        out_mods = {"image"}
    elif mode == "embedding":
        caps.add("embedding")
        caps.discard("text")
        out_mods = {"embedding"}
    elif mode == "audio_transcription":
        caps.add("audio")
        in_mods.add("audio")
    elif mode == "audio_speech":
        caps.add("audio")
        out_mods.add("audio")
    elif mode == "rerank":
        # Rerank endpoints accept (query, documents) and return scores;
        # the UI doesn't model a "rerank" cap, so leave it as text.
        pass

    display_caps = sorted(caps & DISPLAY_CAPABILITIES)
    return display_caps, _sorted_modalities(in_mods), _sorted_modalities(out_mods)


def convert_pricing_field(value: Any) -> Optional[float]:
    """LiteLLM stores per-token USD. Convert to per-1M.

    Rounded to 4 decimal places: enough precision for all public prices
    (smallest real-world cache-hit price is ~$0.0001/M), and close to
    float-representable values so the JSON doesn't show 0.19999…
    """
    if value is None:
        return None
    try:
        return round(float(value) * PER_TOKEN_TO_PER_MTOKEN, 4)
    except (TypeError, ValueError):
        return None


# Modes where an all-zero pricing row is a placeholder (not real free
# pricing): LiteLLM routinely publishes $0 rows for new model releases
# before upstream USD pricing is known, and for providers whose public
# prices aren't in USD (volcengine Doubao, dashscope Qwen, etc.). For
# these modes we null out the prices so downstream code can't mistake
# the placeholder for a real "free" tier. Rerank / moderation are
# excluded because their pricing model is per-request, not per-token,
# so we legitimately have no per-token value to show.
_STUB_CHECKED_MODES = frozenset(
    {"chat", "completion", "embedding", "image_generation", "audio_transcription", "audio_speech"}
)


def _is_zero_stub_row(raw: Dict[str, Any]) -> bool:
    """True if the LiteLLM entry looks like a $0 placeholder row.

    Heuristic: both input and output cost are literal 0 (not missing),
    no other per-token pricing signal is present, and the mode is one
    where a true $0 is not plausible. Brand-new releases and non-USD
    pricing from volcengine/dashscope land here.
    """
    mode = raw.get("mode") or ""
    if mode not in _STUB_CHECKED_MODES:
        return False

    input_cost = raw.get("input_cost_per_token")
    output_cost = raw.get("output_cost_per_token")
    if input_cost != 0 or output_cost not in (0, None):
        return False

    # If the row at least has a cache, batch, or image cost, it's not a stub.
    for field in (
        "cache_read_input_token_cost",
        "cache_creation_input_token_cost",
        "input_cost_per_image",
        "input_cost_per_audio_token",
        "output_cost_per_audio_token",
        "input_cost_per_token_batches",
        "output_cost_per_token_batches",
    ):
        value = raw.get(field)
        if value is not None and value != 0:
            return False
    return True


@dataclass
class LiteLLMEntry:
    """A single LiteLLM registry entry, normalized for our use."""

    raw_key: str
    canonical_id: str  # slug derived from raw_key (sans provider prefix)
    slug: str  # == canonical_id, kept separate for frontend compat
    name: str  # human-readable
    family: str
    maker: str
    litellm_provider: str
    is_canonical: bool  # True if litellm_provider ∈ CANONICAL_LITELLM_PROVIDERS
    # Normalized pricing (per 1M tokens)
    input_price: Optional[float]
    output_price: Optional[float]
    cache_read_price: Optional[float]
    cache_write_price: Optional[float]
    image_input_price: Optional[float]
    audio_input_price: Optional[float]
    audio_output_price: Optional[float]
    embedding_price: Optional[float]
    batch_input_price: Optional[float]
    batch_output_price: Optional[float]
    context_length: Optional[int]
    max_output_tokens: Optional[int]
    capabilities: List[str]
    input_modalities: List[str]
    output_modalities: List[str]
    mode: str
    raw: Dict[str, Any] = field(default_factory=dict)

    def as_pricing_dict(self) -> Dict[str, Optional[float]]:
        return {
            "input": self.input_price,
            "output": self.output_price,
            "cache_read": self.cache_read_price,
            "cache_write": self.cache_write_price,
            "image_input": self.image_input_price,
            "audio_input": self.audio_input_price,
            "audio_output": self.audio_output_price,
            "embedding": self.embedding_price,
        }


class LiteLLMRegistry:
    """In-memory LiteLLM registry with reverse-lookup tables."""

    def __init__(self) -> None:
        self._entries: Dict[str, LiteLLMEntry] = {}  # canonical_id → canonical entry
        self._alias: Dict[str, str] = {}  # any lookup key → canonical_id
        self._aggregator_entries: Dict[str, LiteLLMEntry] = {}  # aggregator key → entry
        self._raw_count = 0
        self._loaded_at: Optional[datetime] = None

    @property
    def loaded_at(self) -> Optional[datetime]:
        return self._loaded_at

    @property
    def canonical_count(self) -> int:
        return len(self._entries)

    @property
    def aggregator_count(self) -> int:
        return len(self._aggregator_entries)

    @property
    def raw_count(self) -> int:
        return self._raw_count

    def get(self, canonical_id: str) -> Optional[LiteLLMEntry]:
        return self._entries.get(canonical_id)

    def resolve_alias(self, key: str) -> Optional[str]:
        """Return canonical_id for a lookup key, or None.

        Follows alias chains so a two-hop path like
        ``vertex_ai/mistralai/codestral-2`` → ``mistralai-codestral-2``
        → ``codestral-2`` lands on the final target rather than
        stopping at the intermediate one. A ``seen`` guard stops any
        accidental cycles.
        """
        if not key:
            return None

        seen: set[str] = set()
        current: Optional[str] = key
        last_resolved: Optional[str] = None
        while current is not None and current not in seen:
            seen.add(current)
            # Direct hit; normalize + strip fallbacks come last so an
            # exact alias always wins over a normalized one.
            target = self._alias.get(current)
            if target is None:
                normalized = slugify(current)
                if normalized in self._alias:
                    target = self._alias[normalized]
                else:
                    stripped = strip_version_suffix(normalized)
                    if stripped in self._alias:
                        target = self._alias[stripped]
            if target is None:
                break
            last_resolved = target
            if target == current:
                break
            current = target
        return last_resolved

    def register_alias(
        self,
        key: str,
        canonical_id: str,
        *,
        allow_dangling: bool = False,
    ) -> None:
        """Register an alias pointing at ``canonical_id``.

        Normally the target must be a real canonical entry; otherwise
        we'd pollute the alias table with broken links. The
        ``allow_dangling`` escape hatch is for
        :data:`canonical.KNOWN_AGGREGATOR_ALIASES` entries whose target
        is itself a synthesized canonical created by the merger's
        dangling-target clustering — e.g. ``codestral-2`` exists only
        as a synthetic, never as a LiteLLM canonical, so the Vertex
        ``mistralai-codestral-2`` alias has to be allowed to point at
        the dangling id for the merger to collapse both into one
        synthetic entity.
        """
        if not key:
            return
        if canonical_id not in self._entries and not allow_dangling:
            return
        self._alias[key] = canonical_id
        self._alias[slugify(key)] = canonical_id
        self._alias[strip_version_suffix(slugify(key))] = canonical_id

    def iter_canonical(self) -> Iterable[LiteLLMEntry]:
        return self._entries.values()

    def get_aggregator_entry(self, raw_key: str) -> Optional[LiteLLMEntry]:
        return self._aggregator_entries.get(raw_key)

    # ─── Loading ─────────────────────────────────────────────────

    async def load(self, force_network: bool = False) -> None:
        """Populate registry from cache then network (or both)."""
        raw = None
        if not force_network:
            raw = self._load_cache()

        if raw is None:
            raw = await self._fetch_network()
            if raw is not None:
                self._save_cache(raw)

        if raw is None:
            raise RuntimeError("LiteLLM registry unavailable: no cache and no network")

        self._parse(raw)

    def _load_cache(self) -> Optional[Dict[str, Any]]:
        if not CACHE_PATH.exists():
            return None
        try:
            with CACHE_PATH.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("LiteLLM cache unreadable: %s", exc)
            return None

    def _save_cache(self, raw: Dict[str, Any]) -> None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with CACHE_PATH.open("w", encoding="utf-8") as handle:
                json.dump(raw, handle)
        except OSError as exc:
            logger.warning("LiteLLM cache write failed: %s", exc)

    async def _fetch_network(self) -> Optional[Dict[str, Any]]:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(LITELLM_URL)
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            logger.warning("LiteLLM network fetch failed: %s", exc)
            return None

    def _parse(self, raw: Dict[str, Any]) -> None:
        self._entries.clear()
        self._alias.clear()
        self._aggregator_entries.clear()
        self._raw_count = 0

        # First split into two buckets so we can guarantee canonical
        # entries are registered before aggregators get a chance to
        # overwrite any alias keys.
        canonical_inputs: List[tuple[str, Dict[str, Any], str, str, str]] = []
        aggregator_inputs: List[tuple[str, Dict[str, Any], str, str, str]] = []

        for raw_key, payload in raw.items():
            if not isinstance(payload, dict) or raw_key == "sample_spec":
                continue
            self._raw_count += 1

            litellm_provider = str(payload.get("litellm_provider") or "").strip()
            raw_slug = slugify(raw_key)
            if not raw_slug:
                continue
            base_id = strip_version_suffix(raw_slug) or raw_slug

            is_canonical = (
                litellm_provider in CANONICAL_LITELLM_PROVIDERS
                or (not litellm_provider and "/" not in raw_key)
            )
            bucket = canonical_inputs if is_canonical else aggregator_inputs
            bucket.append((raw_key, payload, litellm_provider, raw_slug, base_id))

        # Pass 1: canonical entries.
        for raw_key, payload, litellm_provider, raw_slug, base_id in canonical_inputs:
            entry = self._build_entry(
                raw_key=raw_key,
                canonical_id=base_id,
                litellm_provider=litellm_provider,
                raw=payload,
            )
            entry.is_canonical = True

            # Keep the first entry we see for a given base_id. LiteLLM
            # sometimes has both "gpt-4o" and "gpt-4o-2024-11-20" as
            # canonical entries — we treat them as the same entity and
            # prefer the unversioned form since it arrives first in the
            # iteration order and matches user expectations.
            if base_id not in self._entries:
                self._entries[base_id] = entry

            # Every alias form points to the entity's base_id.
            self._alias[raw_key] = base_id
            self._alias[raw_slug] = base_id
            self._alias[base_id] = base_id

        # Pass 2: aggregator entries, best-effort wiring into canonicals.
        for raw_key, payload, litellm_provider, raw_slug, base_id in aggregator_inputs:
            entry = self._build_entry(
                raw_key=raw_key,
                canonical_id=base_id,
                litellm_provider=litellm_provider,
                raw=payload,
            )
            entry.is_canonical = False
            self._aggregator_entries[raw_key] = entry

            # Only add aliases that don't conflict with canonical entries.
            self._alias.setdefault(raw_key, base_id)
            self._alias.setdefault(raw_slug, base_id)
            self._alias.setdefault(base_id, base_id)

        # Third pass: for every aggregator, try to find the canonical
        # entity whose pricing it represents. Prefix-stripping handles
        # forms like "anthropic-claude-sonnet-4-5" → "claude-sonnet-4-5".
        maker_prefixes = (
            "anthropic-",
            "google-",
            "amazon-",
            "meta-",
            "meta-llama-",
            "mistral-",
            "mistralai-",
            "cohere-",
            "ai21-",
            "deepseek-",
            "x-ai-",
            "xai-",
            "stability-",
            "nvidia-",
            "microsoft-",
            "openai-",
            "qwen-",
            "alibaba-",
        )

        def find_canonical_for(slug: str) -> Optional[str]:
            if slug in self._entries:
                return slug
            stripped = strip_version_suffix(slug)
            if stripped in self._entries:
                return stripped
            for prefix in maker_prefixes:
                if slug.startswith(prefix):
                    tail = slug[len(prefix):]
                    if tail in self._entries:
                        return tail
                    tail_stripped = strip_version_suffix(tail)
                    if tail_stripped in self._entries:
                        return tail_stripped
            return None

        for raw_key, entry in self._aggregator_entries.items():
            target = find_canonical_for(entry.canonical_id)
            if target is None:
                continue
            # Aggregator raw_key and slug point at the canonical entity.
            # We do NOT touch the bare base_id alias if it was set by a
            # canonical entry in Pass 1; that would turn canonical lookups
            # into aggregator round-trips.
            self._alias[raw_key] = target
            self._alias[slugify(raw_key)] = target
            if entry.canonical_id not in self._entries:
                self._alias[entry.canonical_id] = target
                stripped_agg = strip_version_suffix(entry.canonical_id)
                if stripped_agg not in self._entries:
                    self._alias[stripped_agg] = target

        self._loaded_at = datetime.utcnow()
        logger.info(
            "LiteLLM registry loaded: %s canonical, %s aggregators, %s raw entries",
            self.canonical_count,
            self.aggregator_count,
            self.raw_count,
        )

    def _build_entry(
        self,
        raw_key: str,
        canonical_id: str,
        litellm_provider: str,
        raw: Dict[str, Any],
    ) -> LiteLLMEntry:
        name = str(raw.get("model_name") or raw_key.split("/")[-1])
        family, maker = detect_family_maker(canonical_id, name)
        capabilities, input_mods, output_mods = derive_capability_modality(raw)
        mode = str(raw.get("mode") or "chat")

        if _is_zero_stub_row(raw):
            # LiteLLM placeholder: both input and output are literal 0
            # with no other pricing signal. Null everything so downstream
            # merger treats this entry as "no price known" instead of a
            # "real $0" that pollutes alternatives / drift / comparisons.
            input_price = output_price = None
            cache_read = cache_write = None
            image_in = audio_in = audio_out = None
            embedding_price = None
            batch_in = batch_out = None
        else:
            batch = raw.get("batch_pricing") or {}
            batch_in = (
                convert_pricing_field(raw.get("input_cost_per_token_batches"))
                or convert_pricing_field(batch.get("input"))
            )
            batch_out = (
                convert_pricing_field(raw.get("output_cost_per_token_batches"))
                or convert_pricing_field(batch.get("output"))
            )
            input_price = convert_pricing_field(raw.get("input_cost_per_token"))
            output_price = convert_pricing_field(raw.get("output_cost_per_token"))
            cache_read = convert_pricing_field(raw.get("cache_read_input_token_cost"))
            cache_write = convert_pricing_field(raw.get("cache_creation_input_token_cost"))
            image_in = convert_pricing_field(raw.get("input_cost_per_image"))
            audio_in = convert_pricing_field(raw.get("input_cost_per_audio_token"))
            audio_out = convert_pricing_field(raw.get("output_cost_per_audio_token"))
            embedding_price = (
                convert_pricing_field(raw.get("input_cost_per_token"))
                if mode == "embedding"
                else None
            )

        return LiteLLMEntry(
            raw_key=raw_key,
            canonical_id=canonical_id,
            slug=canonical_id,
            name=_pretty_name(raw_key, name),
            family=family,
            maker=maker,
            litellm_provider=litellm_provider,
            is_canonical=False,
            input_price=input_price,
            output_price=output_price,
            cache_read_price=cache_read,
            cache_write_price=cache_write,
            image_input_price=image_in,
            audio_input_price=audio_in,
            audio_output_price=audio_out,
            embedding_price=embedding_price,
            batch_input_price=batch_in,
            batch_output_price=batch_out,
            context_length=_safe_int(raw.get("max_input_tokens") or raw.get("max_tokens")),
            max_output_tokens=_safe_int(raw.get("max_output_tokens")),
            capabilities=capabilities,
            input_modalities=input_mods,
            output_modalities=output_mods,
            mode=mode,
            raw=raw,
        )


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pretty_name(raw_key: str, fallback: str) -> str:
    """Make a reasonably human-looking model name from a raw_key."""
    base = raw_key.split("/")[-1]
    # Keep versioned names as-is, just title-case segments
    if base == fallback:
        pretty = base
    else:
        pretty = fallback
    return pretty


# Module-level singleton so multiple services share the same parsed tree.
_registry = LiteLLMRegistry()


async def get_registry(force_network: bool = False) -> LiteLLMRegistry:
    if _registry.canonical_count == 0 or force_network:
        await _registry.load(force_network=force_network)
    return _registry


def reset_registry_for_tests() -> None:
    global _registry
    _registry = LiteLLMRegistry()
