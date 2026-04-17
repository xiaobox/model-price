"""Tests for slug normalization and version suffix stripping.

These functions are the backbone of canonical resolution — a change to
their behaviour cascades to entity identity, so they get thorough
coverage including regression cases captured during Phase 1.
"""

from services.litellm_registry import slugify, strip_version_suffix


class TestSlugify:
    def test_lowercases(self):
        assert slugify("Claude-Sonnet-4-5") == "claude-sonnet-4-5"

    def test_drops_leading_provider_prefix(self):
        assert slugify("bedrock/anthropic.claude-sonnet-4-5") == "anthropic-claude-sonnet-4-5"
        assert slugify("openrouter/google/gemini-2.5-pro") == "google-gemini-2-5-pro"

    def test_flattens_slashes(self):
        assert slugify("meta-llama/llama-4-maverick") == "llama-4-maverick"

    def test_flattens_dots(self):
        assert slugify("claude-sonnet-4.5") == "claude-sonnet-4-5"
        assert slugify("gpt-3.5-turbo") == "gpt-3-5-turbo"

    def test_collapses_special_chars(self):
        assert slugify("claude-sonnet-4-5:beta") == "claude-sonnet-4-5-beta"
        assert slugify("claude sonnet 4-5") == "claude-sonnet-4-5"

    def test_strips_leading_trailing_dashes(self):
        assert slugify("--claude--") == "claude"

    def test_empty_and_whitespace(self):
        assert slugify("") == ""
        assert slugify("   ") == ""

    def test_preserves_multi_slash_prefix_style(self):
        # "openrouter/anthropic/claude-3.5-sonnet" → drops only first segment
        assert slugify("openrouter/anthropic/claude-3.5-sonnet") == "anthropic-claude-3-5-sonnet"


class TestStripVersionSuffix:
    def test_bedrock_v1_colon_0(self):
        assert strip_version_suffix("claude-sonnet-4-5-v1-0") == "claude-sonnet-4-5"
        assert strip_version_suffix("claude-sonnet-4-5-v2-0") == "claude-sonnet-4-5"

    def test_eight_digit_date(self):
        assert strip_version_suffix("claude-sonnet-4-5-20250929") == "claude-sonnet-4-5"
        assert strip_version_suffix("gpt-4o-20241120") == "gpt-4o"

    def test_yyyy_mm_dd(self):
        assert strip_version_suffix("gpt-4o-2024-11-20") == "gpt-4o"

    def test_six_digit_yymmdd_compact_date(self):
        """Volcengine / some Chinese vendors stamp releases with a
        YYMMDD compact date (``-251201`` = 2025-12-01). Without
        stripping we end up with orphan canonicals like
        deepseek-v3-2-251201 living next to deepseek-v3-2."""
        assert strip_version_suffix("deepseek-v3-2-251201") == "deepseek-v3-2"
        assert strip_version_suffix("glm-4-7-251222") == "glm-4-7"
        assert strip_version_suffix("kimi-k2-thinking-251104") == "kimi-k2-thinking"
        assert strip_version_suffix("doubao-seed-2-0-pro-260215") == "doubao-seed-2-0-pro"

    def test_default_suffix_stripped(self):
        """Vertex AI uses ``@default`` which slugifies to ``-default``.
        It means "pick this model's default variant" — same identity
        as the bare canonical. Without stripping,
        ``claude-opus-4-7@default`` became its own orphan entity."""
        assert strip_version_suffix("claude-opus-4-7-default") == "claude-opus-4-7"
        assert (
            strip_version_suffix("claude-sonnet-4-6-default") == "claude-sonnet-4-6"
        )

    def test_default_suffix_chains_with_date(self):
        """If the model_id has both ``@default`` and a date suffix,
        strip both in one pass."""
        # Contrived but realistic once Vertex stamps releases like
        # claude-sonnet-4-5@default-20251001.
        assert (
            strip_version_suffix("claude-sonnet-4-5-default-20251001")
            == "claude-sonnet-4-5"
        )

    def test_six_digit_non_date_not_stripped(self):
        """Only plausible 21st-century YYMMDD dates strip. Random
        six-digit tails that happen not to fit the calendar stay
        attached, so model names with legit numeric suffixes (e.g.
        a hypothetical ``foo-193456``) are safe."""
        # month 34 → not a date
        assert strip_version_suffix("foo-123456") == "foo-123456"
        # day 99 → not a date
        assert strip_version_suffix("foo-250199") == "foo-250199"
        # year leading 1 → not 21st century
        assert strip_version_suffix("foo-190101") == "foo-190101"
        # 5 digits (YMMDD) → too short to be a date, left alone
        assert strip_version_suffix("foo-51201") == "foo-51201"

    def test_preview_exp_instruct_carry_identity(self):
        """-preview / -exp / -experimental / -instruct carry real
        product identity and must survive normalization.

        (Historically ``-latest`` lived in this list too; it has since
        been moved into ``_VARIANT_TAGS`` because the ``-latest``
        floating alias was creating duplicate entities alongside every
        product for negligible gain. See ``test_latest_stripped``.)"""
        assert strip_version_suffix("gemini-3-pro-preview") == "gemini-3-pro-preview"
        assert strip_version_suffix("claude-3-5-sonnet-experimental") == "claude-3-5-sonnet-experimental"
        assert strip_version_suffix("deepseek-v3-2-exp") == "deepseek-v3-2-exp"
        assert strip_version_suffix("gpt-3-5-turbo-instruct") == "gpt-3-5-turbo-instruct"

    def test_latest_stripped(self):
        """``-latest`` is a floating alias that every Mistral / Grok /
        Qwen / Devstral / Codex product used to duplicate into a
        second entity with identical pricing. Strip it."""
        assert strip_version_suffix("mistral-large-latest") == "mistral-large"
        assert strip_version_suffix("grok-4-latest") == "grok-4"
        assert strip_version_suffix("qwen-plus-latest") == "qwen-plus"
        assert strip_version_suffix("codex-mini-latest") == "codex-mini"
        assert strip_version_suffix("devstral-small-latest") == "devstral-small"

    def test_gemma_it_suffix_stripped(self):
        """Gemma's ``-it`` (instruction-tuned) is the default product
        variant. The bare ``gemma-3-12b`` and ``gemma-3-12b-it`` refer
        to the same model."""
        assert strip_version_suffix("gemma-3-12b-it") == "gemma-3-12b"
        assert strip_version_suffix("gemma-3-27b-it") == "gemma-3-27b"
        assert strip_version_suffix("gemma-3-4b-it") == "gemma-3-4b"

    def test_resolution_prefix_stripped(self):
        """LiteLLM prices OpenAI image models per-resolution:
        ``1024-x-1024-gpt-image-1.5`` / ``1024-x-1536-gpt-image-1.5``
        / ``1792-x-1024-dall-e-3``. These are the same logical model
        at different sizes, not distinct entities."""
        assert (
            strip_version_suffix("1024-x-1024-gpt-image-1-5")
            == "gpt-image-1-5"
        )
        assert (
            strip_version_suffix("1024-x-1536-gpt-image-1-mini")
            == "gpt-image-1-mini"
        )
        assert (
            strip_version_suffix("1792-x-1024-dall-e-3") == "dall-e-3"
        )
        # Guard: ``-x-`` elsewhere in the slug must not be touched.
        assert (
            strip_version_suffix("claude-x-ai-not-a-res") == "claude-x-ai-not-a-res"
        )

    def test_maas_suffix_stripped(self):
        """Vertex AI Model Garden / Alibaba Dashscope mark hosted-
        service SKUs with ``@maas`` ("Model as a Service"). After
        slugify it's ``-maas`` and carries no identity — strip so
        ``deepseek-v3-2-maas`` collapses into ``deepseek-v3-2``."""
        assert strip_version_suffix("deepseek-v3-2-maas") == "deepseek-v3-2"
        assert (
            strip_version_suffix("kimi-k2-thinking-maas")
            == "kimi-k2-thinking"
        )
        # chains with 8-digit date and `-v1-0` Bedrock suffix:
        assert (
            strip_version_suffix("deepseek-v3-2-20260215-maas")
            == "deepseek-v3-2"
        )

    def test_four_digit_compact_date_stripped(self):
        """``-MM-DD$`` at end of slug (``gemini-2-5-pro-preview-05-06``
        / ``pixtral-large-25-02``) strips only when both halves are
        zero-padded two digits so version numbers like ``-4-5`` stay."""
        assert (
            strip_version_suffix("gemini-2-5-pro-preview-05-06")
            == "gemini-2-5-pro-preview"
        )
        assert (
            strip_version_suffix("qwen3-5-plus-02-15") == "qwen3-5-plus"
        )
        # Guard: single-digit month/day looks like a version number,
        # must not strip.
        assert strip_version_suffix("claude-sonnet-4-5") == "claude-sonnet-4-5"
        assert strip_version_suffix("gpt-4-1") == "gpt-4-1"

    def test_free_suffix_stripped(self):
        """OpenRouter's ``-free`` variants are the same model with a
        $0 rate-limited tier. Merge them so users see one entity with
        both the paid and free offerings."""
        assert strip_version_suffix("gemma-3-12b-it-free") == "gemma-3-12b"
        assert strip_version_suffix("gpt-oss-120b-free") == "gpt-oss-120b"
        assert strip_version_suffix("llama-guard-4-12b-free") == "llama-guard-4-12b"
        # Chain with -instruct (which is NOT stripped) and -free:
        assert (
            strip_version_suffix("llama-3-3-70b-instruct-free")
            == "llama-3-3-70b-instruct"
        )

    def test_quantization_tags_stripped(self):
        """Pure storage/hardware variants ARE the same logical model
        and still strip — fp8/fp16/bf16/int4/int8/rlhf and the MoE
        expert-count tags."""
        assert strip_version_suffix("llama-3-70b-fp8") == "llama-3-70b"
        assert strip_version_suffix("llama-3-70b-bf16") == "llama-3-70b"
        assert strip_version_suffix("llama-3-70b-int4") == "llama-3-70b"
        # MoE expert counts
        assert strip_version_suffix("llama-4-maverick-17b-128e") == "llama-4-maverick-17b"

    def test_chained_quantization_and_version_collapse(self):
        """Storage variant + Bedrock version suffix should both strip."""
        assert (
            strip_version_suffix("llama-4-maverick-17b-128e-v1-0")
            == "llama-4-maverick-17b"
        )

    def test_instruct_no_longer_stripped_in_chain(self):
        """The -instruct tag carries identity (gpt-3.5-turbo-instruct
        is a separate product), so a chain like 128e-instruct-v1-0 only
        strips the -v1-0, leaving -instruct intact for downstream
        canonical lookup."""
        assert (
            strip_version_suffix("llama-4-maverick-17b-128e-instruct-v1-0")
            == "llama-4-maverick-17b-128e-instruct"
        )

    def test_preserves_bare_version_in_name(self):
        # "deepseek-v3" is a product name; the -v3 must NOT be treated as a
        # Bedrock version suffix. This is the regression we fixed in Phase 1.
        assert strip_version_suffix("deepseek-v3") == "deepseek-v3"
        assert strip_version_suffix("moonshot-v1-8k") == "moonshot-v1-8k"
        assert strip_version_suffix("deepseek-v3-2") == "deepseek-v3-2"

    def test_preserves_chat_and_base(self):
        # -chat / -base are legitimate product names and must not be stripped.
        assert strip_version_suffix("deepseek-chat") == "deepseek-chat"
        assert strip_version_suffix("qwen-base") == "qwen-base"

    def test_idempotent(self):
        already_clean = "claude-sonnet-4-5"
        assert strip_version_suffix(already_clean) == already_clean
