"""Tests for the entity display-name polisher.

Regression coverage for the list-view duplicate-looking rows where the
same GPT-4o family showed up as "GPT 4O Realtime Preview" (canonical
side, via ``_pretty_model_name`` uppercasing the ``4o``) next to
"GPT-4o Realtime" (synthetic side, passing the scraper name through
unchanged). One styling rule on the way out makes both paths agree.
"""

from services.offering_merger import _polish_display_name, _style_name_token


class TestStyleNameToken:
    def test_brand_suffix_4o_stays_lowercase(self):
        """The ``o`` in GPT-4o stands for omni, not an acronym letter.
        Uppercasing it (``4O``) is wrong."""
        assert _style_name_token("4o") == "4o"
        assert _style_name_token("4O") == "4o"
        assert _style_name_token("3n") == "3n"  # Gemma 3n

    def test_acronyms_uppercase(self):
        assert _style_name_token("gpt") == "GPT"
        assert _style_name_token("ai") == "AI"
        assert _style_name_token("tts") == "TTS"
        assert _style_name_token("r1") == "R1"
        assert _style_name_token("ocr") == "OCR"
        assert _style_name_token("oss") == "OSS"
        assert _style_name_token("vl") == "VL"
        assert _style_name_token("ft") == "FT"
        assert _style_name_token("hd") == "HD"

    def test_short_non_acronyms_title_case(self):
        """Regression: ``max`` / ``pro`` / ``air`` / ``exp`` / ``her`` /
        ``non`` used to default to UPPER (via a "<=3 letters → UPPER"
        rule), which turned real English words into fake acronyms
        ("Qwen MAX" / "GLM 4.5 AIR" / "Claude Opus 4.6 FAST"). They
        are just adjectives — Title Case them."""
        assert _style_name_token("max") == "Max"
        assert _style_name_token("pro") == "Pro"
        assert _style_name_token("air") == "Air"
        assert _style_name_token("exp") == "Exp"
        assert _style_name_token("her") == "Her"
        assert _style_name_token("non") == "Non"
        assert _style_name_token("ada") == "Ada"

    def test_brand_case_compressed_words(self):
        """Compressed-word brands keep their mixed case so they don't
        read like ``Chatgpt`` / ``Deepseek`` / ``Minimax``."""
        assert _style_name_token("chatgpt") == "ChatGPT"
        assert _style_name_token("deepseek") == "DeepSeek"
        assert _style_name_token("minimax") == "MiniMax"
        assert _style_name_token("openrouter") == "OpenRouter"
        assert _style_name_token("bytedance") == "ByteDance"
        assert _style_name_token("mistralai") == "MistralAI"
        assert _style_name_token("moonshotai") == "MoonshotAI"
        assert _style_name_token("openai") == "OpenAI"
        assert _style_name_token("qwq") == "QwQ"

    def test_parameter_sizes_digit_plus_upper(self):
        """Parameter counts like 70B / 480M / 8K follow the convention
        of digit followed by an uppercase unit letter."""
        assert _style_name_token("70b") == "70B"
        assert _style_name_token("480m") == "480M"
        assert _style_name_token("8k") == "8K"
        assert _style_name_token("1.5t") == "1.5T"

    def test_active_parameter_tokens_uppercase_both_letters(self):
        """MoE-style ``A22b`` / ``A3b`` / ``R7b`` name the active
        parameter count. Both the leading letter and the trailing
        unit letter render uppercase for consistency with plain
        ``70B`` / ``235B`` tokens."""
        assert _style_name_token("a22b") == "A22B"
        assert _style_name_token("A22b") == "A22B"
        assert _style_name_token("a3b") == "A3B"
        assert _style_name_token("r7b") == "R7B"
        assert _style_name_token("A47b") == "A47B"

    def test_long_words_title_case(self):
        assert _style_name_token("claude") == "Claude"
        assert _style_name_token("sonnet") == "Sonnet"
        assert _style_name_token("realtime") == "Realtime"

    def test_mixed_alphanumeric_capitalised(self):
        """Tokens like ``qwen3`` / ``llama2`` capitalise only the first
        letter, keeping the trailing digit run intact."""
        assert _style_name_token("qwen3") == "Qwen3"
        assert _style_name_token("glm4") == "Glm4"
        assert _style_name_token("llama2") == "Llama2"

    def test_pure_digits_preserved(self):
        assert _style_name_token("3") == "3"
        assert _style_name_token("12") == "12"


class TestPolishDisplayName:
    def test_rescues_uppercased_omni(self):
        """The concrete user complaint: ``GPT 4O Realtime Preview``
        and ``GPT-4o Realtime`` both collapse to the same styling so
        the list view no longer reads them as duplicates."""
        assert (
            _polish_display_name("GPT 4O Realtime Preview")
            == "GPT 4o Realtime Preview"
        )
        assert (
            _polish_display_name("GPT-4o Realtime") == "GPT 4o Realtime"
        )

    def test_recovers_version_dot(self):
        """Canonical-side slug-derived names have digit runs that must
        glue back together with dots: ``4 5`` → ``4.5``."""
        assert (
            _polish_display_name("claude-sonnet-4-5") == "Claude Sonnet 4.5"
        )
        assert (
            _polish_display_name("claude-opus-4-7") == "Claude Opus 4.7"
        )
        assert (
            _polish_display_name("llama-3-3-70b-instruct")
            == "Llama 3.3 70B Instruct"
        )

    def test_gpt_omni_from_slug(self):
        assert _polish_display_name("gpt-4o") == "GPT 4o"
        assert _polish_display_name("gpt-4o-mini") == "GPT 4o Mini"
        assert (
            _polish_display_name("gpt-4o-realtime-preview")
            == "GPT 4o Realtime Preview"
        )

    def test_hyphenated_input_collapsed_to_spaces(self):
        """Synthetic-path names come in hyphenated (`GPT-4o Realtime`).
        Polish produces a consistent space-separated output."""
        assert (
            _polish_display_name("GPT-4o Realtime") == "GPT 4o Realtime"
        )
        assert _polish_display_name("Qwen3-Coder") == "Qwen3 Coder"

    def test_maker_prefix_preserved(self):
        """We don't strip maker prefixes here — that's the canonical
        resolver's job. Anything that reaches polish is assumed to
        already represent the model's own name."""
        # ``deepseek`` is a compressed brand name so the styling pass
        # renders it ``DeepSeek``, not ``Deepseek``.
        assert _polish_display_name("deepseek-v3") == "DeepSeek V3"

    def test_empty_input(self):
        assert _polish_display_name("") == ""
        assert _polish_display_name("   ") == ""

    def test_parentheses_unwrapped(self):
        """OpenRouter tails ``(free)`` / ``(fast)`` / ``(thinking)`` /
        ``(extended)`` are product-variant tags, not annotation, and
        should flow into the name as normal words."""
        assert (
            _polish_display_name("Gemma 3n 2B (free)") == "Gemma 3n 2B Free"
        )
        assert (
            _polish_display_name("Claude Opus 4.6 (fast)")
            == "Claude Opus 4.6 Fast"
        )
        assert (
            _polish_display_name("Qwen Plus 0728 (thinking)")
            == "Qwen Plus 0728 Thinking"
        )

    def test_active_param_token_in_full_name(self):
        """Regression: Qwen3's MoE entities rendered ``A22b``; the
        styling rule now produces ``A22B`` consistently."""
        assert (
            _polish_display_name("Qwen3 235B A22b Instruct")
            == "Qwen3 235B A22B Instruct"
        )

    def test_ft_prefix_rewritten_to_fine_tuned_suffix(self):
        """Regression: OpenAI fine-tuning endpoints came in as
        ``FT GPT 4.1 Mini`` which reads like another acronym. Rewrite
        so users immediately see it's a fine-tuning variant."""
        assert (
            _polish_display_name("ft-gpt-4-1-mini")
            == "GPT 4.1 Mini (Fine-tuned)"
        )
        assert (
            _polish_display_name("ft-gpt-4o-mini")
            == "GPT 4o Mini (Fine-tuned)"
        )

    def test_brand_names_in_full_name(self):
        """End-to-end: slugified inputs containing compressed brand
        words render with the proper mixed case."""
        assert _polish_display_name("deepseek-v3") == "DeepSeek V3"
        assert _polish_display_name("chatgpt-4o") == "ChatGPT 4o"
        assert (
            _polish_display_name("minimax-m2-5") == "MiniMax M2.5"
        )

    def test_short_adjective_not_uppercased(self):
        """Regression: ``Qwen Max`` / ``GLM 4.5 Air`` / ``Claude Opus
        4.6 Fast`` no longer render ``MAX`` / ``AIR`` / ``FAST``."""
        assert _polish_display_name("qwen-max") == "Qwen Max"
        assert _polish_display_name("glm-4-5-air") == "GLM 4.5 Air"
        assert (
            _polish_display_name("claude-opus-4-6-fast")
            == "Claude Opus 4.6 Fast"
        )
