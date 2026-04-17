"""OpenAI pricing provider — mirrored via the LiteLLM registry.

The old Playwright scraper against ``platform.openai.com/docs/pricing``
degraded badly after OpenAI's 2025 pricing page rewrite: it ended up
emitting only ~16 offerings and most of those were edge variants with
slugify bugs (``gpt-5.4`` landing as ``gpt-54``). Mainstream models —
GPT-5 / GPT-4o / o1 / o3 — had no ``openai`` offering at all, only
aggregator rows from Azure and OpenRouter.

LiteLLM tracks OpenAI's own model_prices file and carries every GPT /
o-series / codex / DALL-E / Sora / embedding entry. Covers two
LiteLLM tags: ``openai`` (chat / responses) and
``text-completion-openai`` (legacy ``gpt-3.5-turbo-instruct`` etc.).
"""

from ._litellm_first_party import LiteLLMFirstPartyProvider
from .registry import ProviderRegistry


class OpenAIProvider(LiteLLMFirstPartyProvider):
    name = "openai"
    display_name = "OpenAI"
    litellm_tags = frozenset({"openai", "text-completion-openai"})
    is_open_source = False


ProviderRegistry.register(OpenAIProvider())
