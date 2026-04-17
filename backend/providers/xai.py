"""xAI (Grok) pricing provider — mirrored via the LiteLLM registry.

xAI does not publish a scraped-friendly pricing page and our old
static data (``data/fallback/xai.json``, last verified 2026-01)
rotted — it missed Grok 4, Grok 4 Fast, Grok 4.1, Grok code-fast-1.
LiteLLM tracks xAI's own model_prices file and carries ~35 Grok
entries today, updated within hours of a release.
"""

from ._litellm_first_party import LiteLLMFirstPartyProvider
from .registry import ProviderRegistry


class XAIProvider(LiteLLMFirstPartyProvider):
    name = "xai"
    display_name = "xAI"
    litellm_tags = frozenset({"xai"})
    is_open_source = False


ProviderRegistry.register(XAIProvider())
