"""Anthropic pricing provider — mirrored via the LiteLLM registry.

Anthropic does not publish a scraped-friendly pricing page or a public
pricing API, so Claude data reaches us through a two-hop chain:

    Anthropic official pricing docs
        ↓  community contributors update LiteLLM
    LiteLLM `model_prices_and_context_window.json`
        ↓  we HTTP GET once per refresh cycle
    this provider

Offerings carry ``source="via_litellm"`` after merger so the UI labels
the pricing row as a mirror rather than a direct first-party fetch.
"""

from ._litellm_first_party import LiteLLMFirstPartyProvider
from .registry import ProviderRegistry


class AnthropicProvider(LiteLLMFirstPartyProvider):
    name = "anthropic"
    display_name = "Anthropic"
    litellm_tags = frozenset({"anthropic"})
    is_open_source = False


ProviderRegistry.register(AnthropicProvider())
