"""Moonshot AI (Kimi) pricing provider — mirrored via the LiteLLM registry.

Moonshot runs first-party API at platform.moonshot.cn with official
pricing for Kimi K1 / K2 / K2 Thinking / Kimi Latest variants.
LiteLLM tracks the ``moonshot`` tag.
"""

from ._litellm_first_party import LiteLLMFirstPartyProvider
from .registry import ProviderRegistry


class MoonshotProvider(LiteLLMFirstPartyProvider):
    name = "moonshot"
    display_name = "Moonshot AI"
    litellm_tags = frozenset({"moonshot"})
    # Kimi K2 weights are openly released.
    is_open_source = True


ProviderRegistry.register(MoonshotProvider())
