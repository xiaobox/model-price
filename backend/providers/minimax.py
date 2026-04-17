"""MiniMax pricing provider — mirrored via the LiteLLM registry.

MiniMax runs first-party API at minimaxi.com / minimax.io for the
abab / M1 chat family plus speech synthesis. LiteLLM tag: ``minimax``.
"""

from ._litellm_first_party import LiteLLMFirstPartyProvider
from .registry import ProviderRegistry


class MiniMaxProvider(LiteLLMFirstPartyProvider):
    name = "minimax"
    display_name = "MiniMax"
    litellm_tags = frozenset({"minimax"})
    # MiniMax-M1 and MiniMax-Text-01 weights are open.
    is_open_source = True


ProviderRegistry.register(MiniMaxProvider())
