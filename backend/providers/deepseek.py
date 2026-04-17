"""DeepSeek pricing provider — mirrored via the LiteLLM registry.

DeepSeek publishes first-party pricing on its own docs page without a
scrape-friendly API. Data reaches us via the LiteLLM community
registry two-hop chain (see ``_litellm_first_party.py``).
"""

from ._litellm_first_party import LiteLLMFirstPartyProvider
from .registry import ProviderRegistry


class DeepSeekProvider(LiteLLMFirstPartyProvider):
    name = "deepseek"
    display_name = "DeepSeek"
    litellm_tags = frozenset({"deepseek"})
    is_open_source = True


ProviderRegistry.register(DeepSeekProvider())
