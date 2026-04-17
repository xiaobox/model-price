"""Z.AI (GLM) pricing provider — mirrored via the LiteLLM registry.

Z.AI (formerly Zhipu AI) runs the official GLM family API at z.ai /
bigmodel.cn. LiteLLM tags these canonical entries as ``zai``.
"""

from ._litellm_first_party import LiteLLMFirstPartyProvider
from .registry import ProviderRegistry


class ZAIProvider(LiteLLMFirstPartyProvider):
    name = "zai"
    display_name = "Z.AI"
    litellm_tags = frozenset({"zai"})
    # GLM-4.6 weights are open.
    is_open_source = True


ProviderRegistry.register(ZAIProvider())
