"""Meta Llama pricing provider — mirrored via the LiteLLM registry.

Meta's official Llama API (``llama.developer.meta.com``) covers the
latest Llama 3.x / 4 Instruct models. LiteLLM tags these canonical
entries as ``meta_llama`` (and occasionally ``meta``).
"""

from ._litellm_first_party import LiteLLMFirstPartyProvider
from .registry import ProviderRegistry


class MetaLlamaProvider(LiteLLMFirstPartyProvider):
    name = "meta_llama"
    display_name = "Meta Llama API"
    litellm_tags = frozenset({"meta_llama", "meta"})
    is_open_source = True  # Llama weights are open.


ProviderRegistry.register(MetaLlamaProvider())
