"""Cohere pricing provider — mirrored via the LiteLLM registry.

Cohere ships Command (chat), Embed, and Rerank models through its own
API plus Bedrock / Azure AI aggregations. LiteLLM covers two tags:
``cohere`` (the umbrella) and ``cohere_chat`` (Command-family chat).
"""

from ._litellm_first_party import LiteLLMFirstPartyProvider
from .registry import ProviderRegistry


class CohereProvider(LiteLLMFirstPartyProvider):
    name = "cohere"
    display_name = "Cohere"
    litellm_tags = frozenset({"cohere", "cohere_chat"})
    is_open_source = None


ProviderRegistry.register(CohereProvider())
