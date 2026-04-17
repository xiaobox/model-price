"""Sber GigaChat pricing provider — mirrored via the LiteLLM registry."""

from ._litellm_first_party import LiteLLMFirstPartyProvider
from .registry import ProviderRegistry


class GigaChatProvider(LiteLLMFirstPartyProvider):
    name = "gigachat"
    display_name = "Sber GigaChat"
    litellm_tags = frozenset({"gigachat"})
    is_open_source = False


ProviderRegistry.register(GigaChatProvider())
