"""AI21 (Jamba / Jurassic) pricing provider — mirrored via LiteLLM."""

from ._litellm_first_party import LiteLLMFirstPartyProvider
from .registry import ProviderRegistry


class AI21Provider(LiteLLMFirstPartyProvider):
    name = "ai21"
    display_name = "AI21"
    litellm_tags = frozenset({"ai21"})
    is_open_source = False


ProviderRegistry.register(AI21Provider())
