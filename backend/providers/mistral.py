"""Mistral AI pricing provider — mirrored via the LiteLLM registry.

Mistral publishes first-party pricing on its own docs. LiteLLM covers
both the main ``mistral`` tag (Mistral / Ministral / Mixtral /
Pixtral / Magistral / Devstral) and the ``codestral`` subtag for
dedicated code models.
"""

from ._litellm_first_party import LiteLLMFirstPartyProvider
from .registry import ProviderRegistry


class MistralProvider(LiteLLMFirstPartyProvider):
    name = "mistral"
    display_name = "Mistral AI"
    litellm_tags = frozenset({"mistral", "codestral"})
    # Weights for older Mistral models are open, but commercial
    # flagships (Medium / Large / Pixtral Large) are not. Leave None
    # so per-entity metadata wins.
    is_open_source = None


ProviderRegistry.register(MistralProvider())
