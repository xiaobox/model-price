"""Volcengine pricing provider — ByteDance's cloud platform.

Volcengine is ByteDance's public cloud — like AWS / Azure / GCP —
and hosts:

- ByteDance's own Doubao models (doubao-seed / doubao-embedding)
- Third-party foundation models redistributed on ByteDance's cloud:
  DeepSeek V3.x (as ``deepseek-v3-2-251201``), GLM 4.x (as
  ``glm-4-7-251222``), Kimi K2 (as ``kimi-k2-thinking-251104``).

The slug here therefore names the **distribution channel**, not the
maker. A "Volcengine" row on the DeepSeek detail page means
"DeepSeek is also available via ByteDance's cloud with these prices",
same pattern as AWS Bedrock / Azure AI / OpenRouter rows.

Pricing comes via the LiteLLM ``volcengine`` tag. Note that many
Volcengine entries publish RMB-only pricing upstream, which LiteLLM
surfaces as ``input=null / output=null``; those rows get pruned by
``_is_stub_offering_set`` in the merger.
"""

from ._litellm_first_party import LiteLLMFirstPartyProvider
from .registry import ProviderRegistry


class VolcengineProvider(LiteLLMFirstPartyProvider):
    name = "volcengine"
    display_name = "Volcengine"
    litellm_tags = frozenset({"volcengine"})
    is_open_source = None  # Mixed catalog — some open, some not.


ProviderRegistry.register(VolcengineProvider())
