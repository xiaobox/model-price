"""Alibaba Qwen pricing provider (first-party via Model Studio / Dashscope).

Alibaba Cloud's official Qwen API is branded "Model Studio" /
"Dashscope". LiteLLM tags those canonical entries as ``dashscope``.
"""

from ._litellm_first_party import LiteLLMFirstPartyProvider
from .registry import ProviderRegistry


class AlibabaQwenProvider(LiteLLMFirstPartyProvider):
    name = "alibaba_qwen"
    display_name = "Alibaba Qwen"
    litellm_tags = frozenset({"dashscope"})
    # Many Qwen weights are openly released (Qwen2.5, Qwen3), but the
    # flagship Max / Plus variants on Dashscope are commercial.
    is_open_source = None


ProviderRegistry.register(AlibabaQwenProvider())
