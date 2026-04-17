"""Google Vertex AI pricing provider — mirrored via the LiteLLM registry.

Google Vertex AI is GCP's managed platform that hosts Gemini plus
third-party models (Claude, Mistral, Llama, DeepSeek, Qwen, etc.).
LiteLLM splits Vertex's catalog across many ``vertex_ai-*`` tags;
we consolidate all of them under one app slug.

**Scope: non-Google makers only.**

Google maintains price parity between AI Studio (``google_gemini``
provider) and Vertex AI, so emitting Gemini rows from both would just
add "$1.25/$10 × 2" duplicates. We skip the Google maker here — Gemini
appears under ``google_gemini`` only. Vertex's real value for the user
is "Claude / Mistral / Llama / DeepSeek / Qwen on Vertex" as a
distribution alternative to Bedrock / Azure / OpenRouter.

Most vertex_ai-* entries land in LiteLLM's aggregator bucket (not
canonical), so we set ``include_aggregator_bucket = True`` to walk
both sides.
"""

from services.litellm_registry import LiteLLMEntry

from ._litellm_first_party import LiteLLMFirstPartyProvider
from .registry import ProviderRegistry


_VERTEX_TAGS = frozenset({
    "vertex_ai",
    "vertex_ai-language-models",
    "vertex_ai-anthropic_models",
    "vertex_ai-mistral_models",
    "vertex_ai-llama_models",
    "vertex_ai-deepseek_models",
    "vertex_ai-qwen_models",
    "vertex_ai-ai21_models",
    "vertex_ai-zai_models",
    "vertex_ai-moonshot_models",
    "vertex_ai-minimax_models",
    "vertex_ai-openai_models",
    "vertex_ai-image-models",
    "vertex_ai-video-models",
    "vertex_ai-embedding-models",
    "vertex_ai-text-models",
    "vertex_ai-chat-models",
    "vertex_ai-code-chat-models",
})


class GoogleVertexAIProvider(LiteLLMFirstPartyProvider):
    name = "google_vertex_ai"
    display_name = "Google Vertex AI"
    litellm_tags = _VERTEX_TAGS
    is_open_source = None
    include_aggregator_bucket = True

    def maker_filter(self, entry: LiteLLMEntry) -> bool:
        # Gemini is already covered by google_gemini at identical
        # prices — see the module docstring for the rationale.
        return entry.maker != "Google"


ProviderRegistry.register(GoogleVertexAIProvider())
