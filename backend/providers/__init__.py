from .base import BaseProvider
from .registry import ProviderRegistry

# Import providers to trigger registration
from . import ai21
from . import alibaba_qwen
from . import anthropic
from . import aws_bedrock
# azure_openai.py now registers as slug "azure_ai" — see the
# module docstring for the Azure rebrand history.
from . import azure_openai
from . import volcengine
from . import cohere
from . import deepseek
from . import gigachat
from . import google_gemini
from . import google_vertex_ai
from . import meta_llama
from . import minimax
from . import mistral
from . import moonshot
from . import openai
from . import openrouter
from . import xai
from . import zai

__all__ = ["BaseProvider", "ProviderRegistry"]
