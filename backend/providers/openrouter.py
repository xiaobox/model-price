"""OpenRouter pricing provider.

Fetches pricing data from OpenRouter's public API.
https://openrouter.ai/api/v1/models
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Union

import httpx

from config import settings
from models import ModelPricing, Pricing
from .base import BaseProvider, detect_modalities
from .registry import ProviderRegistry

logger = logging.getLogger(__name__)


GPT_IMAGE_MODEL_RE = re.compile(
    r"(?:^|[/:\s])gpt[-_.]?\d[\w.-]*[-_.]image(?:$|[-_.\s/]\w*)"
)

IMAGE_GENERATION_MODEL_PATTERNS = (
    "chatgpt-image",
    "gpt-image",
    "dall-e",
    "dalle",
    "imagen",
    "flux",
    "stable-diffusion",
    "sdxl",
    "recraft",
    "ideogram",
    "midjourney",
    "kandinsky",
    "nova-canvas",
    "titan-image",
    "image-generator",
)


class OpenRouterProvider(BaseProvider):
    """Provider for OpenRouter pricing data."""

    name = "openrouter"
    display_name = "OpenRouter"

    async def fetch(self) -> List[ModelPricing]:
        """Fetch pricing from OpenRouter API."""
        async with httpx.AsyncClient(timeout=settings.http_timeout) as client:
            resp = await client.get(settings.openrouter_url)
            resp.raise_for_status()
            data = resp.json()

        models: Dict[str, ModelPricing] = {}
        now = datetime.now()

        for model_data in data.get("data", []):
            try:
                model = self._parse_model(model_data, now)
                if model:
                    models[model.id] = model
            except Exception as e:
                logger.warning(f"Failed to parse model {model_data.get('id')}: {e}")
                continue

        all_models = list(models.values())
        logger.info(f"OpenRouter: fetched {len(all_models)} models")
        return all_models

    def _parse_model(self, data: dict, now: datetime) -> Optional[ModelPricing]:
        """Parse a single model from API response."""
        model_id = data.get("id")
        if not model_id:
            return None

        # Get display name, fallback to id
        model_name = data.get("name") or model_id

        # Parse pricing - OpenRouter uses per-token pricing, convert to per-million
        pricing_data = data.get("pricing", {})
        pricing = self._parse_pricing(pricing_data)

        # Get modalities from API response (OpenRouter provides these directly).
        # Some newly listed image models omit image output initially, so keep a
        # conservative name-based fallback before capability derivation.
        input_modalities = data.get("input_modalities", [])
        output_modalities = data.get("output_modalities", [])
        is_image_generation = self._is_image_generation_model(data)
        if is_image_generation:
            input_modalities = self._image_generation_input_modalities(input_modalities)
            output_modalities = ["image"]

        capability_data = {
            **data,
            "input_modalities": input_modalities,
            "output_modalities": output_modalities,
        }

        # Parse capabilities from modalities
        capabilities = self._parse_capabilities(capability_data)

        # Get context info
        context_length = data.get("context_length")
        top_provider = data.get("top_provider", {})
        max_output_tokens = top_provider.get("max_completion_tokens")

        # If not provided by API, fall back to detection from capabilities
        if not input_modalities or not output_modalities:
            detected_input, detected_output = detect_modalities(capabilities, model_name)
            if not input_modalities:
                input_modalities = detected_input
            if not output_modalities:
                output_modalities = detected_output

        full_id = f"{self.name}:{model_id}"

        return ModelPricing(
            id=full_id,
            provider=self.name,
            model_id=model_id,
            model_name=model_name,
            pricing=pricing,
            context_length=context_length,
            max_output_tokens=max_output_tokens,
            capabilities=capabilities,
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            last_updated=now,
        )

    def _is_image_generation_model(self, data: dict) -> bool:
        """Return True for models whose primary output is generated images."""
        output_modalities = data.get("output_modalities", [])
        if "image" in output_modalities:
            return True

        model_id = str(data.get("id", "")).lower()
        model_name = str(data.get("name", "")).lower()
        haystack = f"{model_id} {model_name}"

        if GPT_IMAGE_MODEL_RE.search(haystack):
            return True
        return any(pattern in haystack for pattern in IMAGE_GENERATION_MODEL_PATTERNS)

    def _image_generation_input_modalities(self, modalities: List[str]) -> List[str]:
        """Normalize image generation inputs to prompt text plus optional image."""
        normalized = list(modalities or [])
        if "text" not in normalized:
            normalized.insert(0, "text")
        if "image" not in normalized:
            normalized.append("image")
        return list(dict.fromkeys(normalized))

    def _parse_pricing(self, pricing_data: dict) -> Pricing:
        """Parse pricing and convert from per-token to per-million tokens."""
        # OpenRouter pricing is per token, multiply by 1M for our format

        def to_per_million(value: Optional[Union[str, float]]) -> Optional[float]:
            if value is None:
                return None
            try:
                v = float(value)
                # -1 is a sentinel value for "variable/unknown pricing"
                if v < 0:
                    return None
                if v == 0:
                    return 0.0
                return v * 1_000_000
            except (ValueError, TypeError):
                return None

        return Pricing(
            input=to_per_million(pricing_data.get("prompt")),
            output=to_per_million(pricing_data.get("completion")),
            cached_input=to_per_million(pricing_data.get("input_cache_read")),
            cached_write=to_per_million(pricing_data.get("input_cache_write")),
            reasoning=to_per_million(pricing_data.get("internal_reasoning")),
            image_input=to_per_million(pricing_data.get("image")),
            audio_input=to_per_million(pricing_data.get("audio")),
        )

    def _parse_capabilities(self, data: dict) -> List[str]:
        """Parse capabilities from input/output modalities.

        Based on official documentation and third-party verification (artificialanalysis.ai).
        """
        capabilities: List[str] = []

        input_modalities = data.get("input_modalities", [])
        output_modalities = data.get("output_modalities", [])
        pricing_data = data.get("pricing", {})
        model_id = data.get("id", "").lower()

        if self._is_image_generation_model(data):
            capabilities.append("image_generation")
            if "image" in input_modalities:
                capabilities.append("vision")
            return capabilities

        # Text capability - check modalities or assume text for chat models
        has_text = "text" in input_modalities or "text" in output_modalities
        # Most models support text even if not explicitly listed
        if has_text or not input_modalities:
            capabilities.append("text")

        # Vision capability
        if "image" in input_modalities:
            capabilities.append("vision")
        else:
            # Many models have vision but OpenRouter doesn't list it in modalities
            # Claude 3/3.5/4+ have vision
            if "claude" in model_id:
                if any(x in model_id for x in ["claude-3", "claude-4", "claude-5", "haiku-4", "sonnet-4", "opus-4"]):
                    capabilities.append("vision")
            # GPT-4o, GPT-4.1, GPT-4.5, GPT-5 have vision (except codex/nano variants)
            elif any(x in model_id for x in ["gpt-4o", "gpt-4.1", "gpt-4.5", "gpt-5", "chatgpt-4o"]):
                # Skip audio/realtime variants and codex/nano (no vision)
                no_vision_variants = ["realtime", "audio", "transcribe", "codex", "nano"]
                if not any(x in model_id for x in no_vision_variants):
                    capabilities.append("vision")
            # O3, O4-mini have vision ("think with images"), O1 has limited vision
            elif any(x in model_id for x in ["o3", "o4-mini", "o1"]):
                # O1-mini, O1-pro, O3-mini do NOT have vision
                if not any(x in model_id for x in ["o1-mini", "o1-pro", "o3-mini"]):
                    capabilities.append("vision")
            # Gemini models have vision
            elif "gemini" in model_id and "embed" not in model_id:
                capabilities.append("vision")
            # Llama 4 has native multimodal vision
            elif "llama-4" in model_id or "llama4" in model_id:
                capabilities.append("vision")
            # Grok 3 has vision
            elif "grok-3" in model_id or "grok3" in model_id:
                capabilities.append("vision")
            # Mistral Large 3 and Ministral have vision
            elif "mistral-large-3" in model_id or "ministral" in model_id:
                capabilities.append("vision")

        # Audio capability
        if "audio" in input_modalities or "audio" in output_modalities:
            capabilities.append("audio")
        # Gemini 2.x models support audio
        elif "gemini-2" in model_id:
            capabilities.append("audio")

        # Image generation
        if "image" in output_modalities:
            capabilities.append("image_generation")

        # Video input
        if "video" in input_modalities:
            capabilities.append("video")

        # File input
        if "file" in input_modalities:
            capabilities.append("file")

        # Reasoning capability - models with explicit chain-of-thought or extended thinking
        # Check internal_reasoning pricing (must be > 0, not just present)
        has_reasoning_pricing = False
        internal_reasoning = pricing_data.get("internal_reasoning")
        if internal_reasoning is not None:
            try:
                has_reasoning_pricing = float(internal_reasoning) > 0
            except (ValueError, TypeError):
                pass

        # Reasoning patterns - models explicitly designed for chain-of-thought reasoning
        reasoning_patterns = [
            "o1", "o3", "o4",  # OpenAI reasoning models (all o-series)
            "gpt-5",          # GPT-5 series (all have reasoning capability)
            "deepseek-r1", "deepseek/r1",  # DeepSeek R1
            "deepseek-v3.1", "deepseek-v3-1",  # DeepSeek V3.1 (hybrid thinking)
            "qwq",            # Qwen QwQ
            "-think",         # Various thinking models
            "thinking",       # Explicit thinking models
            "-r1",            # R1 variants
        ]
        has_reasoning_pattern = any(x in model_id for x in reasoning_patterns)

        # Claude advanced models with extended thinking capability
        # Claude Opus 4/4.5, Sonnet 4, Claude 3.5 Sonnet (extended thinking available)
        claude_reasoning_patterns = [
            "claude-3.5-sonnet", "claude-3-5-sonnet",  # Claude 3.5 Sonnet
            "claude-3.7", "claude-3-7",                 # Claude 3.7
            "claude-4", "claude-opus-4", "claude-sonnet-4",  # Claude 4
            "claude-opus-4.5", "claude-4.5",            # Claude 4.5
            "claude-haiku-4",                           # Claude Haiku 4
        ]
        if any(x in model_id for x in claude_reasoning_patterns):
            has_reasoning_pattern = True

        # Gemini 2.5 models have thinking mode
        if "gemini-2.5" in model_id or "gemini-2-5" in model_id:
            has_reasoning_pattern = True

        # Grok 3 has "Think" and "Big Brain" reasoning modes
        if "grok-3" in model_id or "grok3" in model_id:
            has_reasoning_pattern = True

        # Cohere Command A Reasoning
        if "command-a" in model_id and "reasoning" in model_id:
            has_reasoning_pattern = True

        # Ministral reasoning variants
        if "ministral" in model_id and "reason" in model_id:
            has_reasoning_pattern = True

        if has_reasoning_pricing or has_reasoning_pattern:
            capabilities.append("reasoning")

        # Tool use capability - based on official documentation
        # Check supported_parameters if available, otherwise detect by model family
        supported_params = data.get("supported_parameters", [])
        if "tools" in supported_params or "tool_choice" in supported_params:
            capabilities.append("tool_use")
        else:
            # Fallback: detect tool_use by model family
            tool_use_patterns = [
                "gpt-4", "gpt-3.5", "gpt-5",
                "claude",
                "gemini",
                "mistral", "ministral",
                "llama-3", "llama-4", "llama3", "llama4",
                "command",  # Cohere
                "grok",
                "qwen",
                "deepseek",
                "o3", "o4", "o1",  # O-series with tool use
            ]
            # Models without tool use
            no_tool_use = ["o1-mini", "o1-pro", "o3-mini", "embed", "whisper", "tts"]
            if any(x in model_id for x in tool_use_patterns):
                if not any(x in model_id for x in no_tool_use):
                    capabilities.append("tool_use")

        # If no capabilities detected, default to text
        if not capabilities:
            capabilities = ["text"]

        return capabilities


# Register provider
ProviderRegistry.register(OpenRouterProvider())
