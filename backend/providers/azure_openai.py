"""Azure OpenAI pricing provider."""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import httpx

from config import settings
from models import ModelPricing, Pricing, BatchPricing
from .base import BaseProvider, detect_modalities
from .registry import ProviderRegistry

logger = logging.getLogger(__name__)

API_VERSION = "2023-01-01-preview"

# Model name patterns: (regex, model_id, display_name)
# Order matters - more specific patterns first
MODEL_PATTERNS: List[Tuple[str, str, str]] = [
    # GPT-4.1 variants
    (r"gpt.?4\.1.?nano", "gpt-4.1-nano", "GPT-4.1 Nano"),
    (r"gpt.?4\.1.?mini", "gpt-4.1-mini", "GPT-4.1 Mini"),
    (r"gpt.?4\.1\b", "gpt-4.1", "GPT-4.1"),
    # GPT-4o mini variants
    (r"gpt.?4o.?mini.?rt.?aud", "gpt-4o-mini-realtime-audio", "GPT-4o Mini Realtime Audio"),
    (r"gpt.?4o.?mini.?rt", "gpt-4o-mini-realtime", "GPT-4o Mini Realtime"),
    (r"gpt.?4o.?mini.?(tts|aud)", "gpt-4o-mini-audio", "GPT-4o Mini Audio"),
    (r"gpt.?4o.?mini.?transcribe", "gpt-4o-mini-transcribe", "GPT-4o Mini Transcribe"),
    (r"gpt.?4o.?mini", "gpt-4o-mini", "GPT-4o Mini"),
    # GPT-4o realtime/audio (consolidate versions)
    (r"gpt.?4o.?rt.?aud|realtimeprvw.?audio", "gpt-4o-realtime-audio", "GPT-4o Realtime Audio"),
    (r"gpt.?4o.?rt.?txt|gpt.?4o.?rt(?!.?aud)", "gpt-4o-realtime", "GPT-4o Realtime"),
    (r"gpt.?4o.?transcribe|trscb", "gpt-4o-transcribe", "GPT-4o Transcribe"),
    (r"gpt.?4o.?aud|tts", "gpt-4o-audio", "GPT-4o Audio"),
    # GPT-4o base (consolidate versions to latest)
    (r"gpt.?4o.*(1120|0806|0513)", "gpt-4o", "GPT-4o"),
    (r"gpt.?4o\b", "gpt-4o", "GPT-4o"),
    # GPT-4 Turbo
    (r"gpt.?4.?turbo.?vision", "gpt-4-turbo-vision", "GPT-4 Turbo Vision"),
    (r"gpt.?4.?turbo", "gpt-4-turbo", "GPT-4 Turbo"),
    # GPT-4 base
    (r"gpt.?4.?32k", "gpt-4-32k", "GPT-4 32K"),
    (r"gpt.?4.?8k", "gpt-4", "GPT-4"),
    # GPT-5
    (r"gpt.?5\.2.?pro", "gpt-5.2-pro", "GPT-5.2 Pro"),
    (r"gpt.?5\.2.?chat", "gpt-5.2-chat", "GPT-5.2 Chat"),
    (r"gpt.?5\.2", "gpt-5.2", "GPT-5.2"),
    (r"5\.1.?codex", "gpt-5.1-codex", "GPT-5.1 Codex"),
    (r"gpt.?5\.1", "gpt-5.1", "GPT-5.1"),
    (r"gpt.?5.?nano", "gpt-5-nano", "GPT-5 Nano"),
    (r"gpt.?5.?mini", "gpt-5-mini", "GPT-5 Mini"),
    (r"gpt.?5.?pro", "gpt-5-pro", "GPT-5 Pro"),
    (r"gpt.?5.?codex", "gpt-5-codex", "GPT-5 Codex"),
    (r"gpt.?5\b", "gpt-5", "GPT-5"),
    # GPT-3.5 (consolidate versions)
    (r"gpt.?35.?turbo.?instruct", "gpt-3.5-turbo-instruct", "GPT-3.5 Turbo Instruct"),
    (r"gpt.?35.?turbo", "gpt-3.5-turbo", "GPT-3.5 Turbo"),
    # GPT Image (1.5 must come before 1)
    (r"gpt.?img.?1\.5|gpt.?image.?1\.5", "gpt-image-1.5", "GPT Image 1.5"),
    (r"gpt.?img.?1\b|gpt.?image.?1\b", "gpt-image-1", "GPT Image 1"),
    (r"gpt.?img.?1.?mini|gpt.?image.?1.?mini", "gpt-image-1-mini", "GPT Image 1 Mini"),
    # O-series reasoning (order matters - more specific patterns first)
    (r"o4.?mini", "o4-mini", "O4 Mini"),
    (r"o4\b", "o4", "O4"),
    (r"o3.?pro", "o3-pro", "O3 Pro"),  # Must be before o3 to avoid mismatching
    (r"o3.?deep.?research", "o3-deep-research", "O3 Deep Research"),
    (r"o3.?mini.*0131", "o3-mini", "O3 Mini"),
    (r"o3.?mini", "o3-mini", "O3 Mini"),
    (r"o3.*0416", "o3", "O3"),  # o3 0416 is the standard o3
    (r"o3\b", "o3", "O3"),
    (r"o1.?pro", "o1-pro", "O1 Pro"),  # Must be before o1 to avoid mismatching
    (r"o1.?mini", "o1-mini", "O1 Mini"),
    (r"o1.?preview", "o1-preview", "O1 Preview"),
    (r"o1.*1217", "o1", "O1"),
    (r"o1\b", "o1", "O1"),
    # Codex
    (r"codex.?mini", "codex-mini", "Codex Mini"),
    # Computer use (Claude via Azure)
    (r"computer.?use", "computer-use-preview", "Computer Use Preview"),
    # Embeddings
    (r"text.?embedding.?3.?large", "text-embedding-3-large", "Text Embedding 3 Large"),
    (r"text.?embedding.?3.?small", "text-embedding-3-small", "Text Embedding 3 Small"),
    (r"text.?embedding.?ada", "text-embedding-ada-002", "Text Embedding Ada"),
    # Llama
    (r"llama.?4.?scout", "llama-4-scout", "Llama 4 Scout"),
    (r"llama.?4.?maverick", "llama-4-maverick", "Llama 4 Maverick"),
    (r"llama.?3\.3.*70b", "llama-3.3-70b", "Llama 3.3 70B"),
    (r"llama.?3\.2", "llama-3.2", "Llama 3.2"),
    (r"llama.?3\.1", "llama-3.1", "Llama 3.1"),
    (r"llama", "llama", "Llama"),
    # Mistral
    (r"mistral.?large", "mistral-large", "Mistral Large"),
    (r"mistral.?small", "mistral-small", "Mistral Small"),
    (r"mistral.?nemo", "mistral-nemo", "Mistral Nemo"),
    (r"pixtral", "pixtral", "Pixtral"),
    (r"ministral", "ministral", "Ministral"),
    (r"mistral", "mistral", "Mistral"),
    # DeepSeek
    (r"deepseek.?r1", "deepseek-r1", "DeepSeek R1"),
    (r"deepseek.?v3", "deepseek-v3", "DeepSeek V3"),
    (r"deepseek", "deepseek", "DeepSeek"),
    # Phi
    (r"phi.?4", "phi-4", "Phi-4"),
    (r"phi.?3\.5", "phi-3.5", "Phi-3.5"),
    (r"phi.?3", "phi-3", "Phi-3"),
    # Cohere
    (r"command.?r.?plus", "command-r-plus", "Command R+"),
    (r"command.?r\b", "command-r", "Command R"),
    (r"command", "command", "Command"),
    (r"embed.?v3", "cohere-embed-v3", "Cohere Embed V3"),
    # Grok (more specific patterns first to avoid mismatch)
    (r"grok.?4\.1", "grok-4.1", "Grok 4.1"),
    (r"grok.?4.?fast", "grok-4-fast", "Grok 4 Fast"),
    (r"grok.?4\b", "grok-4", "Grok 4"),
    (r"grok.?3.?mini", "grok-3-mini", "Grok 3 Mini"),
    (r"grok.?3\b", "grok-3", "Grok 3"),
    (r"grok.?2", "grok-2", "Grok 2"),
    (r"grok", "grok", "Grok"),
    # Kimi
    (r"k2.?thinking", "kimi-k2", "Kimi K2"),
    (r"kimi.?k2", "kimi-k2", "Kimi K2"),
    (r"kimi", "kimi", "Kimi"),
    # Qwen
    (r"qwen", "qwen", "Qwen"),
    # Flux (image)
    (r"flux.?1\.1.?pro", "flux-1.1-pro", "Flux 1.1 Pro"),
    (r"flux.?pro", "flux-pro", "Flux Pro"),
    (r"flux", "flux", "Flux"),
    # Legacy
    (r"davinci.?002", "davinci-002", "Davinci 002"),
    (r"babbage.?002", "babbage-002", "Babbage 002"),
]


class AzureOpenAIProvider(BaseProvider):
    """Provider for Azure OpenAI pricing data."""

    name = "azure_openai"
    display_name = "Azure OpenAI"

    # Products to include (AI models only)
    INCLUDED_PRODUCTS = {
        "Azure OpenAI",
        "Azure OpenAI GPT5",
        "Azure OpenAI Media",
        "Azure OpenAI Reasoning",
        "Azure Deepseek Models",
        "Azure Llama Models",
        "Azure Mistral Models",
        "Azure Phi Models",
        "Azure Grok Models",
        "Azure Kimi",
        "Cohere Models",
        "Qwen models",
        "Azure BFL Flux Models",
    }

    async def fetch(self) -> List[ModelPricing]:
        """Fetch pricing from Azure Retail Prices API."""
        models: Dict[str, ModelPricing] = {}

        async with httpx.AsyncClient(timeout=settings.http_timeout) as client:
            # Filter for Foundry Models service (AI models)
            params = {
                "api-version": API_VERSION,
                "$filter": "serviceName eq 'Foundry Models'",
            }

            next_url: Optional[str] = settings.azure_prices_url
            page = 0

            while next_url:
                if page == 0:
                    resp = await client.get(next_url, params=params)
                else:
                    # NextPageLink already contains all params
                    resp = await client.get(next_url)

                resp.raise_for_status()
                data = resp.json()

                items = data.get("Items", [])
                self._parse_items(items, models)

                next_url = data.get("NextPageLink")
                page += 1

                # Safety limit
                if page > 50:
                    logger.warning("Azure API pagination limit reached")
                    break

        logger.info(f"Azure: parsed {len(models)} models from {page} pages")
        return list(models.values())

    def _parse_items(
        self, items: List[dict], models: Dict[str, ModelPricing]
    ) -> None:
        """Parse Azure pricing items."""
        for item in items:
            product_name = item.get("productName", "")

            # Skip products we don't care about
            if product_name not in self.INCLUDED_PRODUCTS:
                continue

            sku_name = item.get("skuName", "")

            # Only take Global pricing (skip regional/data zone)
            if not self._is_global_pricing(sku_name):
                continue

            # Skip non-primary regions to avoid duplicates
            if not item.get("isPrimaryMeterRegion", False):
                continue

            # Skip non-token pricing (training, sessions, etc.)
            unit = item.get("unitOfMeasure", "")
            if unit not in ("1K", "1M"):
                continue

            # Skip fine-tuning, training, grader entries
            sku_lower = sku_name.lower()
            if any(x in sku_lower for x in ["-ft ", "-ft-", " ft ", "trng", "grdr", "grader"]):
                continue

            # Extract model info
            model_info = self._parse_model_info(sku_name, product_name)
            if not model_info:
                continue

            model_id = model_info["model_id"]
            full_id = f"{self.name}:{model_id}"

            # Get price, normalize to per-million tokens
            price = item.get("retailPrice", 0)
            if unit == "1K":
                price = price * 1000  # Convert to per-million

            # Create or update model
            if full_id not in models:
                capabilities = model_info["capabilities"]
                input_mods, output_mods = detect_modalities(capabilities, model_info["display_name"])

                models[full_id] = ModelPricing(
                    id=full_id,
                    provider=self.name,
                    model_id=model_id,
                    model_name=model_info["display_name"],
                    pricing=Pricing(),
                    batch_pricing=None,
                    capabilities=capabilities,
                    input_modalities=input_mods,
                    output_modalities=output_mods,
                    last_updated=datetime.now(),
                )

            model = models[full_id]
            price_type = model_info["price_type"]

            # Update prices based on type
            if model_info["is_batch"]:
                if model.batch_pricing is None:
                    model.batch_pricing = BatchPricing()
                if price_type == "input":
                    if model.batch_pricing.input is None:
                        model.batch_pricing.input = price
                elif price_type == "output":
                    if model.batch_pricing.output is None:
                        model.batch_pricing.output = price
            else:
                if price_type == "input":
                    if model.pricing.input is None:
                        model.pricing.input = price
                elif price_type == "output":
                    if model.pricing.output is None:
                        model.pricing.output = price
                elif price_type == "cached_input":
                    if model.pricing.cached_input is None:
                        model.pricing.cached_input = price

    def _is_global_pricing(self, sku_name: str) -> bool:
        """Check if this is global pricing (not regional/data zone)."""
        sku_lower = sku_name.lower()

        # Reject regional and data zone explicitly
        if "rgnl" in sku_lower or "regnl" in sku_lower or "regional" in sku_lower:
            return False
        if "data zone" in sku_lower or " dz " in sku_lower or "-dz-" in sku_lower:
            return False
        if "dzone" in sku_lower or " dzn" in sku_lower or "-dzn" in sku_lower:
            return False
        # "DZ" at end of string
        if sku_lower.endswith(" dz") or sku_lower.endswith("-dz"):
            return False

        return True

    def _parse_model_info(
        self, sku_name: str, product_name: str
    ) -> Optional[dict]:
        """Parse model information from SKU name."""
        sku_lower = sku_name.lower()

        # Determine price type
        is_batch = "batch" in sku_lower
        is_cached = "cchd" in sku_lower or "cached" in sku_lower or " cd " in sku_lower

        if is_cached:
            price_type = "cached_input"
        elif "inp" in sku_lower or "input" in sku_lower or " in " in sku_lower:
            price_type = "input"
        elif "outp" in sku_lower or "output" in sku_lower or " out " in sku_lower or " opt " in sku_lower:
            price_type = "output"
        else:
            return None  # Unknown price type

        # Match against known patterns
        model_id = None
        display_name = None

        for pattern, mid, dname in MODEL_PATTERNS:
            if re.search(pattern, sku_lower):
                model_id = mid
                display_name = dname
                break

        if not model_id:
            # Fallback: skip unknown models
            return None

        # Detect capabilities
        capabilities = self._detect_capabilities(sku_name, product_name, model_id)

        return {
            "model_id": model_id,
            "display_name": display_name,
            "price_type": price_type,
            "is_batch": is_batch,
            "capabilities": capabilities,
        }

    def _detect_capabilities(
        self, sku_name: str, product_name: str, model_id: str
    ) -> List[str]:
        """Detect model capabilities.

        Based on official documentation and third-party verification (artificialanalysis.ai).
        """
        sku_lower = sku_name.lower()

        # Image generation models - no text capability
        if any(x in model_id for x in ["flux", "gpt-image"]):
            return ["image_generation"]

        # Embedding models - no text capability
        if "embedding" in model_id:
            return ["embedding"]

        capabilities = ["text"]

        # Vision capability - based on official documentation
        if "vision" in sku_lower or "media" in product_name.lower():
            capabilities.append("vision")
        else:
            # Models with CONFIRMED vision capability:
            # - GPT-4o (all variants except realtime/audio-only)
            # - GPT-4.1 (all variants including nano)
            # - GPT-4.5
            # - GPT-5 (all variants except nano/codex)
            # - O3, O4-mini (first reasoning models with "think with images")
            # - O1 (has vision but limited)
            # - Llama 4 (native multimodal)
            # - Grok 3 (vision capability)
            # - Mistral Large 3, Pixtral, Ministral (vision variants)
            # Vision NOT available: O1-mini, O1-pro, O3-mini, GPT-5-nano, GPT-5-codex
            vision_models = [
                "gpt-4o", "gpt-4.1", "gpt-4.5", "gpt-5",
                "o3", "o4-mini", "o1",
                "llama-4", "grok-3", "grok-2",
                "pixtral", "mistral-large",
            ]
            no_vision = [
                "realtime", "audio", "transcribe",
                "o1-mini", "o1-pro", "o3-mini",
                "gpt-5-nano", "gpt-5-codex", "5.1-codex", "5-codex",
            ]
            if any(x in model_id for x in vision_models):
                if not any(x in model_id for x in no_vision):
                    capabilities.append("vision")

        # Audio capability
        if "aud" in sku_lower or "audio" in sku_lower or "transcribe" in model_id:
            capabilities.append("audio")

        # Reasoning capability - models with chain-of-thought or extended thinking
        # Based on official documentation and artificialanalysis.ai
        # O-series: All are reasoning models
        # GPT-5 series: All have reasoning capability
        # DeepSeek R1, V3.1: Reasoning models
        # Kimi K2: Has reasoning capability
        # Qwen QwQ: Reasoning model
        # NOT reasoning: GPT-4.1, GPT-4o, Llama, Mistral (fast response, no thinking)
        reasoning_models = [
            "o1", "o3", "o4",
            "gpt-5",
            "deepseek-r1", "deepseek-v3.1",
            "kimi-k2",
            "qwq",
        ]
        if any(x in model_id for x in reasoning_models):
            capabilities.append("reasoning")

        # Tool use capability (function calling)
        # Based on official documentation
        # Exclude: o1-mini, o1-pro, o3-mini (limited tool support), embeddings, audio-only
        tool_use_models = [
            "gpt-5", "gpt-4", "gpt-3.5", "gpt-4o",
            "o3", "o4-mini", "o1",
            "llama", "mistral", "command",
            "deepseek", "grok", "kimi", "qwen",
        ]
        no_tool_use = ["o1-mini", "o1-pro", "o3-mini", "transcribe", "whisper", "tts", "embed"]

        if any(x in model_id for x in tool_use_models):
            if not any(x in model_id for x in no_tool_use):
                capabilities.append("tool_use")

        return capabilities


# Register provider
ProviderRegistry.register(AzureOpenAIProvider())
