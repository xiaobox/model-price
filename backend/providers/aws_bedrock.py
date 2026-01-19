"""AWS Bedrock pricing provider."""

import asyncio
import logging
import re
from datetime import datetime
from typing import Dict, List, Tuple

import httpx

from config import settings
from models import ModelPricing, Pricing, BatchPricing
from .base import BaseProvider, detect_modalities
from .registry import ProviderRegistry

logger = logging.getLogger(__name__)

# Model name patterns: (regex, model_id, display_name)
# Used to normalize model names from AWS API which sometimes omit vendor prefixes
# Order matters - more specific patterns first
MODEL_PATTERNS: List[Tuple[str, str, str]] = [
    # DeepSeek - AWS API returns just "R1" without vendor prefix
    (r"^r1$", "deepseek-r1", "DeepSeek R1"),
    (r"^v3$", "deepseek-v3", "DeepSeek V3"),
    (r"deepseek.?r1", "deepseek-r1", "DeepSeek R1"),
    (r"deepseek.?v3", "deepseek-v3", "DeepSeek V3"),
    (r"deepseek", "deepseek", "DeepSeek"),
]


class AWSBedrockProvider(BaseProvider):
    """Provider for AWS Bedrock pricing data."""

    name = "aws_bedrock"
    display_name = "AWS Bedrock"

    async def fetch(self) -> List[ModelPricing]:
        """Fetch pricing from both Bedrock sources."""
        async with httpx.AsyncClient(timeout=settings.http_timeout) as client:
            # Fetch both sources concurrently
            bedrock_resp, fm_resp = await asyncio.gather(
                client.get(settings.bedrock_url),
                client.get(settings.bedrock_fm_url),
            )
            bedrock_resp.raise_for_status()
            fm_resp.raise_for_status()

            bedrock_data = bedrock_resp.json()
            fm_data = fm_resp.json()

        models: Dict[str, ModelPricing] = {}

        # Parse AmazonBedrock data
        self._parse_bedrock_data(bedrock_data, models)

        # Parse AmazonBedrockFoundationModels data
        self._parse_fm_data(fm_data, models)

        return list(models.values())

    def _parse_bedrock_data(
        self, data: dict, models: Dict[str, ModelPricing]
    ) -> None:
        """Parse AmazonBedrock pricing data.

        NOTE: This data source prices are per 1K tokens, need to convert to per Million.
        """
        products = data.get("products", {})
        terms = data.get("terms", {}).get("OnDemand", {})

        for sku, product in products.items():
            attrs = product.get("attributes", {})
            model_name = attrs.get("model", "")
            if not model_name:
                continue

            # Skip non-model products (Guardrails, etc.)
            usage_type = attrs.get("usagetype", "")
            if "Guardrail" in usage_type or "CustomModel" in usage_type:
                continue
            # Skip provisioned throughput
            if "ProvisionedThroughput" in usage_type:
                continue
            # Skip customization (training/storage)
            if "Customization" in usage_type:
                continue
            # Skip special pricing tiers: flex, priority, latency-optimized, custom-model
            # These are not standard on-demand pricing
            skip_patterns = ["-flex", "-priority", "-latency-optimized", "-custom-model"]
            if any(p in usage_type.lower() for p in skip_patterns):
                continue

            # Get price
            term_data = terms.get(sku)
            if not term_data:
                continue

            term = list(term_data.values())[0]
            price_dim = list(term["priceDimensions"].values())[0]
            # Price is per 1K tokens in this data source - convert to per Million
            price_per_1k = float(price_dim["pricePerUnit"].get("USD", "0"))
            price_usd = price_per_1k * 1000  # Convert to per Million tokens
            description = price_dim.get("description", "")

            # Determine price type from description/usagetype
            is_input = "input" in usage_type.lower() or "input" in description.lower()
            is_output = "output" in usage_type.lower() or "output" in description.lower()
            is_batch = "batch" in usage_type.lower() or "batch" in description.lower()
            is_cache_read = "cache-read" in usage_type.lower()
            is_cache_write = "cache-write" in usage_type.lower()

            # Create or update model
            model_id, display_name = self._normalize_model_id(model_name)
            full_id = f"{self.name}:{model_id}"

            if full_id not in models:
                # Detect capabilities from model name
                capabilities = self._detect_capabilities(display_name)
                input_mods, output_mods = detect_modalities(capabilities, display_name)

                models[full_id] = ModelPricing(
                    id=full_id,
                    provider=self.name,
                    model_id=model_id,
                    model_name=display_name,
                    pricing=Pricing(),
                    batch_pricing=None,
                    capabilities=capabilities,
                    input_modalities=input_mods,
                    output_modalities=output_mods,
                    last_updated=datetime.now(),
                )

            model = models[full_id]

            # Update prices - only if not already set (first value wins)
            if is_batch:
                if model.batch_pricing is None:
                    model.batch_pricing = BatchPricing()
                if is_input and model.batch_pricing.input is None:
                    model.batch_pricing.input = price_usd
                elif is_output and model.batch_pricing.output is None:
                    model.batch_pricing.output = price_usd
            elif is_cache_read:
                if model.pricing.cached_input is None:
                    model.pricing.cached_input = price_usd
            elif is_cache_write:
                if model.pricing.cached_write is None:
                    model.pricing.cached_write = price_usd
            elif is_input:
                if model.pricing.input is None:
                    model.pricing.input = price_usd
            elif is_output:
                if model.pricing.output is None:
                    model.pricing.output = price_usd

    def _parse_fm_data(
        self, data: dict, models: Dict[str, ModelPricing]
    ) -> None:
        """Parse AmazonBedrockFoundationModels pricing data.

        NOTE: This data source prices are per Million tokens (standard unit).
        """
        products = data.get("products", {})
        terms = data.get("terms", {}).get("OnDemand", {})

        for sku, product in products.items():
            attrs = product.get("attributes", {})
            service_name = attrs.get("servicename", "")
            if not service_name:
                continue

            # Extract model name from service name
            # e.g., "Claude 3.5 Sonnet (Amazon Bedrock Edition)" -> "Claude 3.5 Sonnet"
            model_name = re.sub(r"\s*\(Amazon Bedrock Edition\)\s*$", "", service_name)

            usage_type = attrs.get("usagetype", "")

            # Get price
            term_data = terms.get(sku)
            if not term_data:
                continue

            term = list(term_data.values())[0]
            price_dim = list(term["priceDimensions"].values())[0]
            price_usd = float(price_dim["pricePerUnit"].get("USD", "0"))
            description = price_dim.get("description", "")

            # Determine price type
            is_input = "Input" in usage_type
            is_output = "Output" in usage_type or "Response" in description
            is_batch = "batch" in usage_type.lower() or "Batch" in description
            is_cache_read = "CacheRead" in usage_type or "Cache Read" in description
            is_cache_write = "CacheWrite" in usage_type or "Cache Write" in description
            # Long context pricing is more expensive (2x) - skip it, use standard pricing
            is_long_context = "LCtx" in usage_type or "Long Context" in description

            # Skip provisioned throughput, reserved capacity, and other non-token pricing
            if "ProvisionedThroughput" in usage_type:
                continue
            # Skip reserved capacity pricing (e.g., Reserved_1Month, Reserved_3Month)
            if "Reserved" in usage_type:
                continue
            # Skip long context pricing - use standard pricing as the base rate
            if is_long_context:
                continue
            # Skip regional pricing - prefer global pricing (_Global) which matches standard rates
            # Regional pricing has a ~10% premium (e.g., $5.5 vs $5.0 for input)
            is_regional = "Global" not in usage_type and "_Batch" not in usage_type

            # Create or update model
            model_id, display_name = self._normalize_model_id(model_name)
            full_id = f"{self.name}:{model_id}"

            if full_id not in models:
                # Detect capabilities from model name
                capabilities = self._detect_capabilities(display_name)
                input_mods, output_mods = detect_modalities(capabilities, display_name)

                models[full_id] = ModelPricing(
                    id=full_id,
                    provider=self.name,
                    model_id=model_id,
                    model_name=display_name,
                    pricing=Pricing(),
                    batch_pricing=None,
                    capabilities=capabilities,
                    input_modalities=input_mods,
                    output_modalities=output_mods,
                    last_updated=datetime.now(),
                )

            model = models[full_id]

            # Update prices based on type
            # Prefer global pricing over regional (global is standard, regional has ~10% premium)
            if is_batch:
                if model.batch_pricing is None:
                    model.batch_pricing = BatchPricing()
                if is_input:
                    # Prefer global batch pricing
                    if model.batch_pricing.input is None or not is_regional:
                        model.batch_pricing.input = price_usd
                elif is_output:
                    if model.batch_pricing.output is None or not is_regional:
                        model.batch_pricing.output = price_usd
            elif is_cache_read:
                # Prefer global cache pricing
                if model.pricing.cached_input is None or not is_regional:
                    model.pricing.cached_input = price_usd
            elif is_cache_write:
                if model.pricing.cached_write is None or not is_regional:
                    model.pricing.cached_write = price_usd
            elif is_input and not is_cache_read and not is_cache_write:
                # Prefer global pricing over regional
                if model.pricing.input is None or not is_regional:
                    model.pricing.input = price_usd
            elif is_output:
                if model.pricing.output is None or not is_regional:
                    model.pricing.output = price_usd

    def _detect_capabilities(self, display_name: str) -> List[str]:
        """Detect model capabilities from display name.

        Based on official documentation and third-party verification (artificialanalysis.ai).

        Returns:
            List of capability strings.
        """
        name_lower = display_name.lower()

        # Image generation models - no text capability
        if any(x in name_lower for x in ["stable diffusion", "sdxl", "titan image"]):
            return ["image_generation"]

        # Embedding models - no text capability
        if "embed" in name_lower:
            return ["embedding"]

        # Start with text as base
        capabilities = ["text"]

        # Vision capability - based on official documentation
        if any(x in name_lower for x in ["vision", "vl", "image"]):
            capabilities.append("vision")
        # Claude 3/3.5/4+ have vision; older versions (2.x, instant) don't
        elif "claude" in name_lower:
            vision_patterns = [
                "claude 3", "claude-3", "claude 4", "claude-4", "claude 5", "claude-5",
                "haiku 4", "haiku-4", "sonnet 4", "sonnet-4", "opus 4", "opus-4",
            ]
            if any(x in name_lower for x in vision_patterns):
                capabilities.append("vision")
        # Llama 4 has native multimodal vision
        elif "llama 4" in name_lower or "llama-4" in name_lower:
            capabilities.append("vision")
        # Mistral Large 3, Pixtral have vision
        elif any(x in name_lower for x in ["mistral large 3", "mistral-large-3", "pixtral"]):
            capabilities.append("vision")

        # Audio capability
        if any(x in name_lower for x in ["audio", "sonic", "voxtral"]):
            capabilities.append("audio")

        # Reasoning capability - models with chain-of-thought or extended thinking
        # Based on official documentation and artificialanalysis.ai
        reasoning_patterns = [
            "deepseek r1", "deepseek-r1", " r1",
            "deepseek v3.1", "deepseek-v3.1",  # V3.1 has hybrid thinking
        ]
        # Claude advanced models with extended thinking capability
        # Opus 4/4.5, Sonnet 4, Claude 3.5 Sonnet, Claude 3.7
        claude_reasoning_patterns = [
            "claude 3.5 sonnet", "claude-3.5-sonnet", "claude-3-5-sonnet",
            "claude 3.7", "claude-3.7", "claude-3-7",
            "claude 4", "claude-4",
            "claude opus 4", "claude sonnet 4", "claude haiku 4",
            "opus 4", "sonnet 4",
        ]
        if any(x in name_lower for x in reasoning_patterns):
            capabilities.append("reasoning")
        elif any(x in name_lower for x in claude_reasoning_patterns):
            capabilities.append("reasoning")

        # Tool use capability (function calling)
        # Based on official documentation
        tool_use_patterns = [
            "claude", "llama", "mistral", "command",
            "nova", "titan",  # Amazon models
            "deepseek",  # DeepSeek models support function calling
        ]
        if any(x in name_lower for x in tool_use_patterns):
            capabilities.append("tool_use")

        return capabilities

    def _normalize_model_id(self, name: str) -> Tuple[str, str]:
        """Normalize model name to ID format and return display name.

        Returns:
            Tuple of (model_id, display_name)
        """
        name_lower = name.lower()
        
        # Check against known patterns first
        for pattern, model_id, display_name in MODEL_PATTERNS:
            if re.search(pattern, name_lower, re.IGNORECASE):
                return model_id, display_name
        
        # Default: lowercase, replace spaces with hyphens, remove special chars
        model_id = re.sub(r"[^a-z0-9\s\-\.]", "", name_lower)
        model_id = re.sub(r"\s+", "-", model_id)
        return model_id, name


# Register provider
ProviderRegistry.register(AWSBedrockProvider())
