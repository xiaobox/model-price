"""Pricing data service for CRUD and query operations."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from models import (
    IndexFile,
    ModelPricing,
    PricingDatabase,
    ProviderFile,
    ProviderIndexEntry,
    ProviderInfo,
)
from services.metadata_fetcher import MetadataFetcher

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
PROVIDERS_DIR = DATA_DIR / "providers"
INDEX_FILE = DATA_DIR / "index.json"
# Legacy single file (for backward compatibility)
LEGACY_DATA_FILE = DATA_DIR / "pricing.json"


class PricingService:
    """Service for managing pricing data."""

    # In-memory cache for loaded models
    _cache: Optional[List[ModelPricing]] = None
    _cache_index: Optional[IndexFile] = None

    @classmethod
    def _ensure_data_dir(cls) -> None:
        """Ensure data directories exist."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        PROVIDERS_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def invalidate_cache(cls) -> None:
        """Invalidate the in-memory cache after data changes."""
        cls._cache = None
        cls._cache_index = None

    # ========== New split-file methods ==========

    @classmethod
    def _load_index(cls) -> Optional[IndexFile]:
        """Load index.json if it exists."""
        if not INDEX_FILE.exists():
            return None
        try:
            with open(INDEX_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return IndexFile.model_validate(data)
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return None

    @classmethod
    def _save_index(cls, index: IndexFile) -> None:
        """Save index.json atomically."""
        cls._ensure_data_dir()
        temp_file = INDEX_FILE.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(index.model_dump_json(indent=2))
        temp_file.replace(INDEX_FILE)
        cls._cache_index = None

    @classmethod
    def _load_provider_file(cls, provider_name: str) -> Optional[ProviderFile]:
        """Load a single provider's JSON file."""
        provider_file = PROVIDERS_DIR / f"{provider_name}.json"
        if not provider_file.exists():
            return None
        try:
            with open(provider_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ProviderFile.model_validate(data)
        except Exception as e:
            logger.error(f"Failed to load provider {provider_name}: {e}")
            return None

    @classmethod
    def _save_provider_file(
        cls, provider_name: str, models: List[ModelPricing]
    ) -> datetime:
        """Save models to a provider-specific JSON file atomically.

        Returns the last_updated timestamp.
        """
        cls._ensure_data_dir()
        provider_file = PROVIDERS_DIR / f"{provider_name}.json"
        temp_file = provider_file.with_suffix(".tmp")

        last_updated = (
            max((m.last_updated for m in models), default=datetime.now())
            if models
            else datetime.now()
        )

        provider_data = ProviderFile(
            provider=provider_name,
            last_updated=last_updated,
            models=models,
        )

        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(provider_data.model_dump_json(indent=2))
        temp_file.replace(provider_file)

        logger.info(f"Saved {len(models)} models to {provider_file}")
        return last_updated

    @classmethod
    def _load_all_from_split_files(cls) -> List[ModelPricing]:
        """Load all models from split provider files."""
        index = cls._load_index()
        if not index:
            return []

        all_models: List[ModelPricing] = []
        for provider_name in index.providers:
            provider_file = cls._load_provider_file(provider_name)
            if provider_file:
                all_models.extend(provider_file.models)

        return all_models

    @classmethod
    def _uses_split_files(cls) -> bool:
        """Check if we're using the new split-file format."""
        return INDEX_FILE.exists()

    # ========== Legacy compatibility ==========

    @classmethod
    def _load_database(cls) -> PricingDatabase:
        """Load database, preferring split files over legacy single file."""
        cls._ensure_data_dir()

        # Use cached data if available
        if cls._cache is not None:
            index = cls._load_index()
            return PricingDatabase(
                last_refresh=index.last_refresh if index else datetime.now(),
                models=cls._cache,
            )

        # Prefer new split-file format
        if cls._uses_split_files():
            index = cls._load_index()
            models = cls._load_all_from_split_files()
            cls._cache = models
            return PricingDatabase(
                version="2.0",
                last_refresh=index.last_refresh if index else datetime.now(),
                models=models,
            )

        # Fall back to legacy single file
        if LEGACY_DATA_FILE.exists():
            try:
                with open(LEGACY_DATA_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                db = PricingDatabase.model_validate(data)
                cls._cache = db.models
                return db
            except Exception as e:
                logger.error(f"Failed to load legacy database: {e}")

        return PricingDatabase(last_refresh=datetime.now(), models=[])

    @classmethod
    def _save_database(cls, db: PricingDatabase) -> None:
        """Save database - routes to split files if using new format."""
        if cls._uses_split_files():
            # Group models by provider and save each
            by_provider: Dict[str, List[ModelPricing]] = {}
            for model in db.models:
                by_provider.setdefault(model.provider, []).append(model)

            index_providers: Dict[str, ProviderIndexEntry] = {}
            total = 0

            for provider_name, models in by_provider.items():
                last_updated = cls._save_provider_file(provider_name, models)
                index_providers[provider_name] = ProviderIndexEntry(
                    file=f"providers/{provider_name}.json",
                    model_count=len(models),
                    last_updated=last_updated,
                )
                total += len(models)

            # Update index
            index = IndexFile(
                last_refresh=db.last_refresh,
                providers=index_providers,
                total_models=total,
            )
            cls._save_index(index)
            cls.invalidate_cache()
        else:
            # Legacy: save to single file
            cls._ensure_data_dir()
            temp_file = LEGACY_DATA_FILE.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(db.model_dump_json(indent=2))
            temp_file.replace(LEGACY_DATA_FILE)
            cls.invalidate_cache()
            logger.info(f"Saved {len(db.models)} models to {LEGACY_DATA_FILE}")

    @staticmethod
    def extract_model_family(model_name: str) -> str:
        """Extract model family from model name.

        Examples:
        - 'Claude 3.5 Sonnet' -> 'Claude'
        - 'GPT-4o Mini' -> 'GPT'
        - 'Llama 3.1 70B' -> 'Llama'
        - 'Gemini 2.5 Flash' -> 'Gemini'
        - 'Nova Pro' -> 'Nova'
        - 'Anthropic: Claude 3.5 Haiku' -> 'Claude'
        - 'OpenAI: o3-mini' -> 'o3'
        """
        name = model_name.strip()

        # Remove common provider prefixes (e.g., "Anthropic: Claude 3.5")
        provider_prefixes = [
            "Anthropic:", "OpenAI:", "Google:", "Meta:", "Mistral:", "Cohere:",
            "AI21:", "Amazon:", "Microsoft:", "NVIDIA:", "xAI:", "DeepSeek:",
            "Baidu:", "Alibaba:", "ByteDance:", "Tencent:", "Perplexity:",
            "AllenAI:", "Arcee AI:", "Arcee:", "Nous:", "TheDrummer:", "Z.AI:",
            "AionLabs:", "MiniMax:", "Inflection:", "Inception:", "Kwaipilot:",
            "Morph:", "Relace:", "TNG:", "Mancer:", "Meituan:", "EssentialAI:",
            "EleutherAI:", "Venice:", "OpenGVLab:", "StepFun:", "THUDM:",
            "IBM:", "NeverSleep:", "Xiaomi:", "AlfredPros:",
            "ByteDance Seed:", "Deep Cogito:", "Nex AGI:", "Prime Intellect:",
            "NousResearch:", "Qwen:", "TwelveLabs:",
        ]

        name_lower = name.lower()
        for prefix in provider_prefixes:
            if name_lower.startswith(prefix.lower()):
                name = name[len(prefix):].strip()
                name_lower = name.lower()
                break

        # Common model family patterns (order matters - more specific first)
        # Patterns starting with ^ mean "starts with"
        # A model family is defined by the original developer/company, not fine-tuners
        family_patterns = [
            # ===== Deep Cogito (separate company, check FIRST before Llama) =====
            ("Cogito", ["cogito"]),

            # ===== Anthropic =====
            ("Claude", ["claude"]),

            # ===== OpenAI =====
            # O-Series: reasoning models (o1, o3, o4 are the same product line)
            ("OpenAI O-Series", ["^o1", "^o3", "^o4"]),
            # GPT: general-purpose models (includes Codex as GPT variant)
            # Note: "^computer use preview" matches OpenAI's CUA model (starts with)
            ("GPT", ["gpt-", "gpt ", "chatgpt", "gpt4", "gpt3", "gpt5",
                     "babbage", "davinci", "omni moderation", "codex",
                     "^computer use preview"]),
            ("DALL-E", ["dall-e", "dall·e"]),
            ("Whisper", ["whisper"]),
            ("OpenAI TTS", ["^tts", "-tts"]),
            ("Sora", ["sora"]),

            # ===== Google =====
            # Note: "nano banana" models also contain "gemini" in their names
            ("Gemini", ["gemini"]),
            ("Gemma", ["gemma"]),
            ("Imagen", ["imagen"]),
            ("Veo", ["^veo"]),

            # ===== Meta =====
            # Only official Meta Llama models, not third-party fine-tunes
            ("Llama", ["llama", "llamaguard"]),

            # ===== Mistral =====
            ("Mistral", ["mixtral", "ministral", "pixtral", "codestral",
                         "magistral", "voxtral", "devstral", "mistral", "saba"]),

            # ===== Amazon =====
            ("Nova", ["^nova"]),
            ("Titan", ["titan"]),

            # ===== Cohere =====
            ("Command", ["command"]),
            ("Cohere Embed", ["cohere embed", "embed 3", "embed 4"]),
            ("Rerank", ["rerank"]),

            # ===== AI21 =====
            ("Jamba", ["jamba"]),
            ("Jurassic", ["jurassic"]),

            # ===== xAI =====
            ("Grok", ["grok"]),

            # ===== DeepSeek =====
            # DeepSeek's own models including R1 series
            ("DeepSeek", ["deepseek", "deep-seek", "^r1"]),

            # ===== Qwen/Alibaba =====
            ("Qwen", ["qwen", "qwq", "tongyi"]),

            # ===== NVIDIA =====
            ("Nemotron", ["nemotron"]),

            # ===== Stability AI =====
            ("Stable Diffusion", ["stable diffusion", "stable image", "sdxl"]),

            # ===== Moonshot =====
            ("Kimi", ["kimi"]),

            # ===== 01.AI =====
            ("Yi", ["yi-", "yi "]),

            # ===== Microsoft =====
            ("Phi", ["phi-", "phi "]),

            # ===== Inflection =====
            ("Inflection", ["inflection"]),

            # ===== Perplexity =====
            ("Sonar", ["sonar"]),

            # ===== Allen AI =====
            ("OLMo", ["olmo"]),

            # ===== MiniMax =====
            ("MiniMax", ["minimax", "abab"]),

            # ===== Video models =====
            ("Kling", ["kling"]),
            ("Runway", ["runway"]),
            ("Pegasus", ["pegasus"]),

            # ===== Nous Research (fine-tunes, but notable) =====
            ("Hermes", ["hermes", "deephermes"]),

            # ===== Third-party fine-tuned/merged models =====
            ("MythoMax", ["mythomax"]),
            ("Goliath", ["goliath"]),
            ("Noromaid", ["noromaid"]),
            ("WizardLM", ["wizardlm"]),
            ("Dolphin", ["dolphin"]),
            ("Lumimaid", ["lumimaid"]),
            ("ReMM", ["remm "]),
            ("UnslopNemo", ["unslopnemo"]),
            ("Rocinante", ["rocinante"]),
            ("Skyfall", ["skyfall"]),
            ("Cydonia", ["cydonia"]),
            ("Magnum", ["magnum"]),
            ("Euryale", ["euryale"]),
            ("SorcererLM", ["sorcererlm"]),
            ("LongCat", ["longcat"]),
            ("Llemma", ["llemma"]),

            # ===== LiquidAI =====
            ("LFM", ["lfm-", "lfm2", "liquidai"]),

            # ===== Zhipu/GLM =====
            ("GLM", ["glm"]),

            # ===== Baidu =====
            ("ERNIE", ["ernie"]),

            # ===== ByteDance =====
            ("Seed", ["^seed ", "seed 1", "ui-tars"]),

            # ===== Tencent =====
            ("Hunyuan", ["hunyuan"]),

            # ===== IBM =====
            ("Granite", ["granite"]),

            # ===== OpenGVLab =====
            ("InternVL", ["internvl"]),

            # ===== Arcee AI =====
            ("Arcee", ["arcee", "coder large", "maestro", "spotlight",
                       "trinity", "virtuoso"]),

            # ===== Inception =====
            ("Mercury", ["mercury"]),

            # ===== OpenAI Embeddings =====
            ("OpenAI Embed", ["text-embedding"]),

            # ===== Twelve Labs =====
            ("TwelveLabs", ["marengo", "pegasus"]),

            # ===== Step (StepFun) =====
            ("Step", ["^step"]),

            # ===== Xiaomi =====
            ("MiMo", ["mimo"]),

            # ===== Kuaishou =====
            ("KAT", ["kat-coder"]),

            # ===== Morph =====
            ("Morph", ["morph "]),

            # ===== EssentialAI =====
            ("Rnj", ["rnj "]),

            # ===== Relace =====
            ("Relace", ["relace "]),

            # ===== Mancer =====
            ("Weaver", ["weaver"]),

            # ===== Prime Intellect =====
            ("INTELLECT", ["intellect"]),

            # ===== AionLabs =====
            ("Aion", ["aion"]),

            # ===== Writer (NOT Google PaLM) =====
            ("Palmyra", ["palmyra"]),

            # ===== TNG/Chimera =====
            ("Chimera", ["chimera"]),

            # ===== Nex AGI =====
            ("Nex", ["nex n1"]),

            # ===== Router/aggregator =====
            ("Router", ["router", "switchpoint"]),

            # ===== Venice =====
            ("Venice", ["uncensored"]),

            # ===== Other =====
            ("Other", ["body builder"]),
        ]

        for family, patterns in family_patterns:
            for pattern in patterns:
                # Patterns starting with ^ mean "starts with"
                if pattern.startswith("^"):
                    if name_lower.startswith(pattern[1:]):
                        return family
                elif pattern in name_lower:
                    return family

        # Fallback: use first word, clean it up
        first_word = name.split()[0] if name else "Unknown"
        # Remove trailing punctuation
        first_word = first_word.rstrip(":/-")
        return first_word if first_word else "Other"

    @classmethod
    def get_all(
        cls,
        provider: Optional[str] = None,
        capability: Optional[str] = None,
        family: Optional[str] = None,
        search: Optional[str] = None,
        sort_by: str = "model_name",
        sort_order: str = "asc",
    ) -> List[ModelPricing]:
        """Get all models with optional filters and sorting."""
        db = cls._load_database()
        models = db.models

        # Filter by provider
        if provider:
            models = [m for m in models if m.provider == provider]

        # Filter by capability
        if capability:
            models = [m for m in models if capability in m.capabilities]

        # Filter by model family
        if family:
            models = [m for m in models if cls.extract_model_family(m.model_name) == family]

        # Search by model name
        if search:
            search_lower = search.lower()
            models = [m for m in models if search_lower in m.model_name.lower()]

        # Sort
        reverse = sort_order == "desc"
        if sort_by == "model_name":
            models.sort(key=lambda m: m.model_name.lower(), reverse=reverse)
        elif sort_by == "input":
            models.sort(key=lambda m: m.pricing.input or 0, reverse=reverse)
        elif sort_by == "output":
            models.sort(key=lambda m: m.pricing.output or 0, reverse=reverse)
        elif sort_by == "context_length":
            models.sort(key=lambda m: m.context_length or 0, reverse=reverse)

        return models

    @classmethod
    def get_by_id(cls, model_id: str) -> Optional[ModelPricing]:
        """Get a single model by ID."""
        db = cls._load_database()
        for model in db.models:
            if model.id == model_id:
                return model
        return None

    @classmethod
    def get_providers(
        cls,
        capability: Optional[str] = None,
        family: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[ProviderInfo]:
        """Get list of all providers with stats, filtered by other conditions."""
        db = cls._load_database()
        models = db.models

        # Apply filters (except provider itself)
        if capability:
            models = [m for m in models if capability in m.capabilities]
        if family:
            models = [m for m in models if cls.extract_model_family(m.model_name) == family]
        if search:
            search_lower = search.lower()
            models = [m for m in models if search_lower in m.model_name.lower()]

        provider_stats: Dict[str, dict] = {}

        for model in models:
            if model.provider not in provider_stats:
                provider_stats[model.provider] = {
                    "count": 0,
                    "last_updated": model.last_updated,
                }
            provider_stats[model.provider]["count"] += 1
            if model.last_updated > provider_stats[model.provider]["last_updated"]:
                provider_stats[model.provider]["last_updated"] = model.last_updated

        # Map provider names to display names
        display_names = {
            "aws_bedrock": "AWS Bedrock",
            "openai": "OpenAI",
            "azure_openai": "Azure OpenAI",
            "google_vertex_ai": "Google Vertex AI",
            "google_gemini": "Google Vertex",
            "openrouter": "OpenRouter",
            "anthropic": "Anthropic",
            "xai": "xAI",
        }

        return [
            ProviderInfo(
                name=name,
                display_name=display_names.get(name, name),
                model_count=stats["count"],
                last_updated=stats["last_updated"],
            )
            for name, stats in sorted(provider_stats.items())
        ]

    @classmethod
    def get_model_families(
        cls,
        provider: Optional[str] = None,
        capability: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get list of all model families with counts, filtered by other conditions."""
        db = cls._load_database()
        models = db.models

        # Apply filters (except family itself)
        if provider:
            models = [m for m in models if m.provider == provider]
        if capability:
            models = [m for m in models if capability in m.capabilities]
        if search:
            search_lower = search.lower()
            models = [m for m in models if search_lower in m.model_name.lower()]

        family_stats: Dict[str, int] = {}

        for model in models:
            family = cls.extract_model_family(model.model_name)
            family_stats[family] = family_stats.get(family, 0) + 1

        # Sort by count descending, then by name
        sorted_families = sorted(
            family_stats.items(),
            key=lambda x: (-x[1], x[0].lower())
        )

        return [
            {"name": name, "count": count}
            for name, count in sorted_families
        ]

    @classmethod
    def save_models(cls, models: List[ModelPricing]) -> None:
        """Save models to database (full replace)."""
        db = PricingDatabase(
            last_refresh=datetime.now(),
            models=models,
        )
        cls._save_database(db)

    @classmethod
    def update_provider(cls, provider_name: str, models: List[ModelPricing]) -> None:
        """Update models for a single provider (keep others)."""
        if cls._uses_split_files():
            # Directly save to provider file and update index
            last_updated = cls._save_provider_file(provider_name, models)

            # Update index
            index = cls._load_index() or IndexFile(
                last_refresh=datetime.now(), providers={}, total_models=0
            )
            index.providers[provider_name] = ProviderIndexEntry(
                file=f"providers/{provider_name}.json",
                model_count=len(models),
                last_updated=last_updated,
            )
            index.last_refresh = datetime.now()
            index.total_models = sum(p.model_count for p in index.providers.values())
            cls._save_index(index)
            cls.invalidate_cache()
        else:
            # Legacy: load all, replace provider, save all
            db = cls._load_database()
            other_models = [m for m in db.models if m.provider != provider_name]
            db.models = other_models + models
            db.last_refresh = datetime.now()
            cls._save_database(db)

    @classmethod
    def get_stats(cls) -> dict:
        """Get overall statistics."""
        db = cls._load_database()
        models = db.models

        if not models:
            return {
                "total_models": 0,
                "providers": 0,
                "avg_input_price": 0,
                "avg_output_price": 0,
                "last_refresh": db.last_refresh.isoformat(),
            }

        input_prices = [m.pricing.input for m in models if m.pricing.input is not None]
        output_prices = [m.pricing.output for m in models if m.pricing.output is not None]

        return {
            "total_models": len(models),
            "providers": len(set(m.provider for m in models)),
            "avg_input_price": sum(input_prices) / len(input_prices) if input_prices else 0,
            "avg_output_price": sum(output_prices) / len(output_prices) if output_prices else 0,
            "last_refresh": db.last_refresh.isoformat(),
        }

    @classmethod
    def update_model(cls, model_id: str, updates: Dict[str, Any]) -> Optional[ModelPricing]:
        """Update a single model with user overrides."""
        db = cls._load_database()

        for i, model in enumerate(db.models):
            if model.id == model_id:
                # Apply updates to the model
                model_data = model.model_dump()
                if "context_length" in updates:
                    model_data["context_length"] = updates["context_length"]
                if "max_output_tokens" in updates:
                    model_data["max_output_tokens"] = updates["max_output_tokens"]
                if "is_open_source" in updates:
                    model_data["is_open_source"] = updates["is_open_source"]
                if "capabilities" in updates:
                    model_data["capabilities"] = updates["capabilities"]

                # Handle pricing updates
                if "pricing" in updates and updates["pricing"]:
                    pricing_updates = updates["pricing"]
                    if "input" in pricing_updates:
                        model_data["pricing"]["input"] = pricing_updates["input"]
                    if "output" in pricing_updates:
                        model_data["pricing"]["output"] = pricing_updates["output"]
                    if "cached_input" in pricing_updates:
                        model_data["pricing"]["cached_input"] = pricing_updates["cached_input"]

                # Update the model in database
                updated_model = ModelPricing.model_validate(model_data)
                db.models[i] = updated_model
                cls._save_database(db)

                # Save to user overrides for persistence
                override_data: Dict[str, Any] = {}
                if "context_length" in updates:
                    override_data["context_length"] = updates["context_length"]
                if "max_output_tokens" in updates:
                    override_data["max_output_tokens"] = updates["max_output_tokens"]
                if "is_open_source" in updates:
                    override_data["is_open_source"] = updates["is_open_source"]
                if "capabilities" in updates:
                    override_data["capabilities"] = updates["capabilities"]
                if "pricing" in updates and updates["pricing"]:
                    if "pricing" not in override_data:
                        override_data["pricing"] = {}
                    pricing_updates = updates["pricing"]
                    if "input" in pricing_updates:
                        override_data["pricing"]["input"] = pricing_updates["input"]
                    if "output" in pricing_updates:
                        override_data["pricing"]["output"] = pricing_updates["output"]
                    if "cached_input" in pricing_updates:
                        override_data["pricing"]["cached_input"] = pricing_updates["cached_input"]
                if override_data:
                    MetadataFetcher.save_user_override(model_id, override_data)

                return updated_model
        return None

    @classmethod
    async def refresh_metadata(cls) -> int:
        """Refresh metadata for all models from LiteLLM."""
        MetadataFetcher.clear_cache()
        db = cls._load_database()

        models_data = [m.model_dump() for m in db.models]
        enriched = await MetadataFetcher.enrich_models(models_data)

        updated_models = [ModelPricing.model_validate(m) for m in enriched]
        db.models = updated_models
        cls._save_database(db)

        return len(updated_models)

    @classmethod
    def migrate_to_split_files(cls) -> dict:
        """Migrate from legacy single pricing.json to split provider files.

        Returns migration statistics.
        """
        if not LEGACY_DATA_FILE.exists():
            return {"status": "skipped", "reason": "No legacy file found"}

        if cls._uses_split_files():
            return {"status": "skipped", "reason": "Already using split files"}

        # Load from legacy file
        with open(LEGACY_DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        legacy_db = PricingDatabase.model_validate(data)

        # Group by provider
        by_provider: Dict[str, List[ModelPricing]] = {}
        for model in legacy_db.models:
            by_provider.setdefault(model.provider, []).append(model)

        # Save each provider file and build index
        cls._ensure_data_dir()
        index_providers: Dict[str, ProviderIndexEntry] = {}
        total = 0

        for provider_name, models in by_provider.items():
            last_updated = cls._save_provider_file(provider_name, models)
            index_providers[provider_name] = ProviderIndexEntry(
                file=f"providers/{provider_name}.json",
                model_count=len(models),
                last_updated=last_updated,
            )
            total += len(models)

        # Create index
        index = IndexFile(
            last_refresh=legacy_db.last_refresh,
            providers=index_providers,
            total_models=total,
        )
        cls._save_index(index)
        cls.invalidate_cache()

        return {
            "status": "success",
            "providers": len(by_provider),
            "total_models": total,
            "files_created": list(by_provider.keys()),
        }
