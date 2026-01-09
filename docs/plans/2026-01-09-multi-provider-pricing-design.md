# Multi-Provider Pricing System Design

## Overview

Refactor the model-price application to fetch pricing data from multiple AI providers (AWS Bedrock, Azure, OpenAI, Google, OpenRouter, xAI, etc.) with a flexible, extensible architecture.

## Decisions

| Decision | Choice |
|----------|--------|
| Data Storage | JSON file (`data/pricing.json`) |
| Update Mechanism | On startup + manual trigger via API |
| Provider Architecture | Hybrid (plugin for complex, config for simple) |
| Data Fields | Complete (multimodal + batch pricing) |
| Frontend Display | Dual mode (table/card toggle) |

## Architecture

```
backend/
├── main.py                    # FastAPI entry point
├── config.py                  # Configuration management
├── models/
│   └── pricing.py             # Pydantic data models
├── providers/
│   ├── base.py                # Provider abstract base class
│   ├── registry.py            # Provider registry
│   ├── configurable.py        # Config-driven provider
│   └── aws_bedrock.py         # AWS Bedrock (plugin)
├── services/
│   ├── pricing.py             # Pricing data service (CRUD + query)
│   └── fetcher.py             # Data fetch orchestrator
└── data/
    └── pricing.json           # Persistent storage

frontend/src/
├── components/
│   ├── ModelTable.tsx         # Table view
│   ├── ModelCard.tsx          # Single card
│   ├── ModelGrid.tsx          # Card grid container
│   ├── ViewToggle.tsx         # Table/card switch
│   ├── FilterBar.tsx          # Provider, capability, search filters
│   ├── StatsBar.tsx           # Statistics bar
│   └── RefreshButton.tsx      # Manual refresh button
├── hooks/
│   └── useModels.ts           # Data fetching + state
├── types/
│   └── pricing.ts             # TypeScript type definitions
└── App.tsx                    # Main layout

providers.yaml                 # Simple provider configurations
```

## Data Models

```python
class Pricing(BaseModel):
    """Price info (USD per million tokens/units)"""
    input: float | None = None
    output: float | None = None
    cached_input: float | None = None
    cached_write: float | None = None
    reasoning: float | None = None
    image_input: float | None = None
    audio_input: float | None = None
    audio_output: float | None = None
    embedding: float | None = None

class BatchPricing(BaseModel):
    """Batch processing discounted prices"""
    input: float | None = None
    output: float | None = None

class ModelPricing(BaseModel):
    """Complete pricing info for a single model"""
    id: str                              # Unique: "{provider}:{model_id}"
    provider: str                        # aws_bedrock, openai, azure, etc.
    model_id: str                        # Original model ID
    model_name: str                      # Display name
    pricing: Pricing
    batch_pricing: BatchPricing | None = None
    context_length: int | None = None
    max_output_tokens: int | None = None
    capabilities: list[str] = []         # ["text", "vision", "audio", "embedding"]
    last_updated: datetime

class PricingDatabase(BaseModel):
    """JSON file root structure"""
    version: str = "1.0"
    last_refresh: datetime
    models: list[ModelPricing]
```

## Provider Abstraction

### Base Class (Plugin Pattern)

```python
class BaseProvider(ABC):
    name: str                    # "aws_bedrock"
    display_name: str            # "AWS Bedrock"

    @abstractmethod
    async def fetch(self) -> list[ModelPricing]:
        """Fetch all model prices from this provider"""
        pass
```

### Registry

```python
class ProviderRegistry:
    _providers: dict[str, BaseProvider] = {}

    @classmethod
    def register(cls, provider: BaseProvider):
        cls._providers[provider.name] = provider

    @classmethod
    async def fetch_all(cls) -> list[ModelPricing]:
        """Concurrently fetch from all providers"""
        results = await asyncio.gather(
            *[p.fetch() for p in cls._providers.values()],
            return_exceptions=True
        )
        # Merge results, skip failed providers
        ...
```

### Config-Driven Provider

```yaml
# providers.yaml
openrouter:
  display_name: "OpenRouter"
  api_url: "https://openrouter.ai/api/v1/models"
  auth: null
  mapping:
    model_id: "id"
    model_name: "name"
    input_price: "pricing.prompt"
    output_price: "pricing.completion"
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/models` | List models with filters (provider, capability, search, sort) |
| GET | `/api/models/{model_id}` | Get single model |
| GET | `/api/providers` | List all providers with stats |
| POST | `/api/refresh` | Manual refresh (optional: single provider) |
| GET | `/api/stats` | Overall statistics |

### Query Parameters for `/api/models`

- `provider`: Filter by provider name
- `capability`: Filter by capability (text, vision, audio, embedding)
- `search`: Search model name
- `sort_by`: Sort field (model_name, input, output, context_length)
- `sort_order`: asc / desc

## Frontend Components

### Table View

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Provider │ Model        │ Input  │ Output │ Cache  │ Context │ Caps     │
├──────────────────────────────────────────────────────────────────────────┤
│ AWS      │ Claude 3.5   │ $3.00  │ $15.00 │ $0.30  │ 200K    │ text img │
│ AWS      │ Llama 3.1    │ $0.80  │ $0.80  │ -      │ 128K    │ text     │
│ OpenAI   │ GPT-4o       │ $2.50  │ $10.00 │ $1.25  │ 128K    │ text img │
└──────────────────────────────────────────────────────────────────────────┘
```

### Card View

Current style with additions:
- Expand/collapse button for full price details
- Capability icons (vision, audio, embedding)

### State Management

```typescript
function useModels() {
  const [models, setModels] = useState<ModelPricing[]>([])
  const [filters, setFilters] = useState({ provider: null, capability: null, search: '' })
  const [view, setView] = useState<'table' | 'card'>('card')
  const [sortConfig, setSortConfig] = useState({ field: 'model_name', order: 'asc' })

  return { models, filters, setFilters, view, setView, refresh, loading }
}
```

## Error Handling

| Scenario | Handling |
|----------|----------|
| Provider API unavailable | Log error, keep old data, other providers update normally |
| JSON file missing | Create empty file on first startup |
| Data parsing failure | Skip entry, log error |
| Request during refresh | Return old data (atomic file replacement) |
| Missing price field | Frontend shows `-` or hides column |

## First Implementation: AWS Bedrock

Merge two data sources:
- `AmazonBedrock`: Nova, Llama, Gemma, Mistral, Qwen, DeepSeek, etc.
- `AmazonBedrockFoundationModels`: Claude, Cohere, AI21, Stability AI, etc.

API URLs:
```
https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/current/us-east-1/index.json
https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrockFoundationModels/current/us-east-1/index.json
```
