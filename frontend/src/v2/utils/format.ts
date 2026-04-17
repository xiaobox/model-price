// Shared display formatters for v2 UI.

const MAKER_PALETTE: Record<string, string> = {
  Anthropic: 'var(--maker-anthropic)',
  OpenAI: 'var(--maker-openai)',
  Google: 'var(--maker-google)',
  xAI: 'var(--maker-xai)',
  DeepSeek: 'var(--maker-deepseek)',
  Meta: 'var(--maker-meta)',
  Mistral: 'var(--maker-mistral)',
  Cohere: 'var(--maker-cohere)',
};

export function makerColor(maker: string | null | undefined): string {
  if (!maker) return 'var(--maker-default)';
  return MAKER_PALETTE[maker] ?? 'var(--maker-default)';
}

export function formatPrice(price: number | null | undefined): string {
  if (price === null || price === undefined) return '—';
  if (price === 0) return '$0';
  if (price < 0.01) return `$${price.toFixed(4)}`;
  if (price < 1) return `$${price.toFixed(3)}`;
  return `$${price.toFixed(2)}`;
}

export function formatContext(ctx: number | null | undefined): string {
  if (ctx === null || ctx === undefined || ctx === 0) return '—';
  if (ctx >= 1_000_000) return `${(ctx / 1_000_000).toFixed(ctx % 1_000_000 === 0 ? 0 : 1)}M`;
  if (ctx >= 1_000) return `${Math.round(ctx / 1_000)}K`;
  return String(ctx);
}

export function formatPct(delta: number): string {
  const sign = delta > 0 ? '+' : '';
  return `${sign}${delta.toFixed(0)}%`;
}

export function formatOverlap(overlap: number): string {
  return `${Math.round(overlap * 100)}%`;
}

const CAP_LABEL: Record<string, string> = {
  text: 'Text',
  vision: 'Vision',
  audio: 'Audio',
  tool_use: 'Tools',
  reasoning: 'Reasoning',
  function_calling: 'Functions',
  image_generation: 'Images',
  embedding: 'Embedding',
};

export function capabilityLabel(cap: string): string {
  return CAP_LABEL[cap] ?? cap;
}

const PROVIDER_LABEL: Record<string, string> = {
  // First-party maker-operators (pricing mirrored via LiteLLM, except
  // google_gemini which scrapes ai.google.dev directly).
  ai21: 'AI21',
  alibaba_qwen: 'Alibaba Qwen',
  anthropic: 'Anthropic',
  volcengine: 'Volcengine',
  // Retained for older cached snapshots; current slug is `volcengine`.
  bytedance_doubao: 'Volcengine',
  cohere: 'Cohere',
  deepseek: 'DeepSeek',
  gigachat: 'Sber GigaChat',
  google_gemini: 'Google Gemini',
  google_vertex_ai: 'Google Vertex AI',
  meta_llama: 'Meta Llama API',
  minimax: 'MiniMax',
  mistral: 'Mistral AI',
  moonshot: 'Moonshot AI',
  openai: 'OpenAI',
  xai: 'xAI',
  zai: 'Z.AI',
  // Cloud distribution channels.
  aws_bedrock: 'AWS Bedrock',
  azure_ai: 'Azure AI',
  openrouter: 'OpenRouter',
  // Retained for older cached snapshots; current pipeline emits
  // `azure_ai` for every Azure-hosted model (formerly `azure_openai`
  // and briefly `azure_ai_foundry`).
  azure_openai: 'Azure AI',
  azure_ai_foundry: 'Azure AI',
  // Placeholder when no real provider attaches.
  litellm_fallback: 'LiteLLM fallback',
  litellm: 'LiteLLM',
};

export function providerLabel(provider: string): string {
  return PROVIDER_LABEL[provider] ?? provider;
}
