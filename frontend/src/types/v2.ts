// Frozen contract types for /api/v2/*.
// Mirrors backend/data/v2/fixtures/sample.json and
// docs/plans/v2-api-contract.md. Keep field names byte-for-byte.
// All prices are per 1,000,000 tokens.

export interface PricingV2 {
  input: number | null;
  output: number | null;
  cache_read: number | null;
  cache_write: number | null;
  image_input: number | null;
  audio_input: number | null;
  audio_output: number | null;
  embedding: number | null;
}

export interface BatchPricingV2 {
  input: number | null;
  output: number | null;
}

export type OfferingSource =
  | 'provider_api'      // Directly from the provider's own API / scrape
  | 'provider_scrape'   // Playwright scrape of the provider's pricing page
  | 'via_litellm'       // Mirror of provider's first-party pricing via LiteLLM
  | 'litellm_fallback'; // Placeholder when no real provider is attached

export interface OfferingV2 {
  provider: string;
  provider_model_id: string;
  pricing: PricingV2;
  batch_pricing: BatchPricingV2 | null;
  availability: string;
  region: string | null;
  notes: string | null;
  last_updated: string;
  source: OfferingSource;
}

export interface EntityCoreV2 {
  canonical_id: string;
  slug: string;
  name: string;
  family: string;
  maker: string;
  context_length: number | null;
  max_output_tokens: number | null;
  capabilities: string[];
  input_modalities: string[];
  output_modalities: string[];
  mode: string;
  is_open_source: boolean | null;
  primary_offering_provider: string;
  sources: string[];
  last_refreshed: string;
}

export interface AlternativeV2 {
  canonical_id: string;
  name: string;
  delta_input_pct: number;
  delta_output_pct: number;
  capability_overlap: number;
}

export interface EntityListItemV2 extends EntityCoreV2 {
  primary_offering: OfferingV2 | null;
}

export interface EntityDetailV2 {
  entity: EntityCoreV2;
  offerings: OfferingV2[];
  alternatives: AlternativeV2[];
}

export interface SearchResultV2 {
  canonical_id: string;
  slug: string;
  name: string;
  family: string | null;
  maker: string | null;
  primary_input_price: number | null;
  primary_output_price: number | null;
}

export interface CompareResultV2 {
  entities: EntityDetailV2[];
  common_capabilities: string[];
  requested_ids: string[];
  missing_ids: string[];
}

export interface StatsV2 {
  total_entities: number;
  total_offerings: number;
  makers: number;
  families: number;
  last_refresh: string;
  fixture?: boolean;
}

export interface DriftCounts {
  entities_total: number;
  entities_new: number;
  entities_removed: number;
  offerings_total: number;
  unmatched_provider_models: number;
  orphan_entities: number;
  price_drift_items: number;
}

export interface DriftReportV2 {
  generated_at: string;
  counts: DriftCounts;
  unmatched_provider_models: Array<{
    provider: string;
    model_id: string;
    tried_aliases: string[];
  }>;
  price_drift: Array<{
    entity: string;
    provider: string;
    field: string;
    provider_value: number;
    litellm_value: number;
    delta_pct: number;
  }>;
  new_entities: string[];
  removed_entities: string[];
  orphan_entities: string[];
  note?: string;
}

export interface EntitiesListQuery {
  q?: string;
  family?: string;
  maker?: string;
  capability?: string;
  min_context?: number;
  max_input_price?: number;
  sort?: 'name' | 'input' | 'output' | 'context';
  order?: 'asc' | 'desc';
}
