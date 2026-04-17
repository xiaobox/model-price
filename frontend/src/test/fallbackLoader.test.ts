import { describe, expect, it } from 'vitest';
import {
  compareFromFallback,
  detailFromFallback,
  listFromFallback,
} from '../v2/fallbackLoader';
import type { EntityCoreV2, OfferingV2 } from '../types/v2';

function entity(overrides: Partial<EntityCoreV2> & { slug: string }): EntityCoreV2 {
  return {
    canonical_id: overrides.slug,
    slug: overrides.slug,
    name: overrides.slug,
    family: 'Claude',
    maker: 'Anthropic',
    context_length: 200_000,
    max_output_tokens: 64_000,
    capabilities: ['text'],
    input_modalities: ['text'],
    output_modalities: ['text'],
    mode: 'chat',
    is_open_source: false,
    primary_offering_provider: 'anthropic',
    sources: ['anthropic'],
    last_refreshed: '2026-04-17T00:00:00.000Z',
    ...overrides,
  };
}

function offering(
  provider: string,
  input: number | null,
  output: number | null,
): OfferingV2 {
  return {
    provider,
    provider_model_id: `${provider}:x`,
    pricing: {
      input,
      output,
      cache_read: null,
      cache_write: null,
      image_input: null,
      audio_input: null,
      audio_output: null,
      embedding: null,
    },
    batch_pricing: null,
    availability: 'ga',
    region: null,
    notes: null,
    last_updated: '2026-04-17T00:00:00.000Z',
    source: 'provider_api',
  };
}

function snapshot() {
  const opus = entity({
    slug: 'claude-opus-4-7',
    name: 'Claude Opus 4.7',
    capabilities: ['text', 'vision', 'reasoning'],
  });
  const sonnet = entity({
    slug: 'claude-sonnet-4-6',
    name: 'Claude Sonnet 4.6',
    capabilities: ['text', 'vision'],
  });
  const gpt = entity({
    slug: 'gpt-5',
    name: 'GPT-5',
    family: 'GPT',
    maker: 'OpenAI',
    primary_offering_provider: 'openai',
    sources: ['openai'],
    capabilities: ['text', 'function_calling'],
  });
  return {
    version: 'test',
    generated_at: '2026-04-17T00:00:00.000Z',
    entity_count: 3,
    source_last_refresh: '2026-04-17T00:00:00.000Z',
    entities: [opus, sonnet, gpt],
    offerings_by_entity: {
      'claude-opus-4-7': [offering('anthropic', 5, 25)],
      'claude-sonnet-4-6': [offering('anthropic', 3, 15)],
      'gpt-5': [offering('openai', 2, 10)],
    },
    alternatives_by_entity: {
      'claude-opus-4-7': [],
      'claude-sonnet-4-6': [],
      'gpt-5': [],
    },
  };
}

describe('detailFromFallback', () => {
  it('returns entity with offerings and alternatives for a known slug', () => {
    const detail = detailFromFallback(snapshot(), 'claude-opus-4-7');
    expect(detail).not.toBeNull();
    expect(detail!.entity.name).toBe('Claude Opus 4.7');
    expect(detail!.offerings).toHaveLength(1);
    expect(detail!.alternatives).toEqual([]);
  });

  it('returns null for an unknown slug', () => {
    expect(detailFromFallback(snapshot(), 'nonexistent')).toBeNull();
  });

  it('returns empty offerings when the snapshot has none for the slug', () => {
    const snap = snapshot();
    // @ts-expect-error test covers a degraded snapshot where offerings are missing
    delete snap.offerings_by_entity['claude-opus-4-7'];
    const detail = detailFromFallback(snap, 'claude-opus-4-7');
    expect(detail!.offerings).toEqual([]);
  });
});

describe('listFromFallback', () => {
  it('returns every entity when query is empty', () => {
    const list = listFromFallback(snapshot(), {});
    expect(list.map((e) => e.slug)).toContain('claude-opus-4-7');
    expect(list.map((e) => e.slug)).toContain('gpt-5');
  });

  it('filters by maker', () => {
    const list = listFromFallback(snapshot(), { maker: 'Anthropic' });
    expect(list.map((e) => e.slug).sort()).toEqual([
      'claude-opus-4-7',
      'claude-sonnet-4-6',
    ]);
  });

  it('filters by capability', () => {
    const list = listFromFallback(snapshot(), { capability: 'reasoning' });
    expect(list.map((e) => e.slug)).toEqual(['claude-opus-4-7']);
  });

  it('filters by q against name / canonical / family', () => {
    expect(
      listFromFallback(snapshot(), { q: 'opus' }).map((e) => e.slug),
    ).toEqual(['claude-opus-4-7']);
    expect(listFromFallback(snapshot(), { q: 'GPT' })).toHaveLength(1);
  });

  it('filters by max_input_price using primary offering', () => {
    const list = listFromFallback(snapshot(), { max_input_price: 3 });
    // Sonnet input=3, GPT-5 input=2. Opus input=5 excluded.
    expect(list.map((e) => e.slug).sort()).toEqual(['claude-sonnet-4-6', 'gpt-5']);
  });

  it('sorts by input price asc by default when sort=input', () => {
    const list = listFromFallback(snapshot(), { sort: 'input' });
    expect(list.map((e) => e.slug)).toEqual([
      'gpt-5',
      'claude-sonnet-4-6',
      'claude-opus-4-7',
    ]);
  });

  it('sorts desc when order=desc', () => {
    const list = listFromFallback(snapshot(), { sort: 'input', order: 'desc' });
    expect(list.map((e) => e.slug)).toEqual([
      'claude-opus-4-7',
      'claude-sonnet-4-6',
      'gpt-5',
    ]);
  });
});

describe('compareFromFallback', () => {
  it('returns entities in the same order as ids when all are present', () => {
    const result = compareFromFallback(snapshot(), ['gpt-5', 'claude-opus-4-7']);
    expect(result.entities.map((e) => e.entity.slug)).toEqual([
      'gpt-5',
      'claude-opus-4-7',
    ]);
    expect(result.missing_ids).toEqual([]);
    expect(result.requested_ids).toEqual(['gpt-5', 'claude-opus-4-7']);
  });

  it('puts unknown ids in missing_ids and skips them from entities', () => {
    const result = compareFromFallback(snapshot(), [
      'claude-opus-4-7',
      'ghost-model',
      'gpt-5',
    ]);
    expect(result.entities.map((e) => e.entity.slug)).toEqual([
      'claude-opus-4-7',
      'gpt-5',
    ]);
    expect(result.missing_ids).toEqual(['ghost-model']);
  });

  it('returns empty result for empty ids', () => {
    const result = compareFromFallback(snapshot(), []);
    expect(result.entities).toEqual([]);
    expect(result.missing_ids).toEqual([]);
    expect(result.common_capabilities).toEqual([]);
  });

  it('returns empty result when every id is unknown', () => {
    const result = compareFromFallback(snapshot(), ['a', 'b']);
    expect(result.entities).toEqual([]);
    expect(result.missing_ids).toEqual(['a', 'b']);
    expect(result.common_capabilities).toEqual([]);
  });

  it('intersects capabilities across compared entities', () => {
    // opus: [text, vision, reasoning]
    // sonnet: [text, vision]
    // gpt: [text, function_calling]
    const both = compareFromFallback(snapshot(), [
      'claude-opus-4-7',
      'claude-sonnet-4-6',
    ]);
    expect(both.common_capabilities).toEqual(['text', 'vision']);

    const all = compareFromFallback(snapshot(), [
      'claude-opus-4-7',
      'claude-sonnet-4-6',
      'gpt-5',
    ]);
    expect(all.common_capabilities).toEqual(['text']);
  });

  it('trims whitespace and drops empty ids', () => {
    const result = compareFromFallback(snapshot(), [
      ' claude-opus-4-7 ',
      '',
      '  ',
    ]);
    expect(result.requested_ids).toEqual(['claude-opus-4-7']);
    expect(result.entities).toHaveLength(1);
  });
});
