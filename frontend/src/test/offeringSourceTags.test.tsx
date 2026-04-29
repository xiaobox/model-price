/**
 * The offerings table must label each pricing row by its data-source
 * honesty level so users are never misled about where a number comes
 * from.
 *
 *   - `source: 'provider_api'` / `'provider_scrape'` → no tag
 *   - `source: 'via_litellm'`                        → "via LiteLLM"
 *   - `source: 'litellm_fallback'`                   → "placeholder"
 *     (the entity has no real provider attached; the row is grayed out)
 */

import { render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter, Route, Routes } from 'react-router-dom';

import { EntityPage } from '../v2/pages/EntityPage';
import { CompareBasketProvider } from '../v2/compareBasketContext';
import { LocaleProvider } from '../v2/i18n/localeContext';
import { resetFallbackCacheForTests } from '../v2/fallbackLoader';
import type { EntityDetailV2, OfferingSource } from '../types/v2';

function buildDetail(sources: OfferingSource[]): EntityDetailV2 {
  const slug = 'test-model';
  const entity = {
    canonical_id: slug,
    slug,
    name: 'Test Model',
    family: 'Test',
    maker: 'Anthropic',
    context_length: 100_000,
    max_output_tokens: 8_000,
    capabilities: ['text'],
    input_modalities: ['text'],
    output_modalities: ['text'],
    mode: 'chat',
    is_open_source: false,
    primary_offering_provider: 'anthropic',
    sources: ['anthropic'],
    last_refreshed: '2026-04-17T00:00:00.000Z',
  };
  const offerings = sources.map((source, i) => ({
    provider: ['anthropic', 'aws_bedrock', 'litellm'][i] ?? `provider-${i}`,
    provider_model_id: `${slug}-${i}`,
    pricing: {
      input: 5,
      output: 25,
      cache_read: 0.5,
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
    source,
  }));
  return { entity, offerings, alternatives: [] };
}

function installFetchReturning(detail: EntityDetailV2) {
  vi.stubGlobal(
    'fetch',
    vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.endsWith('/api/v2/snapshot') || url.endsWith('/v2-fallback.json')) {
        // Empty snapshot — the backend response is what the page renders.
        return Promise.resolve(
          new Response(
            JSON.stringify({
              version: 'test',
              generated_at: '2026-04-17T00:00:00.000Z',
              entity_count: 0,
              source_last_refresh: null,
              entities: [],
              offerings_by_entity: {},
              alternatives_by_entity: {},
            }),
            { status: 200, headers: { 'content-type': 'application/json' } },
          ),
        );
      }
      if (url.includes('/entities/test-model')) {
        return Promise.resolve(
          new Response(JSON.stringify(detail), {
            status: 200,
            headers: { 'content-type': 'application/json' },
          }),
        );
      }
      return new Promise<Response>(() => {});
    }),
  );
}

function renderPage() {
  return render(
    <MemoryRouter initialEntries={['/m/test-model']}>
      <LocaleProvider>
        <CompareBasketProvider>
          <Routes>
            <Route path="/m/:slug" element={<EntityPage />} />
          </Routes>
        </CompareBasketProvider>
      </LocaleProvider>
    </MemoryRouter>,
  );
}

describe('offering source honesty labels', () => {
  beforeEach(() => {
    resetFallbackCacheForTests();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('does not tag direct provider_api rows', async () => {
    installFetchReturning(buildDetail(['provider_api']));
    renderPage();
    // Page hydrates once the backend call resolves.
    await screen.findByRole('heading', { level: 1, name: /Test Model/ });
    expect(screen.queryByText('via LiteLLM')).not.toBeInTheDocument();
    expect(screen.queryByText('placeholder')).not.toBeInTheDocument();
  });

  it('tags via_litellm rows so users know it is a mirror', async () => {
    installFetchReturning(buildDetail(['via_litellm']));
    renderPage();
    expect(await screen.findByText('via LiteLLM')).toBeInTheDocument();
  });

  it('marks litellm_fallback rows with the is-placeholder class (no text tag)', async () => {
    installFetchReturning(buildDetail(['litellm_fallback']));
    renderPage();
    // Wait for the hydrated page, then assert the row style — the
    // "placeholder" / "占位" text tag was removed in the name-polish
    // pass because the gray row style is enough visual signal.
    const title = await screen.findByRole('heading', {
      level: 1,
      name: /Test Model/,
    });
    expect(title).toBeInTheDocument();
    expect(screen.queryByText('placeholder')).not.toBeInTheDocument();
    expect(screen.queryByText('占位')).not.toBeInTheDocument();
    // The row itself keeps the .is-placeholder class so CSS grays it.
    const row = document.querySelector('.v2-offer.is-placeholder');
    expect(row).not.toBeNull();
  });

  it('labels mixed rows independently (direct + mirror, placeholder gets no tag)', async () => {
    installFetchReturning(
      buildDetail(['provider_api', 'via_litellm', 'litellm_fallback']),
    );
    renderPage();
    await screen.findByRole('heading', { level: 1, name: /Test Model/ });
    expect(screen.getByText('via LiteLLM')).toBeInTheDocument();
    // Only one of each meaningful tag — direct rows carry nothing,
    // placeholder rows no longer carry a text tag either.
    expect(screen.getAllByText('via LiteLLM')).toHaveLength(1);
    expect(screen.queryByText('placeholder')).not.toBeInTheDocument();
    // The placeholder row's class still marks it for graying.
    const placeholderRow = document.querySelector('.v2-offer.is-placeholder');
    expect(placeholderRow).not.toBeNull();
  });
});
