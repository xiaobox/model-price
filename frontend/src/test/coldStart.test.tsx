import { render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import type { ReactNode } from 'react';

import { EntityPage } from '../v2/pages/EntityPage';
import { ComparePage } from '../v2/pages/ComparePage';
import { EntityDrawer } from '../v2/components/EntityDrawer';
import { CompareBasketProvider } from '../v2/compareBasketContext';
import { LocaleProvider } from '../v2/i18n/localeContext';
import { resetFallbackCacheForTests } from '../v2/fallbackLoader';

// Minimal snapshot covering the two slugs the tests reference below.
// Shape mirrors public/v2-fallback.json so loadFallback() parses it.
const SNAPSHOT = {
  version: 'test',
  generated_at: '2026-04-17T00:00:00.000Z',
  entity_count: 2,
  source_last_refresh: '2026-04-17T00:00:00.000Z',
  entities: [
    {
      canonical_id: 'claude-opus-4-7',
      slug: 'claude-opus-4-7',
      name: 'Claude Opus 4.7',
      family: 'Claude',
      maker: 'Anthropic',
      context_length: 1_000_000,
      max_output_tokens: 128_000,
      capabilities: ['text', 'vision', 'reasoning'],
      input_modalities: ['text', 'image'],
      output_modalities: ['text'],
      mode: 'chat',
      is_open_source: false,
      primary_offering_provider: 'anthropic',
      sources: ['anthropic'],
      last_refreshed: '2026-04-17T00:00:00.000Z',
    },
    {
      canonical_id: 'gpt-5',
      slug: 'gpt-5',
      name: 'GPT-5',
      family: 'GPT',
      maker: 'OpenAI',
      context_length: 400_000,
      max_output_tokens: 100_000,
      capabilities: ['text', 'vision'],
      input_modalities: ['text', 'image'],
      output_modalities: ['text'],
      mode: 'chat',
      is_open_source: false,
      primary_offering_provider: 'openai',
      sources: ['openai'],
      last_refreshed: '2026-04-17T00:00:00.000Z',
    },
  ],
  offerings_by_entity: {
    'claude-opus-4-7': [
      {
        provider: 'anthropic',
        provider_model_id: 'claude-opus-4-7',
        pricing: {
          input: 5.0,
          output: 25.0,
          cache_read: 0.5,
          cache_write: 6.25,
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
      },
    ],
    'gpt-5': [
      {
        provider: 'openai',
        provider_model_id: 'gpt-5',
        pricing: {
          input: 2.0,
          output: 10.0,
          cache_read: 0.2,
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
      },
    ],
  },
  alternatives_by_entity: {
    'claude-opus-4-7': [],
    'gpt-5': [],
  },
};

type BackendHandler = (url: string) => Response | Promise<Response> | null;

// Install a fetch stub that always serves the bundled snapshot
// instantly, and delegates every backend call to `backend`. Returning
// null means "stall forever" — the Render free-tier cold-boot case.
function installFetch(backend: BackendHandler = () => null) {
  const fetchMock = vi.fn((input: RequestInfo | URL) => {
    const url = typeof input === 'string' ? input : input.toString();
    if (url.endsWith('/v2-fallback.json')) {
      return Promise.resolve(
        new Response(JSON.stringify(SNAPSHOT), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        }),
      );
    }
    const result = backend(url);
    if (result === null) {
      return new Promise<Response>(() => {
        /* never resolves */
      });
    }
    return Promise.resolve(result);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

function renderWithProviders(children: ReactNode, initialPath: string, routePattern?: string) {
  return render(
    <MemoryRouter initialEntries={[initialPath]}>
      <LocaleProvider>
        <CompareBasketProvider>
          {routePattern ? (
            <Routes>
              <Route path={routePattern} element={children} />
            </Routes>
          ) : (
            children
          )}
        </CompareBasketProvider>
      </LocaleProvider>
    </MemoryRouter>,
  );
}

describe('cold-start snapshot paint', () => {
  beforeEach(() => {
    resetFallbackCacheForTests();
    installFetch();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('EntityPage shows the model name immediately from snapshot even while the backend stalls', async () => {
    renderWithProviders(<EntityPage />, '/m/claude-opus-4-7', '/m/:slug');

    expect(await screen.findByRole('heading', { level: 1, name: /Claude Opus 4\.7/ })).toBeInTheDocument();
    // Pricing from the snapshot is visible without waiting for the backend.
    expect(screen.getByText('$5.00')).toBeInTheDocument();
    expect(screen.getByText('$25.00')).toBeInTheDocument();
  });

  it('EntityDrawer shows the model name immediately from snapshot even while the backend stalls', async () => {
    renderWithProviders(
      <EntityDrawer
        slug="gpt-5"
        onClose={() => {}}
        isInBasket={() => false}
        onToggleBasket={() => {}}
        onNavigateSlug={() => {}}
      />,
      '/',
    );

    expect(await screen.findByRole('heading', { level: 2, name: /GPT-5/ })).toBeInTheDocument();
    expect(screen.getByText('$2.00')).toBeInTheDocument();
    expect(screen.getByText('$10.00')).toBeInTheDocument();
  });

  it('ComparePage shows both model names immediately from snapshot even while the backend stalls', async () => {
    renderWithProviders(
      <ComparePage />,
      '/compare/claude-opus-4-7,gpt-5',
      '/compare/:ids',
    );

    expect(await screen.findByRole('link', { name: /Claude Opus 4\.7/ })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /GPT-5/ })).toBeInTheDocument();
  });
});

describe('cold-start negative paths', () => {
  beforeEach(() => {
    resetFallbackCacheForTests();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('EntityPage shows "not found" when the slug is absent from snapshot and backend returns 404', async () => {
    installFetch(() => new Response('{}', { status: 404 }));
    renderWithProviders(<EntityPage />, '/m/ghost-model', '/m/:slug');

    expect(
      await screen.findByText(/Model "ghost-model" not found/),
    ).toBeInTheDocument();
  });

  it('EntityPage swaps snapshot price with backend response when the live fetch succeeds', async () => {
    installFetch((url) => {
      if (url.includes('/entities/claude-opus-4-7')) {
        return new Response(
          JSON.stringify({
            entity: SNAPSHOT.entities[0],
            offerings: [
              {
                ...SNAPSHOT.offerings_by_entity['claude-opus-4-7'][0],
                pricing: {
                  ...SNAPSHOT.offerings_by_entity['claude-opus-4-7'][0].pricing,
                  input: 6.0,
                  output: 30.0,
                },
              },
            ],
            alternatives: [],
          }),
          { status: 200, headers: { 'content-type': 'application/json' } },
        );
      }
      return null;
    });
    renderWithProviders(<EntityPage />, '/m/claude-opus-4-7', '/m/:slug');

    // Backend-updated price wins the final render.
    expect(await screen.findByText('$6.00')).toBeInTheDocument();
    expect(await screen.findByText('$30.00')).toBeInTheDocument();
    // Snapshot values are no longer visible.
    expect(screen.queryByText('$5.00')).not.toBeInTheDocument();
    expect(screen.queryByText('$25.00')).not.toBeInTheDocument();
  });

  it('ComparePage paints snapshot hits immediately even when some ids are missing', async () => {
    installFetch(); // backend stalls — snapshot must carry the page on its own
    renderWithProviders(
      <ComparePage />,
      '/compare/claude-opus-4-7,ghost-model,gpt-5',
      '/compare/:ids',
    );

    // The two snapshot-known models render their titles.
    expect(
      await screen.findByRole('link', { name: /Claude Opus 4\.7/ }),
    ).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /GPT-5/ })).toBeInTheDocument();
    // The unknown id surfaces in the "missing" notice.
    expect(screen.getByText(/ghost-model/)).toBeInTheDocument();
  });
});
