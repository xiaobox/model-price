import { useEffect, useState } from 'react';
import { API_V2_BASE } from '../config';
import type { EntityDetailV2, EntityListItemV2 } from '../types/v2';
import {
  readApiCache,
  removeApiCache,
  writeApiCache,
} from '../v2/apiResponseCache';
import { detailFromFallback, loadFallback } from '../v2/fallbackLoader';

const BACKEND_TIMEOUT_MS = 15000;

interface State {
  detail: EntityDetailV2 | null;
  loading: boolean;
  error: string | null;
  notFound: boolean;
  fromFallback: boolean;
}

function detailFromListItem(item: EntityListItemV2): EntityDetailV2 {
  return {
    entity: {
      canonical_id: item.canonical_id,
      slug: item.slug,
      name: item.name,
      family: item.family,
      maker: item.maker,
      context_length: item.context_length,
      max_output_tokens: item.max_output_tokens,
      capabilities: item.capabilities,
      input_modalities: item.input_modalities,
      output_modalities: item.output_modalities,
      mode: item.mode,
      is_open_source: item.is_open_source,
      primary_offering_provider: item.primary_offering_provider,
      sources: item.sources,
      last_refreshed: item.last_refreshed,
    },
    offerings: item.primary_offering ? [item.primary_offering] : [],
    alternatives: [],
  };
}

function readCachedListDetail(slug: string): EntityDetailV2 | null {
  const item = readApiCache<EntityListItemV2>(`entity-list-item:${slug}`);
  if (item) return detailFromListItem(item);

  const all = readApiCache<EntityListItemV2[]>('entities:all');
  const fromAll = all?.find((entity) => entity.slug === slug);
  return fromAll ? detailFromListItem(fromAll) : null;
}

export function useEntityV2(slug: string | null | undefined): State {
  const [state, setState] = useState<State>({
    detail: null,
    loading: Boolean(slug),
    error: null,
    notFound: false,
    fromFallback: false,
  });

  useEffect(() => {
    if (!slug) {
      setState({
        detail: null,
        loading: false,
        error: null,
        notFound: false,
        fromFallback: false,
      });
      return;
    }

    let cancelled = false;
    setState({
      detail: null,
      loading: true,
      error: null,
      notFound: false,
      fromFallback: false,
    });

    (async () => {
      // Stage 1: prefer the last live response for this model. It is
      // usually newer than the bundled snapshot after a previous session
      // successfully warmed the backend.
      const cacheKey = `entity:${slug}`;
      const cachedDetail = readApiCache<EntityDetailV2>(cacheKey);
      if (cancelled) return;
      let paintedFallback = false;
      if (cachedDetail) {
        setState({
          detail: cachedDetail,
          loading: true,
          error: null,
          notFound: false,
          fromFallback: true,
        });
        paintedFallback = true;
      }

      const listDetail = paintedFallback ? null : readCachedListDetail(slug);
      if (listDetail) {
        setState({
          detail: listDetail,
          loading: true,
          error: null,
          notFound: false,
          fromFallback: true,
        });
        paintedFallback = true;
      }

      // Stage 2: snapshot paint when there is no newer live cache.
      const snapshot = paintedFallback ? null : await loadFallback();
      if (cancelled) return;
      if (snapshot) {
        const fallback = detailFromFallback(snapshot, slug);
        if (fallback) {
          setState({
            detail: fallback,
            loading: true,
            error: null,
            notFound: false,
            fromFallback: true,
          });
          paintedFallback = true;
        }
      }

      // Stage 3: real backend fetch. Swap in live data if it arrives;
      // leave fallback/cache content in place on timeout or transient
      // errors. A 404 is authoritative, so it clears stale cached detail.
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), BACKEND_TIMEOUT_MS);
      try {
        const response = await fetch(
          `${API_V2_BASE}/entities/${encodeURIComponent(slug)}`,
          { signal: controller.signal },
        );
        if (cancelled) return;
        if (response.status === 404) {
          removeApiCache(cacheKey);
          setState({
            detail: null,
            loading: false,
            error: null,
            notFound: true,
            fromFallback: false,
          });
          return;
        }
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = (await response.json()) as EntityDetailV2;
        writeApiCache(cacheKey, data);
        setState({
          detail: data,
          loading: false,
          error: null,
          notFound: false,
          fromFallback: false,
        });
      } catch (err) {
        if (cancelled) return;
        setState((prev) => ({
          ...prev,
          loading: false,
          error: paintedFallback
            ? null
            : err instanceof Error
              ? err.message
              : 'fetch failed',
        }));
      } finally {
        clearTimeout(timeout);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [slug]);

  return state;
}
