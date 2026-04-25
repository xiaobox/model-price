import { useCallback, useEffect, useState } from 'react';
import { API_V2_BASE } from '../config';
import type { EntitiesListQuery, EntityListItemV2 } from '../types/v2';
import { readApiCache, writeApiCache } from '../v2/apiResponseCache';
import { listFromFallback, loadFallback } from '../v2/fallbackLoader';

const BACKEND_TIMEOUT_MS = 15000;

interface State {
  entities: EntityListItemV2[];
  loading: boolean;
  error: string | null;
  fromFallback: boolean;
}

function buildQueryString(query: EntitiesListQuery): string {
  const params = new URLSearchParams();
  if (query.q) params.set('q', query.q);
  if (query.family) params.set('family', query.family);
  if (query.maker) params.set('maker', query.maker);
  if (query.capability) params.set('capability', query.capability);
  if (typeof query.min_context === 'number') {
    params.set('min_context', String(query.min_context));
  }
  if (typeof query.max_input_price === 'number') {
    params.set('max_input_price', String(query.max_input_price));
  }
  if (query.sort) params.set('sort', query.sort);
  if (query.order) params.set('order', query.order);
  return params.toString();
}

function cacheEntityListItems(items: EntityListItemV2[]): void {
  for (const item of items) {
    writeApiCache(`entity-list-item:${item.slug}`, item);
  }
}

export function useEntitiesV2(query: EntitiesListQuery): State & {
  refetch: () => Promise<void>;
} {
  const [state, setState] = useState<State>({
    entities: [],
    loading: true,
    error: null,
    fromFallback: false,
  });

  const queryString = buildQueryString(query);

  const fetchEntities = useCallback(async () => {
    // Stage 1: paint from the last successful live response if we have
    // one. That keeps yesterday's warmed backend data from regressing to
    // an older bundled snapshot on the next cold start.
    const cacheKey = `entities:${queryString || 'all'}`;
    let paintedFallback = false;
    const cachedList = readApiCache<EntityListItemV2[]>(cacheKey);
    if (cachedList) {
      cacheEntityListItems(cachedList);
      setState({
        entities: cachedList,
        loading: true,
        error: null,
        fromFallback: true,
      });
      paintedFallback = true;
    }

    // Stage 2: paint from the bundled snapshot when no live cache is
    // available. During a first-visit cold boot this is what keeps the
    // page interactive while Render wakes up.
    const snapshot = paintedFallback ? null : await loadFallback();
    if (snapshot) {
      const fallbackList = listFromFallback(snapshot, query);
      setState({
        entities: fallbackList,
        loading: true,
        error: null,
        fromFallback: true,
      });
      paintedFallback = true;
    } else {
      setState((prev) => ({ ...prev, loading: true, error: null }));
    }

    // Stage 3: fire the real backend request and swap in live data
    // when it arrives. If it times out or errors, keep the fallback
    // visible and swallow the error — the user still has content.
    const url = queryString
      ? `${API_V2_BASE}/entities?${queryString}`
      : `${API_V2_BASE}/entities`;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), BACKEND_TIMEOUT_MS);
    try {
      const response = await fetch(url, { signal: controller.signal });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = (await response.json()) as EntityListItemV2[];
      writeApiCache(cacheKey, data);
      if (!queryString) writeApiCache('entities:all', data);
      cacheEntityListItems(data);
      setState({
        entities: data,
        loading: false,
        error: null,
        fromFallback: false,
      });
    } catch (err) {
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
    // queryString is the serialized cache key.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queryString]);

  useEffect(() => {
    fetchEntities();
  }, [fetchEntities]);

  return { ...state, refetch: fetchEntities };
}
