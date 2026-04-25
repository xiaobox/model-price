import { useCallback, useEffect, useMemo, useState } from 'react';
import { API_V2_BASE } from '../config';
import type { EntitiesListQuery, EntityListItemV2 } from '../types/v2';
import { listFromFallback, loadFallback } from '../v2/fallbackLoader';
import {
  readEntityListCache,
  writeEntityListCache,
} from '../v2/liveDataCache';

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
  if (query.sort && query.sort !== 'name') params.set('sort', query.sort);
  if (query.order && query.order !== 'asc') params.set('order', query.order);
  return params.toString();
}

function queryFromString(queryString: string): EntitiesListQuery {
  if (!queryString) return {};
  const params = new URLSearchParams(queryString);
  const query: EntitiesListQuery = {};
  const minContext = params.get('min_context');
  const maxInputPrice = params.get('max_input_price');
  const sort = params.get('sort');
  const order = params.get('order');

  query.q = params.get('q') ?? undefined;
  query.family = params.get('family') ?? undefined;
  query.maker = params.get('maker') ?? undefined;
  query.capability = params.get('capability') ?? undefined;
  if (minContext) query.min_context = Number(minContext);
  if (maxInputPrice) query.max_input_price = Number(maxInputPrice);
  if (sort === 'input' || sort === 'output' || sort === 'context') query.sort = sort;
  if (order === 'desc') query.order = order;
  return query;
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
  const fallbackQuery = useMemo(() => queryFromString(queryString), [queryString]);

  const fetchEntities = useCallback(async () => {
    // Stage 1: paint from the last successful live response if we have
    // one. That keeps yesterday's warmed backend data from regressing to
    // an older bundled snapshot on the next cold start.
    let paintedFallback = false;
    const cachedList = readEntityListCache(queryString);
    if (cachedList) {
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
      const fallbackList = listFromFallback(snapshot, fallbackQuery);
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
      writeEntityListCache(queryString, data);
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
  }, [fallbackQuery, queryString]);

  useEffect(() => {
    fetchEntities();
  }, [fetchEntities]);

  return { ...state, refetch: fetchEntities };
}
