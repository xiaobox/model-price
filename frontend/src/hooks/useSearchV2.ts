import { useEffect, useRef, useState } from 'react';
import type { EntityListItemV2, SearchResultV2 } from '../types/v2';
import { readApiCache } from '../v2/apiResponseCache';
import { loadFallback, searchFallback } from '../v2/fallbackLoader';

const DEBOUNCE_MS = 80;

interface State {
  results: SearchResultV2[];
  loading: boolean;
}

function searchCachedEntities(
  entities: EntityListItemV2[],
  query: string,
  limit: number,
): SearchResultV2[] {
  const ql = query.toLowerCase().trim();
  if (!ql) return [];
  const scored: Array<[number, SearchResultV2]> = [];
  for (const entity of entities) {
    const name = (entity.name ?? '').toLowerCase();
    const canon = (entity.canonical_id ?? '').toLowerCase();
    const family = (entity.family ?? '').toLowerCase();
    let rank: number;
    if (name === ql || canon === ql) rank = 0;
    else if (name.startsWith(ql) || canon.startsWith(ql)) rank = 1;
    else if (name.includes(ql) || canon.includes(ql)) rank = 2;
    else if (family.includes(ql)) rank = 3;
    else continue;
    scored.push([
      rank,
      {
        canonical_id: entity.canonical_id,
        slug: entity.slug,
        name: entity.name,
        family: entity.family ?? null,
        maker: entity.maker ?? null,
        primary_input_price: entity.primary_offering?.pricing?.input ?? null,
        primary_output_price: entity.primary_offering?.pricing?.output ?? null,
      },
    ]);
  }
  scored.sort((a, b) => {
    if (a[0] !== b[0]) return a[0] - b[0];
    return a[1].name.toLowerCase().localeCompare(b[1].name.toLowerCase());
  });
  return scored.slice(0, limit).map((pair) => pair[1]);
}

/**
 * Cmd+K search runs entirely client-side against the v2 fallback
 * snapshot. At ~650 entities substring matching is sub-millisecond
 * per keystroke, and it avoids a network round-trip — which matters
 * most when the Render free-tier backend is cold. The backend
 * /api/v2/search endpoint remains part of the public contract for
 * API consumers, but the UI no longer calls it.
 */
export function useSearchV2(query: string, limit = 10): State {
  const [state, setState] = useState<State>({ results: [], loading: false });
  const tokenRef = useRef(0);

  useEffect(() => {
    const trimmed = query.trim();
    const token = ++tokenRef.current;

    const timeout = setTimeout(async () => {
      if (!trimmed) {
        setState({ results: [], loading: false });
        return;
      }

      setState((prev) => ({ ...prev, loading: true }));
      const cachedEntities =
        readApiCache<EntityListItemV2[]>('entities:all') ??
        readApiCache<EntityListItemV2[]>('entities:sort=name&order=asc');
      if (tokenRef.current !== token) return;
      if (cachedEntities) {
        setState({
          results: searchCachedEntities(cachedEntities, trimmed, limit),
          loading: false,
        });
        return;
      }

      const snapshot = await loadFallback();
      if (tokenRef.current !== token) return;
      if (!snapshot) {
        setState({ results: [], loading: false });
        return;
      }
      const results = searchFallback(snapshot, trimmed, limit);
      setState({ results, loading: false });
    }, trimmed ? DEBOUNCE_MS : 0);

    return () => clearTimeout(timeout);
  }, [query, limit]);

  return state;
}
