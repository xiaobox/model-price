import { useEffect, useRef, useState } from 'react';
import type { SearchResultV2 } from '../types/v2';
import { loadFallback, searchFallback } from '../v2/fallbackLoader';
import { searchEntityListCache } from '../v2/liveDataCache';

const DEBOUNCE_MS = 80;

interface State {
  results: SearchResultV2[];
  loading: boolean;
}

/**
 * Cmd+K search runs client-side against the last live list cache first,
 * then the bundled snapshot. It avoids network round trips in the exact
 * cold-start path where search needs to feel instant.
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
      const cachedResults = searchEntityListCache(trimmed, limit);
      if (tokenRef.current !== token) return;
      if (cachedResults) {
        setState({ results: cachedResults, loading: false });
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
