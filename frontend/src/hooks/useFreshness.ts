import { useEffect, useState } from 'react';
import { API_V2_BASE } from '../config';
import { readApiCache, writeApiCache } from '../v2/apiResponseCache';
import { loadFallback } from '../v2/fallbackLoader';

interface State {
  lastRefresh: string | null;
}

// Exposes the newest known `last_refresh` timestamp. The hook paints from
// the last live stats response first, then falls back to the bundled
// snapshot, and finally swaps in /api/v2/stats when the backend responds.
export function useFreshness(): State {
  const [lastRefresh, setLastRefresh] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const cached = readApiCache<{ last_refresh?: string }>('stats');
      if (cached?.last_refresh) setLastRefresh(cached.last_refresh);

      const snapshot = cached?.last_refresh ? null : await loadFallback();
      if (cancelled) return;
      if (snapshot?.generated_at) setLastRefresh(snapshot.generated_at);

      try {
        const res = await fetch(`${API_V2_BASE}/stats`);
        if (cancelled || !res.ok) return;
        const data = (await res.json()) as { last_refresh?: string };
        if (data?.last_refresh) {
          writeApiCache('stats', data);
          setLastRefresh(data.last_refresh);
        }
      } catch {
        // Keep the snapshot value — the topbar stays populated even if
        // the backend is unreachable.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return { lastRefresh };
}
