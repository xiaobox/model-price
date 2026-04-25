import { useEffect, useState } from 'react';
import { API_V2_BASE } from '../config';
import { readApiCache, writeApiCache } from '../v2/apiResponseCache';
import { loadFallback } from '../v2/fallbackLoader';

interface State {
  lastRefresh: string | null;
}

// Exposes the backend's `last_refresh` timestamp so the UI can show
// users when the pricing snapshot was built. Paints from the bundled
// snapshot first (so the stamp is visible before the cold backend
// replies), then swaps in the live value from /api/v2/stats when it
// arrives.
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
