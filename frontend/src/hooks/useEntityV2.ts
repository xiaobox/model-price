import { useEffect, useState } from 'react';
import { API_V2_BASE } from '../config';
import type { EntityDetailV2 } from '../types/v2';
import {
  readEntityDetailCache,
  removeEntityDetailCache,
  writeEntityDetailCache,
} from '../v2/liveDataCache';
import { detailFromFallback, loadFallback } from '../v2/fallbackLoader';

const BACKEND_TIMEOUT_MS = 15000;

interface State {
  detail: EntityDetailV2 | null;
  loading: boolean;
  error: string | null;
  notFound: boolean;
  fromFallback: boolean;
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
      const cachedDetail = readEntityDetailCache(slug);
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
          removeEntityDetailCache(slug);
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
        writeEntityDetailCache(slug, data);
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
