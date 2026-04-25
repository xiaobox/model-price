const CACHE_PREFIX = 'model-price-v2:api-cache:';
const MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000;

interface CacheEntry<T> {
  saved_at: string;
  data: T;
}

function getStorage(): Storage | null {
  if (typeof window === 'undefined') return null;
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

export function readApiCache<T>(key: string): T | null {
  const storage = getStorage();
  if (!storage) return null;

  try {
    const raw = storage.getItem(`${CACHE_PREFIX}${key}`);
    if (!raw) return null;
    const entry = JSON.parse(raw) as CacheEntry<T>;
    const savedAt = Date.parse(entry.saved_at);
    if (!Number.isFinite(savedAt) || Date.now() - savedAt > MAX_AGE_MS) {
      storage.removeItem(`${CACHE_PREFIX}${key}`);
      return null;
    }
    return entry.data ?? null;
  } catch {
    return null;
  }
}

export function writeApiCache<T>(key: string, data: T): void {
  const storage = getStorage();
  if (!storage) return;

  try {
    const entry: CacheEntry<T> = {
      saved_at: new Date().toISOString(),
      data,
    };
    storage.setItem(`${CACHE_PREFIX}${key}`, JSON.stringify(entry));
  } catch {
    // Quota/privacy-mode failures should never break live data rendering.
  }
}

export function removeApiCache(key: string): void {
  const storage = getStorage();
  if (!storage) return;

  try {
    storage.removeItem(`${CACHE_PREFIX}${key}`);
  } catch {
    // Ignore storage failures.
  }
}
