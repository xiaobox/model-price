import { useCallback, useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { CompareBasketContext } from './compareBasketContextValue';
import type { BasketValue } from './compareBasketContextValue';

const STORAGE_KEY = 'model-price-v2:compare-basket';
const MAX_ITEMS = 4;

function readInitial(): string[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = window.sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    return Array.isArray(parsed) ? (parsed as string[]).slice(0, MAX_ITEMS) : [];
  } catch {
    return [];
  }
}

export function CompareBasketProvider({ children }: { children: ReactNode }) {
  const [slugs, setSlugs] = useState<string[]>(readInitial);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(slugs));
    } catch {
      // ignore storage quota errors
    }
  }, [slugs]);

  const toggle = useCallback((slug: string) => {
    let outcome: { added: boolean; full: boolean } = { added: false, full: false };
    setSlugs((prev) => {
      if (prev.includes(slug)) {
        outcome = { added: false, full: false };
        return prev.filter((s) => s !== slug);
      }
      if (prev.length >= MAX_ITEMS) {
        outcome = { added: false, full: true };
        return prev;
      }
      outcome = { added: true, full: false };
      return [...prev, slug];
    });
    return outcome;
  }, []);

  const add = useCallback((slug: string) => {
    setSlugs((prev) => {
      if (prev.includes(slug) || prev.length >= MAX_ITEMS) return prev;
      return [...prev, slug];
    });
  }, []);

  const remove = useCallback((slug: string) => {
    setSlugs((prev) => prev.filter((s) => s !== slug));
  }, []);

  const clear = useCallback(() => setSlugs([]), []);

  const has = useCallback((slug: string) => slugs.includes(slug), [slugs]);

  const value: BasketValue = {
    slugs,
    count: slugs.length,
    capacity: MAX_ITEMS,
    isFull: slugs.length >= MAX_ITEMS,
    toggle,
    add,
    remove,
    clear,
    has,
  };

  return (
    <CompareBasketContext.Provider value={value}>
      {children}
    </CompareBasketContext.Provider>
  );
}
