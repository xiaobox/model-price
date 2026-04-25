import { useEffect, useState, useCallback } from 'react';
import type { ReactNode } from 'react';
import { ThemeContext } from './themeContextValue';
import type { ResolvedTheme, ThemeMode } from './themeContextValue';

const STORAGE_KEY = 'model-price-v2:theme';

function readInitial(): ThemeMode {
  if (typeof window === 'undefined') return 'dark';
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw === 'dark' || raw === 'light' || raw === 'system') return raw;
  } catch {
    // ignore
  }
  return 'dark';
}

function resolveTheme(mode: ThemeMode): ResolvedTheme {
  if (mode === 'dark' || mode === 'light') return mode;
  if (typeof window === 'undefined' || !window.matchMedia) return 'dark';
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [mode, setModeState] = useState<ThemeMode>(readInitial);
  const [systemTheme, setSystemTheme] = useState<ResolvedTheme>(() => resolveTheme('system'));
  const resolved: ResolvedTheme = mode === 'system' ? systemTheme : mode;

  // Persist + re-resolve whenever the user picks a mode.
  const setMode = useCallback((next: ThemeMode) => {
    setModeState(next);
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // ignore
    }
  }, []);

  const cycle = useCallback(() => {
    setMode(mode === 'dark' ? 'light' : mode === 'light' ? 'system' : 'dark');
  }, [mode, setMode]);

  // Keep the OS setting hot so switching to "system" never flashes stale theme.
  useEffect(() => {
    if (!window.matchMedia) return;
    const media = window.matchMedia('(prefers-color-scheme: light)');
    const handler = () => setSystemTheme(media.matches ? 'light' : 'dark');
    media.addEventListener('change', handler);
    return () => media.removeEventListener('change', handler);
  }, []);

  // Reflect to <html data-theme="…"> so CSS custom properties switch.
  useEffect(() => {
    const root = document.documentElement;
    root.setAttribute('data-theme', resolved);
  }, [resolved]);

  return (
    <ThemeContext.Provider value={{ mode, resolved, setMode, cycle }}>
      {children}
    </ThemeContext.Provider>
  );
}
