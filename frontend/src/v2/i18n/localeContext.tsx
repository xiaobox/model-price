import { useCallback, useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { MESSAGES } from './messages';
import type { MessageKey } from './messages';
import { LocaleContext } from './localeContextValue';
import type { Locale } from './localeContextValue';

const STORAGE_KEY = 'model-price-v2:locale';

function detectInitial(): Locale {
  if (typeof window === 'undefined') return 'en';
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored === 'en' || stored === 'zh') return stored;
  } catch {
    // ignore
  }
  // Always default to English. Users toggle to Chinese explicitly
  // via the topbar chip; the choice persists in localStorage.
  return 'en';
}

function interpolate(template: string, vars?: Record<string, string | number>): string {
  if (!vars) return template;
  return template.replace(/\{(\w+)\}/g, (_match, name: string) => {
    const value = vars[name];
    return value === undefined ? `{${name}}` : String(value);
  });
}

export function LocaleProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(detectInitial);

  const setLocale = useCallback((next: Locale) => {
    setLocaleState(next);
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // ignore
    }
  }, []);

  const toggle = useCallback(() => {
    setLocale(locale === 'en' ? 'zh' : 'en');
  }, [locale, setLocale]);

  const t = useCallback(
    (key: MessageKey, vars?: Record<string, string | number>) => {
      const bundle = MESSAGES[locale];
      const template = bundle[key] ?? MESSAGES.en[key] ?? key;
      return interpolate(template, vars);
    },
    [locale],
  );

  useEffect(() => {
    if (typeof document !== 'undefined') {
      document.documentElement.lang = locale === 'zh' ? 'zh-CN' : 'en';
    }
  }, [locale]);

  return (
    <LocaleContext.Provider value={{ locale, setLocale, toggle, t }}>
      {children}
    </LocaleContext.Provider>
  );
}
