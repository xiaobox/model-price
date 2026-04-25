import { createContext } from 'react';
import type { MessageKey } from './messages';

export type Locale = 'en' | 'zh';

export interface LocaleValue {
  locale: Locale;
  setLocale: (next: Locale) => void;
  toggle: () => void;
  t: (key: MessageKey, vars?: Record<string, string | number>) => string;
}

export const LocaleContext = createContext<LocaleValue | null>(null);
