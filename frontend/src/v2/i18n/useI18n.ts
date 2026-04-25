import { useContext } from 'react';
import { LocaleContext } from './localeContextValue';
import type { LocaleValue } from './localeContextValue';

export function useI18n(): LocaleValue {
  const ctx = useContext(LocaleContext);
  if (!ctx) {
    throw new Error('useI18n must be used within LocaleProvider');
  }
  return ctx;
}
