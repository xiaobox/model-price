import { useContext } from 'react';
import { ThemeContext } from './themeContextValue';
import type { ThemeValue } from './themeContextValue';

export function useTheme(): ThemeValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) {
    throw new Error('useTheme must be used within ThemeProvider');
  }
  return ctx;
}
