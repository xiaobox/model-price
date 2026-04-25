import { useTheme } from '../useTheme';
import type { ThemeMode } from '../themeContextValue';
import { useI18n } from '../i18n/useI18n';

const ICONS: Record<ThemeMode, string> = {
  dark: '●',
  light: '○',
  system: '◐',
};

const NEXT: Record<ThemeMode, ThemeMode> = {
  dark: 'light',
  light: 'system',
  system: 'dark',
};

export function ThemeToggle() {
  const { mode, cycle } = useTheme();
  const { t } = useI18n();
  const currentLabel = t(`theme.${mode}` as 'theme.dark' | 'theme.light' | 'theme.system');
  const nextLabel = t(`theme.${NEXT[mode]}` as 'theme.dark' | 'theme.light' | 'theme.system');
  const title = `${currentLabel} · ${t('theme.next_fmt', { next: nextLabel })}`;
  return (
    <button
      type="button"
      className="v2-topbar-chip"
      onClick={cycle}
      title={title}
      aria-label={title}
    >
      {ICONS[mode]}
    </button>
  );
}
