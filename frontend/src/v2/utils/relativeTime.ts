import type { Locale } from '../i18n/localeContext';

const MINUTE_MS = 60_000;
const HOUR_MS = 60 * MINUTE_MS;
const DAY_MS = 24 * HOUR_MS;

export function formatRelativeTime(iso: string | null, locale: Locale): string {
  if (!iso) return '';
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return '';
  const diff = Math.max(0, Date.now() - then);

  const bcp = locale === 'zh' ? 'zh-CN' : 'en';
  const rtf = new Intl.RelativeTimeFormat(bcp, { numeric: 'auto' });

  if (diff < MINUTE_MS) return rtf.format(0, 'second');
  if (diff < HOUR_MS) return rtf.format(-Math.floor(diff / MINUTE_MS), 'minute');
  if (diff < DAY_MS) return rtf.format(-Math.floor(diff / HOUR_MS), 'hour');
  return rtf.format(-Math.floor(diff / DAY_MS), 'day');
}
