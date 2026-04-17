import { Link, useLocation } from 'react-router-dom';
import type { ReactNode } from 'react';
import { CompareBasket } from './CompareBasket';
import { ThemeToggle } from './ThemeToggle';
import { LocaleToggle } from './LocaleToggle';
import { useI18n } from '../i18n/localeContext';
import { useFreshness } from '../../hooks/useFreshness';
import { formatRelativeTime } from '../utils/relativeTime';
import './Layout.css';

interface LayoutProps {
  children: ReactNode;
  onOpenPalette: () => void;
}

export function Layout({ children, onOpenPalette }: LayoutProps) {
  const location = useLocation();
  const { t, locale } = useI18n();
  const isHome = location.pathname === '/';
  const { lastRefresh } = useFreshness();
  const relative = formatRelativeTime(lastRefresh, locale);
  const absolute = lastRefresh
    ? new Date(lastRefresh).toLocaleString(locale === 'zh' ? 'zh-CN' : 'en')
    : '';

  return (
    <div className="v2-shell">
      <header className="v2-topbar">
        <Link to="/" className="v2-brand" aria-label="Model Price — home">
          <img
            src="/logo.png"
            alt=""
            className="v2-brand-logo"
            width={28}
            height={28}
          />
          <span className="v2-brand-name">Model Price</span>
        </Link>
        <div className="v2-topbar-right">
          {relative && lastRefresh && (
            <time
              className="v2-topbar-updated"
              dateTime={lastRefresh}
              title={t('nav.updated_tooltip_fmt', { absolute })}
            >
              {t('nav.updated_fmt', { relative })}
            </time>
          )}
          {!isHome && (
            <button
              type="button"
              className="v2-topbar-search"
              onClick={onOpenPalette}
              aria-label={t('nav.open_palette')}
            >
              <span className="v2-topbar-search-icon">⌕</span>
              <span className="v2-topbar-search-label">
                {t('nav.search_placeholder')}
              </span>
              <kbd>⌘K</kbd>
            </button>
          )}
          <LocaleToggle />
          <ThemeToggle />
          <a
            href="https://github.com/xiaobox/model-price"
            target="_blank"
            rel="noreferrer"
            className="v2-topbar-chip"
            aria-label="GitHub"
            title="GitHub"
          >
            <svg
              viewBox="0 0 16 16"
              width="16"
              height="16"
              aria-hidden
              fill="currentColor"
            >
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z" />
            </svg>
          </a>
        </div>
      </header>

      <main className="v2-main">{children}</main>

      <footer className="v2-footer">
        <span>{t('footer.tagline')}</span>
        <span className="v2-footer-sep">·</span>
        <span>{t('footer.unit')}</span>
      </footer>

      <CompareBasket />
    </div>
  );
}
