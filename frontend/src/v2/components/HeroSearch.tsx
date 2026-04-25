import { useEffect, useRef } from 'react';
import { useI18n } from '../i18n/useI18n';
import './HeroSearch.css';

interface HeroSearchProps {
  value: string;
  onChange: (next: string) => void;
  resultCount: number;
  totalCount: number;
}

export function HeroSearch({
  value,
  onChange,
  resultCount,
  totalCount,
}: HeroSearchProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const { t } = useI18n();

  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    if (typeof window !== 'undefined' && window.innerWidth < 720) return;
    if (document.activeElement === document.body) {
      el.focus();
    }
  }, []);

  const meta = value
    ? t('hero.meta_matching_fmt', {
        count: resultCount,
        query: value,
        total: totalCount,
      })
    : t('hero.meta_shown_fmt', { count: resultCount, total: totalCount });

  return (
    <section className="v2-hero">
      <h1 className="v2-hero-title">
        {t('hero.title_prefix')}
        <span className="num">{totalCount}</span>
        {t('hero.title_suffix')}
      </h1>
      <p className="v2-hero-sub">{t('hero.subtitle')}</p>

      <div className="v2-hero-search">
        <span className="v2-hero-search-icon">⌕</span>
        <input
          ref={inputRef}
          type="text"
          placeholder={t('hero.search_placeholder')}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          autoCorrect="off"
          autoCapitalize="off"
          spellCheck={false}
        />
        <span className="v2-hero-search-hint" aria-hidden>
          ⌘K
        </span>
      </div>

      <div className="v2-hero-meta">{meta}</div>
    </section>
  );
}
