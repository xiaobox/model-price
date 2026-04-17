import { useMemo, useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import type { EntitiesListQuery } from '../../types/v2';
import { useEntitiesV2 } from '../../hooks/useEntitiesV2';
import { useCompareBasket } from '../compareBasketContext';
import { useI18n } from '../i18n/localeContext';
import { HeroSearch } from '../components/HeroSearch';
import { FilterBar } from '../components/FilterBar';
import { EntityTable } from '../components/EntityTable';
import { EntityDrawer } from '../components/EntityDrawer';
import { exportEntitiesToCsv } from '../utils/exportCsv';
import './HomePage.css';

function cleanLabel(raw: string | null | undefined): string {
  if (!raw) return '';
  // Strip trailing punctuation and whitespace, title-case
  const stripped = raw.trim().replace(/[\s:;,./]+$/, '');
  if (!stripped) return '';
  return stripped;
}

interface HomePageProps {
  onOpenPalette?: () => void;
}

export function HomePage(_props: HomePageProps) {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const basket = useCompareBasket();
  const { t } = useI18n();

  const drawerSlug = searchParams.get('m');

  const [query, setQuery] = useState<EntitiesListQuery>(() => ({
    q: searchParams.get('q') ?? undefined,
    family: searchParams.get('family') ?? undefined,
    maker: searchParams.get('maker') ?? undefined,
    capability: searchParams.get('capability') ?? undefined,
    sort: (searchParams.get('sort') as EntitiesListQuery['sort']) ?? 'name',
    order: (searchParams.get('order') as EntitiesListQuery['order']) ?? 'asc',
  }));

  useEffect(() => {
    const next = new URLSearchParams(searchParams);
    const set = (key: string, value: string | undefined) => {
      if (value === undefined || value === '' || value === null) next.delete(key);
      else next.set(key, value);
    };
    set('q', query.q);
    set('family', query.family);
    set('maker', query.maker);
    set('capability', query.capability);
    set('sort', query.sort && query.sort !== 'name' ? query.sort : undefined);
    set(
      'order',
      query.order && query.order !== 'asc' ? query.order : undefined,
    );
    setSearchParams(next, { replace: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query]);

  const { entities, loading, error, fromFallback } = useEntitiesV2(query);

  const { totalEntities, makers, families } = useTotals();

  const handleOpen = (slug: string) => {
    const next = new URLSearchParams(searchParams);
    next.set('m', slug);
    setSearchParams(next, { replace: true });
  };

  const handleCloseDrawer = () => {
    const next = new URLSearchParams(searchParams);
    next.delete('m');
    setSearchParams(next, { replace: true });
  };

  const handleNavigateSlug = (slug: string) => {
    navigate(`/m/${slug}`);
  };

  return (
    <div className="v2-home">
      <HeroSearch
        value={query.q ?? ''}
        onChange={(next) => setQuery((prev) => ({ ...prev, q: next || undefined }))}
        resultCount={entities.length}
        totalCount={totalEntities}
      />

      <FilterBar
        query={query}
        onChange={setQuery}
        makers={makers}
        families={families}
      />

      <div className="v2-actions-bar">
        <button
          type="button"
          className="v2-export-btn"
          onClick={() => exportEntitiesToCsv(entities)}
          disabled={entities.length === 0}
          title={t('export.csv_tooltip')}
        >
          <span aria-hidden="true">↓</span>
          {t('export.csv')}
          <span className="v2-export-count">({entities.length})</span>
        </button>
      </div>

      {error ? (
        <div className="v2-error">
          <p>Backend unreachable. Start uvicorn on :8000.</p>
          <pre>{error}</pre>
        </div>
      ) : loading && !fromFallback && entities.length === 0 ? (
        <div className="v2-loading">Loading…</div>
      ) : (
        <EntityTable
          entities={entities}
          onSelect={handleOpen}
          selectedSlug={drawerSlug}
          isInBasket={basket.has}
          onToggleBasket={(slug) => basket.toggle(slug)}
          basketCount={basket.count}
          basketCapacity={basket.capacity}
          basketFull={basket.isFull}
        />
      )}

      <EntityDrawer
        slug={drawerSlug}
        onClose={handleCloseDrawer}
        isInBasket={basket.has}
        onToggleBasket={(slug) => basket.toggle(slug)}
        onNavigateSlug={handleNavigateSlug}
      />
    </div>
  );
}

function useTotals() {
  const { entities } = useEntitiesV2({});
  return useMemo(() => {
    // Dedupe case-insensitively and strip trailing punctuation so
    // "Bytedance" and "Bytedance:" don't both show up in the dropdown.
    const makersSet = new Map<string, string>();
    const familiesSet = new Map<string, string>();
    for (const e of entities) {
      const maker = cleanLabel(e.maker);
      if (maker && maker !== 'Unknown' && maker !== 'Other') {
        makersSet.set(maker.toLowerCase(), maker);
      }
      const family = cleanLabel(e.family);
      if (family && family !== 'Unknown' && family !== 'Other') {
        familiesSet.set(family.toLowerCase(), family);
      }
    }
    const makers = [...makersSet.values()].sort();
    const families = [...familiesSet.values()].sort();
    return {
      totalEntities: entities.length,
      makers,
      families,
    };
  }, [entities]);
}
