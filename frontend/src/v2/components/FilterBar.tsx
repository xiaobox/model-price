import type { EntitiesListQuery } from '../../types/v2';
import { useI18n } from '../i18n/useI18n';
import type { MessageKey } from '../i18n/messages';
import './FilterBar.css';

const CAPABILITIES = [
  'text',
  'vision',
  'audio',
  'tool_use',
  'reasoning',
  'function_calling',
  'image_generation',
  'embedding',
] as const;

const SORTS: { value: NonNullable<EntitiesListQuery['sort']>; labelKey: MessageKey }[] = [
  { value: 'name', labelKey: 'filter.sort.name' },
  { value: 'input', labelKey: 'filter.sort.input' },
  { value: 'output', labelKey: 'filter.sort.output' },
  { value: 'context', labelKey: 'filter.sort.context' },
];

interface FilterBarProps {
  query: EntitiesListQuery;
  onChange: (next: EntitiesListQuery) => void;
  makers: string[];
  families: string[];
}

export function FilterBar({ query, onChange, makers, families }: FilterBarProps) {
  const { t } = useI18n();
  const patch = (diff: Partial<EntitiesListQuery>) => {
    onChange({ ...query, ...diff });
  };

  const toggleCapability = (cap: string) => {
    patch({ capability: query.capability === cap ? undefined : cap });
  };

  const toggleSort = (field: NonNullable<EntitiesListQuery['sort']>) => {
    if (query.sort === field) {
      patch({ order: query.order === 'asc' ? 'desc' : 'asc' });
    } else {
      patch({ sort: field, order: field === 'name' ? 'asc' : 'asc' });
    }
  };

  return (
    <div className="v2-filters">
      <div className="v2-filters-row">
        <select
          className="v2-select"
          value={query.maker ?? ''}
          onChange={(e) => patch({ maker: e.target.value || undefined })}
        >
          <option value="">{t('filter.all_makers')}</option>
          {makers.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
        <select
          className="v2-select"
          value={query.family ?? ''}
          onChange={(e) => patch({ family: e.target.value || undefined })}
        >
          <option value="">{t('filter.all_families')}</option>
          {families.map((f) => (
            <option key={f} value={f}>
              {f}
            </option>
          ))}
        </select>
        <div className="v2-filters-sort">
          {SORTS.map((s) => {
            const active = query.sort === s.value;
            const arrow = active ? (query.order === 'desc' ? '↓' : '↑') : '';
            return (
              <button
                key={s.value}
                type="button"
                className={`v2-sort-btn${active ? ' is-active' : ''}`}
                onClick={() => toggleSort(s.value)}
              >
                {t(s.labelKey)} <span className="v2-sort-arrow">{arrow}</span>
              </button>
            );
          })}
        </div>
      </div>
      <div className="v2-filters-row">
        <button
          type="button"
          className={`v2-chip${!query.capability ? ' is-active' : ''}`}
          onClick={() => patch({ capability: undefined })}
        >
          {t('filter.cap.all')}
        </button>
        {CAPABILITIES.map((cap) => (
          <button
            key={cap}
            type="button"
            className={`v2-chip${query.capability === cap ? ' is-active' : ''}`}
            onClick={() => toggleCapability(cap)}
          >
            {t(`cap.${cap}` as MessageKey)}
          </button>
        ))}
      </div>
    </div>
  );
}
