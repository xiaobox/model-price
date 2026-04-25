import { memo } from 'react';
import type { EntityListItemV2 } from '../../types/v2';
import { useI18n } from '../i18n/useI18n';
import type { MessageKey } from '../i18n/messages';
import {
  formatContext,
  formatPrice,
  makerColor,
} from '../utils/format';
import './EntityTable.css';

interface EntityTableProps {
  entities: EntityListItemV2[];
  onSelect: (slug: string) => void;
  selectedSlug: string | null;
  isInBasket: (slug: string) => boolean;
  onToggleBasket: (slug: string) => void;
  basketCount: number;
  basketCapacity: number;
  basketFull: boolean;
}

const CAP_ORDER = [
  'text',
  'vision',
  'audio',
  'reasoning',
  'tool_use',
  'function_calling',
  'image_generation',
];

function orderedCaps(caps: string[]): string[] {
  const set = new Set(caps);
  return CAP_ORDER.filter((c) => set.has(c));
}

function Row({
  entity,
  onSelect,
  isSelected,
  inBasket,
  onToggleBasket,
  basketCount,
  basketCapacity,
  basketFull,
}: {
  entity: EntityListItemV2;
  onSelect: (slug: string) => void;
  isSelected: boolean;
  inBasket: boolean;
  onToggleBasket: (slug: string) => void;
  basketCount: number;
  basketCapacity: number;
  basketFull: boolean;
}) {
  const { t } = useI18n();
  const primary = entity.primary_offering;
  const pricing = primary?.pricing;
  const input = pricing?.input ?? null;
  const output = pricing?.output ?? null;
  const context = entity.context_length;
  const caps = orderedCaps(entity.capabilities);

  return (
    <div
      className={`v2-row${isSelected ? ' is-selected' : ''}${inBasket ? ' is-basket' : ''}`}
      role="button"
      tabIndex={0}
      onClick={() => onSelect(entity.slug)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onSelect(entity.slug);
        }
      }}
    >
      <div className="v2-row-main">
        <div className="v2-row-name">
          <span
            className="v2-maker-dot"
            style={{ background: makerColor(entity.maker) }}
            aria-hidden
          />
          <span className="v2-row-title">{entity.name}</span>
          {entity.is_open_source ? <span className="v2-row-oss">OSS</span> : null}
        </div>
        <div className="v2-row-maker">{entity.maker}</div>
      </div>
      <div className="v2-row-col num">{formatContext(context)}</div>
      <div className="v2-row-col num v2-row-price">{formatPrice(input)}</div>
      <div className="v2-row-col num v2-row-price">{formatPrice(output)}</div>
      <div className="v2-row-col v2-row-caps">
        {caps.slice(0, 5).map((cap) => {
          const label = t(`cap.${cap}` as MessageKey);
          return (
            <span key={cap} className={`v2-cap v2-cap-${cap}`} title={label}>
              {label}
            </span>
          );
        })}
      </div>
      <div className="v2-row-col v2-row-actions">
        <button
          type="button"
          className={`v2-row-add${inBasket ? ' is-added' : ''}${
            !inBasket && basketFull ? ' is-disabled' : ''
          }`}
          disabled={!inBasket && basketFull}
          onClick={(e) => {
            e.stopPropagation();
            if (!inBasket && basketFull) return;
            onToggleBasket(entity.slug);
          }}
          title={
            inBasket
              ? t('table.remove_from_compare')
              : basketFull
              ? t('table.compare_full_fmt', { max: basketCapacity })
              : t('table.add_to_compare_fmt', {
                  count: basketCount,
                  max: basketCapacity,
                })
          }
          aria-label={
            inBasket
              ? t('table.remove_from_compare')
              : basketFull
              ? t('table.compare_full_fmt', { max: basketCapacity })
              : t('table.add_to_compare_fmt', {
                  count: basketCount,
                  max: basketCapacity,
                })
          }
        >
          {inBasket ? '✓' : '+'}
        </button>
      </div>
    </div>
  );
}

const MemoRow = memo(Row);

function EntityTableImpl({
  entities,
  onSelect,
  selectedSlug,
  isInBasket,
  onToggleBasket,
  basketCount,
  basketCapacity,
  basketFull,
}: EntityTableProps) {
  const { t } = useI18n();
  if (entities.length === 0) {
    return (
      <div className="v2-empty">
        <span className="v2-empty-icon">∅</span>
        <p>{t('table.empty')}</p>
      </div>
    );
  }

  return (
    <div className="v2-table" role="table" aria-label={t('table.col.model')}>
      <div className="v2-table-head" role="row">
        <div className="v2-row-main">{t('table.col.model')}</div>
        <div className="v2-row-col">{t('table.col.context')}</div>
        <div className="v2-row-col">{t('table.col.input')}</div>
        <div className="v2-row-col">{t('table.col.output')}</div>
        <div className="v2-row-col">{t('table.col.capabilities')}</div>
        <div className="v2-row-col v2-row-cap-hint" title={t('basket.hint_fmt', { max: basketCapacity })}>
          {basketCount}/{basketCapacity}
        </div>
      </div>
      <div className="v2-table-body">
        {entities.map((entity) => (
          <MemoRow
            key={entity.slug}
            entity={entity}
            onSelect={onSelect}
            isSelected={entity.slug === selectedSlug}
            inBasket={isInBasket(entity.slug)}
            onToggleBasket={onToggleBasket}
            basketCount={basketCount}
            basketCapacity={basketCapacity}
            basketFull={basketFull}
          />
        ))}
      </div>
    </div>
  );
}

export const EntityTable = memo(EntityTableImpl);
