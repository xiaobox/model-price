import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { PUBLIC_BASE_URL } from '../../config';
import { useEntityV2 } from '../../hooks/useEntityV2';
import { AlternativesList } from './AlternativesList';
import { OfferingRow } from './OfferingRow';
import { ShareActions } from './ShareActions';
import {
  formatContext,
  makerColor,
} from '../utils/format';
import { useI18n } from '../i18n/useI18n';
import type { MessageKey } from '../i18n/messages';
import { LITELLM_REGISTRY_URL, officialLinkForMaker } from '../utils/officialLinks';
import './EntityDrawer.css';

interface EntityDrawerProps {
  slug: string | null;
  onClose: () => void;
  onToggleBasket: (slug: string) => void;
  isInBasket: (slug: string) => boolean;
  onNavigateSlug: (slug: string) => void;
}

export function EntityDrawer({
  slug,
  onClose,
  onToggleBasket,
  isInBasket,
  onNavigateSlug,
}: EntityDrawerProps) {
  const { detail, loading, notFound } = useEntityV2(slug);
  const { t } = useI18n();

  useEffect(() => {
    if (!slug) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [slug, onClose]);

  if (!slug) return null;

  return (
    <>
      <div className="v2-drawer-scrim" onClick={onClose} />
      <aside className="v2-drawer" role="dialog" aria-label="Model detail">
        <header className="v2-drawer-head">
          <button
            type="button"
            className="v2-drawer-close"
            onClick={onClose}
            aria-label="Close"
          >
            ×
          </button>
          <Link
            to={`/m/${slug}`}
            className="v2-drawer-expand"
            title={t('detail.open_full_page')}
          >
            {t('detail.open_full_page')}
          </Link>
        </header>

        {detail ? (
          <DrawerContent
            detail={detail}
            onToggleBasket={onToggleBasket}
            isInBasket={isInBasket}
            onNavigateSlug={onNavigateSlug}
          />
        ) : notFound ? (
          <div className="v2-drawer-empty">
            {t('detail.not_found_fmt', { slug: slug ?? '' })}
          </div>
        ) : loading ? (
          <div className="v2-drawer-loading">{t('detail.loading')}</div>
        ) : null}
      </aside>
    </>
  );
}

function DrawerContent({
  detail,
  onToggleBasket,
  isInBasket,
  onNavigateSlug,
}: {
  detail: ReturnType<typeof useEntityV2>['detail'] extends infer T
    ? NonNullable<T>
    : never;
  onToggleBasket: (slug: string) => void;
  isInBasket: (slug: string) => boolean;
  onNavigateSlug: (slug: string) => void;
}) {
  const { t } = useI18n();
  const { entity, offerings, alternatives } = detail;
  const inBasket = isInBasket(entity.slug);
  const primary = offerings.find((o) => o.provider === entity.primary_offering_provider)
    ?? offerings[0];
  const [copied, setCopied] = useState(false);

  const copyModelId = async () => {
    if (!primary) return;
    try {
      await navigator.clipboard.writeText(primary.provider_model_id);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // ignore
    }
  };

  return (
    <div className="v2-drawer-body">
      <div className="v2-drawer-title">
        <span
          className="v2-maker-dot"
          style={{ background: makerColor(entity.maker) }}
          aria-hidden
        />
        <h2>{entity.name}</h2>
        {entity.is_open_source ? <span className="v2-row-oss">OSS</span> : null}
      </div>
      <div className="v2-drawer-meta">
        <span>{entity.maker}</span>
        <span className="v2-drawer-sep">·</span>
        <span>{entity.family}</span>
        <span className="v2-drawer-sep">·</span>
        <span>
          {formatContext(entity.context_length)} {t('detail.context_suffix')}
        </span>
        {entity.max_output_tokens ? (
          <>
            <span className="v2-drawer-sep">·</span>
            <span>
              {formatContext(entity.max_output_tokens)} {t('detail.max_output_suffix')}
            </span>
          </>
        ) : null}
      </div>

      <div className="v2-drawer-actions">
        <button
          type="button"
          className={`v2-btn v2-btn-primary${copied ? ' is-copied' : ''}`}
          onClick={copyModelId}
          disabled={!primary}
        >
          {copied ? t('detail.copied') : t('detail.copy_model_id')}
          <span className="v2-btn-sub mono">
            {primary?.provider_model_id ?? ''}
          </span>
        </button>
        <button
          type="button"
          className={`v2-btn${inBasket ? ' is-active' : ''}`}
          onClick={() => onToggleBasket(entity.slug)}
        >
          {inBasket ? t('detail.in_compare') : t('detail.add_to_compare')}
        </button>
      </div>

      <OfficialLinks maker={entity.maker} />
      <ShareActions
        url={`${PUBLIC_BASE_URL}/m/${entity.slug}`}
        name={entity.name}
        maker={entity.maker}
      />

      <section className="v2-drawer-section">
        <h3>{t('detail.capabilities')}</h3>
        <div className="v2-caps-grid">
          {entity.capabilities.map((cap) => (
            <span key={cap} className={`v2-cap v2-cap-${cap}`}>
              {t(`cap.${cap}` as MessageKey)}
            </span>
          ))}
        </div>
        <div className="v2-drawer-modality">
          <div>
            <span className="v2-drawer-label">{t('detail.modality_input')}</span>
            <span>{entity.input_modalities.join(', ')}</span>
          </div>
          <div>
            <span className="v2-drawer-label">{t('detail.modality_output')}</span>
            <span>{entity.output_modalities.join(', ')}</span>
          </div>
        </div>
      </section>

      <section className="v2-drawer-section">
        <h3>{t('detail.pricing_across_providers')}</h3>
        <div className="v2-offerings">
          <div className="v2-offer-head">
            <span>{t('detail.col.provider')}</span>
            <span>{t('detail.col.input')}</span>
            <span>{t('detail.col.output')}</span>
            <span>{t('detail.col.cache_read')}</span>
            <span>{t('detail.col.batch_in')}</span>
          </div>
          {offerings.map((o) => (
            <OfferingRow
              key={`${o.provider}-${o.provider_model_id}`}
              offering={o}
              primaryProvider={entity.primary_offering_provider}
            />
          ))}
        </div>
      </section>

      {alternatives.length > 0 ? (
        <section className="v2-drawer-section">
          <h3>{t('detail.alternatives')}</h3>
          <AlternativesList
            alternatives={alternatives}
            onNavigate={onNavigateSlug}
          />
        </section>
      ) : null}
    </div>
  );
}

function OfficialLinks({ maker }: { maker: string }) {
  const { t } = useI18n();
  const link = officialLinkForMaker(maker);
  return (
    <div className="v2-official-links">
      <span className="v2-official-label">{t('detail.official_label')}</span>
      {link ? (
        <>
          <a href={link.pricing} target="_blank" rel="noreferrer" className="v2-official-link">
            {t('detail.official_pricing_fmt', { maker })}
          </a>
          {link.docs ? (
            <a href={link.docs} target="_blank" rel="noreferrer" className="v2-official-link">
              {t('detail.official_docs_fmt', { maker })}
            </a>
          ) : null}
        </>
      ) : null}
      <a href={LITELLM_REGISTRY_URL} target="_blank" rel="noreferrer" className="v2-official-link v2-official-link-muted">
        {t('detail.litellm_source')}
      </a>
    </div>
  );
}
