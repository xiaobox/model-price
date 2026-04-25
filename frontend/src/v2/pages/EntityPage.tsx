import { useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { useEntityV2 } from '../../hooks/useEntityV2';
import { useCompareBasket } from '../useCompareBasket';
import { PUBLIC_BASE_URL } from '../../config';
import { AlternativesList } from '../components/AlternativesList';
import { OfferingRow } from '../components/OfferingRow';
import { ShareActions } from '../components/ShareActions';
import {
  formatContext,
  makerColor,
} from '../utils/format';
import { useI18n } from '../i18n/useI18n';
import type { MessageKey } from '../i18n/messages';
import { LITELLM_REGISTRY_URL, officialLinkForMaker } from '../utils/officialLinks';
import './EntityPage.css';

export function EntityPage() {
  const { slug } = useParams<{ slug: string }>();
  const { detail, loading, notFound } = useEntityV2(slug);
  const basket = useCompareBasket();
  const navigate = useNavigate();
  const { t } = useI18n();
  // All hooks must run unconditionally — early returns come AFTER.
  const [copied, setCopied] = useState(false);

  // Render the Stage-1 snapshot from useEntityV2 as soon as `detail` is
  // populated, even while `loading` is still true waiting for the cold
  // backend. Only fall back to spinner / 404 when we genuinely have
  // nothing to paint — mirrors the EntityDrawer fix (c8ed8ca).
  if (!detail) {
    if (notFound) {
      return (
        <div className="v2-error">
          <p>
            {t('detail.not_found_fmt', { slug: slug ?? '' })}{' '}
            <Link to="/" className="v2-link-accent">
              {t('detail.back_to_home')}
            </Link>
          </p>
        </div>
      );
    }
    if (loading) return <div className="v2-loading">{t('detail.loading')}</div>;
    return null;
  }

  const { entity, offerings, alternatives } = detail;
  const inBasket = basket.has(entity.slug);
  const primary = offerings.find((o) => o.provider === entity.primary_offering_provider);

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
    <article className="v2-entity-page">
      <Link to="/" className="v2-entity-back">
        {t('detail.back_to_home')}
      </Link>

      <header className="v2-entity-head">
        <div className="v2-entity-title">
          <span
            className="v2-maker-dot v2-entity-dot"
            style={{ background: makerColor(entity.maker) }}
            aria-hidden
          />
          <h1>{entity.name}</h1>
          {entity.is_open_source ? <span className="v2-row-oss">OSS</span> : null}
        </div>
        <p className="v2-entity-meta">
          {entity.maker} · {entity.family} · {formatContext(entity.context_length)}{' '}
          {t('detail.context_suffix')}
          {entity.max_output_tokens ? (
            <>
              {' '}
              · {formatContext(entity.max_output_tokens)}{' '}
              {t('detail.max_output_suffix')}
            </>
          ) : null}
        </p>

        <div className="v2-entity-actions">
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
            onClick={() => basket.toggle(entity.slug)}
          >
            {inBasket ? t('detail.in_compare') : t('detail.add_to_compare')}
          </button>
        </div>

        <OfficialLinksStrip maker={entity.maker} />
        <ShareActions
          url={`${PUBLIC_BASE_URL}/m/${entity.slug}`}
          name={entity.name}
          maker={entity.maker}
        />
      </header>

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
            onNavigate={(s) => navigate(`/m/${s}`)}
          />
        </section>
      ) : null}
    </article>
  );
}

function OfficialLinksStrip({ maker }: { maker: string }) {
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
      <a
        href={LITELLM_REGISTRY_URL}
        target="_blank"
        rel="noreferrer"
        className="v2-official-link v2-official-link-muted"
      >
        {t('detail.litellm_source')}
      </a>
    </div>
  );
}
