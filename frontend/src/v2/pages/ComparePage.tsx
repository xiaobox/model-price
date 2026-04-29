import { useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { API_V2_BASE } from '../../config';
import type { CompareResultV2 } from '../../types/v2';
import { useCompareBasket } from '../useCompareBasket';
import { useI18n } from '../i18n/useI18n';
import type { MessageKey } from '../i18n/messages';
import {
  formatContext,
  formatPrice,
  makerColor,
  providerLabel,
} from '../utils/format';
import { compareFromFallback, loadFallback } from '../fallbackLoader';
import { readApiCache, writeApiCache } from '../apiResponseCache';
import { compareCacheKey, compareFromEntityListCache } from '../liveDataCache';
import { noStoreFetch } from '../noStoreFetch';
import './ComparePage.css';

const BACKEND_TIMEOUT_MS = 15000;

export function ComparePage() {
  const { ids = '' } = useParams<{ ids: string }>();
  const [state, setState] = useState<{
    data: CompareResultV2 | null;
    loading: boolean;
    error: string | null;
  }>({ data: null, loading: Boolean(ids), error: null });
  const basket = useCompareBasket();
  const navigate = useNavigate();
  const { t } = useI18n();

  useEffect(() => {
    if (!ids) return;
    let cancelled = false;
    setState({ data: null, loading: true, error: null });

    (async () => {
      // Stage 1: prefer the last live compare response for these ids.
      const cacheKey = compareCacheKey(ids);
      const cached = readApiCache<CompareResultV2>(cacheKey);
      if (cancelled) return;
      let paintedFallback = false;
      if (cached) {
        setState({ data: cached, loading: true, error: null });
        paintedFallback = true;
      }

      const listCache = paintedFallback ? null : compareFromEntityListCache(ids);
      if (listCache) {
        setState({ data: listCache, loading: true, error: null });
        paintedFallback = true;
      }

      // Stage 2: synth compare result from the bundled snapshot when
      // there is no newer live cache. The backend fetch below swaps in
      // fresh data when it arrives.
      const snapshot = paintedFallback ? null : await loadFallback();
      if (cancelled) return;
      if (snapshot) {
        const requested = ids.split(',');
        const fallback = compareFromFallback(snapshot, requested);
        if (fallback.entities.length > 0) {
          setState({ data: fallback, loading: true, error: null });
          paintedFallback = true;
        }
      }

      // Stage 2: real backend call. Keep the snapshot visible on
      // timeout / error so the user is never stranded on a spinner.
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), BACKEND_TIMEOUT_MS);
      try {
        const response = await noStoreFetch(
          `${API_V2_BASE}/compare?ids=${encodeURIComponent(ids)}`,
          { signal: controller.signal },
        );
        if (cancelled) return;
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = (await response.json()) as CompareResultV2;
        writeApiCache(cacheKey, data);
        setState({ data, loading: false, error: null });
      } catch (err) {
        if (cancelled) return;
        setState((prev) => ({
          data: prev.data,
          loading: false,
          error: paintedFallback
            ? null
            : err instanceof Error
              ? err.message
              : 'fetch failed',
        }));
      } finally {
        clearTimeout(timeout);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [ids]);

  // Render whatever data we have first; only fall back to spinner /
  // error when we genuinely have nothing to paint.
  if (!state.data) {
    if (state.error)
      return (
        <div className="v2-error">
          <p>{t('compare.failed_fmt', { error: state.error })}</p>
          <Link to="/" className="v2-link-accent">
            {t('detail.back_to_home')}
          </Link>
        </div>
      );
    if (state.loading)
      return <div className="v2-loading">{t('detail.loading')}</div>;
    return null;
  }

  const { entities, common_capabilities, missing_ids } = state.data;

  if (entities.length === 0) {
    return (
      <div className="v2-error">
        <p>{t('compare.none_valid')}</p>
        <Link to="/" className="v2-link-accent">
          {t('detail.back_to_home')}
        </Link>
      </div>
    );
  }

  const removeAndNav = (slug: string) => {
    basket.remove(slug);
    const remaining = entities
      .map((e) => e.entity.slug)
      .filter((s) => s !== slug);
    if (remaining.length === 0) {
      navigate('/');
    } else {
      navigate(`/compare/${remaining.join(',')}`);
    }
  };

  return (
    <div className="v2-compare">
      <header className="v2-compare-head">
        <Link to="/" className="v2-entity-back">
          {t('compare.back_to_home')}
        </Link>
        <h1>{t('compare.heading_fmt', { count: entities.length })}</h1>
        {missing_ids.length > 0 ? (
          <p className="v2-compare-missing">
            {t('compare.missing_fmt', { ids: missing_ids.join(', ') })}
          </p>
        ) : null}
      </header>

      <div
        className="v2-compare-grid"
        style={{ gridTemplateColumns: `160px repeat(${entities.length}, 1fr)` }}
      >
        <div className="v2-compare-label" />
        {entities.map(({ entity }) => (
          <div key={entity.slug} className="v2-compare-col-head">
            <div className="v2-compare-col-title">
              <span
                className="v2-maker-dot"
                style={{ background: makerColor(entity.maker) }}
                aria-hidden
              />
              <Link to={`/m/${entity.slug}`}>{entity.name}</Link>
              <button
                type="button"
                className="v2-compare-remove"
                onClick={() => removeAndNav(entity.slug)}
                aria-label={t('compare.remove')}
                title={t('compare.remove_tooltip')}
              >
                ×
              </button>
            </div>
            <div className="v2-compare-col-maker">{entity.maker}</div>
          </div>
        ))}

        <Row label={t('compare.row.family')}>
          {entities.map(({ entity }) => (
            <span key={entity.slug}>{entity.family}</span>
          ))}
        </Row>
        <Row label={t('compare.row.context')}>
          {entities.map(({ entity }) => (
            <span key={entity.slug} className="num">
              {formatContext(entity.context_length)}
            </span>
          ))}
        </Row>
        <Row label={t('compare.row.max_output')}>
          {entities.map(({ entity }) => (
            <span key={entity.slug} className="num">
              {formatContext(entity.max_output_tokens)}
            </span>
          ))}
        </Row>
        <Row label={t('compare.row.input')}>
          {entities.map(({ entity, offerings }) => {
            const p = primary(offerings, entity.primary_offering_provider);
            return (
              <span key={entity.slug} className="num v2-compare-price">
                {formatPrice(p?.pricing.input ?? null)}
              </span>
            );
          })}
        </Row>
        <Row label={t('compare.row.output')}>
          {entities.map(({ entity, offerings }) => {
            const p = primary(offerings, entity.primary_offering_provider);
            return (
              <span key={entity.slug} className="num v2-compare-price">
                {formatPrice(p?.pricing.output ?? null)}
              </span>
            );
          })}
        </Row>
        <Row label={t('compare.row.cache_read')}>
          {entities.map(({ entity, offerings }) => {
            const p = primary(offerings, entity.primary_offering_provider);
            return (
              <span key={entity.slug} className="num v2-muted">
                {formatPrice(p?.pricing.cache_read ?? null)}
              </span>
            );
          })}
        </Row>
        <Row label={t('compare.row.batch_in')}>
          {entities.map(({ entity, offerings }) => {
            const p = primary(offerings, entity.primary_offering_provider);
            return (
              <span key={entity.slug} className="num v2-muted">
                {formatPrice(p?.batch_pricing?.input ?? null)}
              </span>
            );
          })}
        </Row>
        <Row label={t('compare.row.primary_provider')}>
          {entities.map(({ entity }) => (
            <span key={entity.slug}>
              {providerLabel(entity.primary_offering_provider)}
            </span>
          ))}
        </Row>
        <Row label={t('compare.row.capabilities')}>
          {entities.map(({ entity }) => (
            <div key={entity.slug} className="v2-compare-caps">
              {entity.capabilities.map((cap) => (
                <span
                  key={cap}
                  className={`v2-cap v2-cap-${cap}${
                    common_capabilities.includes(cap) ? ' is-shared' : ''
                  }`}
                >
                  {t(`cap.${cap}` as MessageKey)}
                </span>
              ))}
            </div>
          ))}
        </Row>
      </div>

      {common_capabilities.length > 0 ? (
        <p className="v2-compare-common">
          {t('compare.shared_fmt', {
            caps: common_capabilities.map((c) => t(`cap.${c}` as MessageKey)).join(', '),
          })}
        </p>
      ) : null}
    </div>
  );
}

function Row({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <>
      <div className="v2-compare-label">{label}</div>
      {children}
    </>
  );
}

function primary(
  offerings: CompareResultV2['entities'][number]['offerings'],
  provider: string,
) {
  return offerings.find((o) => o.provider === provider) ?? offerings[0];
}
