import type { ReactNode } from 'react';
import type { OfferingV2 } from '../../types/v2';
import { useI18n } from '../i18n/localeContext';
import { formatPrice, providerLabel } from '../utils/format';

interface OfferingRowProps {
  offering: OfferingV2;
  primaryProvider: string;
}

/**
 * Single row in the "pricing across providers" grid.
 *
 * Layout:
 *   Provider name (bold, accent color if primary)
 *   ⓘ notes icon (if any)
 *   └─ small meta line: "primary · via LiteLLM"
 *
 * We deliberately avoid pill/badge shapes after the review pass —
 * they pushed the provider name onto a second line for long labels
 * like "Google Vertex AI" and created uneven row heights. Subtitles
 * stay out of the way while the name column breathes.
 *
 * ``litellm_fallback`` rows (no real provider attached; LiteLLM
 * registry used as the only price source) render as a gray ``.is-
 * placeholder`` row with no subtitle. The earlier "placeholder" tag
 * read awkwardly in Chinese ("占位") and the gray row style is
 * already sufficient visual signal.
 */
export function OfferingRow({ offering, primaryProvider }: OfferingRowProps) {
  const { t } = useI18n();
  const isPrimary = offering.provider === primaryProvider;
  const isPlaceholder = offering.source === 'litellm_fallback';
  const isViaLitellm = offering.source === 'via_litellm';

  const subParts: ReactNode[] = [];
  // Placeholder rows don't carry primary/via-litellm subtitles —
  // the whole point of the row is "this is reference pricing, not
  // a real provider offering", and a "primary" tag on the entity's
  // only row is meaningless. The gray row style carries the signal.
  if (!isPlaceholder) {
    if (isPrimary) {
      subParts.push(
        <span key="primary" className="v2-offer-sub-primary">
          {t('detail.primary_tag')}
        </span>,
      );
    }
    if (isViaLitellm) {
      subParts.push(
        <span
          key="via"
          className="v2-offer-sub-src"
          title={t('detail.source_via_litellm_tooltip')}
        >
          {t('detail.source_via_litellm')}
        </span>,
      );
    }
  }

  const rowClass = [
    'v2-offer',
    isPrimary ? 'is-primary' : '',
    isPlaceholder ? 'is-placeholder' : '',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div className={rowClass}>
      <div className="v2-offer-provider">
        <div className="v2-offer-name-row">
          <span
            className={`v2-offer-name${isPrimary ? ' is-primary' : ''}`}
          >
            {providerLabel(offering.provider)}
          </span>
          {offering.notes ? (
            <span className="v2-offer-note" title={offering.notes}>
              ⓘ
            </span>
          ) : null}
        </div>
        {subParts.length > 0 ? (
          <div className="v2-offer-sub">
            {joinWithSeparator(subParts)}
          </div>
        ) : null}
      </div>
      <span className="num">{formatPrice(offering.pricing.input)}</span>
      <span className="num">{formatPrice(offering.pricing.output)}</span>
      <span className="num v2-muted">
        {formatPrice(offering.pricing.cache_read)}
      </span>
      <span className="num v2-muted">
        {formatPrice(offering.batch_pricing?.input ?? null)}
      </span>
    </div>
  );
}

function joinWithSeparator(nodes: ReactNode[]): ReactNode[] {
  return nodes.reduce<ReactNode[]>((acc, node, idx) => {
    if (idx > 0) {
      acc.push(
        <span
          key={`sep-${idx}`}
          className="v2-offer-sub-sep"
          aria-hidden
        >
          ·
        </span>,
      );
    }
    acc.push(node);
    return acc;
  }, []);
}
