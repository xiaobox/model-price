import type { AlternativeV2 } from '../../types/v2';
import { useI18n } from '../i18n/useI18n';
import { formatOverlap, formatPct } from '../utils/format';
import './AlternativesList.css';

interface AlternativesListProps {
  alternatives: AlternativeV2[];
  onNavigate: (slug: string) => void;
}

export function AlternativesList({ alternatives, onNavigate }: AlternativesListProps) {
  const { t } = useI18n();
  return (
    <div className="v2-alts">
      {alternatives.map((alt) => {
        const cheaper = alt.delta_input_pct < 0;
        return (
          <button
            key={alt.canonical_id}
            type="button"
            className="v2-alt"
            onClick={() => onNavigate(alt.canonical_id)}
          >
            <div className="v2-alt-head">
              <span className="v2-alt-name">{alt.name}</span>
              <span className="v2-alt-overlap">
                {t('alt.match_fmt', { pct: formatOverlap(alt.capability_overlap) })}
              </span>
            </div>
            <div className="v2-alt-delta">
              <div>
                <span className="v2-alt-label">{t('alt.input')}</span>
                <span className={`v2-alt-pct${cheaper ? ' is-cheaper' : ''}`}>
                  {formatPct(alt.delta_input_pct)}
                </span>
              </div>
              <div>
                <span className="v2-alt-label">{t('alt.output')}</span>
                <span className={`v2-alt-pct${alt.delta_output_pct < 0 ? ' is-cheaper' : ''}`}>
                  {formatPct(alt.delta_output_pct)}
                </span>
              </div>
            </div>
          </button>
        );
      })}
    </div>
  );
}
