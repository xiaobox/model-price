import { Link } from 'react-router-dom';
import { useCompareBasket } from '../useCompareBasket';
import { useI18n } from '../i18n/useI18n';
import './CompareBasket.css';

export function CompareBasket() {
  const basket = useCompareBasket();
  const { t } = useI18n();

  if (basket.count === 0) return null;

  const href = `/compare/${basket.slugs.join(',')}`;

  return (
    <div className="v2-basket" role="region" aria-label={t('basket.in_compare')}>
      <div className="v2-basket-label">
        <span className="v2-basket-count num">
          {basket.count}
          <span className="v2-basket-max">/{basket.capacity}</span>
        </span>
        <span>{t('basket.in_compare')}</span>
      </div>
      <div className="v2-basket-chips">
        {basket.slugs.map((slug) => (
          <button
            key={slug}
            type="button"
            className="v2-basket-chip"
            onClick={() => basket.remove(slug)}
            title={t('basket.remove_fmt', { slug })}
          >
            {slug}
            <span className="v2-basket-chip-x">×</span>
          </button>
        ))}
      </div>
      <div className="v2-basket-actions">
        <button type="button" className="v2-btn" onClick={basket.clear}>
          {t('basket.clear')}
        </button>
        <Link to={href} className="v2-btn v2-btn-primary">
          {t('basket.go')}
        </Link>
      </div>
    </div>
  );
}
