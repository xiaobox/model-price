import { useI18n } from '../i18n/useI18n';

export function LocaleToggle() {
  const { locale, toggle, t } = useI18n();
  return (
    <button
      type="button"
      className="v2-topbar-chip"
      onClick={toggle}
      title={t('locale.switch_tooltip')}
      aria-label={t('locale.switch_tooltip')}
    >
      {locale === 'en' ? t('locale.zh') : t('locale.en')}
    </button>
  );
}
