import { useState } from 'react';
import { useI18n } from '../i18n/useI18n';
import './ShareActions.css';

interface ShareActionsProps {
  /** Public URL to share. Defaults to PUBLIC_BASE_URL + path if omitted. */
  url: string;
  /** Short name for display in the X/Twitter intent. */
  name: string;
  /** Maker ("Anthropic", "OpenAI", …) for social copy. */
  maker: string;
}

export function ShareActions({ url, name, maker }: ShareActionsProps) {
  const { t } = useI18n();
  const [copied, setCopied] = useState(false);

  const onCopyLink = async () => {
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // ignore clipboard errors
    }
  };

  const xIntentUrl = (() => {
    const text = t('share.x_text_fmt', { name, maker });
    const params = new URLSearchParams({ text, url });
    return `https://twitter.com/intent/tweet?${params.toString()}`;
  })();

  return (
    <div className="v2-share">
      <span className="v2-share-label">{t('share.label')}</span>
      <button
        type="button"
        className={`v2-share-btn${copied ? ' is-copied' : ''}`}
        onClick={onCopyLink}
      >
        <ShareIcon type="link" />
        <span>{copied ? t('share.link_copied') : t('share.copy_link')}</span>
      </button>
      <a
        href={xIntentUrl}
        target="_blank"
        rel="noreferrer"
        className="v2-share-btn"
      >
        <ShareIcon type="x" />
        <span>{t('share.x_button')}</span>
      </a>
    </div>
  );
}

function ShareIcon({ type }: { type: 'link' | 'x' }) {
  const common = {
    width: 14,
    height: 14,
    viewBox: '0 0 16 16',
    fill: 'currentColor',
    'aria-hidden': true as const,
  };
  if (type === 'link') {
    return (
      <svg {...common}>
        <path d="M7.05 2.95a3.5 3.5 0 014.95 4.95l-2 2a3.5 3.5 0 01-4.94 0 .75.75 0 111.06-1.06 2 2 0 002.82 0l2-2a2 2 0 00-2.83-2.83l-.76.77a.75.75 0 11-1.06-1.06l.76-.77zM8.95 13.05a3.5 3.5 0 01-4.95-4.95l2-2a3.5 3.5 0 014.94 0 .75.75 0 01-1.06 1.06 2 2 0 00-2.82 0l-2 2a2 2 0 002.83 2.83l.76-.77a.75.75 0 111.06 1.06l-.76.77z" />
      </svg>
    );
  }
  // x / twitter
  return (
    <svg {...common}>
      <path d="M9.29 7.13 14.23 1.5h-1.17L8.77 6.39 5.34 1.5H1.38l5.18 7.39-5.18 5.61h1.17l4.53-4.9 3.62 4.9h3.96L9.29 7.13zm-1.6 1.73-.53-.74L2.97 2.38h1.8l3.37 4.72.53.74 4.38 6.12h-1.8L7.7 8.86z" />
    </svg>
  );
}
