import type { ViewMode } from '../types/pricing';

interface ViewToggleProps {
  view: ViewMode;
  onViewChange: (view: ViewMode) => void;
}

export function ViewToggle({ view, onViewChange }: ViewToggleProps) {
  return (
    <div className="view-toggle">
      <button
        className={`view-btn ${view === 'table' ? 'active' : ''}`}
        onClick={() => onViewChange('table')}
        title="表格视图"
      >
        ☰
      </button>
      <button
        className={`view-btn ${view === 'card' ? 'active' : ''}`}
        onClick={() => onViewChange('card')}
        title="卡片视图"
      >
        ▦
      </button>
    </div>
  );
}
