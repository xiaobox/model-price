import { useRef, useState, useCallback } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import type { ModelPricing, ModelUpdate, SortConfig } from '../types/pricing';
import { CapabilityList, EditableCapabilityList } from './CapabilityBadge';
import { ModalityList } from './ModalityIcons';
import { getProviderDisplayName } from '../config';
import './VirtualTable.css';

interface Props {
  models: ModelPricing[];
  sortConfig: SortConfig;
  onSort: (field: SortConfig['field']) => void;
  onUpdateModel?: (modelId: string, updates: ModelUpdate) => Promise<boolean>;
  updating?: string | null;
}

function formatPrice(price: number | null): string {
  if (price === null) return '-';
  if (price === 0) return 'Free';
  return '$' + price.toFixed(2);
}

function formatNumber(num: number | null): string {
  if (num === null) return '-';
  if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
  if (num >= 1000) return (num / 1000).toFixed(0) + 'K';
  return num.toString();
}

const ROW_HEIGHT = 52;

// Fixed column widths in pixels - explicitly list all 11 columns
const COLUMNS = [
  { key: 'provider', width: 140 },
  { key: 'model', width: 220 },
  { key: 'priceInput', width: 95 },
  { key: 'priceOutput', width: 95 },
  { key: 'priceCached', width: 85 },
  { key: 'context', width: 80 },
  { key: 'maxOutput', width: 80 },
  { key: 'modalityInput', width: 110 },
  { key: 'modalityOutput', width: 110 },
  { key: 'opensource', width: 65 },
  { key: 'capabilities', width: 200 },
];

// Total table width
const TABLE_WIDTH = COLUMNS.reduce((sum, col) => sum + col.width, 0);

// Helper to get column width
const W = Object.fromEntries(COLUMNS.map(c => [c.key, c.width])) as Record<string, number>;

export function VirtualTable({ models, sortConfig, onSort, onUpdateModel, updating }: Props) {
  const parentRef = useRef<HTMLDivElement>(null);
  const [editing, setEditing] = useState<{ modelId: string; field: string; value: string } | null>(null);

  const virtualizer = useVirtualizer({
    count: models.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 10,
  });

  const renderSortIndicator = useCallback((field: SortConfig['field']) => {
    if (sortConfig.field !== field) return null;
    return <span className="sort-indicator">{sortConfig.order === 'asc' ? '↑' : '↓'}</span>;
  }, [sortConfig]);

  const handleDoubleClick = (modelId: string, field: string, value: number | null) => {
    if (!onUpdateModel) return;
    setEditing({ modelId, field, value: value?.toString() || '' });
  };

  const handleEditSubmit = async () => {
    if (!editing || !onUpdateModel) return;
    const isPricing = editing.field.startsWith('pricing_');
    const numValue = editing.value ? (isPricing ? parseFloat(editing.value) : parseInt(editing.value, 10)) : null;
    if (editing.value && (isNaN(numValue!) || numValue! < 0)) { setEditing(null); return; }

    const updates = isPricing
      ? { pricing: { [editing.field.replace('pricing_', '')]: numValue } }
      : { [editing.field]: numValue };
    await onUpdateModel(editing.modelId, updates as ModelUpdate);
    setEditing(null);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleEditSubmit();
    else if (e.key === 'Escape') setEditing(null);
  };

  const handleOpenSourceClick = async (model: ModelPricing) => {
    if (!onUpdateModel) return;
    const newValue = model.is_open_source === true ? false : model.is_open_source === false ? null : true;
    await onUpdateModel(model.id, { is_open_source: newValue });
  };

  const renderOpenSourceBadge = (value: boolean | null) => {
    if (value === true) return <span className="badge badge-open">开源</span>;
    if (value === false) return <span className="badge badge-closed">闭源</span>;
    return <span className="badge badge-unknown">未知</span>;
  };

  const renderEditableCell = (model: ModelPricing, field: string, value: number | null, isPrice = false) => {
    const isEditing = editing?.modelId === model.id && editing?.field === field;
    if (isEditing) {
      return (
        <input
          type="number"
          step={isPrice ? '0.01' : '1'}
          className="edit-input"
          value={editing.value}
          onChange={(e) => setEditing({ ...editing, value: e.target.value })}
          onBlur={handleEditSubmit}
          onKeyDown={handleKeyDown}
          autoFocus
          disabled={updating === model.id}
        />
      );
    }
    return (
      <span
        className={onUpdateModel ? 'editable' : ''}
        onDoubleClick={() => handleDoubleClick(model.id, field, value)}
        title={onUpdateModel ? '双击编辑' : ''}
      >
        {isPrice ? formatPrice(value) : formatNumber(value)}
      </span>
    );
  };

  // Cell style helper - fixed width, no shrink/grow
  const cellStyle = (width: number): React.CSSProperties => ({
    width, minWidth: width, maxWidth: width, flexShrink: 0, flexGrow: 0,
  });

  return (
    <div className="vtable-container" ref={parentRef}>
      {/* Inner wrapper: min-width prevents compression */}
      <div className="vtable-inner" style={{ minWidth: TABLE_WIDTH }}>
        {/* Header row */}
        <div className="vtable-header" style={{ minWidth: TABLE_WIDTH }}>
          <div className="vtable-cell vtable-cell-text" style={cellStyle(W.provider)}>提供商</div>
          <div className="vtable-cell vtable-cell-text sortable" style={cellStyle(W.model)} onClick={() => onSort('model_name')}>
            模型 {renderSortIndicator('model_name')}
          </div>
          <div className="vtable-cell vtable-cell-numeric sortable" style={cellStyle(W.priceInput)} onClick={() => onSort('input')}>
            输入<span className="unit-hint">$/M</span>{renderSortIndicator('input')}
          </div>
          <div className="vtable-cell vtable-cell-numeric sortable" style={cellStyle(W.priceOutput)} onClick={() => onSort('output')}>
            输出<span className="unit-hint">$/M</span>{renderSortIndicator('output')}
          </div>
          <div className="vtable-cell vtable-cell-numeric" style={cellStyle(W.priceCached)}>
            缓存<span className="unit-hint">$/M</span>
          </div>
          <div className="vtable-cell vtable-cell-numeric sortable" style={cellStyle(W.context)} onClick={() => onSort('context_length')}>
            上下文{renderSortIndicator('context_length')}
          </div>
          <div className="vtable-cell vtable-cell-numeric" style={cellStyle(W.maxOutput)}>最大输出</div>
          <div className="vtable-cell vtable-cell-wrap" style={cellStyle(W.modalityInput)}>输入</div>
          <div className="vtable-cell vtable-cell-wrap" style={cellStyle(W.modalityOutput)}>输出</div>
          <div className="vtable-cell vtable-cell-wrap" style={cellStyle(W.opensource)}>开源</div>
          <div className="vtable-cell vtable-cell-wrap" style={{ minWidth: W.capabilities, flexGrow: 1, borderRight: 'none' }}>能力</div>
        </div>

        {/* Virtualized body */}
        <div style={{ height: virtualizer.getTotalSize(), position: 'relative' }}>
          {virtualizer.getVirtualItems().map((row) => {
            const model = models[row.index];
            const isEven = row.index % 2 === 0;
            return (
              <div
                key={model.id}
                className="vtable-row"
                style={{
                  minWidth: TABLE_WIDTH,
                  transform: `translateY(${row.start}px)`,
                  background: isEven ? 'var(--bg-card)' : 'var(--bg-table-stripe)',
                }}
              >
                <div className="vtable-cell vtable-cell-text provider-cell" style={cellStyle(W.provider)} title={getProviderDisplayName(model.provider)}>
                  <span className="vtable-cell-text-inner">{getProviderDisplayName(model.provider)}</span>
                </div>
                <div className="vtable-cell vtable-cell-text model-name-cell" style={cellStyle(W.model)} title={model.model_name}>
                  <span className="vtable-cell-text-inner">{model.model_name}</span>
                </div>
                <div className="vtable-cell vtable-cell-numeric" style={cellStyle(W.priceInput)}>
                  {renderEditableCell(model, 'pricing_input', model.pricing.input, true)}
                </div>
                <div className="vtable-cell vtable-cell-numeric" style={cellStyle(W.priceOutput)}>
                  {renderEditableCell(model, 'pricing_output', model.pricing.output, true)}
                </div>
                <div className="vtable-cell vtable-cell-numeric vtable-cell-muted" style={cellStyle(W.priceCached)}>
                  {renderEditableCell(model, 'pricing_cached_input', model.pricing.cached_input, true)}
                </div>
                <div className="vtable-cell vtable-cell-numeric" style={cellStyle(W.context)}>
                  {renderEditableCell(model, 'context_length', model.context_length)}
                </div>
                <div className="vtable-cell vtable-cell-numeric" style={cellStyle(W.maxOutput)}>
                  {renderEditableCell(model, 'max_output_tokens', model.max_output_tokens)}
                </div>
                <div className="vtable-cell vtable-cell-wrap" style={cellStyle(W.modalityInput)}>
                  <ModalityList modalities={model.input_modalities || []} />
                </div>
                <div className="vtable-cell vtable-cell-wrap" style={cellStyle(W.modalityOutput)}>
                  <ModalityList modalities={model.output_modalities || []} />
                </div>
                <div
                  className={`vtable-cell vtable-cell-wrap ${onUpdateModel ? 'clickable' : ''}`}
                  style={cellStyle(W.opensource)}
                  onClick={() => handleOpenSourceClick(model)}
                  title={onUpdateModel ? '点击切换' : ''}
                >
                  {renderOpenSourceBadge(model.is_open_source)}
                </div>
                <div className="vtable-cell vtable-cell-wrap capabilities-cell" style={{ minWidth: W.capabilities, flexGrow: 1, borderRight: 'none' }}>
                  {onUpdateModel ? (
                    <EditableCapabilityList
                      capabilities={model.capabilities}
                      onUpdate={async (caps) => { await onUpdateModel(model.id, { capabilities: caps }); }}
                      editable={true}
                      updating={updating === model.id}
                    />
                  ) : (
                    <CapabilityList capabilities={model.capabilities} />
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
