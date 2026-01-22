import './App.css';
import { useModels } from './hooks/useModels';
import {
  FilterBar,
  ModelCard,
  VirtualTable,
  ViewToggle,
} from './components';
import { APP_VERSION } from './config';

function formatPrice(price: number): string {
  if (price === 0) return 'Free';
  return '$' + price.toFixed(2);
}

function App() {
  const {
    models,
    providers,
    families,
    stats,
    loading,
    error,
    view,
    filters,
    sortConfig,
    setView,
    setFilters,
    handleSort,
    refresh,
  } = useModels();

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <span className="logo-icon">⬡</span>
            <h1>Model Price</h1>
            <span className="version mono">v{APP_VERSION}</span>
          </div>
          <p className="tagline">AI 模型定价一览表</p>
        </div>
        <div className="header-glow"></div>
      </header>

      {/* Stats Bar */}
      <section className="stats-bar">
        <div className="stat-item">
          <span className="stat-label">模型数量</span>
          <span className="stat-value mono">{stats?.total_models || 0}</span>
        </div>
        <div className="stat-divider"></div>
        <div className="stat-item">
          <span className="stat-label">提供商数量</span>
          <span className="stat-value mono">{stats?.providers || 0}</span>
        </div>
        <div className="stat-divider"></div>
        <div className="stat-item">
          <span className="stat-label">平均输入价格</span>
          <span className="stat-value mono">
            {formatPrice(stats?.avg_input_price || 0)}
            <span className="stat-unit">/M</span>
          </span>
        </div>
        <div className="stat-divider"></div>
        <div className="stat-item">
          <span className="stat-label">平均输出价格</span>
          <span className="stat-value mono">
            {formatPrice(stats?.avg_output_price || 0)}
            <span className="stat-unit">/M</span>
          </span>
        </div>
      </section>

      {/* Main Content */}
      <main className="main-content">
        {loading ? (
          <div className="loading">
            <div className="loading-spinner"></div>
            <p>正在加载数据...</p>
          </div>
        ) : error ? (
          <div className="error-card">
            <span className="error-icon">⚠</span>
            <p>{error}</p>
            <button onClick={() => refresh()} className="retry-btn">
              重试连接
            </button>
          </div>
        ) : (
          <>
            {/* Controls */}
            <div className="controls">
              <FilterBar
                filters={filters}
                onFiltersChange={setFilters}
                providers={providers}
                families={families}
              />
              <div className="controls-right">
                <ViewToggle view={view} onViewChange={setView} />
              </div>
            </div>

            {/* Model Display */}
            {models.length === 0 ? (
              <div className="empty-state">
                <span className="empty-icon">📭</span>
                <p>没有找到匹配的模型</p>
              </div>
            ) : view === 'card' ? (
              <div className="model-grid">
                {models.map((model, index) => (
                  <ModelCard key={model.id} model={model} index={index} />
                ))}
              </div>
            ) : (
              <VirtualTable
                models={models}
                sortConfig={sortConfig}
                onSort={handleSort}
              />
            )}

            {/* Result count */}
            <div className="result-count">
              显示 {models.length} 个模型
            </div>
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>
          <span className="mono">{'<'}</span>
          Built with FastAPI + React
          <span className="mono">{'>'}</span>
        </p>
        <p className="footer-note">价格单位：$/百万 tokens</p>
      </footer>
    </div>
  );
}

export default App;
