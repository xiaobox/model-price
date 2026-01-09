import { useState, useEffect, useCallback } from 'react';
import type { ModelPricing, ProviderInfo, Stats, Filters, SortConfig, ViewMode } from '../types/pricing';

const API_BASE = '/api';

export function useModels() {
  const [models, setModels] = useState<ModelPricing[]>([]);
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [view, setView] = useState<ViewMode>('card');
  const [filters, setFilters] = useState<Filters>({
    provider: null,
    capability: null,
    search: '',
  });
  const [sortConfig, setSortConfig] = useState<SortConfig>({
    field: 'model_name',
    order: 'asc',
  });

  const buildQueryString = useCallback(() => {
    const params = new URLSearchParams();
    if (filters.provider) params.set('provider', filters.provider);
    if (filters.capability) params.set('capability', filters.capability);
    if (filters.search) params.set('search', filters.search);
    params.set('sort_by', sortConfig.field);
    params.set('sort_order', sortConfig.order);
    return params.toString();
  }, [filters, sortConfig]);

  const fetchModels = useCallback(async () => {
    try {
      const queryString = buildQueryString();
      const response = await fetch(`${API_BASE}/models?${queryString}`);
      if (!response.ok) throw new Error('Failed to fetch models');
      const data = await response.json();
      setModels(data);
      setError(null);
    } catch (err) {
      setError('无法连接到后端服务，请确保后端已启动 (port 8000)');
      console.error(err);
    }
  }, [buildQueryString]);

  const fetchProviders = async () => {
    try {
      const response = await fetch(`${API_BASE}/providers`);
      if (!response.ok) throw new Error('Failed to fetch providers');
      const data = await response.json();
      setProviders(data);
    } catch (err) {
      console.error('Failed to fetch providers:', err);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_BASE}/stats`);
      if (!response.ok) throw new Error('Failed to fetch stats');
      const data = await response.json();
      setStats(data);
    } catch (err) {
      console.error('Failed to fetch stats:', err);
    }
  };

  const refresh = async (provider?: string) => {
    setRefreshing(true);
    try {
      const url = provider
        ? `${API_BASE}/refresh?provider=${provider}`
        : `${API_BASE}/refresh`;
      const response = await fetch(url, { method: 'POST' });
      if (!response.ok) throw new Error('Failed to refresh');

      // Reload data after refresh
      await Promise.all([fetchModels(), fetchProviders(), fetchStats()]);
    } catch (err) {
      setError('刷新失败');
      console.error(err);
    } finally {
      setRefreshing(false);
    }
  };

  const handleSort = (field: SortConfig['field']) => {
    setSortConfig(prev => ({
      field,
      order: prev.field === field && prev.order === 'asc' ? 'desc' : 'asc',
    }));
  };

  // Initial load
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([fetchModels(), fetchProviders(), fetchStats()]);
      setLoading(false);
    };
    loadData();
  }, []);

  // Reload when filters or sort change
  useEffect(() => {
    if (!loading) {
      fetchModels();
    }
  }, [filters, sortConfig, fetchModels, loading]);

  return {
    models,
    providers,
    stats,
    loading,
    refreshing,
    error,
    view,
    setView,
    filters,
    setFilters,
    sortConfig,
    handleSort,
    refresh,
    retry: fetchModels,
  };
}
