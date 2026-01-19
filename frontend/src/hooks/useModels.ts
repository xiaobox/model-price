import { useState, useEffect, useCallback } from 'react';
import type { ModelPricing, ModelUpdate, ProviderInfo, ModelFamily, Stats, Filters, SortConfig, ViewMode } from '../types/pricing';
import { API_BASE } from '../config';

export function useModels() {
  const [models, setModels] = useState<ModelPricing[]>([]);
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [families, setFamilies] = useState<ModelFamily[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [updating, setUpdating] = useState<string | null>(null);

  const [view, setView] = useState<ViewMode>('table');
  const [filters, setFilters] = useState<Filters>({
    provider: null,
    capability: null,
    family: null,
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
    if (filters.family) params.set('family', filters.family);
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

  const fetchProviders = useCallback(async () => {
    try {
      // Build query string excluding provider filter (to show available providers)
      const params = new URLSearchParams();
      if (filters.capability) params.set('capability', filters.capability);
      if (filters.family) params.set('family', filters.family);
      if (filters.search) params.set('search', filters.search);
      const queryString = params.toString();
      const url = queryString ? `${API_BASE}/providers?${queryString}` : `${API_BASE}/providers`;

      const response = await fetch(url);
      if (!response.ok) throw new Error('Failed to fetch providers');
      const data = await response.json();
      setProviders(data);
    } catch (err) {
      console.error('Failed to fetch providers:', err);
    }
  }, [filters.capability, filters.family, filters.search]);

  const fetchFamilies = useCallback(async () => {
    try {
      // Build query string excluding family filter (to show available families)
      const params = new URLSearchParams();
      if (filters.provider) params.set('provider', filters.provider);
      if (filters.capability) params.set('capability', filters.capability);
      if (filters.search) params.set('search', filters.search);
      const queryString = params.toString();
      const url = queryString ? `${API_BASE}/families?${queryString}` : `${API_BASE}/families`;

      const response = await fetch(url);
      if (!response.ok) throw new Error('Failed to fetch families');
      const data = await response.json();
      setFamilies(data);
    } catch (err) {
      console.error('Failed to fetch families:', err);
    }
  }, [filters.provider, filters.capability, filters.search]);

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
      await Promise.all([fetchModels(), fetchProviders(), fetchFamilies(), fetchStats()]);
    } catch (err) {
      setError('刷新失败');
      console.error(err);
    } finally {
      setRefreshing(false);
    }
  };

  const updateModel = async (modelId: string, updates: ModelUpdate): Promise<boolean> => {
    setUpdating(modelId);
    try {
      const response = await fetch(`${API_BASE}/models/${encodeURIComponent(modelId)}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates),
      });
      if (!response.ok) throw new Error('Failed to update model');

      const updatedModel: ModelPricing = await response.json();
      setModels(prev => prev.map(m => m.id === modelId ? updatedModel : m));
      return true;
    } catch (err) {
      setError('更新失败');
      console.error(err);
      return false;
    } finally {
      setUpdating(null);
    }
  };

  const handleSort = (field: SortConfig['field']) => {
    setSortConfig(prev => ({
      field,
      order: prev.field === field && prev.order === 'asc' ? 'desc' : 'asc',
    }));
  };

  // Initial load - intentionally empty deps to run only on mount
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([fetchModels(), fetchProviders(), fetchFamilies(), fetchStats()]);
      setLoading(false);
    };
    loadData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Reload models, providers and families when filters change
  useEffect(() => {
    if (!loading) {
      fetchModels();
      fetchProviders();
      fetchFamilies();
    }
  }, [filters, sortConfig, fetchModels, fetchProviders, fetchFamilies, loading]);

  return {
    models,
    providers,
    families,
    stats,
    loading,
    refreshing,
    updating,
    error,
    view,
    setView,
    filters,
    setFilters,
    sortConfig,
    handleSort,
    refresh,
    updateModel,
    retry: fetchModels,
  };
}
