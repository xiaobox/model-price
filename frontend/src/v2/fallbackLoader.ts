// Client-side v2 snapshot loader.
//
// Hooks call loadFallback() on mount while the main live API request
// is already in flight. We first try a CDN-shareable live snapshot
// produced by the backend; that gives first-time visitors the latest
// successful data another user warmed. If that shared cache is empty
// or slow, we fall back to the static Vite-bundled snapshot.

import type {
  AlternativeV2,
  CompareResultV2,
  EntitiesListQuery,
  EntityCoreV2,
  EntityDetailV2,
  EntityListItemV2,
  OfferingV2,
  SearchResultV2,
} from '../types/v2';
import { API_V2_BASE } from '../config';

interface V2Snapshot {
  version: string;
  generated_at: string;
  entity_count: number;
  source_last_refresh: string | null;
  entities: EntityCoreV2[];
  offerings_by_entity: Record<string, OfferingV2[]>;
  alternatives_by_entity: Record<string, AlternativeV2[]>;
}

const FALLBACK_URL = '/v2-fallback.json';
const LIVE_SNAPSHOT_URL = `${API_V2_BASE}/snapshot`;
const LIVE_SNAPSHOT_TIMEOUT_MS = 2500;

let cached: V2Snapshot | null = null;
let loading: Promise<V2Snapshot | null> | null = null;

async function fetchJsonSnapshot(
  url: string,
  init?: RequestInit,
): Promise<V2Snapshot | null> {
  const response = await fetch(url, init);
  if (!response.ok) return null;
  return (await response.json()) as V2Snapshot;
}

async function loadLiveSnapshot(): Promise<V2Snapshot | null> {
  const controller = new AbortController();
  let timeout: ReturnType<typeof setTimeout> | null = null;
  try {
    const request = fetchJsonSnapshot(LIVE_SNAPSHOT_URL, {
      signal: controller.signal,
    }).catch(() => null);
    const deadline = new Promise<null>((resolve) => {
      timeout = setTimeout(() => {
        controller.abort();
        resolve(null);
      }, LIVE_SNAPSHOT_TIMEOUT_MS);
    });
    return await Promise.race([request, deadline]);
  } catch {
    return null;
  } finally {
    if (timeout) clearTimeout(timeout);
  }
}

async function loadStaticSnapshot(): Promise<V2Snapshot | null> {
  try {
    return await fetchJsonSnapshot(FALLBACK_URL, { cache: 'no-store' });
  } catch {
    return null;
  }
}

export async function loadFallback(): Promise<V2Snapshot | null> {
  if (cached) return cached;
  if (loading) return loading;
  loading = (async () => {
    const data = (await loadLiveSnapshot()) ?? (await loadStaticSnapshot());
    cached = data;
    loading = null;
    return data;
  })();
  return loading;
}

export function resetFallbackCacheForTests(): void {
  cached = null;
  loading = null;
}

// ─── Primary offering helper ────────────────────────────────

function primaryOffering(
  entity: EntityCoreV2,
  offerings: OfferingV2[] | undefined,
): OfferingV2 | null {
  if (!offerings || offerings.length === 0) return null;
  return (
    offerings.find((o) => o.provider === entity.primary_offering_provider) ??
    offerings[0]
  );
}

function toListItem(
  entity: EntityCoreV2,
  offs: OfferingV2[] | undefined,
): EntityListItemV2 {
  return {
    ...entity,
    primary_offering: primaryOffering(entity, offs),
  };
}

// ─── Query replicas of the backend endpoints ────────────────

export function listFromFallback(
  snapshot: V2Snapshot,
  query: EntitiesListQuery,
): EntityListItemV2[] {
  let list = snapshot.entities as EntityCoreV2[];

  if (query.q) {
    const ql = query.q.toLowerCase();
    list = list.filter(
      (e) =>
        (e.name ?? '').toLowerCase().includes(ql) ||
        (e.canonical_id ?? '').toLowerCase().includes(ql) ||
        (e.family ?? '').toLowerCase().includes(ql),
    );
  }
  if (query.family) list = list.filter((e) => e.family === query.family);
  if (query.maker) list = list.filter((e) => e.maker === query.maker);
  if (query.capability) {
    list = list.filter((e) => (e.capabilities ?? []).includes(query.capability!));
  }
  if (query.min_context != null) {
    list = list.filter(
      (e) => (e.context_length ?? 0) >= query.min_context!,
    );
  }

  let items = list.map((e) => toListItem(e, snapshot.offerings_by_entity[e.slug]));

  if (query.max_input_price != null) {
    items = items.filter((item) => {
      const price = item.primary_offering?.pricing?.input;
      return price != null && price <= query.max_input_price!;
    });
  }

  const sort = query.sort ?? 'name';
  const reverse = query.order === 'desc';
  const getPrice = (item: EntityListItemV2, field: 'input' | 'output'): number => {
    const value = item.primary_offering?.pricing?.[field];
    return value != null ? value : Infinity;
  };
  const sorter = (a: EntityListItemV2, b: EntityListItemV2): number => {
    let d = 0;
    if (sort === 'input') d = getPrice(a, 'input') - getPrice(b, 'input');
    else if (sort === 'output') d = getPrice(a, 'output') - getPrice(b, 'output');
    else if (sort === 'context')
      d = (a.context_length ?? 0) - (b.context_length ?? 0);
    else d = (a.name ?? '').toLowerCase().localeCompare((b.name ?? '').toLowerCase());
    return reverse ? -d : d;
  };
  return [...items].sort(sorter);
}

export function detailFromFallback(
  snapshot: V2Snapshot,
  slug: string,
): EntityDetailV2 | null {
  const entity = snapshot.entities.find((e) => e.slug === slug);
  if (!entity) return null;
  return {
    entity,
    offerings: snapshot.offerings_by_entity[slug] ?? [],
    alternatives: snapshot.alternatives_by_entity[slug] ?? [],
  };
}

export function compareFromFallback(
  snapshot: V2Snapshot,
  ids: string[],
): CompareResultV2 {
  const cleaned = ids.map((s) => s.trim()).filter(Boolean);
  const entities: EntityDetailV2[] = [];
  const missing: string[] = [];
  const capSets: Set<string>[] = [];
  for (const slug of cleaned) {
    const detail = detailFromFallback(snapshot, slug);
    if (!detail) {
      missing.push(slug);
      continue;
    }
    entities.push(detail);
    capSets.push(new Set(detail.entity.capabilities ?? []));
  }
  let common: string[] = [];
  if (capSets.length > 0) {
    const [head, ...rest] = capSets;
    common = [...head].filter((cap) => rest.every((s) => s.has(cap))).sort();
  }
  return {
    entities,
    common_capabilities: common,
    requested_ids: cleaned,
    missing_ids: missing,
  };
}

export function searchFallback(
  snapshot: V2Snapshot,
  query: string,
  limit = 10,
): SearchResultV2[] {
  const ql = query.toLowerCase().trim();
  if (!ql) return [];
  const scored: Array<[number, SearchResultV2]> = [];
  for (const entity of snapshot.entities) {
    const name = (entity.name ?? '').toLowerCase();
    const canon = (entity.canonical_id ?? '').toLowerCase();
    const family = (entity.family ?? '').toLowerCase();
    let rank: number;
    if (name === ql || canon === ql) rank = 0;
    else if (name.startsWith(ql) || canon.startsWith(ql)) rank = 1;
    else if (name.includes(ql) || canon.includes(ql)) rank = 2;
    else if (family.includes(ql)) rank = 3;
    else continue;
    const primary = primaryOffering(
      entity,
      snapshot.offerings_by_entity[entity.slug],
    );
    scored.push([
      rank,
      {
        canonical_id: entity.canonical_id,
        slug: entity.slug,
        name: entity.name,
        family: entity.family ?? null,
        maker: entity.maker ?? null,
        primary_input_price: primary?.pricing?.input ?? null,
        primary_output_price: primary?.pricing?.output ?? null,
      },
    ]);
  }
  scored.sort((a, b) => {
    if (a[0] !== b[0]) return a[0] - b[0];
    return a[1].name.toLowerCase().localeCompare(b[1].name.toLowerCase());
  });
  return scored.slice(0, limit).map((pair) => pair[1]);
}
