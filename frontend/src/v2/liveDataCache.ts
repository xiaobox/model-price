import type {
  CompareResultV2,
  EntityDetailV2,
  EntityListItemV2,
  SearchResultV2,
} from '../types/v2';
import { readApiCache, removeApiCache, writeApiCache } from './apiResponseCache';

const ENTITIES_ALL_KEY = 'entities:all';

export function entitiesCacheKey(queryString: string): string {
  return `entities:${queryString || 'all'}`;
}

export function entityDetailCacheKey(slug: string): string {
  return `entity:${slug}`;
}

export function compareCacheKey(ids: string): string {
  return `compare:${ids}`;
}

function entityListItemCacheKey(slug: string): string {
  return `entity-list-item:${slug}`;
}

export function readEntityListCache(queryString: string): EntityListItemV2[] | null {
  return readApiCache<EntityListItemV2[]>(entitiesCacheKey(queryString));
}

export function readDefaultEntityListCache(): EntityListItemV2[] | null {
  return readApiCache<EntityListItemV2[]>(ENTITIES_ALL_KEY);
}

export function writeEntityListCache(
  queryString: string,
  items: EntityListItemV2[],
): void {
  writeApiCache(entitiesCacheKey(queryString), items);
  if (!queryString) writeApiCache(ENTITIES_ALL_KEY, items);
  for (const item of items) {
    writeApiCache(entityListItemCacheKey(item.slug), item);
  }
}

export function detailFromListItem(item: EntityListItemV2): EntityDetailV2 {
  return {
    entity: {
      canonical_id: item.canonical_id,
      slug: item.slug,
      name: item.name,
      family: item.family,
      maker: item.maker,
      context_length: item.context_length,
      max_output_tokens: item.max_output_tokens,
      capabilities: item.capabilities,
      input_modalities: item.input_modalities,
      output_modalities: item.output_modalities,
      mode: item.mode,
      is_open_source: item.is_open_source,
      primary_offering_provider: item.primary_offering_provider,
      sources: item.sources,
      last_refreshed: item.last_refreshed,
    },
    offerings: item.primary_offering ? [item.primary_offering] : [],
    alternatives: [],
  };
}

export function readEntityDetailCache(slug: string): EntityDetailV2 | null {
  const detail = readApiCache<EntityDetailV2>(entityDetailCacheKey(slug));
  if (detail) return detail;

  const item = readApiCache<EntityListItemV2>(entityListItemCacheKey(slug));
  if (item) return detailFromListItem(item);

  const all = readDefaultEntityListCache();
  const fromAll = all?.find((entity) => entity.slug === slug);
  return fromAll ? detailFromListItem(fromAll) : null;
}

export function writeEntityDetailCache(slug: string, detail: EntityDetailV2): void {
  writeApiCache(entityDetailCacheKey(slug), detail);
}

export function removeEntityDetailCache(slug: string): void {
  removeApiCache(entityDetailCacheKey(slug));
}

export function compareFromEntityListCache(ids: string): CompareResultV2 | null {
  const requested = ids.split(',').map((s) => s.trim()).filter(Boolean);
  if (requested.length === 0) return null;

  const all = readDefaultEntityListCache() ?? [];
  const bySlug = new Map(all.map((entity) => [entity.slug, entity]));
  const entities: EntityDetailV2[] = [];
  const missing: string[] = [];
  const capSets: Set<string>[] = [];

  for (const slug of requested) {
    const item = readApiCache<EntityListItemV2>(entityListItemCacheKey(slug)) ?? bySlug.get(slug);
    if (!item) {
      missing.push(slug);
      continue;
    }
    const detail = detailFromListItem(item);
    entities.push(detail);
    capSets.push(new Set(detail.entity.capabilities ?? []));
  }

  if (entities.length === 0) return null;

  let common: string[] = [];
  if (capSets.length > 0) {
    const [head, ...rest] = capSets;
    common = [...head].filter((cap) => rest.every((s) => s.has(cap))).sort();
  }

  return {
    entities,
    common_capabilities: common,
    requested_ids: requested,
    missing_ids: missing,
  };
}

export function searchEntityListCache(
  query: string,
  limit: number,
): SearchResultV2[] | null {
  const entities = readDefaultEntityListCache();
  if (!entities) return null;

  const ql = query.toLowerCase().trim();
  if (!ql) return [];
  const scored: Array<[number, SearchResultV2]> = [];
  for (const entity of entities) {
    const name = (entity.name ?? '').toLowerCase();
    const canon = (entity.canonical_id ?? '').toLowerCase();
    const family = (entity.family ?? '').toLowerCase();
    let rank: number;
    if (name === ql || canon === ql) rank = 0;
    else if (name.startsWith(ql) || canon.startsWith(ql)) rank = 1;
    else if (name.includes(ql) || canon.includes(ql)) rank = 2;
    else if (family.includes(ql)) rank = 3;
    else continue;
    scored.push([
      rank,
      {
        canonical_id: entity.canonical_id,
        slug: entity.slug,
        name: entity.name,
        family: entity.family ?? null,
        maker: entity.maker ?? null,
        primary_input_price: entity.primary_offering?.pricing?.input ?? null,
        primary_output_price: entity.primary_offering?.pricing?.output ?? null,
      },
    ]);
  }
  scored.sort((a, b) => {
    if (a[0] !== b[0]) return a[0] - b[0];
    return a[1].name.toLowerCase().localeCompare(b[1].name.toLowerCase());
  });
  return scored.slice(0, limit).map((pair) => pair[1]);
}
