#!/usr/bin/env node
// Generates frontend/public/v2-fallback.json at build time.
//
// This is the cold-start defense: when a user opens the site and
// Render's free-tier backend is asleep, the v2 frontend hydrates
// instantly from this static snapshot and then silently upgrades to
// live backend data once it's warm.
//
// The snapshot shape mirrors what /api/v2 returns:
//   - entities: EntityCoreV2[]                 (for /api/v2/entities)
//   - offerings_by_entity: {slug: OfferingV2[]}(for /api/v2/entities/:slug)
//   - alternatives_by_entity: {slug: Alt[]}    (precomputed here so the
//                                              frontend doesn't need to
//                                              re-run the ranking)
//
// Keep the alternatives algorithm in sync with
// backend/services/alternatives.py.

import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const FRONTEND_ROOT = resolve(__dirname, '..');
const ENTITIES_PATH = resolve(FRONTEND_ROOT, '../backend/data/v2/entities.json');
const OFFERINGS_PATH = resolve(FRONTEND_ROOT, '../backend/data/v2/offerings.json');
const OUTPUT_PATH = resolve(FRONTEND_ROOT, 'public/v2-fallback.json');
const LIVE_SOURCE_URL =
  process.env.V2_FALLBACK_SOURCE_URL ||
  'https://modelprice.boxtech.icu/api/v2/snapshot';
const LIVE_SOURCE_TIMEOUT_MS = 10000;

const OVERLAP_MIN = 0.5;
const LIMIT = 3;

function jaccard(a, b) {
  const setA = new Set(a);
  const setB = new Set(b);
  if (setA.size === 0 && setB.size === 0) return 1;
  const union = new Set([...setA, ...setB]);
  if (union.size === 0) return 0;
  let intersect = 0;
  for (const x of setA) if (setB.has(x)) intersect += 1;
  return intersect / union.size;
}

function primaryOffering(entity, offerings) {
  if (!offerings || offerings.length === 0) return null;
  const primary = offerings.find(
    (o) => o.provider === entity.primary_offering_provider,
  );
  return primary ?? offerings[0];
}

// Percent change from reference → candidate. Returns null when the
// delta is undefined (reference 0 and candidate non-zero), matching
// backend/services/alternatives.py::_delta_pct so the fallback
// snapshot never emits Infinity (which JSON.stringify coerces to
// null quietly and corrupts the static data).
function deltaPct(reference, candidate) {
  if (reference == null || candidate == null) return null;
  if (reference === 0) return candidate === 0 ? 0 : null;
  return ((candidate - reference) / reference) * 100;
}

function computeAlternatives(target, entities, offsByEntity) {
  const targetOffs = offsByEntity[target.slug] || [];
  const targetPrimary = primaryOffering(target, targetOffs);
  const refIn = targetPrimary?.pricing?.input;
  if (refIn == null) return [];
  const refOut = targetPrimary?.pricing?.output;
  const targetCaps = target.capabilities || [];
  const targetMode = target.mode || 'chat';

  const scored = [];
  for (const entity of entities) {
    if (entity.slug === target.slug) continue;
    if ((entity.mode || 'chat') !== targetMode) continue;
    const overlap = jaccard(targetCaps, entity.capabilities || []);
    if (overlap < OVERLAP_MIN) continue;
    const cOffs = offsByEntity[entity.slug] || [];
    const cPrimary = primaryOffering(entity, cOffs);
    const cIn = cPrimary?.pricing?.input;
    if (cIn == null) continue;
    const cOut = cPrimary?.pricing?.output;
    const dIn = deltaPct(refIn, cIn);
    const dOut = deltaPct(refOut, cOut);
    if (dIn == null) continue;
    const savings = Math.max(0, -dIn / 100);
    const score = overlap * (1 + savings);
    scored.push({ entity, overlap, dIn, dOut: dOut ?? 0, score });
  }

  const cheaper = scored
    .filter((s) => s.dIn < 0)
    .sort((a, b) => b.score - a.score || a.dIn - b.dIn);
  const picks = cheaper.slice(0, LIMIT);
  if (picks.length < LIMIT) {
    const backup = scored
      .filter((s) => s.dIn >= 0)
      .sort((a, b) => b.score - a.score || a.dIn - b.dIn);
    picks.push(...backup.slice(0, LIMIT - picks.length));
  }

  return picks.map((s) => ({
    canonical_id: s.entity.canonical_id,
    name: s.entity.name,
    delta_input_pct: Math.round(s.dIn * 10) / 10,
    delta_output_pct: Math.round(s.dOut * 10) / 10,
    capability_overlap: Math.round(s.overlap * 1000) / 1000,
  }));
}

async function loadLiveSource() {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), LIVE_SOURCE_TIMEOUT_MS);
  try {
    const response = await fetch(LIVE_SOURCE_URL, {
      signal: controller.signal,
      headers: { accept: 'application/json' },
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const raw = await response.json();
    const entities = raw.entities || [];
    const offsByEntity = raw.offerings_by_entity || {};
    if (entities.length === 0 || Object.keys(offsByEntity).length === 0) {
      throw new Error('live snapshot is empty');
    }
    return {
      entities,
      offsByEntity,
      sourceLastRefresh:
        raw.source_last_refresh || raw.generated_at || new Date().toISOString(),
      label: LIVE_SOURCE_URL,
    };
  } catch (err) {
    console.warn(
      `[build-v2-fallback] live source unavailable (${err.message}); ` +
        'falling back to backend/data/v2/*.json',
    );
    return null;
  } finally {
    clearTimeout(timeout);
  }
}

async function loadLocalSource() {
  if (!existsSync(ENTITIES_PATH) || !existsSync(OFFERINGS_PATH)) {
    throw new Error(
      `v2 data files not found. Expected:\n  ${ENTITIES_PATH}\n  ${OFFERINGS_PATH}`,
    );
  }

  const entRaw = JSON.parse(await readFile(ENTITIES_PATH, 'utf-8'));
  const offRaw = JSON.parse(await readFile(OFFERINGS_PATH, 'utf-8'));
  const entities = entRaw.entities || [];
  const offsByEntity = offRaw.by_entity || {};

  if (entities.length === 0) {
    throw new Error('No entities in backend/data/v2/entities.json');
  }
  return {
    entities,
    offsByEntity,
    sourceLastRefresh: entRaw.generated_at ?? null,
    label: 'backend/data/v2/*.json',
  };
}

async function main() {
  const source = (await loadLiveSource()) ?? (await loadLocalSource());
  const { entities, offsByEntity, sourceLastRefresh } = source;

  // Precompute alternatives for every entity so /m/:slug can render
  // instantly from the snapshot without waiting on the backend.
  const alternativesByEntity = {};
  for (const entity of entities) {
    alternativesByEntity[entity.slug] = computeAlternatives(
      entity,
      entities,
      offsByEntity,
    );
  }

  const snapshot = {
    version: 'v2-fallback.1',
    generated_at: new Date().toISOString(),
    entity_count: entities.length,
    source_last_refresh: sourceLastRefresh,
    entities,
    offerings_by_entity: offsByEntity,
    alternatives_by_entity: alternativesByEntity,
  };

  await mkdir(dirname(OUTPUT_PATH), { recursive: true });
  await writeFile(OUTPUT_PATH, JSON.stringify(snapshot));
  const bytes = JSON.stringify(snapshot).length;
  console.log(
    `v2-fallback.json written from ${source.label}: ${entities.length} entities, ` +
      `${Object.keys(offsByEntity).length} offering groups, ` +
      `${(bytes / 1024).toFixed(0)}KB raw`,
  );
}

main().catch((err) => {
  console.error('[build-v2-fallback] failed:', err);
  process.exit(1);
});
