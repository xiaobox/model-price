import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { formatRelativeTime } from '../v2/utils/relativeTime';

// Freeze "now" so Intl.RelativeTimeFormat output is stable across runs.
const NOW = new Date('2026-04-17T12:00:00.000Z').getTime();

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(NOW);
});

afterEach(() => {
  vi.useRealTimers();
});

describe('formatRelativeTime', () => {
  it('returns empty string for null / empty / invalid input', () => {
    expect(formatRelativeTime(null, 'en')).toBe('');
    expect(formatRelativeTime('', 'en')).toBe('');
    expect(formatRelativeTime('not-a-date', 'en')).toBe('');
  });

  it('formats sub-minute deltas as "now" / "现在"', () => {
    const thirtySecAgo = new Date(NOW - 30_000).toISOString();
    expect(formatRelativeTime(thirtySecAgo, 'en')).toBe('now');
    expect(formatRelativeTime(thirtySecAgo, 'zh')).toBe('现在');
  });

  it('formats sub-hour deltas as minutes', () => {
    const iso = new Date(NOW - 5 * 60_000).toISOString();
    expect(formatRelativeTime(iso, 'en')).toMatch(/5 minutes ago/);
    expect(formatRelativeTime(iso, 'zh')).toContain('5');
    expect(formatRelativeTime(iso, 'zh')).toContain('分钟');
  });

  it('formats sub-day deltas as hours (with "yesterday" only when numeric:auto triggers)', () => {
    const twoHours = new Date(NOW - 2 * 3600_000).toISOString();
    expect(formatRelativeTime(twoHours, 'en')).toMatch(/2 hours ago/);
    expect(formatRelativeTime(twoHours, 'zh')).toMatch(/2\s*小时前/);
  });

  it('formats day-level deltas as days', () => {
    const twoDays = new Date(NOW - 2 * 86_400_000).toISOString();
    expect(formatRelativeTime(twoDays, 'en')).toMatch(/2 days ago/);
    expect(formatRelativeTime(twoDays, 'zh')).toMatch(/前天|2\s*天前/);
  });

  it('treats future timestamps as "now" (never shows "in 3 minutes")', () => {
    // Snapshot generated_at can legitimately be a few ms in the future
    // due to clock skew between backend and the user's device. We want
    // that to read as "now", not "in 3 seconds".
    const fiveSecFuture = new Date(NOW + 5000).toISOString();
    expect(formatRelativeTime(fiveSecFuture, 'en')).toBe('now');
  });

  it('rounds down at boundaries so "59 minutes ago" does not round up to 1 hour', () => {
    const fiftyNine = new Date(NOW - 59 * 60_000).toISOString();
    expect(formatRelativeTime(fiftyNine, 'en')).toMatch(/59 minutes ago/);
  });

  it('crosses into hours exactly at 60 minutes', () => {
    const exactly60 = new Date(NOW - 60 * 60_000).toISOString();
    expect(formatRelativeTime(exactly60, 'en')).toMatch(/hour/);
  });
});
