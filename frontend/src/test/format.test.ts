import { describe, it, expect } from 'vitest';
import {
  formatPrice,
  formatContext,
  formatPct,
  formatOverlap,
  capabilityLabel,
  makerColor,
} from '../v2/utils/format';

describe('formatPrice', () => {
  it('renders em-dash for null/undefined', () => {
    expect(formatPrice(null)).toBe('—');
    expect(formatPrice(undefined)).toBe('—');
  });

  it('renders $0 for literal zero', () => {
    expect(formatPrice(0)).toBe('$0');
  });

  it('uses four decimals for sub-cent prices', () => {
    expect(formatPrice(0.0001)).toBe('$0.0001');
    expect(formatPrice(0.0075)).toBe('$0.0075');
  });

  it('uses three decimals for sub-dollar prices', () => {
    expect(formatPrice(0.15)).toBe('$0.150');
    expect(formatPrice(0.5)).toBe('$0.500');
  });

  it('uses two decimals for normal prices', () => {
    expect(formatPrice(3)).toBe('$3.00');
    expect(formatPrice(15.5)).toBe('$15.50');
    expect(formatPrice(720)).toBe('$720.00');
  });
});

describe('formatContext', () => {
  it('renders em-dash for missing context', () => {
    expect(formatContext(null)).toBe('—');
    expect(formatContext(undefined)).toBe('—');
    expect(formatContext(0)).toBe('—');
  });

  it('formats millions with at most one decimal', () => {
    expect(formatContext(1_000_000)).toBe('1M');
    expect(formatContext(2_000_000)).toBe('2M');
    expect(formatContext(1_048_576)).toBe('1.0M');
  });

  it('rounds thousands', () => {
    expect(formatContext(128_000)).toBe('128K');
    expect(formatContext(200_000)).toBe('200K');
    expect(formatContext(1_000)).toBe('1K');
  });

  it('passes through sub-1000', () => {
    expect(formatContext(512)).toBe('512');
  });
});

describe('formatPct', () => {
  it('prefixes positive with +', () => {
    expect(formatPct(10)).toBe('+10%');
    expect(formatPct(0.4)).toBe('+0%'); // rounds
  });

  it('leaves negatives alone', () => {
    expect(formatPct(-50)).toBe('-50%');
    expect(formatPct(-98.3)).toBe('-98%');
  });

  it('renders zero without a plus sign', () => {
    expect(formatPct(0)).toBe('0%');
  });
});

describe('formatOverlap', () => {
  it('converts 0..1 float to integer percent', () => {
    expect(formatOverlap(0.8)).toBe('80%');
    expect(formatOverlap(1)).toBe('100%');
    expect(formatOverlap(0)).toBe('0%');
    expect(formatOverlap(0.834)).toBe('83%');
  });
});

describe('capabilityLabel', () => {
  it('returns known label', () => {
    expect(capabilityLabel('vision')).toBe('Vision');
    expect(capabilityLabel('tool_use')).toBe('Tools');
  });

  it('passes unknown through verbatim', () => {
    expect(capabilityLabel('some_new_capability')).toBe('some_new_capability');
  });
});

describe('makerColor', () => {
  it('returns a CSS var for known makers', () => {
    expect(makerColor('Anthropic')).toContain('var(--maker-anthropic');
    expect(makerColor('OpenAI')).toContain('var(--maker-openai');
  });

  it('returns default for unknown/null', () => {
    expect(makerColor(null)).toContain('var(--maker-default');
    expect(makerColor('NobodyEverHeardOf')).toContain('var(--maker-default');
  });
});
