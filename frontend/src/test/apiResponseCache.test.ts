import { describe, expect, it } from 'vitest';

import { readApiCache, writeApiCache } from '../v2/apiResponseCache';

const PREFIX = 'model-price-v2:api-cache:';

describe('apiResponseCache', () => {
  it('reads entries written with the current cache schema', () => {
    writeApiCache('entities:all', [{ slug: 'gpt-5' }]);

    expect(readApiCache('entities:all')).toEqual([{ slug: 'gpt-5' }]);
  });

  it('drops legacy entries without the current cache schema version', () => {
    localStorage.setItem(
      `${PREFIX}entities:all`,
      JSON.stringify({
        saved_at: new Date().toISOString(),
        data: [{ slug: 'old-model' }],
      }),
    );

    expect(readApiCache('entities:all')).toBeNull();
    expect(localStorage.getItem(`${PREFIX}entities:all`)).toBeNull();
  });

  it('drops entries older than the short stale-while-refresh window', () => {
    localStorage.setItem(
      `${PREFIX}entities:all`,
      JSON.stringify({
        version: 2,
        saved_at: new Date(Date.now() - 11 * 60 * 1000).toISOString(),
        data: [{ slug: 'old-model' }],
      }),
    );

    expect(readApiCache('entities:all')).toBeNull();
    expect(localStorage.getItem(`${PREFIX}entities:all`)).toBeNull();
  });
});
