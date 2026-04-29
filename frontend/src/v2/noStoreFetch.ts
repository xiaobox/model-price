export function noStoreFetch(
  input: RequestInfo | URL,
  init: RequestInit = {},
): Promise<Response> {
  const headers = new Headers(init.headers);
  headers.set('Cache-Control', 'no-cache');
  headers.set('Pragma', 'no-cache');

  return fetch(input, {
    ...init,
    cache: 'no-store',
    headers,
  });
}
