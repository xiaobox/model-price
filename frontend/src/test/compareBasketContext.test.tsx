import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { userEvent } from '@testing-library/user-event';
import { CompareBasketProvider } from '../v2/compareBasketContext';
import { useCompareBasket } from '../v2/useCompareBasket';

function Probe() {
  const basket = useCompareBasket();
  return (
    <div>
      <span data-testid="count">{basket.count}</span>
      <span data-testid="isFull">{basket.isFull ? 'yes' : 'no'}</span>
      <span data-testid="slugs">{basket.slugs.join(',')}</span>
      <button onClick={() => basket.toggle('alpha')}>toggle alpha</button>
      <button onClick={() => basket.add('beta')}>add beta</button>
      <button onClick={() => basket.add('gamma')}>add gamma</button>
      <button onClick={() => basket.add('delta')}>add delta</button>
      <button onClick={() => basket.add('epsilon')}>add epsilon</button>
      <button onClick={() => basket.remove('beta')}>remove beta</button>
      <button onClick={() => basket.clear()}>clear</button>
    </div>
  );
}

function setup() {
  const user = userEvent.setup();
  render(
    <CompareBasketProvider>
      <Probe />
    </CompareBasketProvider>,
  );
  return { user };
}

describe('CompareBasketProvider', () => {
  it('starts empty', () => {
    setup();
    expect(screen.getByTestId('count').textContent).toBe('0');
    expect(screen.getByTestId('isFull').textContent).toBe('no');
  });

  it('toggle adds then removes', async () => {
    const { user } = setup();
    await user.click(screen.getByText('toggle alpha'));
    expect(screen.getByTestId('count').textContent).toBe('1');
    await user.click(screen.getByText('toggle alpha'));
    expect(screen.getByTestId('count').textContent).toBe('0');
  });

  it('enforces 4-item capacity', async () => {
    const { user } = setup();
    await user.click(screen.getByText('toggle alpha'));
    await user.click(screen.getByText('add beta'));
    await user.click(screen.getByText('add gamma'));
    await user.click(screen.getByText('add delta'));
    expect(screen.getByTestId('count').textContent).toBe('4');
    expect(screen.getByTestId('isFull').textContent).toBe('yes');
    // fifth add should be silently ignored
    await user.click(screen.getByText('add epsilon'));
    expect(screen.getByTestId('count').textContent).toBe('4');
    expect(screen.getByTestId('slugs').textContent).not.toContain('epsilon');
  });

  it('remove frees a slot', async () => {
    const { user } = setup();
    await user.click(screen.getByText('toggle alpha'));
    await user.click(screen.getByText('add beta'));
    await user.click(screen.getByText('add gamma'));
    await user.click(screen.getByText('add delta'));
    await user.click(screen.getByText('remove beta'));
    expect(screen.getByTestId('count').textContent).toBe('3');
    await user.click(screen.getByText('add epsilon'));
    expect(screen.getByTestId('count').textContent).toBe('4');
    expect(screen.getByTestId('slugs').textContent).toContain('epsilon');
  });

  it('clear empties the basket', async () => {
    const { user } = setup();
    await user.click(screen.getByText('toggle alpha'));
    await user.click(screen.getByText('add beta'));
    await user.click(screen.getByText('clear'));
    expect(screen.getByTestId('count').textContent).toBe('0');
  });

  it('persists to sessionStorage', async () => {
    const { user } = setup();
    await user.click(screen.getByText('toggle alpha'));
    await user.click(screen.getByText('add beta'));
    const raw = sessionStorage.getItem('model-price-v2:compare-basket');
    expect(raw).toBeTruthy();
    const parsed = raw ? (JSON.parse(raw) as string[]) : [];
    expect(parsed).toEqual(['alpha', 'beta']);
  });

  it('throws when hook used outside provider', () => {
    // This component deliberately calls the hook with no provider
    function Bad() {
      useCompareBasket();
      return null;
    }
    expect(() => render(<Bad />)).toThrow(/within CompareBasketProvider/);
  });
});
