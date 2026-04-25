import { useContext } from 'react';
import { CompareBasketContext } from './compareBasketContextValue';
import type { BasketValue } from './compareBasketContextValue';

export function useCompareBasket(): BasketValue {
  const ctx = useContext(CompareBasketContext);
  if (!ctx) {
    throw new Error('useCompareBasket must be used within CompareBasketProvider');
  }
  return ctx;
}
