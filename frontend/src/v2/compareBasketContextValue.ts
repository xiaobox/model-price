import { createContext } from 'react';

export interface BasketValue {
  slugs: string[];
  count: number;
  capacity: number;
  isFull: boolean;
  toggle: (slug: string) => { added: boolean; full: boolean };
  add: (slug: string) => void;
  remove: (slug: string) => void;
  clear: () => void;
  has: (slug: string) => boolean;
}

export const CompareBasketContext = createContext<BasketValue | null>(null);
