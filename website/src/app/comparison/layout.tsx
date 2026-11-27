import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'ML vs LBM',
  description:
    'Compare the ML surrogate against Lattice Boltzmann results for drag and lift.',
};

export default function ComparisonLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return children;
}
