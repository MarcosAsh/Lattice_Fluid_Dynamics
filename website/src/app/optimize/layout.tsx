import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Shape Optimization',
  description:
    'Reduce drag in the browser with SPSA optimization over FFD shape parameters.',
};

export default function OptimizeLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return children;
}
