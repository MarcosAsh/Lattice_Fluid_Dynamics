import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: 'Lattice',
  description:
    'GPU-accelerated wind tunnel simulation using Lattice Boltzmann Methods and OpenGL compute shaders',
  openGraph: {
    title: 'Lattice',
    description:
      'GPU-accelerated wind tunnel simulation using LBM and OpenGL compute shaders',
    images: [{ url: '/logo.png', width: 1200, height: 630 }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Lattice',
    description:
      'GPU-accelerated wind tunnel simulation using LBM and OpenGL compute shaders',
    images: ['/logo.png'],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-ctp-base text-ctp-text`}
      >
        {children}
      </body>
    </html>
  );
}
