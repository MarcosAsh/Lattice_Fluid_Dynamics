import type { Metadata, Viewport } from 'next';
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

// Absolute base for OpenGraph URLs, from SITE_URL or Vercel's production domain
function siteUrl(): URL {
  if (process.env.SITE_URL) return new URL(process.env.SITE_URL);
  const vercel = process.env.VERCEL_PROJECT_PRODUCTION_URL;
  return new URL(vercel ? `https://${vercel}` : 'http://localhost:3000');
}

const description =
  'GPU-accelerated wind tunnel simulation using Lattice Boltzmann Methods and OpenGL compute shaders';
const shortDescription =
  'GPU-accelerated wind tunnel simulation using LBM and OpenGL compute shaders';

export const metadata: Metadata = {
  metadataBase: siteUrl(),
  title: { default: 'Lattice', template: '%s | Lattice' },
  description,
  applicationName: 'Lattice',
  openGraph: {
    type: 'website',
    siteName: 'Lattice',
    title: 'Lattice',
    description: shortDescription,
    images: [
      { url: '/logo.png', width: 1536, height: 1024, alt: 'Lattice logo' },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Lattice',
    description: shortDescription,
    images: ['/logo.png'],
  },
};

export const viewport: Viewport = {
  themeColor: '#1e1e2e',
  colorScheme: 'dark',
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
