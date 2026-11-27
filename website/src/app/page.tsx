import Link from 'next/link';
import Image from 'next/image';
import Simulator from '../components/Simulator';

function LearnMore() {
  return (
    <>
      <h2 className="text-xs font-semibold text-ctp-overlay1 uppercase tracking-wider mb-2">
        Learn More
      </h2>
      <div className="flex gap-3">
        <Link
          href="/docs"
          className="flex-1 rounded p-2 hover:bg-ctp-surface0 transition-colors"
        >
          <div className="text-sm font-medium text-ctp-text">White Paper</div>
          <div className="text-xs text-ctp-overlay0 mt-0.5">
            LBM theory, compute shaders, drag validation
          </div>
        </Link>
        <Link
          href="/comparison"
          className="flex-1 rounded p-2 hover:bg-ctp-surface0 transition-colors"
        >
          <div className="text-sm font-medium text-ctp-text">ML vs LBM</div>
          <div className="text-xs text-ctp-overlay0 mt-0.5">
            Surrogate model predictions vs simulation
          </div>
        </Link>
      </div>
    </>
  );
}

export default function Home() {
  return (
    <main className="min-h-screen p-4 md:p-6 lg:p-10 max-w-7xl mx-auto">
      <div className="mb-4 lg:mb-8 flex items-center gap-3 lg:gap-4">
        <Image
          src="/logo.png"
          alt="Lattice"
          width={96}
          height={64}
          loading="eager"
          className="h-12 lg:h-16 w-auto"
        />
        <p className="text-xs lg:text-sm text-ctp-subtext0">
          GPU-accelerated wind tunnel simulation{' '}
          <a
            href="https://github.com/MarcosAsh/3dFluidDynamicsInC"
            target="_blank"
            rel="noopener noreferrer"
            className="text-ctp-blue hover:text-ctp-lavender transition-colors"
          >
            (GitHub)
          </a>
        </p>
      </div>

      <Simulator learnMore={<LearnMore />} />

      <footer className="mt-8 py-4 border-t border-ctp-surface1 flex items-center justify-between text-xs text-ctp-overlay0">
        <span>Built by Marcos Ashton, CS at University of Exeter</span>
        <a
          href="https://github.com/MarcosAsh/3dFluidDynamicsInC"
          target="_blank"
          rel="noopener noreferrer"
          className="text-ctp-blue hover:text-ctp-lavender transition-colors"
        >
          Source Code
        </a>
      </footer>
    </main>
  );
}
