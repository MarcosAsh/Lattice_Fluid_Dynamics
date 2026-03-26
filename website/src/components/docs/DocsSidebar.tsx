'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { type DocGroup } from '../../lib/docs-sections';

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg
      className={`w-3 h-3 text-ctp-overlay0 transition-transform ${open ? 'rotate-90' : ''}`}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
    </svg>
  );
}

function SidebarContent({
  groups,
  activeSection,
  onLinkClick,
}: {
  groups: DocGroup[];
  activeSection: string;
  onLinkClick?: () => void;
}) {
  // Track which groups are expanded
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  // Auto-expand the group containing the active section
  useEffect(() => {
    for (const g of groups) {
      if (g.sections.some((s) => s.id === activeSection)) {
        setExpanded((prev) => ({ ...prev, [g.label]: true }));
        break;
      }
    }
  }, [activeSection, groups]);

  const toggle = (label: string) => {
    setExpanded((prev) => ({ ...prev, [label]: !prev[label] }));
  };

  return (
    <>
      <Link href="/" className="block mb-8" onClick={onLinkClick}>
        <img src="/logo.png" alt="Lattice" className="h-8 w-auto" />
      </Link>

      <nav className="space-y-4">
        {groups.map((g) => {
          const isOpen = expanded[g.label] ?? false;
          return (
            <div key={g.label}>
              <button
                onClick={() => toggle(g.label)}
                className="flex items-center gap-1.5 w-full text-left text-[10px] text-ctp-overlay0 uppercase tracking-wider font-medium hover:text-ctp-subtext0 transition-colors py-1"
              >
                <ChevronIcon open={isOpen} />
                {g.label}
              </button>
              {isOpen && (
                <ul className="mt-1 ml-1 space-y-0.5">
                  {g.sections.map((s) => {
                    const isActive = s.id === activeSection;
                    return (
                      <li key={s.id}>
                        <a
                          href={`#${s.id}`}
                          onClick={onLinkClick}
                          className={`block text-xs py-1 px-3 rounded-r transition-colors border-l-2 ${
                            isActive
                              ? 'border-ctp-mauve text-ctp-text bg-ctp-surface0/50 font-medium'
                              : 'border-transparent text-ctp-overlay1 hover:text-ctp-text hover:border-ctp-surface2'
                          }`}
                        >
                          {s.label}
                        </a>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          );
        })}
      </nav>

      <div className="mt-8 pt-4 border-t border-ctp-surface1 space-y-2">
        <a
          href="https://github.com/MarcosAsh/3dFluidDynamicsInC"
          target="_blank"
          rel="noopener noreferrer"
          className="block text-xs text-ctp-overlay0 hover:text-ctp-blue transition-colors"
        >
          GitHub
        </a>
        <a
          href="/white_paper_CFD.pdf"
          target="_blank"
          className="block text-xs text-ctp-overlay0 hover:text-ctp-blue transition-colors"
        >
          White Paper (PDF)
        </a>
        <Link
          href="/comparison"
          className="block text-xs text-ctp-overlay0 hover:text-ctp-blue transition-colors"
        >
          ML vs LBM
        </Link>
      </div>

      <div className="mt-6 pt-4 border-t border-ctp-surface1">
        <div className="text-[10px] text-ctp-overlay0 uppercase tracking-wider mb-2">
          Made by
        </div>
        <a
          href="https://github.com/MarcosAsh"
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-ctp-subtext1 hover:text-ctp-text transition-colors"
        >
          Marcos Ashton
        </a>
        <p className="text-[10px] text-ctp-overlay0 mt-1">
          University of Exeter
        </p>
        <p className="text-[10px] text-ctp-overlay0 mt-2">
          MIT License -- contributions welcome
        </p>
      </div>
    </>
  );
}

export function DocsSidebar({
  groups,
  activeSection,
}: {
  groups: DocGroup[];
  activeSection: string;
}) {
  const [drawerOpen, setDrawerOpen] = useState(false);

  // Close drawer on escape
  useEffect(() => {
    if (!drawerOpen) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setDrawerOpen(false);
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [drawerOpen]);

  // Prevent body scroll when drawer is open
  useEffect(() => {
    if (drawerOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [drawerOpen]);

  return (
    <>
      {/* Desktop sidebar */}
      <aside className="hidden lg:block w-60 shrink-0 sticky top-0 h-screen overflow-y-auto p-6 border-r border-ctp-surface1">
        <SidebarContent groups={groups} activeSection={activeSection} />
      </aside>

      {/* Mobile hamburger */}
      <button
        onClick={() => setDrawerOpen(true)}
        className="lg:hidden fixed top-4 left-4 z-40 p-2 bg-ctp-surface0 border border-ctp-surface1 rounded-lg text-ctp-text hover:bg-ctp-surface1 transition-colors"
        aria-label="Open navigation"
      >
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>

      {/* Mobile drawer backdrop */}
      {drawerOpen && (
        <div
          className="lg:hidden fixed inset-0 z-40 bg-black/60 backdrop-blur-sm"
          onClick={() => setDrawerOpen(false)}
        />
      )}

      {/* Mobile drawer */}
      <aside
        className={`lg:hidden fixed inset-y-0 left-0 z-50 w-64 bg-ctp-base border-r border-ctp-surface1 p-6 overflow-y-auto transition-transform duration-200 ${
          drawerOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex justify-end mb-4">
          <button
            onClick={() => setDrawerOpen(false)}
            className="text-ctp-overlay1 hover:text-ctp-text transition-colors"
            aria-label="Close navigation"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <SidebarContent
          groups={groups}
          activeSection={activeSection}
          onLinkClick={() => setDrawerOpen(false)}
        />
      </aside>
    </>
  );
}
