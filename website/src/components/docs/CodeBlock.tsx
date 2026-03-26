'use client';

import { useState, useCallback } from 'react';
import { highlight, tokenColor } from '../../lib/highlight';

export function CodeBlock({
  children,
  language,
  title,
}: {
  children: string;
  language?: string;
  title?: string;
}) {
  const [copied, setCopied] = useState(false);

  const copy = useCallback(() => {
    navigator.clipboard.writeText(children).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [children]);

  const tokens = highlight(children, language);
  const label = title ?? language;

  return (
    <div className="bg-ctp-crust rounded-lg border border-ctp-surface1 mb-4 overflow-hidden group">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-1.5 border-b border-ctp-surface1/50">
        <span className="text-[10px] text-ctp-overlay0 uppercase tracking-wider font-mono">
          {label ?? ''}
        </span>
        <button
          onClick={copy}
          className="text-ctp-overlay0 hover:text-ctp-text transition-colors opacity-0 group-hover:opacity-100 focus:opacity-100"
          aria-label="Copy code"
        >
          {copied ? (
            <svg className="w-3.5 h-3.5 text-ctp-green" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
            </svg>
          ) : (
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
              <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
            </svg>
          )}
        </button>
      </div>

      {/* Code content */}
      <div className="overflow-x-auto">
        <pre className="text-xs font-mono p-4 leading-relaxed">
          <code>
            {tokens.map((t, i) => (
              <span key={i} className={tokenColor(t.type)}>
                {t.text}
              </span>
            ))}
          </code>
        </pre>
      </div>
    </div>
  );
}
