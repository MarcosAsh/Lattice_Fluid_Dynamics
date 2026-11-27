import { highlight, tokenColor } from '../../lib/highlight';
import { CopyButton } from './CopyButton';

export function CodeBlock({
  children,
  language,
  title,
}: {
  children: string;
  language?: string;
  title?: string;
}) {
  const tokens = highlight(children, language);
  const label = title ?? language;

  return (
    <div className="bg-ctp-crust rounded-lg border border-ctp-surface1 mb-4 overflow-hidden group">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-1.5 border-b border-ctp-surface1/50">
        <span className="text-[10px] text-ctp-overlay0 uppercase tracking-wider font-mono">
          {label ?? ''}
        </span>
        <CopyButton text={children} />
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
