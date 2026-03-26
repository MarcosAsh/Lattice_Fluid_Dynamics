import { REPO } from '../../lib/docs-sections';

export function SrcLink({ path, label }: { path: string; label: string }) {
  return (
    <a
      href={`${REPO}/blob/master/${path}`}
      target="_blank"
      rel="noopener noreferrer"
      className="text-ctp-blue hover:text-ctp-lavender text-xs transition-colors"
    >
      {label}
    </a>
  );
}
