export function MathBlock({ children }: { children: React.ReactNode }) {
  return (
    <div className="bg-ctp-surface0 rounded-lg p-4 mb-3 overflow-x-auto">
      <div className="text-center text-sm font-mono text-ctp-text">
        {children}
      </div>
    </div>
  );
}
