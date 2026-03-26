export function InlineCode({ children }: { children: string }) {
  return (
    <code className="bg-ctp-crust px-1.5 py-0.5 rounded text-ctp-green text-xs font-mono">
      {children}
    </code>
  );
}
