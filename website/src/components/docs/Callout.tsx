const styles = {
  note: {
    border: 'border-l-ctp-blue',
    icon: (
      <svg className="w-4 h-4 text-ctp-blue shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <circle cx="12" cy="12" r="10" />
        <path d="M12 16v-4M12 8h.01" />
      </svg>
    ),
  },
  warning: {
    border: 'border-l-ctp-yellow',
    icon: (
      <svg className="w-4 h-4 text-ctp-yellow shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
      </svg>
    ),
  },
  tip: {
    border: 'border-l-ctp-green',
    icon: (
      <svg className="w-4 h-4 text-ctp-green shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 18v-5.25m0 0a6.01 6.01 0 001.5-.189m-1.5.189a6.01 6.01 0 01-1.5-.189m3.75 7.478a12.06 12.06 0 01-4.5 0m3.44 2.278a2.25 2.25 0 01-2.88 0M9.349 4.501A3.75 3.75 0 0112 3a3.75 3.75 0 012.651 1.501M6.75 12A5.25 5.25 0 0112 6.75 5.25 5.25 0 0117.25 12" />
      </svg>
    ),
  },
};

export function Callout({
  type,
  children,
}: {
  type: 'note' | 'warning' | 'tip';
  children: React.ReactNode;
}) {
  const s = styles[type];
  return (
    <div
      className={`flex gap-3 items-start bg-ctp-mantle border border-ctp-surface1 border-l-4 ${s.border} rounded-lg p-4 mb-4 text-sm text-ctp-subtext1`}
    >
      {s.icon}
      <div>{children}</div>
    </div>
  );
}
