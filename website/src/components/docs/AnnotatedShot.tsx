import Image from 'next/image';

export interface Shot {
  src: string;
  width: number;
  height: number;
  // Badge positions in percent of the image size
  markers: { n: number; x: number; y: number }[];
}

export function StepBadge({ n }: { n: number }) {
  return (
    <span className="inline-flex shrink-0 items-center justify-center w-6 h-6 rounded-full bg-ctp-peach text-ctp-crust text-xs font-bold">
      {n}
    </span>
  );
}

export function AnnotatedShot({
  shot,
  alt,
  className = '',
}: {
  shot: Shot;
  alt: string;
  className?: string;
}) {
  return (
    <div className={`relative ${className}`} data-testid="annotated-shot">
      <Image
        src={shot.src}
        width={shot.width}
        height={shot.height}
        alt={alt}
        className="w-full h-auto rounded-lg border border-ctp-surface1"
      />
      {shot.markers.map((m) => (
        <span
          key={m.n}
          aria-hidden="true"
          className="absolute -translate-x-1/2 -translate-y-1/2 ring-2 ring-ctp-base rounded-full"
          style={{ left: `${m.x}%`, top: `${m.y}%` }}
        >
          <StepBadge n={m.n} />
        </span>
      ))}
    </div>
  );
}
