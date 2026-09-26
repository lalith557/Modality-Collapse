import type { ReactNode } from "react";

/**
 * A figure that sits directly on the page (not inside a heavy card) with a
 * paper-style caption: "Fig. N — caption text". Optional legend and a
 * provenance/source line.
 */
export function ScientificFigure({
  index,
  caption,
  children,
  legend,
  source,
  className = "",
}: {
  index?: string;
  caption: ReactNode;
  children: ReactNode;
  legend?: ReactNode;
  source?: ReactNode;
  className?: string;
}) {
  return (
    <figure className={`animate-fade-up ${className}`}>
      <div className="rounded-md border border-line bg-surface p-4 sm:p-5">
        {legend && <div className="mb-3 flex flex-wrap items-center gap-x-5 gap-y-1.5">{legend}</div>}
        {children}
      </div>
      <figcaption className="mt-3 flex gap-2 text-2xs leading-relaxed text-muted">
        {index && <span className="whitespace-nowrap font-mono font-medium text-ink">{index}</span>}
        <span>{caption}</span>
      </figcaption>
      {source && <div className="mt-1 font-mono text-[0.6rem] uppercase tracking-wider text-muted/70">{source}</div>}
    </figure>
  );
}

export function LegendSwatch({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-2 font-mono text-2xs text-muted">
      <span className="inline-block h-2.5 w-2.5 rounded-[2px]" style={{ background: color }} />
      {label}
    </span>
  );
}
