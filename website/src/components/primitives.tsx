import type { ReactNode } from "react";

/* ----------------------------------------------------------------- */
/*  Small, reusable scientific UI primitives                          */
/* ----------------------------------------------------------------- */

export function Label({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <span className={`label ${className}`}>{children}</span>;
}

export function SectionHeader({
  eyebrow,
  title,
  subtitle,
  className = "",
}: {
  eyebrow?: string;
  title: string;
  subtitle?: string;
  className?: string;
}) {
  return (
    <header className={`animate-fade-up ${className}`}>
      {eyebrow && <div className="label mb-3">{eyebrow}</div>}
      <h1 className="font-display text-[1.85rem] font-semibold leading-tight text-ink sm:text-[2.15rem]">
        {title}
      </h1>
      {subtitle && <p className="mt-3 max-w-2xl text-[0.975rem] leading-relaxed text-muted">{subtitle}</p>}
    </header>
  );
}

export function Divider({ className = "" }: { className?: string }) {
  return <hr className={`border-0 border-t border-line ${className}`} />;
}

export function Card({
  children,
  className = "",
  as: Tag = "div",
}: {
  children: ReactNode;
  className?: string;
  as?: keyof JSX.IntrinsicElements;
}) {
  return <Tag className={`card ${className}`}>{children}</Tag>;
}

/** A large measurement value with an optional label above and unit/annotation below. */
export function MetricBlock({
  value,
  label,
  sub,
  accent = "ink",
  size = "md",
  className = "",
}: {
  value: ReactNode;
  label: string;
  sub?: ReactNode;
  accent?: "ink" | "forward" | "gradient" | "neutral" | "positive" | "warning" | "critical";
  size?: "sm" | "md" | "lg";
  className?: string;
}) {
  const color = {
    ink: "text-ink",
    forward: "text-forward",
    gradient: "text-gradient",
    neutral: "text-neutral",
    positive: "text-positive",
    warning: "text-warning",
    critical: "text-critical",
  }[accent];
  const valueSize = { sm: "text-2xl", md: "text-[2rem]", lg: "text-[2.75rem]" }[size];
  return (
    <div className={className}>
      <div className="label mb-2">{label}</div>
      <div className={`measure font-semibold leading-none ${valueSize} ${color}`}>{value}</div>
      {sub && <div className="mt-2 text-2xs text-muted">{sub}</div>}
    </div>
  );
}

/** Inline label + monospace value on a hairline row (used in spec sheets). */
export function SpecRow({
  field,
  children,
  accent = "ink",
}: {
  field: string;
  children: ReactNode;
  accent?: "ink" | "forward" | "gradient";
}) {
  const color = { ink: "text-ink", forward: "text-forward", gradient: "text-gradient" }[accent];
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-line py-2.5 last:border-0">
      <span className="text-sm text-muted">{field}</span>
      <span className={`measure text-sm font-medium ${color}`}>{children}</span>
    </div>
  );
}

const TONES = {
  neutral: "border-line bg-black/[0.02] text-muted",
  forward: "border-forward/30 bg-forward/[0.08] text-forward",
  gradient: "border-gradient/30 bg-gradient/[0.08] text-gradient",
  positive: "border-positive/30 bg-positive/[0.08] text-positive",
  warning: "border-warning/30 bg-warning/[0.08] text-warning",
  critical: "border-critical/30 bg-critical/[0.08] text-critical",
};

export function Tag({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: keyof typeof TONES;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-sm border px-2 py-0.5 font-mono text-2xs uppercase tracking-wider ${TONES[tone]}`}
    >
      {children}
    </span>
  );
}

/** An editorial callout for a scoped key insight / takeaway. */
export function Callout({
  title = "Key insight",
  tone = "forward",
  children,
}: {
  title?: string;
  tone?: "forward" | "gradient" | "neutral" | "warning";
  children: ReactNode;
}) {
  const bar = {
    forward: "before:bg-forward",
    gradient: "before:bg-gradient",
    neutral: "before:bg-neutral",
    warning: "before:bg-warning",
  }[tone];
  return (
    <aside
      className={`relative overflow-hidden rounded-md border border-line bg-surface p-5 pl-6 before:absolute before:left-0 before:top-0 before:h-full before:w-[3px] before:content-[''] ${bar}`}
    >
      <div className="label mb-2">{title}</div>
      <p className="text-[0.95rem] leading-relaxed text-ink">{children}</p>
    </aside>
  );
}

/** Small disclaimer chip for illustrative visualizations. */
export function IllustrativeNote({ children = "Illustrative visualization" }: { children?: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 font-mono text-2xs uppercase tracking-wider text-muted">
      <span className="inline-block h-1.5 w-1.5 rounded-full bg-warning/70" />
      {children}
    </span>
  );
}
