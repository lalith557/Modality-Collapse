import { useState } from "react";
import { useNavigate } from "react-router-dom";

type Pathway = "forward" | "gradient";

const C = {
  forward: "#557A95",
  gradient: "#9A7358",
  line: "#DCD9D1",
  ink: "#17232B",
} as const;

// ── Thin vertical connector ────────────────────────────────────────
function VConnector({ h = 14 }: { h?: number }) {
  return (
    <svg width="2" height={h} viewBox={`0 0 2 ${h}`} className="mx-auto block" aria-hidden="true">
      <line x1="1" y1="0" x2="1" y2={h} stroke={C.line} strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

// ── Tiny downward chevron ──────────────────────────────────────────
function DownArrow() {
  return (
    <svg width="8" height="5" viewBox="0 0 8 5" className="mx-auto block" aria-hidden="true">
      <path d="M0 0 L4 5 L8 0" stroke={C.line} strokeWidth="1.2" fill="none" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

// ── Y-split / Y-merge (desktop only, animated on hover) ────────────
function BranchConnector({ direction, hovered }: { direction: "split" | "merge"; hovered: Pathway | null }) {
  const fc = hovered === "forward" ? C.forward : C.line;
  const gc = hovered === "gradient" ? C.gradient : C.line;

  const [fp, gp] =
    direction === "split"
      ? ["M200 0 L200 16 L100 28", "M200 0 L200 16 L300 28"]
      : ["M100 0 L100 12 L200 28", "M300 0 L300 12 L200 28"];

  return (
    <svg viewBox="0 0 400 28" className="hidden w-full sm:block" style={{ height: 28 }}
      preserveAspectRatio="none" aria-hidden="true">
      <path d={fp} stroke={fc} strokeWidth="1.5" fill="none" vectorEffect="non-scaling-stroke"
        className={hovered === "forward" ? "pw-flow" : ""} style={{ transition: "stroke 200ms ease" }} />
      <path d={gp} stroke={gc} strokeWidth="1.5" fill="none" vectorEffect="non-scaling-stroke"
        className={hovered === "gradient" ? "pw-flow" : ""} style={{ transition: "stroke 200ms ease" }} />
    </svg>
  );
}

// ── Mobile vertical gap between stacked cards ──────────────────────
function MobileGap() {
  return <div className="flex justify-center sm:hidden"><VConnector h={12} /></div>;
}

// ── Pathway card data ──────────────────────────────────────────────
const PATHWAY_DATA = {
  forward: {
    route: "/disentanglement",
    routeName: "Disentanglement",
    eq: { main: "αh", sub: "A" },
    concept: "Representation magnitude",
    body: "Scaling changes the representation magnitude entering fusion. Under Adam, this pathway carries almost the entire observed effect.",
    accent: C.forward,
    textClass: "text-forward" as const,
    barClass: "bg-forward" as const,
    borderActiveClass: "border-forward/50" as const,
    bgActiveClass: "bg-forward/[0.06]" as const,
  },
  gradient: {
    route: "/disentanglement?section=adam-vs-sgd",
    routeName: "Optimizer Analysis",
    eq: { main: "∂L/∂θ", sub: "A" },
    concept: "Parameter update",
    body: "Scaling can alter the gradient reaching the modality encoder and its parameter updates — effect depends on the optimizer.",
    accent: C.gradient,
    textClass: "text-gradient" as const,
    barClass: "bg-gradient" as const,
    borderActiveClass: "border-gradient/50" as const,
    bgActiveClass: "bg-gradient/[0.06]" as const,
  },
} as const;

// ── Pathway card: compact button + floating popup tooltip on hover ──
function PathwayCard({ type, hovered, onHover }: {
  type: Pathway;
  hovered: Pathway | null;
  onHover: (p: Pathway | null) => void;
}) {
  const p = PATHWAY_DATA[type];
  const nav = useNavigate();
  const isActive = hovered === type;
  const isDimmed = hovered !== null && hovered !== type;

  return (
    // relative wrapper lets the popup escape the button without overflow-hidden issues
    <div className="relative">
      <button
        type="button"
        tabIndex={0}
        aria-label={`${type === "forward" ? "Forward" : "Gradient"} Pathway — ${p.body} Click to open ${p.routeName}.`}
        className={[
          "group relative w-full overflow-hidden rounded-md border text-left",
          "focus-visible:outline-2 focus-visible:outline-offset-2",
          isActive
            ? `${p.borderActiveClass} ${p.bgActiveClass} shadow-panel -translate-y-px`
            : "border-line bg-surface hover:border-ink/20 hover:-translate-y-px",
        ].join(" ")}
        style={{
          outlineColor: p.accent,
          opacity: isDimmed ? 0.32 : 1,
          transition: "opacity 200ms ease, transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease, background-color 160ms ease",
        }}
        onMouseEnter={() => onHover(type)}
        onMouseLeave={() => onHover(null)}
        onFocus={() => onHover(type)}
        onBlur={() => onHover(null)}
        onClick={() => nav(p.route)}
      >
        {/* Left accent bar */}
        <div
          className={`absolute left-0 top-0 h-full w-[3px] ${p.barClass}`}
          style={{ opacity: isActive ? 0.7 : 0.25, transition: "opacity 160ms ease" }}
        />

        <div className="px-3.5 py-3 pl-5">
          {/* Pathway label */}
          <div className={`label mb-1.5 ${p.textClass}`}>
            {type === "forward" ? "Forward" : "Gradient"} Pathway
          </div>
          {/* Concept + equation — always visible */}
          <div className="text-[0.8rem] font-semibold text-ink">{p.concept}</div>
          <div className={`mt-1 font-mono text-[0.95rem] font-medium leading-none ${p.textClass}`}>
            {p.eq.main}<sub className="text-[0.6em] font-normal">{p.eq.sub}</sub>
          </div>
        </div>
      </button>

      {/* Floating popup — appears above the card, does not shift layout */}
      {isActive && (
        <div
          className="animate-fade-in pointer-events-none absolute bottom-full left-0 right-0 z-20 mb-2
                     rounded-md border bg-surface p-3 shadow-panel"
          style={{ borderColor: p.accent + "50" }}
        >
          <div className={`label mb-1.5 ${p.textClass}`}>
            {type === "forward" ? "Forward" : "Gradient"} Pathway
          </div>
          <p className="text-2xs leading-relaxed text-ink/70">{p.body}</p>
          <div
            className="mt-2 font-mono text-[0.52rem] uppercase tracking-wider"
            style={{ color: p.accent, opacity: 0.6 }}
          >
            click to explore →
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main export ────────────────────────────────────────────────────
export function ResearchMechanismMap() {
  const [hovered, setHovered] = useState<Pathway | null>(null);
  const nav = useNavigate();

  return (
    <div
      className="flex w-full flex-col items-center"
      role="img"
      aria-label="Research mechanism map: feature-scale α propagates through Forward and Gradient pathways to affect Modality Contribution."
    >

      {/* ── Intervention + feature node (combined, compact) ── */}
      <div className="w-full max-w-[200px] rounded-md border-2 border-ink/20 bg-ink px-4 py-3 text-center shadow-panel">
        <div className="font-mono text-[0.54rem] uppercase tracking-[0.14em] text-white/40">
          Feature Scale
        </div>
        <div className="mt-1 font-display text-[1.9rem] font-semibold leading-none text-canvas">
          α
        </div>
        {/* h_A transformation — compact, inline */}
        <div className="mt-1.5 flex items-center justify-center gap-1.5 font-mono text-[0.62rem] text-white/60">
          <span>h<sub className="text-[0.7em]">A</sub></span>
          <span className="text-white/35">→</span>
          <span className="font-semibold text-white/85">αh<sub className="text-[0.7em]">A</sub></span>
        </div>
      </div>

      {/* Connector */}
      <VConnector />
      <DownArrow />

      {/* ── Split connectors ── */}
      <MobileGap />
      <BranchConnector direction="split" hovered={hovered} />

      {/* ── Pathway cards ── */}
      <div className="grid w-full grid-cols-1 gap-2.5 sm:grid-cols-2">
        {(["forward", "gradient"] as Pathway[]).map((type) => (
          <PathwayCard key={type} type={type} hovered={hovered} onHover={setHovered} />
        ))}
      </div>

      {/* ── Merge connectors ── */}
      <MobileGap />
      <BranchConnector direction="merge" hovered={hovered} />

      {/* ── Outcome node ── */}
      <button
        type="button"
        className={[
          "w-full max-w-[232px] rounded-md border border-ink/25 bg-ink/[0.9]",
          "px-5 py-3 text-center shadow-panel",
          "transition-all duration-150",
          "hover:border-ink hover:shadow-[0_2px_6px_rgba(23,35,43,0.16)]",
          "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink",
        ].join(" ")}
        onClick={() => nav("/overview?section=findings")}
        aria-label="Modality Contribution — navigate to Findings page."
      >
        <div className="font-mono text-[0.52rem] uppercase tracking-[0.16em] text-white/35">
          Outcome
        </div>
        <div className="mt-0.5 text-[0.9rem] font-semibold text-canvas">
          Modality Contribution
        </div>
      </button>

      {/* ── Performance label ── */}
      <VConnector h={12} />
      <div className="flex items-center gap-2">
        <span className="h-px w-6 bg-line" />
        <span className="font-mono text-[0.56rem] uppercase tracking-[0.14em] text-muted/70">
          Multimodal Performance
        </span>
        <span className="h-px w-6 bg-line" />
      </div>

    </div>
  );
}
