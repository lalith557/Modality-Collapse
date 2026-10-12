import { ArrowDown } from "lucide-react";

function Node({
  children,
  tone = "plain",
}: {
  children: React.ReactNode;
  tone?: "plain" | "forward" | "gradient" | "ink" | "scale";
}) {
  const styles = {
    plain: "border-line bg-surface text-ink",
    forward: "border-forward/40 bg-forward/[0.08] text-forward",
    gradient: "border-gradient/40 bg-gradient/[0.08] text-gradient",
    ink: "border-ink bg-ink text-canvas",
    scale: "border-forward bg-forward text-canvas",
  }[tone];
  return (
    <div className={`w-full rounded border px-3 py-2 text-center font-mono text-sm ${styles}`}>{children}</div>
  );
}

function Connector({ active = false }: { active?: boolean }) {
  return (
    <div className="flex h-6 items-center justify-center">
      <ArrowDown size={14} className={active ? "text-forward" : "text-muted/50"} strokeWidth={2} />
    </div>
  );
}

/**
 * Vertical model pipeline for one modality. When `scaled`, the α-scaling step
 * is inserted at the fusion boundary and highlighted.
 */
export function ModalityPipeline({
  modality,
  signal,
  symbol,
  scaled = false,
  alpha,
}: {
  modality: string;
  signal: string;
  symbol: string;
  scaled?: boolean;
  alpha?: number;
}) {
  return (
    <div className="w-full">
      <div className="mb-3 flex items-baseline justify-between">
        <span className="text-sm font-semibold text-ink">{modality}</span>
        <span className="font-mono text-2xs uppercase tracking-wider text-muted">{signal}</span>
      </div>
      <div className="flex flex-col items-center">
        <Node tone={scaled ? "forward" : "plain"}>{symbol}</Node>
        <Connector active={scaled} />
        {scaled ? (
          <>
            <Node tone="scale">
              α · {symbol}
              {alpha ? <span className="opacity-80"> = {alpha}·{symbol}</span> : null}
            </Node>
            <Connector active />
          </>
        ) : null}
        <Node tone="ink">Fusion</Node>
        <Connector />
        <Node>Prediction</Node>
      </div>
    </div>
  );
}

/** Small schematic for a fusion mechanism (sum / concat / gated). */
export function FusionDiagram({ kind }: { kind: "sum" | "concat" | "gated" }) {
  const op = { sum: "+", concat: "‖", gated: "⊙" }[kind];
  const color = { sum: "#557A95", concat: "#7A8589", gated: "#9A7358" }[kind];
  return (
    <svg viewBox="0 0 220 112" className="h-auto w-full" role="img" aria-label={`${kind} fusion`}>
      <g fill="none" stroke="#B9B5AA" strokeWidth="1.25">
        <path d="M52 30 H104" />
        <path d="M52 82 H104" />
        <path d="M116 56 H168" />
      </g>
      {/* hA */}
      <rect x="10" y="16" width="42" height="28" rx="5" fill="#557A95" opacity="0.1" stroke="#557A95" strokeWidth="1.3" />
      <text x="31" y="34" textAnchor="middle" fontFamily="'IBM Plex Mono', monospace" fontSize="13" fill="#557A95">
        hₐ
      </text>
      {/* hB */}
      <rect x="10" y="68" width="42" height="28" rx="5" fill="#9A7358" opacity="0.1" stroke="#9A7358" strokeWidth="1.3" />
      <text x="31" y="86" textAnchor="middle" fontFamily="'IBM Plex Mono', monospace" fontSize="13" fill="#9A7358">
        h_b
      </text>
      {/* fusion op */}
      <circle cx="110" cy="56" r="17" fill="#FCFBF8" stroke={color} strokeWidth="1.5" />
      <text x="110" y="62" textAnchor="middle" fontFamily="'IBM Plex Mono', monospace" fontSize="16" fill={color}>
        {op}
      </text>
      {/* prediction */}
      <rect x="168" y="42" width="44" height="28" rx="5" fill="#17232B" />
      <text x="190" y="60" textAnchor="middle" fontFamily="'IBM Plex Mono', monospace" fontSize="10" fill="#F5F3EE">
        ŷ
      </text>
    </svg>
  );
}
