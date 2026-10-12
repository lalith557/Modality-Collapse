import { useTheme } from "../context/theme-context";

export type Emphasis = "forward" | "gradient" | "both" | "none";

interface PathwayDiagramProps {
  animated?: boolean;
  alpha?: string;
  emphasis?: Emphasis;
  onHover?: (e: Emphasis) => void;
  onSelect?: (e: "forward" | "gradient") => void;
  className?: string;
}

export function PathwayDiagram({
  animated = true,
  alpha,
  emphasis = "both",
  onHover,
  onSelect,
  className = "",
}: PathwayDiagramProps) {
  const { theme } = useTheme();

  // Theme-adaptive palette
  const ink      = theme === "dark" ? "#e2e8ef" : "#17232B";
  const surface  = theme === "dark" ? "#1e2b38" : "#FCFBF8";
  const muted    = theme === "dark" ? "#8896a0"  : "#687278";
  const border   = theme === "dark" ? "#263040"  : "#DCD9D1";
  const FWD      = theme === "dark" ? "#6b9ab5"  : "#557A95";
  const GRD      = theme === "dark" ? "#b08a6a"  : "#9A7358";
  const outcomeB = theme === "dark" ? "#1e2b38"  : "#17232B";
  const outcomeT = theme === "dark" ? "#e2e8ef"  : "#F5F3EE";

  const fOn = emphasis === "forward" || emphasis === "both";
  const gOn = emphasis === "gradient" || emphasis === "both";
  const flow = animated ? "pw-flow" : "";
  const fOpacity = fOn ? 1 : 0.22;
  const gOpacity = gOn ? 1 : 0.22;
  const T = { transition: "opacity 320ms ease" };
  const interactive = !!(onHover || onSelect);

  const tip =
    interactive && (emphasis === "forward" || emphasis === "gradient")
      ? ({
          forward: {
            x: 5, color: FWD, label: "FORWARD PATHWAY",
            lines: [
              "Representation magnitude entering fusion.",
              "Under tested Adam settings, this pathway",
              "carries ~all the effect (Δ𝒟 ≈ −0.070).",
            ],
          },
          gradient: {
            x: 312, color: GRD, label: "GRADIENT PATHWAY",
            lines: [
              "100–282× gradient ratio under Adam leaves",
              "update ratio ≈ 1 — this pathway barely",
              "moves contribution (unlike SGD).",
            ],
          },
        } as const)[emphasis]
      : null;

  return (
    <svg
      viewBox="0 0 560 440"
      className={`h-auto w-full ${className}`}
      role="img"
      aria-label="Feature scale splits into a forward representation-magnitude pathway and a gradient/parameter-update pathway."
    >
      {/* base connectors */}
      <g fill="none" stroke={border} strokeWidth="1.25">
        <path d="M280 92 V128" />
        <path d="M280 128 H150 V158" style={{ ...T, opacity: fOpacity }} />
        <path d="M280 128 H410 V158" style={{ ...T, opacity: gOpacity }} />
        <path d="M150 232 V300 H280" style={{ ...T, opacity: fOpacity }} />
        <path d="M410 232 V300 H280" style={{ ...T, opacity: gOpacity }} />
        <path d="M280 300 V336" />
      </g>

      {/* animated signal flow */}
      <g fill="none" strokeWidth="1.75">
        <path className={flow} d="M280 92 V128" stroke={muted} />
        <path className={fOn ? flow : ""} d="M280 128 H150 V158" stroke={FWD} style={{ ...T, opacity: fOpacity }} />
        <path className={fOn ? flow : ""} d="M150 232 V300 H280" stroke={FWD} style={{ ...T, opacity: fOpacity }} />
        <path className={gOn ? flow : ""} d="M280 128 H410 V158" stroke={GRD} style={{ ...T, opacity: gOpacity }} />
        <path className={gOn ? flow : ""} d="M410 232 V300 H280" stroke={GRD} style={{ ...T, opacity: gOpacity }} />
        <path className={flow} d="M280 300 V336" stroke={ink} />
      </g>

      {/* Feature scale node */}
      <text x="280" y="34" textAnchor="middle" fontFamily="'IBM Plex Mono',monospace"
        fontSize="12" letterSpacing="2" fill={muted}>
        FEATURE SCALE
      </text>
      <circle cx="280" cy="66" r="24" fill={surface} stroke={ink} strokeWidth="1.5" />
      <text x="280" y="72" textAnchor="middle" fontFamily="'IBM Plex Mono',monospace"
        fontSize="18" fill={ink}>
        α
      </text>
      {alpha && (
        <text x="280" y="66" dx="34" textAnchor="start" fontFamily="'IBM Plex Mono',monospace"
          fontSize="11" dominantBaseline="middle" fill={muted}>
          = {alpha}
        </text>
      )}

      {/* Forward node */}
      <g
        style={{ ...T, opacity: fOpacity, cursor: interactive ? "pointer" : "default" }}
        onMouseEnter={() => onHover?.("forward")}
        onMouseLeave={() => onHover?.("both")}
        onClick={() => onSelect?.("forward")}
      >
        <rect x="66" y="158" width="168" height="74" rx="6"
          fill={FWD} fillOpacity={fOn ? 0.12 : 0.05}
          stroke={FWD} strokeWidth={fOn ? 1.6 : 1.2} />
        <text x="150" y="183" textAnchor="middle" fontFamily="'IBM Plex Mono',monospace"
          fontSize="11.5" letterSpacing="1.5" fill={FWD}>
          FORWARD PATHWAY
        </text>
        <text x="150" y="205" textAnchor="middle" fontSize="15" fontWeight="600" fill={ink}>
          Representation
        </text>
        <text x="150" y="222" textAnchor="middle" fontSize="15" fontWeight="600" fill={ink}>
          magnitude
        </text>
      </g>

      {/* Gradient node */}
      <g
        style={{ ...T, opacity: gOpacity, cursor: interactive ? "pointer" : "default" }}
        onMouseEnter={() => onHover?.("gradient")}
        onMouseLeave={() => onHover?.("both")}
        onClick={() => onSelect?.("gradient")}
      >
        <rect x="326" y="158" width="168" height="74" rx="6"
          fill={GRD} fillOpacity={gOn ? 0.12 : 0.05}
          stroke={GRD} strokeWidth={gOn ? 1.6 : 1.2} />
        <text x="410" y="183" textAnchor="middle" fontFamily="'IBM Plex Mono',monospace"
          fontSize="11.5" letterSpacing="1.5" fill={GRD}>
          GRADIENT PATHWAY
        </text>
        <text x="410" y="205" textAnchor="middle" fontSize="15" fontWeight="600" fill={ink}>
          Parameter
        </text>
        <text x="410" y="222" textAnchor="middle" fontSize="15" fontWeight="600" fill={ink}>
          update
        </text>
      </g>

      {/* Contribution node */}
      <rect x="164" y="336" width="232" height="64" rx="6" fill={outcomeB} />
      <text x="280" y="361" textAnchor="middle" fontFamily="'IBM Plex Mono',monospace"
        fontSize="11" letterSpacing="1.5" fill={outcomeT} fillOpacity="0.5">
        OUTCOME
      </text>
      <text x="280" y="383" textAnchor="middle" fontSize="16" fontWeight="600" fill={outcomeT}>
        Modality contribution
      </text>

      {/* Inline tooltip bubble */}
      {tip && (
        <g className="animate-fade-in" style={{ pointerEvents: "none" }}>
          <rect x={tip.x} y={93} width={243} height={62} rx={5}
            fill={surface} stroke={tip.color} strokeWidth={0.8} />
          <text x={tip.x + 10} y={107} fontSize={8} fill={tip.color}
            fontFamily="'IBM Plex Mono',monospace" letterSpacing={1.5}>
            {tip.label}
          </text>
          {tip.lines.map((line, i) => (
            <text key={i} x={tip.x + 10} y={121 + i * 14} fontSize={9.5}
              fill={i === 2 ? muted : ink}>
              {line}
            </text>
          ))}
        </g>
      )}
    </svg>
  );
}
