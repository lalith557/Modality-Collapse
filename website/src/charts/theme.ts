/** Shared constants for the restrained, paper-style charts. */
export const C = {
  ink: "#17232B",
  muted: "#687278",
  line: "#DCD9D1",
  grid: "#E7E4DC",
  forward: "#557A95",
  gradient: "#9A7358",
  neutral: "#7A8589",
  positive: "#547A67",
  warning: "#B08354",
  critical: "#A65C52",
  surface: "#FCFBF8",
  canvas: "#F5F3EE",
} as const;

export const axisTick = {
  fontSize: 11,
  fontFamily: "'IBM Plex Mono', monospace",
  fill: C.muted,
} as const;

export const axisLabelStyle = {
  fontSize: 10.5,
  fontFamily: "'IBM Plex Mono', monospace",
  fill: C.muted,
  letterSpacing: "0.06em",
} as const;

/** Shared tooltip styling for Recharts. */
export const tooltipProps = {
  contentStyle: {
    background: C.surface,
    border: `1px solid ${C.line}`,
    borderRadius: 6,
    fontSize: 12,
    fontFamily: "'IBM Plex Mono', monospace",
    color: C.ink,
    boxShadow: "0 1px 3px rgba(23,35,43,0.08)",
  },
  labelStyle: { color: C.muted, fontSize: 11, marginBottom: 2 },
  cursor: { stroke: C.line, strokeWidth: 1 },
} as const;
