/** Shared constants for the restrained, paper-style charts.
 *  All values reference CSS custom properties so they automatically
 *  adapt when the `.dark` class is toggled on <html>.
 */
export const C = {
  ink:      "var(--ink)",
  muted:    "var(--muted)",
  line:     "var(--line)",
  grid:     "var(--line)",
  forward:  "var(--forward)",
  gradient: "var(--gradient)",
  neutral:  "var(--neutral)",
  positive: "var(--positive)",
  warning:  "var(--warning)",
  critical: "var(--critical)",
  surface:  "var(--surface)",
  canvas:   "var(--canvas)",
} as const;

export const axisTick = {
  fontSize: 11,
  fontFamily: "'IBM Plex Mono', monospace",
  fill: "var(--muted)",
} as const;

export const axisLabelStyle = {
  fontSize: 10.5,
  fontFamily: "'IBM Plex Mono', monospace",
  fill: "var(--muted)",
  letterSpacing: "0.06em",
} as const;

/** Shared tooltip styling for Recharts. */
export const tooltipProps = {
  contentStyle: {
    background: "var(--surface)",
    border: "1px solid var(--line)",
    borderRadius: 6,
    fontSize: 12,
    fontFamily: "'IBM Plex Mono', monospace",
    color: "var(--ink)",
    boxShadow: "0 1px 3px rgba(23,35,43,0.08)",
  },
  labelStyle: { color: "var(--muted)", fontSize: 11, marginBottom: 2 },
  cursor: { stroke: "var(--line)", strokeWidth: 1 },
} as const;
