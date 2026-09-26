export interface BarRow {
  label: string;
  /** raw magnitude used for bar length (already absolute) */
  magnitude: number;
  /** the value string shown at the end of the bar */
  value: string;
  color: "forward" | "gradient" | "neutral" | "ink";
  /** optional secondary annotation, e.g. "n.s." */
  note?: string;
}

const COLOR: Record<BarRow["color"], string> = {
  forward: "#557A95",
  gradient: "#9A7358",
  neutral: "#7A8589",
  ink: "#17232B",
};

/**
 * Horizontal measured bars — reads like an instrument readout rather than a
 * dashboard chart. `max` sets the common scale so bars are comparable.
 */
export function PathwayBars({ rows, max, unit }: { rows: BarRow[]; max?: number; unit?: string }) {
  const scale = max ?? Math.max(...rows.map((r) => r.magnitude));
  return (
    <div className="space-y-4">
      {rows.map((r) => {
        const pct = scale > 0 ? Math.max(2, (r.magnitude / scale) * 100) : 0;
        return (
          <div key={r.label}>
            <div className="mb-1.5 flex items-baseline justify-between gap-3">
              <span className="text-sm text-ink">{r.label}</span>
              <span className="measure text-sm font-medium" style={{ color: COLOR[r.color] }}>
                {r.value}
                {r.note && <span className="ml-2 font-mono text-2xs text-muted">{r.note}</span>}
              </span>
            </div>
            <div className="h-2.5 w-full overflow-hidden rounded-[3px] bg-black/[0.04]">
              <div
                className="bar-grow h-full rounded-[3px]"
                style={{ width: `${pct}%`, background: COLOR[r.color] }}
              />
            </div>
          </div>
        );
      })}
      {unit && <div className="pt-1 font-mono text-[0.6rem] uppercase tracking-wider text-muted/70">{unit}</div>}
    </div>
  );
}
