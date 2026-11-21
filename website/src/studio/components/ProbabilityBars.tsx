import type { Prediction } from "../types";

/** Renders exactly the probabilities returned by the server; nothing is computed here. */
export function ProbabilityBars({ prediction, compact = false }: { prediction: Prediction; compact?: boolean }) {
  const rows = [...prediction.probabilities].sort((a, b) => a.index - b.index);
  return (
    <div className={compact ? "space-y-1" : "space-y-1.5"} data-testid="prob-bars">
      {rows.map((c) => {
        const top = c.index === prediction.predicted_index;
        return (
          <div key={c.index} className="flex items-center gap-2">
            <span className={`w-12 shrink-0 font-mono text-2xs ${top ? "text-ink font-semibold" : "text-muted"}`} title={c.name ?? c.code}>
              {c.code}
            </span>
            {!compact && <span className="hidden w-16 shrink-0 truncate text-2xs text-muted sm:inline">{c.name ?? ""}</span>}
            <div className="h-2 flex-1 overflow-hidden rounded-full bg-line/70">
              <div
                className={`h-full rounded-full ${top ? "bg-forward" : "bg-muted/45"}`}
                style={{ width: `${(c.probability * 100).toFixed(2)}%` }}
              />
            </div>
            <span className={`w-12 shrink-0 text-right font-mono text-2xs ${top ? "text-ink" : "text-muted"}`}>
              {c.probability.toFixed(3)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
