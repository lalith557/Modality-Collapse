/**
 * Logged per-epoch training curves of the forward-vs-gradient decomposition runs
 * (trajectories.generated.json ← scripts/export_lab_trajectories.py ← results/*_trajectories.csv).
 * Values are validation-split measurements taken after every epoch; nothing is smoothed or modelled.
 */
import { useEffect, useMemo, useState } from "react";
import { Area, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { C, axisTick, tooltipProps } from "./theme";

export type TrajArm = "baseline" | "forward" | "gradient" | "ordinary";
type Metric = "val_dominance" | "val_gain";
interface Series {
  source: string;
  epochs: number;
  alphas: number[];
  seeds: number[];
  data: Record<string, Partial<Record<TrajArm, Record<Metric, number[][]>>>>;
}
export interface Trajectories {
  metrics: Metric[];
  series: Record<string, Series>;
}

const ARMS: TrajArm[] = ["baseline", "forward", "gradient", "ordinary"];
const COLOR: Record<TrajArm, string> = { baseline: C.neutral, forward: C.forward, gradient: C.gradient, ordinary: C.ink };
const NAME: Record<TrajArm, string> = { baseline: "baseline", forward: "forward-only", gradient: "gradient-only", ordinary: "ordinary" };
const TITLE: Record<Metric, string> = { val_dominance: "Masked dominance 𝒟 = acc_A − acc_B", val_gain: "Multimodal gain" };

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
const sd = (xs: number[]) => {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1));
};

/** Loads the generated curves on demand (kept out of the main bundle). */
export function useTrajectories(): Trajectories | null {
  const [t, setT] = useState<Trajectories | null>(null);
  useEffect(() => {
    let on = true;
    import("../data/trajectories.generated.json").then((m) => on && setT(m.default as unknown as Trajectories));
    return () => {
      on = false;
    };
  }, []);
  return t;
}

function Panel({ s, alpha, metric, arm, seedIx, mode }: { s: Series; alpha: number; metric: Metric; arm: TrajArm; seedIx: number; mode: "mean" | "seed" }) {
  const rows = useMemo(() => {
    const byArm = s.data[String(alpha)];
    return Array.from({ length: s.epochs }, (_, e) => {
      const row: Record<string, number | [number, number] | null> = { epoch: e + 1 };
      for (const a of ARMS) {
        const runs = byArm?.[a]?.[metric];
        if (!runs) continue;
        const at = runs.map((r) => r[e]).filter((v): v is number => typeof v === "number");
        if (!at.length) {
          row[a] = null;
          continue;
        }
        row[a] = mode === "seed" ? (runs[seedIx]?.[e] ?? null) : mean(at);
        if (a === arm && mode === "mean") {
          const m = mean(at);
          const d = sd(at);
          row.band = [m - d, m + d];
        }
      }
      return row;
    });
  }, [s, alpha, metric, arm, seedIx, mode]);
  const ticks = s.epochs === 40 ? [1, 10, 20, 30, 40] : [1, 10, 20, 30];
  return (
    <div className="min-w-0" data-testid={`traj-${metric}`}>
      <div className="mb-1 font-mono text-2xs uppercase tracking-wider text-muted">{TITLE[metric]}</div>
      <ResponsiveContainer width="100%" height={230}>
        <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 22, left: 0 }}>
          <CartesianGrid stroke={C.grid} vertical={false} />
          <ReferenceLine y={0} stroke={C.line} />
          <XAxis
            dataKey="epoch"
            ticks={ticks}
            tick={axisTick}
            tickLine={false}
            axisLine={{ stroke: C.line }}
            label={{ value: "EPOCH", position: "bottom", offset: 6, style: { fontSize: 10, fontFamily: "'IBM Plex Mono', monospace", fill: C.muted, letterSpacing: "0.08em" } }}
          />
          <YAxis tick={axisTick} tickLine={false} axisLine={{ stroke: C.line }} width={46} tickFormatter={(v: number) => v.toFixed(2)} />
          <Tooltip
            {...tooltipProps}
            labelFormatter={(l) => `epoch ${l}`}
            formatter={(v: number | number[], k: string) => (k === "band" ? [`${(v as number[])[0].toFixed(3)} … ${(v as number[])[1].toFixed(3)}`, "±1 SD"] : [(v as number).toFixed(3), NAME[k as TrajArm] ?? k])}
          />
          {mode === "mean" && <Area dataKey="band" stroke="none" fill={COLOR[arm]} fillOpacity={0.14} isAnimationActive={false} />}
          {ARMS.filter((a) => a !== arm).map((a) => (
            <Line key={a} dataKey={a} stroke={COLOR[a]} strokeWidth={1.2} strokeOpacity={0.55} strokeDasharray={a === "baseline" ? "4 3" : undefined} dot={false} isAnimationActive={false} />
          ))}
          <Line dataKey={arm} stroke={COLOR[arm]} strokeWidth={2.4} dot={false} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export function TrajectoryChart({
  traj,
  seriesKey,
  alpha,
  arm,
  seed,
}: {
  traj: Trajectories | null;
  seriesKey: string;
  alpha: number;
  arm: TrajArm;
  seed: number;
}) {
  const [mode, setMode] = useState<"mean" | "seed">("mean");
  if (!traj) return <p className="py-10 text-center font-mono text-2xs text-muted">Loading the logged training curves…</p>;
  const s = traj.series[seriesKey];
  if (!s) return null;
  if (!s.alphas.includes(alpha)) {
    return (
      <p className="py-6 text-center text-2xs text-muted" data-testid="traj-missing">
        No per-epoch log at α = {alpha} for this setting. Logged scales: {s.alphas.map((a) => `${a}×`).join(", ")}.
      </p>
    );
  }
  const seedIx = Math.max(0, Math.min(s.seeds.length - 1, seed - 1));
  return (
    <div data-testid="trajectory-chart">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-[0.62rem] text-muted">
          {ARMS.map((a) => (
            <span key={a} className="inline-flex items-center gap-1.5">
              <span className="inline-block h-0.5 w-5" style={{ background: COLOR[a], opacity: a === arm ? 1 : 0.6, height: a === arm ? 3 : 2 }} />
              <span className={a === arm ? "font-semibold text-ink" : ""}>{NAME[a]}</span>
            </span>
          ))}
          {mode === "mean" && <span>shaded = ±1 SD across {s.seeds.length} seeds ({NAME[arm]})</span>}
        </div>
        <div className="inline-flex rounded border border-line bg-surface p-0.5" role="radiogroup" aria-label="Curve">
          <button type="button" role="radio" aria-checked={mode === "mean"} data-active={mode === "mean"} className="seg" onClick={() => setMode("mean")}>
            Mean of {s.seeds.length} seeds
          </button>
          <button type="button" role="radio" aria-checked={mode === "seed"} data-active={mode === "seed"} className="seg" onClick={() => setMode("seed")}>
            Seed {seedIx + 1} only
          </button>
        </div>
      </div>
      <div className="grid gap-5 md:grid-cols-2">
        <Panel s={s} alpha={alpha} metric="val_dominance" arm={arm} seedIx={seedIx} mode={mode} />
        <Panel s={s} alpha={alpha} metric="val_gain" arm={arm} seedIx={seedIx} mode={mode} />
      </div>
      <p className="mt-2 font-mono text-[0.6rem] leading-relaxed text-muted">
        Logged after every epoch on the validation split (sum fusion, α = {alpha}). Table I reports test-set endpoints, so final values differ
        slightly. Source: {s.source}.
      </p>
    </div>
  );
}
