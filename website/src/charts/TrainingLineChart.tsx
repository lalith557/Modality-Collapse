import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { C, axisTick, tooltipProps } from "./theme";

/**
 * A schematic training curve that eases toward `target` over `epochs`.
 * The endpoint is a real reported value; the trajectory SHAPE is illustrative
 * (we do not store per-epoch measurements), and the UI labels it as such.
 */
export function TrainingLineChart({ target, epochs = 40 }: { target: number; epochs?: number }) {
  const data = Array.from({ length: epochs + 1 }, (_, e) => {
    const t = e / epochs;
    // critically-damped-like easing; deterministic, no randomness
    const ease = 1 - Math.exp(-3.2 * t) * Math.cos(1.4 * t);
    return { epoch: e, value: +(target * Math.min(1, ease)).toFixed(4) };
  });
  const lo = Math.min(0, target) - 0.03;
  const hi = Math.max(0, target) + 0.03;
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 10, right: 14, bottom: 26, left: 2 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <ReferenceLine y={0} stroke={C.line} />
        <ReferenceLine
          y={target}
          stroke={C.forward}
          strokeDasharray="3 4"
          strokeWidth={1}
          label={{ value: `endpoint ${target >= 0 ? "+" : "−"}${Math.abs(target).toFixed(3)}`, position: "insideTopRight", style: { fontSize: 9.5, fontFamily: "'IBM Plex Mono', monospace", fill: C.forward } }}
        />
        <XAxis
          dataKey="epoch"
          ticks={[0, 10, 20, 30, 40]}
          tick={axisTick}
          tickLine={false}
          axisLine={{ stroke: C.line }}
          label={{ value: "EPOCH", position: "bottom", offset: 10, style: { fontSize: 10, fontFamily: "'IBM Plex Mono', monospace", fill: C.muted, letterSpacing: "0.08em" } }}
        />
        <YAxis domain={[lo, hi]} tick={axisTick} tickLine={false} axisLine={{ stroke: C.line }} width={44} />
        <Tooltip {...tooltipProps} formatter={(v: number) => [v.toFixed(3), "Δ𝒟 (schematic)"]} labelFormatter={(l) => `epoch ${l}`} />
        <Line type="monotone" dataKey="value" stroke={C.forward} strokeWidth={1.75} dot={false} isAnimationActive />
      </LineChart>
    </ResponsiveContainer>
  );
}
