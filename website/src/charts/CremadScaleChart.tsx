import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { C, axisTick, tooltipProps } from "./theme";
import { cremad } from "../data/research";

const data = cremad.scale.map((r) => ({
  name: r.condition.replace(" (1×)", ""),
  dom: r.dom,
  gain: r.gain,
}));

/** metric = "dom" | "gain"; `active` dims the non-selected bars. */
export function CremadScaleChart({ metric, active }: { metric: "dom" | "gain"; active?: string }) {
  const color = metric === "dom" ? C.forward : C.positive;
  return (
    <ResponsiveContainer width="100%" height={252}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 28, left: 2 }} barCategoryGap="28%">
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis
          dataKey="name"
          tick={{ ...axisTick, fontSize: 10 }}
          tickLine={false}
          axisLine={{ stroke: C.line }}
          interval={0}
        />
        <YAxis
          domain={metric === "dom" ? [0, 0.35] : [0, 0.32]}
          tick={axisTick}
          tickLine={false}
          axisLine={{ stroke: C.line }}
          width={40}
        />
        <Tooltip {...tooltipProps} formatter={(v: number) => [v.toFixed(3), metric === "dom" ? "𝒟" : "𝒢"]} />
        <Bar dataKey={metric} radius={[2, 2, 0, 0]} maxBarSize={46} isAnimationActive>
          {data.map((d, i) => {
            const dim = active && d.name !== active;
            return <Cell key={i} fill={color} fillOpacity={dim ? 0.28 : i === 0 && !active ? 0.55 : 1} />;
          })}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function CremadScaleLabel({ metric }: { metric: "dom" | "gain" }) {
  return (
    <span className="font-mono text-2xs uppercase tracking-label text-muted">
      {metric === "dom" ? "Dominance 𝒟 (audio − video)" : "Multimodal gain 𝒢"}
    </span>
  );
}
