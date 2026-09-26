import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LabelList,
} from "recharts";
import { C, axisTick, tooltipProps } from "./theme";
import { decomposition } from "../data/research";

/**
 * Gradient-norm ratio vs parameter-update ratio for the gradient-only arm,
 * on a log scale. Adam attenuates a huge gradient ratio down to ~1× at the
 * update; SGD passes it through. Real values from Table I.
 */
const data = [
  {
    name: "Adam",
    rG: decomposition.ratios.adam.rG,
    rTheta: decomposition.ratios.adam.rTheta,
  },
  {
    name: "SGD",
    rG: decomposition.ratios.sgd.rG,
    rTheta: decomposition.ratios.sgd.rTheta,
  },
];

export function OptimizerRatioChart() {
  return (
    <ResponsiveContainer width="100%" height={288}>
      <BarChart data={data} margin={{ top: 16, right: 12, bottom: 8, left: 6 }} barGap={8} barCategoryGap="34%">
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="name" tick={{ ...axisTick, fontSize: 12 }} tickLine={false} axisLine={{ stroke: C.line }} />
        <YAxis
          scale="log"
          domain={[1, 400]}
          ticks={[1, 10, 100]}
          tickFormatter={(v) => `${v}×`}
          tick={axisTick}
          tickLine={false}
          axisLine={{ stroke: C.line }}
          width={46}
        />
        <Tooltip {...tooltipProps} formatter={(v: number, n) => [`${v}×`, n === "rG" ? "gradient-norm ratio" : "update ratio"]} />
        <Bar dataKey="rG" name="rG" fill={C.gradient} radius={[2, 2, 0, 0]} maxBarSize={54}>
          <LabelList dataKey="rG" position="top" formatter={(v: number) => `${v}×`} style={{ fontSize: 10.5, fontFamily: "'IBM Plex Mono', monospace", fill: C.gradient }} />
        </Bar>
        <Bar dataKey="rTheta" name="rTheta" fill={C.forward} radius={[2, 2, 0, 0]} maxBarSize={54}>
          <LabelList dataKey="rTheta" position="top" formatter={(v: number) => `${v}×`} style={{ fontSize: 10.5, fontFamily: "'IBM Plex Mono', monospace", fill: C.forward }} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
