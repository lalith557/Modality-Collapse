import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
  LabelList,
} from "recharts";
import { C, axisTick, tooltipProps } from "./theme";
import { utd } from "../data/research";

const data = [
  { name: "Depth", acc: utd.base.depthAcc, color: C.gradient },
  { name: "Inertial", acc: utd.base.inertialAcc, color: C.forward },
  { name: "Multimodal", acc: utd.base.multimodalAcc, color: C.ink },
];

/** Masked unimodal + multimodal accuracy on UTD-MHAD, with the 27-class chance line. */
export function UtdAccuracyChart() {
  return (
    <ResponsiveContainer width="100%" height={258}>
      <BarChart data={data} margin={{ top: 16, right: 10, bottom: 8, left: 2 }} barCategoryGap="34%">
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="name" tick={axisTick} tickLine={false} axisLine={{ stroke: C.line }} />
        <YAxis domain={[0, 0.7]} ticks={[0, 0.2, 0.4, 0.6]} tick={axisTick} tickLine={false} axisLine={{ stroke: C.line }} width={40} />
        <ReferenceLine
          y={utd.chanceLevel}
          stroke={C.critical}
          strokeDasharray="3 4"
          strokeWidth={1}
          label={{ value: `chance ${utd.chanceLevel}`, position: "insideTopRight", style: { fontSize: 9.5, fontFamily: "'IBM Plex Mono', monospace", fill: C.critical } }}
        />
        <Tooltip {...tooltipProps} formatter={(v: number) => [v.toFixed(3), "accuracy"]} />
        <Bar dataKey="acc" radius={[2, 2, 0, 0]} maxBarSize={64}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.color} />
          ))}
          <LabelList dataKey="acc" position="top" formatter={(v: number) => v.toFixed(3)} style={{ fontSize: 11, fontFamily: "'IBM Plex Mono', monospace", fill: C.ink }} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
