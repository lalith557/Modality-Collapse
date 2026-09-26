import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  LabelList,
} from "recharts";
import { C, axisTick, tooltipProps } from "./theme";
import { cremad } from "../data/research";

/**
 * Contribution landscape: imposed feature-scale factor (log) vs learned
 * audio-video dominance on CREMA-D. Two arms — scaling audio vs scaling video.
 * Real values from Table II; imposed scale and dominance are rank-correlated
 * (Spearman 0.922).
 */
const audioArm = [
  { x: 1, dom: cremad.scale[0].dom, cond: "baseline" },
  { x: 5, dom: cremad.scale[1].dom, cond: "audio ×5" },
  { x: 10, dom: cremad.scale[2].dom, cond: "audio ×10" },
];
const videoArm = [
  { x: 1, dom: cremad.scale[0].dom, cond: "baseline" },
  { x: 5, dom: cremad.scale[3].dom, cond: "video ×5" },
  { x: 10, dom: cremad.scale[4].dom, cond: "video ×10" },
];

export function ContributionScatter() {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <ScatterChart margin={{ top: 12, right: 18, bottom: 34, left: 6 }}>
        <CartesianGrid stroke={C.grid} strokeDasharray="0" vertical={false} />
        <ReferenceLine y={0} stroke={C.line} strokeWidth={1} />
        <XAxis
          type="number"
          dataKey="x"
          scale="log"
          domain={[0.8, 12]}
          ticks={[1, 5, 10]}
          tickFormatter={(v) => `${v}×`}
          tick={axisTick}
          tickLine={false}
          axisLine={{ stroke: C.line }}
          label={{ value: "IMPOSED FEATURE SCALE", position: "bottom", offset: 16, style: { fontSize: 10, fontFamily: "'IBM Plex Mono', monospace", fill: C.muted, letterSpacing: "0.08em" } }}
        />
        <YAxis
          type="number"
          dataKey="dom"
          domain={[-0.05, 0.38]}
          ticks={[0, 0.1, 0.2, 0.3]}
          tick={axisTick}
          tickLine={false}
          axisLine={{ stroke: C.line }}
          width={44}
          label={{ value: "DOMINANCE 𝒟", angle: -90, position: "insideLeft", offset: 14, style: { fontSize: 10, fontFamily: "'IBM Plex Mono', monospace", fill: C.muted, letterSpacing: "0.06em", textAnchor: "middle" } }}
        />
        <ZAxis range={[70, 70]} />
        <Tooltip {...tooltipProps} formatter={(v: number) => [v.toFixed(3), "𝒟"]} labelFormatter={() => ""} />
        <Scatter name="Audio scaled" data={audioArm} fill={C.forward} line={{ stroke: C.forward, strokeWidth: 1.4 }} lineType="joint">
          <LabelList dataKey="cond" position="top" style={{ fontSize: 10, fontFamily: "'IBM Plex Mono', monospace", fill: C.muted }} />
        </Scatter>
        <Scatter name="Video scaled" data={videoArm} fill={C.gradient} line={{ stroke: C.gradient, strokeWidth: 1.4 }} lineType="joint" />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
