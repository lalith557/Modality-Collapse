import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, LabelList,
} from "recharts";
import { C, axisTick, tooltipProps } from "./theme";
import { cremad } from "../data/research";

/**
 * Contribution landscape: imposed feature-scale factor (log) vs learned
 * audio-video dominance on CREMA-D. Real values (Table II). Filterable by
 * modality and scale — points are only ever the verified conditions.
 */
type Pt = { x: number; dom: number; gain: number; cond: string; arm: string };

const audioArm: Pt[] = [
  { x: 1, dom: cremad.scale[0].dom, gain: cremad.scale[0].gain, cond: "baseline", arm: "audio" },
  { x: 5, dom: cremad.scale[1].dom, gain: cremad.scale[1].gain, cond: "audio ×5", arm: "audio" },
  { x: 10, dom: cremad.scale[2].dom, gain: cremad.scale[2].gain, cond: "audio ×10", arm: "audio" },
];
const videoArm: Pt[] = [
  { x: 1, dom: cremad.scale[0].dom, gain: cremad.scale[0].gain, cond: "baseline", arm: "video" },
  { x: 5, dom: cremad.scale[3].dom, gain: cremad.scale[3].gain, cond: "video ×5", arm: "video" },
  { x: 10, dom: cremad.scale[4].dom, gain: cremad.scale[4].gain, cond: "video ×10", arm: "video" },
];

function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const p: Pt = payload[0].payload;
  return (
    <div style={{ background: C.surface, border: `1px solid ${C.line}`, borderRadius: 6, padding: "8px 10px", fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: C.ink }}>
      <div style={{ color: C.muted, marginBottom: 3 }}>{p.cond} · CREMA-D</div>
      <div>scale {p.x}× · 𝒟 {p.dom.toFixed(3)} · 𝒢 {p.gain.toFixed(3)}</div>
    </div>
  );
}

export function ContributionScatter({
  modalities = ["audio", "video"],
  scales = [1, 5, 10],
}: {
  modalities?: ("audio" | "video")[];
  scales?: number[];
}) {
  // Always include baseline (x=1) so connecting lines render even for single-scale selections
  const activeScales = scales.includes(1) ? scales : [1, ...scales];
  const fa = audioArm.filter((p) => activeScales.includes(p.x));
  const fv = videoArm.filter((p) => activeScales.includes(p.x));
  return (
    <ResponsiveContainer width="100%" height={300}>
      <ScatterChart margin={{ top: 12, right: 18, bottom: 34, left: 8 }}>
        <CartesianGrid stroke={C.grid} strokeDasharray="0" vertical={false} />
        <ReferenceLine y={0} stroke={C.line} strokeWidth={1} />
        <XAxis type="number" dataKey="x" scale="log" domain={[0.8, 12]} ticks={[1, 5, 10]} tickFormatter={(v) => `${v}×`} tick={axisTick} tickLine={false} axisLine={{ stroke: C.line }}
          label={{ value: "IMPOSED FEATURE SCALE", position: "bottom", offset: 16, style: { fontSize: 10, fontFamily: "'IBM Plex Mono', monospace", fill: C.muted, letterSpacing: "0.08em" } }} />
        <YAxis type="number" dataKey="dom" domain={[-0.05, 0.38]} ticks={[0, 0.1, 0.2, 0.3]} tick={axisTick} tickLine={false} axisLine={{ stroke: C.line }} width={54}
          label={{
            content: ({ viewBox }: any) => {
              if (!viewBox) return null;
              // Center the rotated label within the Y-axis gutter (left of chart content)
              const cx = viewBox.x + viewBox.width / 2 - 14;
              const cy = viewBox.y + viewBox.height / 2;
              return (
                <text x={cx} y={cy} transform={`rotate(-90, ${cx}, ${cy})`}
                  fill={C.muted} fontSize={10} fontFamily="'IBM Plex Mono', monospace"
                  letterSpacing="0.06em" textAnchor="middle">
                  DOMINANCE 𝒟
                </text>
              );
            }
          }} />
        <ZAxis range={[70, 70]} />
        <Tooltip {...tooltipProps} content={<CustomTooltip />} />
        {modalities.includes("audio") && (
          <Scatter name="Audio scaled" data={fa} fill={C.forward} line={{ stroke: C.forward, strokeWidth: 1.4 }} lineType="joint">
            <LabelList dataKey="cond" position="top" style={{ fontSize: 10, fontFamily: "'IBM Plex Mono', monospace", fill: C.muted }} />
          </Scatter>
        )}
        {modalities.includes("video") && (
          <Scatter name="Video scaled" data={fv} fill={C.gradient} line={{ stroke: C.gradient, strokeWidth: 1.4 }} lineType="joint" />
        )}
      </ScatterChart>
    </ResponsiveContainer>
  );
}
