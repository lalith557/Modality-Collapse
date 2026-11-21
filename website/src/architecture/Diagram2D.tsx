/**
 * 2D workflow view of the same verified graph: modality lanes, real input thumbnails,
 * conv feature-map stacks, pooling funnels, h vectors, the α / 1 intervention markers,
 * fusion (with the synthetic-only alternatives), classifier and class tiles, with
 * animated data flow along every connection. Fully keyboard accessible; also the
 * fallback when WebGL is unavailable. Nothing is executed here.
 */
import type { KeyboardEvent, MouseEvent, ReactNode } from "react";
import media from "../data/architecture.media.json";
import type { SceneState } from "./ArchitectureScene";
import { fmtShape, stageInfo, stageOf, STAGE_ORDER, type ArchGraph, type ArchNode } from "./model";

const VB_W = 1290;
const VB_H = 528;
const LY = { a: 132, b: 396, joint: 264 } as const;
const CX: Record<string, number> = { raw: 74, mid: 188, fin: 300, block1: 418, block2: 503, block3: 582, gap: 652, tmean: 708, fc: 760 };
const HX = 796; // h vector
const AX = 870; // α / 1 markers
const TX = 916; // × markers
const FX = 1012; // fusion
const HDX = 1114; // classifier
const OX = 1220; // output
const ASSET = (f: string) => `${import.meta.env.BASE_URL}arch/${f}`;
const cap = (s: string) => s[0].toUpperCase() + s.slice(1);
const keyOf = (id: string) => id.replace(/^[ab]_/, "");
const COL = { a: "var(--fig-3d-a)", b: "var(--fig-3d-b)", joint: "var(--fig-fus-ink)" } as const;
const WAVE = ["#e07a2f", "#3f8fd0", "#3fa66a"];
/** Operation shown inside a preprocessing pill (the caption below gives the result). */
const PILL: Record<string, string> = { wave: "mono · crop", sample: "sample", resample: "resample" };

function pos(n: ArchNode): { x: number; y: number } {
  if (n.id === "fusion") return { x: FX, y: LY.joint };
  if (n.id === "head") return { x: HDX, y: LY.joint };
  if (n.id === "output") return { x: OX, y: LY.joint };
  const k = keyOf(n.id);
  const pipe = n.segment === "pipeline";
  return { x: pipe ? CX[k === "raw" ? "raw" : ["logmel", "frames", "zscore"].includes(k) ? "fin" : "mid"] : CX[k], y: LY[n.branch as "a" | "b"] };
}

/** Half-width of a node glyph (edge anchors). */
function halfW(n: ArchNode): number {
  const k = keyOf(n.id);
  if (n.id === "fusion") return 30;
  if (n.id === "head") return 38;
  if (n.id === "output") return 34;
  if (k === "raw") return 54;
  if (n.segment === "pipeline") return k === "wave" || k === "sample" || k === "resample" ? 46 : 54;
  if (k.startsWith("block")) return 30;
  if (k === "gap" || k === "tmean") return 16;
  return 8; // FC slab
}

function poly(vals: number[], x: number, y: number, w: number, h: number) {
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  return vals.map((v, i) => `${(x + (i / (vals.length - 1)) * w).toFixed(1)},${(y + h - ((v - lo) / (hi - lo || 1)) * h).toFixed(1)}`).join(" ");
}

/** One frame cut from a horizontal strip image. */
function Frame({ file, i, n, x, y, s, fw, fh }: { file: string; i: number; n: number; x: number; y: number; s: number; fw: number; fh: number }) {
  return (
    <svg x={x} y={y} width={s * (fw / fh)} height={s} viewBox={`${i * fw} 0 ${fw} ${fh}`} preserveAspectRatio="none" overflow="hidden">
      <image href={ASSET(file)} width={n * fw} height={fh} preserveAspectRatio="none" style={{ imageRendering: "pixelated" }} />
    </svg>
  );
}

function Glyph({ g, n, x, y }: { g: ArchGraph; n: ArchNode; x: number; y: number }): ReactNode {
  const k = keyOf(n.id);
  const br = n.branch as "a" | "b";
  const c = COL[n.branch];
  const mod = n.branch === "joint" ? "" : g.modalities[br];
  if (k === "raw") {
    let inner: ReactNode;
    if (mod === "audio") {
      const env = media.audio_envelope as number[];
      inner = env.map((v, i) => {
        const h = Math.max(2, Math.abs(v) * 38);
        return <rect key={i} x={x - 44 + i * (88 / env.length)} y={y - h / 2} width={0.9} height={h} rx={0.4} fill={c} />;
      });
    } else if (mod === "video") {
      inner = (
        <g>
          <rect x={x - 34} y={y - 22} width={68} height={44} rx={6} fill="#2c2f36" />
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <g key={i} fill="#f2efe9">
              <rect x={x - 30 + i * 11} y={y - 19} width={5} height={4} />
              <rect x={x - 30 + i * 11} y={y + 15} width={5} height={4} />
            </g>
          ))}
          <path d={`M${x - 7},${y - 9} L${x + 10},${y} L${x - 7},${y + 9} Z`} fill="#f2efe9" />
        </g>
      );
    } else if (mod === "depth") {
      inner = [0, 1, 2].map((i) => <Frame key={i} file="utd_depth_raw.png" i={i} n={3} fw={320} fh={240} x={x - 40 + i * 10} y={y - 24 + i * 5} s={34} />);
    } else {
      inner = (media.inertial_raw as number[][]).slice(0, 3).map((ch, i) => <polyline key={i} points={poly(ch, x - 44, y - 20 + i * 4, 88, 32)} fill="none" stroke={WAVE[i]} strokeWidth={1.4} />);
    }
    return (
      <g>
        <rect x={x - 54} y={y - 33} width={108} height={66} rx={14} fill="var(--fig-card)" stroke={c} strokeWidth={1.4} />
        {inner}
      </g>
    );
  }
  if (n.segment === "pipeline" && (k === "wave" || k === "sample" || k === "resample")) {
    return (
      <g>
        <rect x={x - 46} y={y - 17} width={92} height={34} rx={17} fill="var(--fig-card)" stroke={c} strokeWidth={1.3} strokeDasharray="4 3" />
        <text x={x} y={y + 4} textAnchor="middle" className="a2-pill">
          {PILL[k] ?? k}
        </text>
      </g>
    );
  }
  if (n.segment === "pipeline") {
    let inner: ReactNode;
    if (mod === "audio") {
      inner = <image href={ASSET("cremad_logmel.png")} x={x - 50} y={y - 18} width={100} height={36} preserveAspectRatio="none" />;
    } else if (mod === "video" || mod === "depth") {
      const [file, nn, fw] = mod === "video" ? ["cremad_frames.png", 8, 64] : ["utd_depth.png", 16, 64];
      inner = [0, 1, 2].map((i) => (
        <g key={i}>
          <rect x={x - 32 + i * 12} y={y - 20 - i * 4} width={38} height={38} rx={3} fill="var(--fig-card)" stroke="var(--fig-border)" />
          <Frame file={file as string} i={[0, Math.floor((nn as number) / 2), (nn as number) - 1][i]} n={nn as number} fw={fw as number} fh={64} x={x - 30 + i * 12} y={y - 18 - i * 4} s={34} />
        </g>
      ));
    } else {
      // standardized inertial tensor [6, 128] as a heat map (every 4th sample)
      const zc = media.inertial_z as number[][];
      inner = zc.flatMap((row, r) =>
        row
          .filter((_, i) => i % 4 === 0)
          .map((v, i) => {
            const t = Math.max(-1, Math.min(1, v / 2.5));
            return <rect key={`${r}-${i}`} x={x - 50 + i * 3.125} y={y - 18 + r * 6} width={3.3} height={6.2} fill={t < 0 ? "#3b6ea5" : "#cf7a3e"} fillOpacity={Math.abs(t)} />;
          }),
      );
    }
    return (
      <g>
        <rect x={x - 54} y={y - 26} width={108} height={52} rx={10} fill="var(--fig-card)" stroke="var(--fig-border)" strokeWidth={1.2} />
        {inner}
      </g>
    );
  }
  if (k.startsWith("block")) {
    // feature-map stack: plates ∝ log2(channels), plate size ∝ log2(map size)
    const s = n.out_shape!;
    const plates = Math.max(3, Math.round(Math.log2(s[0])) - 1);
    const ph = 20 + 7 * Math.log2(s[1]);
    const pw = s.length === 3 ? 6 + 2.6 * Math.log2(s[2]) : 7;
    const dx = 8;
    const x0 = x - ((plates - 1) * dx) / 2 - pw / 2;
    const stack = (ox: number, oy: number, ghost: boolean) =>
      Array.from({ length: plates }, (_, i) => {
        const px = x0 + i * dx + ox;
        const py = y - ph / 2 + oy;
        return (
          <polygon
            key={`${ox}-${i}`}
            points={`${px},${py + pw * 0.55} ${px + pw},${py} ${px + pw},${py + ph} ${px},${py + ph + pw * 0.55}`}
            fill={c}
            fillOpacity={ghost ? 0.12 : 0.45 + 0.5 * (i / (plates - 1))}
            stroke="var(--surface)"
            strokeWidth={0.9}
          />
        );
      });
    return (
      <g>
        {n.per_frame && stack(7, -6, true)}
        {n.per_frame && stack(3.5, -3, true)}
        {stack(0, 0, false)}
      </g>
    );
  }
  if (k === "gap" || k === "tmean") {
    const big = k === "gap" ? 22 : 17;
    return <polygon points={`${x - 15},${y - big} ${x + 15},${y - 7} ${x + 15},${y + 7} ${x - 15},${y + big}`} fill={c} fillOpacity={k === "gap" ? 0.75 : 0.55} stroke={c} />;
  }
  if (k === "fc") {
    return (
      <g>
        <rect x={x - 6} y={y - 30} width={12} height={60} rx={3} fill={c} />
        {Array.from({ length: 8 }, (_, i) => (
          <rect key={i} x={HX - 7} y={y - 32 + i * 8} width={14} height={7} rx={1.5} fill={c} fillOpacity={0.45 + (i % 2) * 0.3} />
        ))}
        <text x={HX} y={y - 40} textAnchor="middle" className="a2-h">
          h<tspan baselineShift="sub" fontSize="9">{br.toUpperCase()}</tspan>
        </text>
      </g>
    );
  }
  if (n.id === "fusion") {
    return (
      <g>
        <circle cx={x} cy={y} r={30} fill="var(--fig-fus)" stroke={c} strokeWidth={2.4} />
        <path d={`M${x - 13},${y} H${x + 13} M${x},${y - 13} V${y + 13}`} stroke={c} strokeWidth={4.5} strokeLinecap="round" />
      </g>
    );
  }
  if (n.id === "head") {
    const ins = [0, 1, 2, 3].map((i) => [x - 20, y - 24 + i * 16]);
    const outs = [0, 1, 2].map((i) => [x + 20, y - 16 + i * 16]);
    return (
      <g>
        <rect x={x - 38} y={y - 38} width={76} height={76} rx={14} fill="var(--fig-card)" stroke="var(--fig-arrow)" strokeWidth={1.3} />
        {ins.flatMap(([ax, ay]) => outs.map(([bx, by]) => <line key={`${ax}${ay}${bx}${by}`} x1={ax} y1={ay} x2={bx} y2={by} stroke="var(--fig-arrow)" strokeOpacity={0.45} />))}
        {ins.map(([ax, ay]) => <circle key={`i${ay}`} cx={ax} cy={ay} r={4.2} fill="var(--fig-fus-ink)" />)}
        {outs.map(([bx, by]) => <circle key={`o${by}`} cx={bx} cy={by} r={4.6} fill="var(--fig-arrow)" />)}
      </g>
    );
  }
  // output: one tile per class (codes only; no scores)
  const codes = g.classCodes;
  if (codes.length <= 9) {
    const th = 17;
    const top = y - (codes.length * (th + 3)) / 2;
    return (
      <g>
        {codes.map((code, i) => (
          <g key={code}>
            <rect x={x - 30} y={top + i * (th + 3)} width={60} height={th} rx={4} fill="var(--fig-card)" stroke="var(--fig-chip)" strokeWidth={1.2} />
            <text x={x} y={top + i * (th + 3) + 12} textAnchor="middle" className="a2-code">
              {code}
            </text>
          </g>
        ))}
      </g>
    );
  }
  const cols = 3;
  const rows = Math.ceil(codes.length / cols);
  return (
    <g>
      {codes.map((code, i) => (
        <rect key={code} x={x - 31 + (i % cols) * 21} y={y - (rows * 13) / 2 + Math.floor(i / cols) * 13} width={19} height={11} rx={2.5} fill="var(--fig-card)" stroke="var(--fig-chip)" strokeWidth={1} />
      ))}
    </g>
  );
}

function nodeTitle(g: ArchGraph, n: ArchNode): string {
  const k = keyOf(n.id);
  const NAMES: Record<string, string> = { block1: "Conv 1", block2: "Conv 2", block3: "Conv 3", gap: "GAP", tmean: "mean over t" };
  if (k === "raw") return cap(g.modalities[n.branch as "a" | "b"]);
  if (NAMES[k]) return NAMES[k];
  if (k === "fc") return `FC → h_${n.branch.toUpperCase()}`;
  if (n.id === "fusion") return `${cap(g.fusion)} fusion`;
  if (n.id === "head") return stageInfo(g, "classifier").detail;
  if (n.id === "output") return "Softmax → ŷ";
  return n.short;
}

function labelY(n: ArchNode): number {
  if (n.branch === "joint") return LY.joint + (n.id === "output" ? 74 : 52);
  return LY[n.branch] + 50;
}

export function Diagram2D({
  graph,
  state,
  onSelect,
  onHover,
}: {
  graph: ArchGraph;
  state: SceneState;
  onSelect: (id: string | null) => void;
  onHover: (id: string | null, x: number, y: number) => void;
}) {
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  const { selected, path, flowNode, flowEdge, overlay } = state;
  const s = graph.scaledBranch;
  const ys = LY[s];
  const d = s === "a" ? 1 : -1; // towards the middle
  const stage = stageOf(graph, flowNode ?? selected);

  const key = (e: KeyboardEvent, id: string) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onSelect(id);
    }
  };

  const edgePath = (from: ArchNode, to: ArchNode) => {
    const p = pos(from);
    const q = pos(to);
    if (to.id === "fusion") {
      const y = p.y;
      const tx = FX - 30;
      const ty = LY.joint + (from.branch === "a" ? -10 : 10);
      return `M${p.x + halfW(from)},${y} L${AX - 15},${y} M${AX + 15},${y} L${TX - 10},${y} M${TX + 10},${y} C${TX + 52},${y} ${FX - 64},${ty} ${tx - 4},${ty}`;
    }
    return `M${p.x + halfW(from)},${p.y} L${q.x - halfW(to) - 5},${q.y}`;
  };
  const flowPath = (from: ArchNode, to: ArchNode) => {
    if (to.id !== "fusion") return edgePath(from, to);
    const p = pos(from);
    const ty = LY.joint + (from.branch === "a" ? -10 : 10);
    return `M${p.x + halfW(from)},${p.y} L${TX + 10},${p.y} C${TX + 52},${p.y} ${FX - 64},${ty} ${FX - 34},${ty}`;
  };

  const concept = (id: string, label: string, child: ReactNode) => (
    <g
      role="button"
      tabIndex={0}
      aria-pressed={selected === id}
      aria-label={label}
      data-concept={id}
      className="a2-node cursor-pointer outline-none"
      onClick={(e: MouseEvent) => {
        e.stopPropagation();
        onSelect(id);
      }}
      onKeyDown={(e) => key(e, id)}
      onMouseEnter={(e) => onHover(id, e.clientX, e.clientY)}
      onMouseLeave={() => onHover(null, 0, 0)}
    >
      {child}
    </g>
  );

  const stageX: Record<string, number> = { inputs: CX.raw, preprocessing: (CX.mid + CX.fin) / 2, encoding: (CX.block1 + CX.fc) / 2, intervention: (AX + TX) / 2, fusion: FX, classifier: HDX, output: OX };

  return (
    <div className="w-full overflow-x-auto" data-testid="diagram-2d">
      <svg
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        className="a2 block h-auto w-full min-w-[960px]"
        role="group"
        aria-label={`${graph.name} model architecture, 2D workflow view`}
        onClick={(e) => {
          if (e.target === e.currentTarget) onSelect(null);
        }}
      >
        <defs>
          {(["a", "b", "joint"] as const).map((b) => (
            <marker key={b} id={`a2-arrow-${b}`} markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
              <path d="M0,0.5 L7,4 L0,7.5 Z" fill={COL[b]} />
            </marker>
          ))}
          <marker id="a2-arrow-ink" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
            <path d="M0,0.5 L7,4 L0,7.5 Z" fill="var(--ink)" />
          </marker>
          <radialGradient id="a2-sphere" cx="35%" cy="30%" r="75%">
            <stop offset="0%" stopColor="#fff" stopOpacity={0.75} />
            <stop offset="45%" stopColor="var(--fig-sphere)" />
            <stop offset="100%" stopColor="var(--fig-int-ink)" />
          </radialGradient>
        </defs>

        {/* stage captions */}
        {STAGE_ORDER.map((st, i) => (
          <g key={st} className={stage === st ? "a2-stage is-active" : "a2-stage"}>
            <circle cx={stageX[st] - 4 - stageInfo(graph, st).label.length * 3.4} cy={22} r={8} />
            <text x={stageX[st] - 4 - stageInfo(graph, st).label.length * 3.4} y={25.5} textAnchor="middle" className="a2-stage-n">
              {i + 1}
            </text>
            <text x={stageX[st] + 8} y={26} textAnchor="middle" className="a2-stage-t">
              {stageInfo(graph, st).label}
            </text>
          </g>
        ))}

        {/* lanes */}
        {(["a", "b"] as const).map((b) => (
          <g key={b}>
            <rect x={14} y={LY[b] - 78} width={AX - 52} height={156} rx={24} fill={`var(--fig-${b})`} stroke={`var(--fig-${b}-border)`} strokeWidth={1.5} />
            <text x={34} y={LY[b] - 56} className="a2-lane">
              Modality {b.toUpperCase()} · {graph.modalities[b]}
            </text>
            <text x={CX.block1 - 34} y={LY[b] - 56} className="a2-lane-sub">
              Encoder ℰ{b === "a" ? "_A" : "_B"} · {graph.encoders[b]} · {graph.params[b === "a" ? "encoder_a" : "encoder_b"].toLocaleString("en-US")} params
              {graph.nodes.find((n) => n.branch === b && n.per_frame)?.per_frame ? ` · per frame × ${graph.nodes.find((n) => n.branch === b && n.per_frame)!.per_frame}, shared weights` : ""}
            </text>
            <line x1={CX.fin + 70} x2={CX.fin + 70} y1={LY[b] - 66} y2={LY[b] + 66} stroke={`var(--fig-${b}-border)`} strokeDasharray="3 4" />
          </g>
        ))}
        <rect x={FX - 52} y={LY.joint - 92} width={VB_W - FX + 40} height={184} rx={24} fill="var(--fig-panel)" stroke="var(--fig-border)" strokeWidth={1.5} />
        {/* intervention band */}
        <rect x={AX - 34} y={LY.a - 78} width={TX - AX + 64} height={LY.b - LY.a + 156} rx={22} fill="var(--fig-int)" fillOpacity={0.8} stroke="var(--fig-int-border)" strokeWidth={1.5} />
        <text x={(AX + TX) / 2} y={VB_H - 22} textAnchor="middle" className="a2-lane-sub">
          Feature-scale intervention · experiments only · α = 1 in this checkpoint
        </text>

        {/* connections with flowing data */}
        {graph.edges.map((e) => {
          const k = `${e.from}->${e.to}`;
          const from = byId.get(e.from)!;
          const to = byId.get(e.to)!;
          const onPath = !path || path.edges.has(k);
          const isFlow = k === flowEdge;
          const b = from.branch;
          return (
            <g key={k} data-edge={k} opacity={onPath ? 1 : 0.12}>
              <path d={edgePath(from, to)} fill="none" stroke={isFlow ? "var(--ink)" : COL[b]} strokeOpacity={isFlow ? 0.9 : 0.5} strokeWidth={isFlow ? 3 : 2} markerEnd={`url(#a2-arrow-${isFlow ? "ink" : b})`} />
              <path d={flowPath(from, to)} fill="none" stroke={isFlow ? "var(--ink)" : COL[b]} strokeWidth={isFlow ? 4.2 : 3.2} strokeLinecap="round" className={`a2-flow${isFlow ? " a2-flow--hot" : ""}`} />
            </g>
          );
        })}

        {/* intervention markers (figure component; not a layer of this checkpoint) */}
        {concept(
          "concept:intervention",
          `Feature-scale intervention: α on ${graph.modalities[s]} in the experiments, 1 on ${graph.modalities[s === "a" ? "b" : "a"]}`,
          <g opacity={selected && selected !== "concept:intervention" && path ? 0.35 : 1}>
            <path d={`M${AX},${LY.a + 22} V${LY.b - 22}`} stroke="var(--fig-int-ink)" strokeWidth={1.6} strokeDasharray="5 5" />
            {(["a", "b"] as const).map((b) => (
              <g key={b}>
                <circle cx={AX} cy={LY[b]} r={b === s ? 16 : 13} fill="url(#a2-sphere)" stroke={selected === "concept:intervention" ? "var(--ink)" : "none"} strokeWidth={2} />
                <text x={AX} y={LY[b] + 5} textAnchor="middle" className="a2-alpha">
                  {b === s ? "α" : "1"}
                </text>
                <circle cx={TX} cy={LY[b]} r={10} fill="#f7f3ee" stroke="var(--fig-int-border)" />
                <text x={TX} y={LY[b] + 4} textAnchor="middle" className="a2-times">
                  ×
                </text>
              </g>
            ))}
          </g>,
        )}

        {/* fusion alternatives tested only in the synthetic experiment */}
        {(
          [
            ["concept:fusion_concat", "Concat", "‖", LY.joint - 70],
            ["concept:fusion_gated", "Gated", "σ", LY.joint + 70],
          ] as const
        ).map(([id, name, glyph, y]) =>
          concept(
            id,
            `${name} fusion, tested only in the synthetic experiment`,
            <g key={id}>
              <circle cx={FX} cy={y} r={15} fill="var(--fig-card)" stroke={selected === id ? "var(--ink)" : "var(--fig-fus-ink)"} strokeDasharray={selected === id ? undefined : "3 3"} strokeWidth={1.4} />
              <text x={FX} y={y + 5} textAnchor="middle" className="a2-ghost">
                {glyph}
              </text>
              <text x={FX + 22} y={y + 1} className="a2-ghost-t">
                {name}
              </text>
              <text x={FX + 22} y={y + 12} className="a2-tag">
                synthetic exp.
              </text>
            </g>,
          ),
        )}

        {/* model nodes */}
        {graph.nodes.map((n) => {
          const p = pos(n);
          const active = n.id === selected || n.id === flowNode;
          const inFocus = !path || path.nodes.has(n.id);
          const shape = n.out_shape ? fmtShape(n.out_shape) : n.id.endsWith("_raw") ? "raw" : "variable";
          const hw = Math.max(halfW(n), 38);
          const ly = labelY(n);
          return (
            <g
              key={n.id}
              role="button"
              tabIndex={0}
              aria-pressed={active}
              aria-label={`${n.label}, ${n.branch === "joint" ? "joint" : graph.modalities[n.branch]} ${n.segment}, output ${shape}`}
              data-node={n.id}
              className="a2-node cursor-pointer outline-none"
              opacity={inFocus ? 1 : 0.25}
              onClick={(e: MouseEvent) => {
                e.stopPropagation();
                onSelect(n.id);
              }}
              onKeyDown={(e) => key(e, n.id)}
              onMouseEnter={(e) => onHover(n.id, e.clientX, e.clientY)}
              onMouseLeave={() => onHover(null, 0, 0)}
            >
              <rect
                x={p.x - hw - 6}
                y={p.y - (n.id === "output" ? 66 : 44)}
                width={2 * hw + 12 + (keyOf(n.id) === "fc" ? HX - CX.fc : 0)}
                height={ly - p.y + (n.id === "output" ? 66 : 44) + 22}
                rx={14}
                className={active ? "a2-halo is-active" : "a2-halo"}
              />
              <Glyph g={graph} n={n} x={p.x} y={p.y} />
              <text x={p.x + (keyOf(n.id) === "fc" ? (HX - CX.fc) / 2 : 0)} y={ly} textAnchor="middle" className="a2-label">
                {nodeTitle(graph, n)}
              </text>
              <text x={p.x + (keyOf(n.id) === "fc" ? (HX - CX.fc) / 2 : 0)} y={ly + 13} textAnchor="middle" className="a2-shape">
                {n.per_frame ? `${n.per_frame}× ` : ""}
                {n.id === "output" && !graph.classNamesAvailable ? `${shape} · a1–a${graph.nClasses}` : shape}
              </text>
            </g>
          );
        })}

        {overlay && (
          <g data-testid="overlay-2d" pointerEvents="none">
            {/* forward pathway (solid): α → sum → logits */}
            <path d={`M${AX},${ys + d * 17} L${AX + 20},${ys + d * 84} L${OX - 42},${ys + d * 84}`} fill="none" stroke="var(--forward)" strokeWidth={2.6} markerEnd="url(#a2-arrow-ink)" />
            <text x={AX - 28} y={ys + d * 84 + 4} textAnchor="end" className="a2-concept" fill="var(--forward)">
              forward pathway (concept)
            </text>
            {/* gradient pathway (dashed): loss → back through sum and α → into the scaled encoder */}
            <path
              d={`M${OX - 42},${ys + d * 100} L${AX + 26},${ys + d * 100} L${AX - 22},${ys + d * 76} L${CX.block1},${ys + d * 76} L${CX.block1},${ys + d * 66}`}
              fill="none"
              stroke="var(--gradient)"
              strokeWidth={2.4}
              strokeDasharray="7 5"
              markerEnd="url(#a2-arrow-ink)"
            />
            <text x={AX - 28} y={ys + d * 100 + 4} textAnchor="end" className="a2-concept" fill="var(--gradient)">
              gradient pathway (concept)
            </text>
            {/* optimizer + update pathway (dotted) */}
            <rect x={CX.block1 + 50} y={s === "a" ? ys + 100 : ys - 126} width={170} height={26} rx={6} fill="var(--surface)" stroke="var(--ink)" />
            <text x={CX.block1 + 135} y={s === "a" ? ys + 117 : ys - 109} textAnchor="middle" className="a2-shape">
              optimizer · not in model
            </text>
            <path
              d={`M${CX.block1 + 70},${s === "a" ? ys + 100 : ys - 100} L${CX.block1},${ys + d * 72} M${CX.block1 + 200},${s === "a" ? ys + 100 : ys - 100} L${CX.block3},${ys + d * 72}`}
              fill="none"
              stroke="var(--ink)"
              strokeWidth={1.8}
              strokeDasharray="1.5 4"
              strokeLinecap="round"
            />
            <text x={CX.block1 + 230} y={s === "a" ? ys + 117 : ys - 109} className="a2-concept" fill="var(--ink)">
              update pathway (concept)
            </text>
          </g>
        )}
      </svg>
    </div>
  );
}
