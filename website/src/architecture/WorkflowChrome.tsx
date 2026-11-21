/**
 * Page furniture around the architecture views: the workflow stage navigator (the
 * figure's data-flow strip, now interactive), the research strip (intervention variants,
 * optimizer analysis, fusion-utility metrics — figure components that are not layers of
 * the checkpoint) and the legend. Values come from the graph and research.ts.
 */
import { Fragment, type ReactNode } from "react";
import { ChevronRight } from "lucide-react";
import { decomposition } from "../data/research";
import { STAGE_ORDER, fmtShape, nodeById, sectionLabel, sectionOf, shapeText, stageInfo, type ArchGraph, type ConceptNode, type SectionKey, type StageKey } from "./model";

const TONE: Partial<Record<StageKey, string>> = {
  encoding: "var(--fig-3d-a)",
  intervention: "var(--fig-int-ink)",
  fusion: "var(--fig-fus-ink)",
  classifier: "var(--fig-arrow)",
  output: "var(--fig-chip)",
};

export function StageNav({ graph, active, onPick }: { graph: ArchGraph; active: StageKey | null; onPick: (s: StageKey) => void }) {
  return (
    <nav aria-label="Workflow stages" data-testid="stage-nav" className="flex items-stretch gap-1 overflow-x-auto pb-1">
      {STAGE_ORDER.map((s, i) => {
        const info = stageInfo(graph, s);
        const on = active === s;
        return (
          <Fragment key={s}>
            <button
              type="button"
              aria-current={on ? "step" : undefined}
              data-testid={`stage-${s}`}
              onClick={() => onPick(s)}
              className={`group flex min-w-[118px] flex-1 items-start gap-2 rounded-lg border px-2.5 py-2 text-left transition-colors ${
                on ? "border-ink/60 bg-ink/[0.06]" : "border-line bg-surface hover:border-ink/30"
              }`}
            >
              <span
                className="mt-px inline-flex h-5 w-5 flex-none items-center justify-center rounded-full font-mono text-[0.62rem] font-semibold text-white"
                style={{ background: TONE[s] ?? "var(--muted)" }}
              >
                {i + 1}
              </span>
              <span className="min-w-0">
                <span className="block text-xs font-semibold text-ink">{info.label}</span>
                <span className="block truncate font-mono text-[0.6rem] text-muted">{info.detail}</span>
              </span>
            </button>
            {i < STAGE_ORDER.length - 1 && <ChevronRight size={14} className="flex-none self-center text-muted" aria-hidden />}
          </Fragment>
        );
      })}
    </nav>
  );
}

export function ResearchStrip({ graph, active, onPick }: { graph: ArchGraph; active: string | null; onPick: (id: string) => void }) {
  const s = graph.scaledBranch.toUpperCase();
  const h = (
    <>
      <i>h</i>
      <sub>{s}</sub>
    </>
  );
  const grad = (
    <>
      ∇<sub>ℰ{s}</sub>ℒ
    </>
  );
  const r = decomposition.ratios;
  const card = (id: string, title: string, tag: string, body: ReactNode) => (
    <button
      type="button"
      onClick={() => onPick(id)}
      aria-pressed={active === id}
      data-testid={`research-${id.slice(8)}`}
      className={`card min-w-0 p-4 text-left transition-colors ${active === id ? "!border-ink/60 bg-ink/[0.04]" : "hover:border-ink/30"}`}
    >
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-sm font-semibold text-ink">{title}</span>
        <span className="font-mono text-[0.6rem] uppercase tracking-wider text-warning">{tag}</span>
      </div>
      <div className="mt-2 text-xs text-ink">{body}</div>
    </button>
  );
  return (
    <section aria-label="Research components" data-testid="research-strip" className="grid gap-3 lg:grid-cols-3">
      {card(
        "concept:variants",
        "Intervention variants",
        "experiments only",
        <table className="w-full border-collapse font-serif text-[0.72rem]">
          <thead>
            <tr className="text-left font-mono text-[0.6rem] uppercase tracking-wider text-muted">
              <th className="py-1 font-normal">Variant</th>
              <th className="py-1 font-normal">Forward ({h}′)</th>
              <th className="py-1 font-normal">Encoder gradient</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-t border-line">
              <td className="py-1 font-sans font-semibold">Forward-only</td>
              <td>α · {h}</td>
              <td>{grad} unchanged</td>
            </tr>
            <tr className="border-t border-line">
              <td className="py-1 font-sans font-semibold">Gradient-only</td>
              <td>{h}</td>
              <td>α · {grad}</td>
            </tr>
            <tr className="border-t border-line">
              <td className="py-1 font-sans font-semibold">Ordinary (both)</td>
              <td>α · {h}</td>
              <td>α · {grad}</td>
            </tr>
          </tbody>
        </table>,
      )}
      {card(
        "concept:optimizer",
        "Optimizer analysis",
        "synthetic · α = 100 · 5 seeds",
        <div className="space-y-1.5">
          {[
            ["Adam", "m = s_f", r.adam, "gradient scale cancels"],
            ["SGD", "m = s_f · s_g", r.sgd, "gradient-only changes the update"],
          ].map(([name, m, v, note]) => {
            const x = v as { rG: number; rTheta: number };
            return (
              <div key={name as string} className="flex items-center gap-2">
                <span className="w-12 flex-none rounded bg-[var(--fig-chip)] py-0.5 text-center text-[0.7rem] font-semibold text-white">{name as string}</span>
                <span className="w-20 flex-none font-serif italic">{m as string}</span>
                <span className="min-w-0 leading-tight">
                  {note as string}
                  <span className="block font-mono text-[0.6rem] text-muted">
                    gradient-only: r_g {x.rG.toFixed(1)}× → r_Δθ {x.rTheta >= 10 ? x.rTheta.toFixed(1) : x.rTheta.toFixed(2)}×
                  </span>
                </span>
              </div>
            );
          })}
        </div>,
      )}
      {card(
        "concept:utility",
        "Fusion utility metrics",
        "analysis",
        <ul className="ml-4 list-disc space-y-0.5 leading-snug">
          <li>Masked-accuracy dominance 𝒟 (other modality zeroed)</li>
          <li>Mean-ablation dominance (other modality at its training mean)</li>
          <li>Two-modality Shapley value</li>
          <li>Frozen-encoder linear probe</li>
        </ul>,
      )}
    </section>
  );
}

export function WorkflowLegend({ view }: { view: "3d" | "2d" }) {
  const sw = (bg: string, round = false) => <span className="inline-block h-2.5 w-2.5 flex-none" style={{ background: bg, borderRadius: round ? 999 : 2 }} />;
  const items: [ReactNode, string][] = [
    [sw("var(--fig-3d-a)"), "modality A"],
    [sw("var(--fig-3d-b)"), "modality B"],
    [sw("var(--fig-fus-ink)"), "fused (joint)"],
    [<span className="inline-flex gap-0.5">{sw("var(--fig-3d-a)")}{sw("var(--fig-3d-a)")}</span>, view === "3d" ? "moving cubes = data flowing (forward pass, animation)" : "moving dashes = data flowing (forward pass, animation)"],
    [<span className="inline-block h-2.5 w-5 flex-none rounded-full border border-dashed border-muted" />, "preprocessing step (not learned)"],
    [<span className="inline-flex items-end gap-px">{[6, 8, 10].map((x) => <span key={x} className="inline-block w-1 bg-[var(--fig-3d-a)]" style={{ height: x }} />)}</span>, "conv feature maps (∝ log₂ channels / size)"],
    [sw("var(--fig-sphere)", true), "α / 1 = feature-scale multiplier (experiments only)"],
    [<span className="inline-block h-2.5 w-2.5 flex-none rounded-full border border-dashed border-[var(--fig-fus-ink)]" />, "tested only in the synthetic experiment"],
  ];
  if (view === "3d") items.push([<span className="inline-block h-2.5 w-2.5 flex-none rotate-45 bg-[var(--fig-chip)]" />, "floating marker = analysis component (click)"]);
  return (
    <div data-testid="workflow-legend" className="flex flex-wrap gap-x-4 gap-y-1.5 font-mono text-[0.62rem] text-muted">
      {items.map(([icon, text]) => (
        <span key={text} className="inline-flex items-center gap-1.5">
          {icon}
          {text}
        </span>
      ))}
    </div>
  );
}

/**
 * Pop-up shown while hovering a part. In the 3D view it explains a part only when the part's
 * section is raised (otherwise it says how to raise it); in the 2D view it always explains.
 */
export function HoverCard({
  graph,
  concepts,
  hover,
  raised,
  view3d,
}: {
  graph: ArchGraph;
  concepts: Record<string, ConceptNode>;
  hover: { id: string; x: number; y: number } | null;
  raised: SectionKey | null;
  view3d: boolean;
}) {
  if (!hover) return null;
  const { id } = hover;
  const sec = sectionOf(graph, id);
  const node = nodeById(graph, id);
  const concept = concepts[id];
  if (!sec && !node && !concept) return null;
  const inRaised = !view3d || (sec !== null && sec === raised);
  const below = hover.y < window.innerHeight * 0.62;
  const style = {
    left: Math.max(8, Math.min(hover.x + 16, window.innerWidth - 336)),
    top: below ? hover.y + 18 : hover.y - 14,
    transform: below ? undefined : "translateY(-100%)",
  };
  const shell = (testid: string, children: ReactNode) => (
    <div className="pointer-events-none fixed z-50 w-[320px] rounded-lg border border-line bg-surface px-3.5 py-3 shadow-panel" style={style} role="tooltip" data-testid={testid}>
      {children}
    </div>
  );
  if (id.startsWith("section:") || !inRaised) {
    const name = id.startsWith("section:") ? sectionLabel(graph, sec!) : (node?.label ?? concept?.label);
    const hint =
      id.startsWith("section:") && sec === raised
        ? "Hover any part on this raised platform to see what it does."
        : `Click to raise ${id.startsWith("section:") ? "this section" : `the ${sec ? sectionLabel(graph, sec) : ""} section`} and see its parts.`;
    return shell(
      "hover-tooltip",
      <>
        <div className="text-xs font-semibold text-ink">{name}</div>
        <div className="mt-0.5 font-mono text-2xs text-muted">{hint}</div>
      </>,
    );
  }
  if (node) {
    const shape =
      node.id === "fusion"
        ? `${fmtShape(node.in_shapes?.a)} + ${fmtShape(node.in_shapes?.b)} → ${fmtShape(node.out_shape)}`
        : `${shapeText(node, "in", node.in_shape)} → ${shapeText(node, "out", node.out_shape)}`;
    return shell(
      "hover-explain",
      <>
        <div className="font-mono text-[0.58rem] uppercase tracking-wider text-muted">{sec ? sectionLabel(graph, sec) : ""}</div>
        <div className="mt-0.5 text-sm font-semibold text-ink">{node.label}</div>
        <p className="mt-1.5 text-xs leading-relaxed text-ink/90">{node.simple}</p>
        <div className="mt-2 border-t border-line pt-1.5 font-mono text-2xs text-muted">
          <div>{shape}</div>
          <div>{node.segment === "pipeline" ? "deterministic step · no learned parameters" : `${node.params.toLocaleString("en-US")} learned parameters`}</div>
        </div>
      </>,
    );
  }
  return shell(
    "hover-explain",
    <>
      <div className="font-mono text-[0.58rem] uppercase tracking-wider text-warning">{concept!.tag}</div>
      <div className="mt-0.5 text-sm font-semibold text-ink">{concept!.label}</div>
      <p className="mt-1.5 text-xs leading-relaxed text-ink/90">{concept!.simple}</p>
      <div className="mt-2 border-t border-line pt-1.5 font-mono text-2xs text-muted">not a layer of this checkpoint · click for details</div>
    </>,
  );
}
