import { Link } from "react-router-dom";
import { ChevronLeft, ChevronRight, Pause, Play, RotateCcw, ArrowRight } from "lucide-react";
import {
  compareRows,
  fmtShape,
  nextStages,
  overlayFacts,
  shapeText,
  type ArchGraph,
  type ArchNode,
  type ConceptNode,
  type FlowStep,
} from "./model";

const fmt = (x: number) => x.toLocaleString("en-US");

function branchName(g: ArchGraph, n: ArchNode) {
  return n.branch === "joint" ? "joint" : `${g.modalities[n.branch]} branch`;
}

function segmentName(n: ArchNode) {
  return {
    pipeline: "Preprocessing (deterministic, not learned)",
    encoder: "Encoder",
    fusion: "Fusion",
    head: "Classifier",
    output: "Output",
  }[n.segment];
}

/* ── Layer inspection ──────────────────────────────────────────────── */

function Field({ k, v }: { k: string; v: string | number | null | undefined }) {
  if (v === null || v === undefined || v === "") return null;
  return (
    <div className="grid grid-cols-[118px_1fr] gap-2 border-t border-line py-1.5 first:border-t-0">
      <dt className="font-mono text-[0.65rem] uppercase tracking-wider text-muted">{k}</dt>
      <dd className="break-words font-mono text-2xs text-ink">{v}</dd>
    </div>
  );
}

export function Inspector({
  graph,
  node,
  onSelect,
}: {
  graph: ArchGraph;
  node: ArchNode | undefined;
  onSelect: (id: string) => void;
}) {
  if (!node) {
    return (
      <div className="card p-5" data-testid="inspector-empty">
        <div className="label mb-2">Layer inspector</div>
        <p className="text-sm leading-relaxed text-muted">
          Select a layer in the view or in the list below to see its type, tensor shapes, parameters and what it does.
          Hover a layer for a short summary.
        </p>
      </div>
    );
  }
  const d = node.details as Record<string, any>;
  const isModel = node.segment !== "pipeline";
  const channels =
    d.channels_in !== undefined
      ? `${d.channels_in} → ${d.channels_out}`
      : d.features_in !== undefined
        ? `${d.features_in} → ${d.features_out} features`
        : undefined;
  const nexts = nextStages(graph, node.id);
  const fused = node.id === "fusion";
  return (
    <div className="card p-5" data-testid="inspector" aria-live="polite">
      <div className="mb-1 flex items-start justify-between gap-2">
        <div className="label">{segmentName(node)}</div>
        <span className="font-mono text-2xs text-muted">{branchName(graph, node)}</span>
      </div>
      <h3 className="font-display text-lg font-semibold leading-snug text-ink" data-testid="inspector-title">
        {node.label}
      </h3>

      <div className="mt-3 space-y-3">
        <div>
          <div className="mb-0.5 font-mono text-[0.6rem] uppercase tracking-wider text-forward">In plain terms</div>
          <p className="text-sm leading-relaxed text-ink">{node.simple}</p>
        </div>
        <div>
          <div className="mb-0.5 font-mono text-[0.6rem] uppercase tracking-wider text-muted">What the code does</div>
          <p className="text-2xs leading-relaxed text-ink/80">{node.technical}</p>
        </div>
      </div>

      <dl className="mt-4 rounded border border-line bg-canvas/40 px-3 py-1" data-testid="inspector-fields">
        <Field k="Layer type" v={node.op} />
        <Field k="Module" v={node.module ?? (isModel ? "functional op (no nn.Module)" : "data pipeline")} />
        <Field
          k="Input shape"
          v={fused ? `h_A ${fmtShape(node.in_shapes?.a)} + h_B ${fmtShape(node.in_shapes?.b)}` : shapeText(node, "in", node.in_shape)}
        />
        <Field k="Output shape" v={shapeText(node, "out", node.out_shape)} />
        <Field k="Channels" v={channels} />
        <Field k="Kernel size" v={d.kernel_size ? (d.kernel_size as number[]).join(" × ") : undefined} />
        <Field k="Stride" v={d.stride ? (d.stride as number[]).join(" × ") : undefined} />
        <Field k="Padding" v={d.padding ? (d.padding as number[]).join(" × ") : undefined} />
        <Field k="Activation" v={d.activation ?? (isModel && node.segment === "encoder" ? "none" : undefined)} />
        <Field k="Pooling" v={d.pooling} />
        <Field
          k="Normalization"
          v={node.segment === "encoder" ? "none (no normalization layer)" : typeof d.normalization === "string" ? d.normalization : undefined}
        />
        <Field k="Fusion" v={fused ? `${d.fusion}: ${d.pre_fusion}` : undefined} />
        <Field k="Parameters" v={isModel ? fmt(node.params) : "none (not learned)"} />
        {node.id === "output" && (
          <Field
            k="Classes"
            v={
              graph.classNamesAvailable
                ? graph.classCodes.join(", ")
                : `${graph.classCodes[0]}–${graph.classCodes[graph.classCodes.length - 1]} (action names unavailable: not stored in the repository)`
            }
          />
        )}
        <Field k="Source" v={node.source} />
      </dl>

      <div className="mt-3 text-2xs text-muted">
        <span className="font-medium text-ink">Purpose:</span> {node.purpose}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-1.5 text-2xs">
        <span className="font-medium text-ink">Next:</span>
        {nexts.length === 0 ? (
          <span className="text-muted">none (final output)</span>
        ) : (
          nexts.map((n) => (
            <button
              key={n.id}
              type="button"
              className="rounded border border-line px-2 py-0.5 font-mono text-2xs text-ink hover:border-ink/40"
              onClick={() => onSelect(n.id)}
            >
              {n.label} →
            </button>
          ))
        )}
      </div>
    </div>
  );
}

/* ── Layer list (keyboard path to every node) ──────────────────────── */

export function LayerList({
  graph,
  activeId,
  onSelect,
  extra = [],
}: {
  graph: ArchGraph;
  activeId: string | null;
  onSelect: (id: string) => void;
  /** figure components that are not layers (intervention, concat/gated, analyses) */
  extra?: { id: string; label: string }[];
}) {
  const groups: { title: string; nodes: { id: string; label: string; segment?: string }[] }[] = [
    { title: `${graph.modalities.a} branch`, nodes: graph.nodes.filter((n) => n.branch === "a") },
    { title: `${graph.modalities.b} branch`, nodes: graph.nodes.filter((n) => n.branch === "b") },
    { title: "Fusion & output", nodes: graph.nodes.filter((n) => n.branch === "joint") },
    ...(extra.length ? [{ title: "Figure components (not checkpoint layers)", nodes: extra.map((e) => ({ ...e, segment: "concept" })) }] : []),
  ];
  return (
    <nav aria-label={`${graph.name} layers`} className="card p-4" data-testid="layer-list">
      <div className="label mb-2">Layers ({graph.nodes.length} stages){extra.length ? " · figure components" : ""}</div>
      <div className="space-y-3">
        {groups.map((gr) => (
          <div key={gr.title}>
            <div className="mb-1 font-mono text-[0.6rem] uppercase tracking-wider text-muted/80">{gr.title}</div>
            <ul className="flex flex-wrap gap-1">
              {gr.nodes.map((n) => (
                <li key={n.id}>
                  <button
                    type="button"
                    data-testid={`layer-${n.id.replace("concept:", "concept-")}`}
                    aria-pressed={activeId === n.id}
                    onClick={() => onSelect(n.id)}
                    className={`rounded border px-2 py-1 text-left font-mono text-[0.68rem] transition-colors ${
                      activeId === n.id
                        ? "border-ink bg-ink text-canvas"
                        : n.segment === "pipeline" || n.segment === "concept"
                          ? "border-dashed border-line text-muted hover:border-ink/40 hover:text-ink"
                          : "border-line text-ink hover:border-ink/40"
                    }`}
                  >
                    {n.label}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </nav>
  );
}

/* ── Data-flow narration + controls ────────────────────────────────── */

export function FlowPanel({
  graph,
  steps,
  step,
  playing,
  setStep,
  setPlaying,
}: {
  graph: ArchGraph;
  steps: FlowStep[];
  step: number;
  playing: boolean;
  setStep: (i: number) => void;
  setPlaying: (p: boolean) => void;
}) {
  const s = steps[step];
  const last = step === steps.length - 1;
  return (
    <div className="card p-5" data-testid="flow-panel" aria-live="polite">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="label">Data flow · step {step + 1} of {steps.length}</div>
        <span className="font-mono text-2xs text-muted">{branchName(graph, s.node)}</span>
      </div>
      <div className="mb-3 h-1 overflow-hidden rounded-full bg-line">
        <div className="h-full bg-ink transition-all duration-300" style={{ width: `${((step + 1) / steps.length) * 100}%` }} />
      </div>
      <h3 className="font-display text-lg font-semibold text-ink" data-testid="flow-stage">
        {s.node.label}
      </h3>
      <div className="mt-2 rounded border border-line bg-canvas/40 px-3 py-2">
        <div className="font-mono text-[0.6rem] uppercase tracking-wider text-muted">Tensor shape</div>
        <div className="font-mono text-xs text-ink" data-testid="flow-shape">
          {s.shape}
        </div>
      </div>
      <p className="mt-3 text-sm leading-relaxed text-ink">{s.node.simple}</p>
      <p className="mt-2 text-2xs leading-relaxed text-muted">{s.node.technical}</p>
      <div className="mt-3 text-2xs">
        <span className="font-medium text-ink">Next destination: </span>
        <span className="text-muted" data-testid="flow-next">
          {s.next.length ? s.next.map((n) => n.label).join(", ") : "end of the forward pass"}
        </span>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-1.5" role="group" aria-label="Data-flow controls">
        <button type="button" className="btn-ghost !px-2.5 !py-1.5" onClick={() => setPlaying(!playing)} aria-label={playing ? "Pause" : "Play data flow"}>
          {playing ? <Pause size={14} /> : <Play size={14} />}
          <span className="text-2xs">{playing ? "Pause" : last ? "Replay" : "Play"}</span>
        </button>
        <button type="button" className="btn-ghost !px-2.5 !py-1.5" onClick={() => { setPlaying(false); setStep(step - 1); }} disabled={step === 0} aria-label="Previous stage">
          <ChevronLeft size={14} />
        </button>
        <button type="button" className="btn-ghost !px-2.5 !py-1.5" onClick={() => { setPlaying(false); setStep(step + 1); }} disabled={last} aria-label="Next stage">
          <ChevronRight size={14} />
        </button>
        <button type="button" className="btn-ghost !px-2.5 !py-1.5" onClick={() => { setPlaying(false); setStep(0); }} aria-label="Restart">
          <RotateCcw size={14} />
        </button>
        <label className="ml-auto flex items-center gap-1.5 text-2xs text-muted">
          <span className="sr-only sm:not-sr-only">Stage</span>
          <select
            value={step}
            onChange={(e) => { setPlaying(false); setStep(Number(e.target.value)); }}
            className="h-7 max-w-[180px] rounded border border-line bg-canvas px-1.5 font-mono text-2xs text-ink"
            aria-label="Select a processing stage"
          >
            {steps.map((st) => (
              <option key={st.node.id} value={st.index}>
                {st.index + 1}. {st.node.label}
                {st.node.branch !== "joint" ? ` (${graph.modalities[st.node.branch]})` : ""}
              </option>
            ))}
          </select>
        </label>
      </div>
      <p className="mt-3 border-t border-line pt-2 font-mono text-[0.6rem] uppercase tracking-wider text-muted/70">
        Order = the forward() call order · walkthrough of verified operations · no tensors are computed here
      </p>
    </div>
  );
}

/* ── Comparison ────────────────────────────────────────────────────── */

export function CompareTable() {
  const rows = compareRows();
  return (
    <div className="card overflow-hidden" data-testid="compare-table">
      <div className="border-b border-line px-5 py-3">
        <div className="label">CREMA-D vs UTD-MHAD · implemented models</div>
        <p className="mt-1 text-2xs text-muted">
          Every cell is read from the generated architecture data; rows marked ≠ differ between the two models.
        </p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] text-left text-2xs">
          <thead>
            <tr className="border-b border-line bg-canvas/50 font-mono uppercase tracking-wider text-muted">
              <th className="w-44 px-4 py-2 font-normal">Field</th>
              <th className="px-4 py-2 font-normal">CREMA-D</th>
              <th className="px-4 py-2 font-normal">UTD-MHAD</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const diff = r.cremad !== r.utd_mhad;
              return (
                <tr key={r.field} className="border-b border-line last:border-0 align-top">
                  <th scope="row" className="px-4 py-2 font-mono font-normal text-muted">
                    {r.field} {diff && <span className="text-warning" title="differs">≠</span>}
                  </th>
                  <td className="px-4 py-2 font-mono text-ink">{r.cremad}</td>
                  <td className="px-4 py-2 font-mono text-ink">{r.utd_mhad}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ── Conceptual overlay explainer ──────────────────────────────────── */

export function OverlayExplainer({ graph }: { graph: ArchGraph }) {
  const tiers = [
    { n: "1", title: "Actual model architecture", tag: "verified from code", body: overlayFacts.actual },
    { n: "2", title: "Experiment-specific intervention", tag: "experiments only", body: overlayFacts.intervention(graph) },
    { n: "3", title: "Conceptual research mechanism", tag: "conceptual", body: overlayFacts.conceptual },
  ];
  return (
    <section className="card p-5" data-testid="overlay-explainer" aria-label="What the overlay shows">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="label">Research pathway overlay: three different things</div>
        <Link to="/disentanglement" className="inline-flex items-center gap-1 text-2xs font-medium text-forward hover:underline">
          Measured pathway effects <ArrowRight size={12} />
        </Link>
      </div>
      <div className="grid gap-3 md:grid-cols-3">
        {tiers.map((t) => (
          <div key={t.n} className="rounded border border-line bg-canvas/40 p-3">
            <div className="mb-1 flex items-center justify-between gap-2">
              <span className="text-sm font-semibold text-ink">
                {t.n}. {t.title}
              </span>
            </div>
            <span className="mb-2 inline-block font-mono text-[0.6rem] uppercase tracking-wider text-muted">{t.tag}</span>
            <p className="text-2xs leading-relaxed text-ink/80">{t.body}</p>
          </div>
        ))}
      </div>
      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 font-mono text-2xs text-muted">
        <span><span className="mr-1.5 inline-block h-0.5 w-6 align-middle bg-forward" />forward pathway (solid)</span>
        <span><span className="mr-1.5 inline-block w-6 align-middle border-t-2 border-dashed border-gradient" />gradient pathway (dashed)</span>
        <span><span className="mr-1.5 inline-block w-6 align-middle border-t-2 border-dotted border-ink" />update pathway (dotted)</span>
        <span>α ring = operator location in the experiments ({overlayFacts.scaledSource[graph.dataset]})</span>
      </div>
    </section>
  );
}

/* ── figure components that are not checkpoint layers ──────────────── */

export function ConceptInspector({ concept }: { concept: ConceptNode }) {
  return (
    <div className="card p-5" data-testid="inspector" aria-live="polite">
      <div className="mb-1 flex items-start justify-between gap-2">
        <div className="label">Figure component</div>
        <span className="rounded-sm border border-warning/40 bg-warning/[0.08] px-1.5 py-0.5 font-mono text-[0.6rem] uppercase tracking-wider text-warning">
          {concept.tag}
        </span>
      </div>
      <h3 className="font-display text-lg font-semibold leading-snug text-ink" data-testid="inspector-title">
        {concept.label}
      </h3>
      <p className="mt-1 text-2xs text-muted">Not a layer of the baseline checkpoint shown here.</p>
      <div className="mt-3 space-y-3">
        <div>
          <div className="mb-0.5 font-mono text-[0.6rem] uppercase tracking-wider text-forward">In plain terms</div>
          <p className="text-sm leading-relaxed text-ink">{concept.simple}</p>
        </div>
        <div>
          <div className="mb-0.5 font-mono text-[0.6rem] uppercase tracking-wider text-muted">Precisely</div>
          <p className="text-2xs leading-relaxed text-ink/80">{concept.technical}</p>
        </div>
      </div>
      <dl className="mt-4 rounded border border-line bg-canvas/40 px-3 py-1" data-testid="inspector-fields">
        {concept.facts.map((f) => (
          <Field key={f.k} k={f.k} v={f.v} />
        ))}
        <Field k="Source" v={concept.source} />
      </dl>
    </div>
  );
}
