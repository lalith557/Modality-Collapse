import { Component, lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { ArrowRight, Box, Crosshair, Loader2, Maximize2, Pause, Play, ScanSearch, Square } from "lucide-react";
import { SectionHeader } from "../components/primitives";
import { usePageContext } from "../state/page-context";
import {
  GRAPHS,
  conceptNodes,
  connectedPath,
  flowSteps,
  fmtShape,
  nodeById,
  sectionOf,
  stageEntry,
  stageOf,
  type ArchDataset,
  type SectionKey,
  type StageKey,
} from "../architecture/model";
import { decomposition, fusion } from "../data/research";
import { pValue } from "../lib/format";
import type { SceneState, ViewPreset } from "../architecture/ArchitectureScene";
import type { ViewportHandle } from "../architecture/Viewport3D";
import { Diagram2D } from "../architecture/Diagram2D";
import { CompareTable, ConceptInspector, FlowPanel, Inspector, LayerList, OverlayExplainer } from "../architecture/panels";
import { HoverCard, ResearchStrip, StageNav, WorkflowLegend } from "../architecture/WorkflowChrome";

const Viewport3D = lazy(() => import("../architecture/Viewport3D"));

type Mode = "explore" | "flow" | "compare";
type View = "3d" | "2d";

export function hasWebGL(): boolean {
  try {
    const c = document.createElement("canvas");
    return !!(window.WebGLRenderingContext && (c.getContext("webgl2") || c.getContext("webgl")));
  } catch {
    return false;
  }
}

/** Falls back to the 2D view if the 3D chunk fails to load or throws while rendering. */
class ViewportBoundary extends Component<{ onError: (m: string) => void; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch(e: Error) {
    this.props.onError(e.message || "The 3D viewer failed to load.");
  }
  render() {
    return this.state.failed ? null : this.props.children;
  }
}

function Seg<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: { id: T; label: string; disabled?: boolean }[];
  onChange: (v: T) => void;
}) {
  return (
    <div>
      <div className="label mb-1.5">{label}</div>
      <div className="inline-flex rounded border border-line bg-surface p-0.5" role="radiogroup" aria-label={label}>
        {options.map((o) => (
          <button
            key={o.id}
            type="button"
            role="radio"
            aria-checked={value === o.id}
            data-active={value === o.id}
            disabled={o.disabled}
            onClick={() => onChange(o.id)}
            className="seg disabled:cursor-not-allowed disabled:opacity-40"
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function ModelArchitecture() {
  const [params, setParams] = useSearchParams();
  const dataset: ArchDataset = params.get("dataset") === "utd_mhad" ? "utd_mhad" : "cremad";
  const graph = GRAPHS[dataset];
  const steps = useMemo(() => flowSteps(graph), [graph]);
  const concepts = useMemo(
    () => conceptNodes(graph, { fusion: fusion.mechanisms as unknown as Parameters<typeof conceptNodes>[1]["fusion"], ratios: decomposition.ratios, pValue }),
    [graph],
  );

  const [webgl] = useState(hasWebGL);
  const [glError, setGlError] = useState<string | null>(webgl ? null : "WebGL is not available in this browser, so the 2D view is shown.");
  // phones start in the 2D view (full labels, horizontal scroll); 3D stays one tap away
  const [view, setView] = useState<View>(webgl && window.innerWidth >= 640 ? "3d" : "2d");
  const [mode, setMode] = useState<Mode>("explore");
  const [overlay, setOverlay] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [hover, setHover] = useState<{ id: string; x: number; y: number } | null>(null);
  const [step, setStepRaw] = useState(0);
  const [playing, setPlaying] = useState(false);
  const viewport = useRef<ViewportHandle>(null);
  const [orbit, setOrbit] = useState(false);
  /** A platform clicked directly (no part selected) stays raised. */
  const [raisedPick, setRaisedPick] = useState<SectionKey | null>(null);
  const setStep = useCallback((i: number) => setStepRaw(Math.max(0, Math.min(steps.length - 1, i))), [steps.length]);

  // switching dataset resets everything that referred to the other model
  useEffect(() => {
    setSelected(null);
    setRaisedPick(null);
    setHover(null);
    setStepRaw(0);
    setPlaying(false);
  }, [dataset]);

  // playback: one stage every 2.6 s, stopping at the output
  useEffect(() => {
    if (!playing || mode !== "flow") return;
    if (step >= steps.length - 1) {
      setPlaying(false);
      return;
    }
    const t = setTimeout(() => setStepRaw((s) => s + 1), 2600);
    return () => clearTimeout(t);
  }, [playing, step, steps.length, mode]);

  // in the data-flow tour the 3D camera flies to each stage as it is reached
  useEffect(() => {
    if (mode === "flow") viewport.current?.focus(steps[step].node.id);
  }, [mode, step, steps]);

  const startPlaying = (p: boolean) => {
    if (p && step >= steps.length - 1) setStepRaw(0);
    setPlaying(p);
  };

  const flow = mode === "flow" ? steps[step] : null;
  const sceneState: SceneState = useMemo(() => {
    if (flow) {
      const visited = new Set(steps.slice(0, step + 1).map((s) => s.node.id));
      const edges = new Set(graph.edges.filter((e) => visited.has(e.from) && visited.has(e.to)).map((e) => `${e.from}->${e.to}`));
      return {
        selected: null,
        path: { nodes: visited, edges },
        flowNode: flow.node.id,
        flowEdge: flow.inEdge ? `${flow.inEdge.from}->${flow.inEdge.to}` : null,
        overlay,
        raised: sectionOf(graph, flow.node.id),
      };
    }
    return {
      selected,
      path: selected && !selected.startsWith("concept:") ? connectedPath(graph, selected) : null,
      flowNode: null,
      flowEdge: null,
      overlay,
      raised: selected ? sectionOf(graph, selected) : raisedPick,
    };
  }, [flow, step, steps, graph, selected, overlay, raisedPick]);

  const select = useCallback(
    (id: string | null) => {
      if (id) viewport.current?.setAutoRotate(false);
      // a platform: raise that section without selecting a part
      if (id?.startsWith("section:")) {
        if (mode === "flow") return;
        setSelected(null);
        setRaisedPick(id.slice(8) as SectionKey);
        return;
      }
      setRaisedPick(null);
      if (id?.startsWith("concept:")) {
        setPlaying(false);
        setMode("explore");
        setSelected(id);
        return;
      }
      if (mode === "flow") {
        const i = id ? steps.findIndex((s) => s.node.id === id) : -1;
        if (i >= 0) {
          setPlaying(false);
          setStepRaw(i);
        }
        return;
      }
      setSelected(id);
    },
    [mode, steps],
  );

  /** Stage navigator: open the stage's first node (or its step in the data-flow tour) and fly there. */
  const goStage = useCallback(
    (st: StageKey) => {
      const entry = stageEntry(graph, st);
      if (mode === "flow" && !entry.startsWith("concept:")) {
        setPlaying(false);
        setStepRaw(Math.max(0, steps.findIndex((x) => x.node.id === entry)));
      } else {
        select(entry);
      }
      viewport.current?.focus(`stage:${st}`);
    },
    [graph, mode, steps, select],
  );
  /** Selection from a list outside the 3D view also moves the camera there. */
  const pickAndFocus = useCallback(
    (id: string) => {
      select(id);
      viewport.current?.focus(id);
    },
    [select],
  );
  const onHover = useCallback((id: string | null, x: number, y: number) => setHover(id ? { id, x, y } : null), []);

  const switchDataset = (d: ArchDataset) => setParams({ dataset: d }, { replace: true });

  usePageContext({
    title: "Model context",
    chips: [
      { k: "Model", v: graph.name },
      { k: "Checkpoint", v: `baseline · seed ${graph.training.seed}` },
      { k: "Intervention", v: "none (α = 1)" },
      { k: "View", v: mode === "compare" ? "comparison" : mode === "flow" ? "data flow" : "explore" },
      { k: "Overlay", v: overlay ? "conceptual (on)" : "off", tone: overlay ? "gradient" : "ink" },
    ],
    note: "Architecture from code · no inference on this page",
  });

  const activeId = flow ? flow.node.id : selected;
  const activeStage = stageOf(graph, activeId);
  const glAvailable = webgl && !glError;

  return (
    <div className="space-y-8">
      <SectionHeader
        eyebrow="Model architecture & data flow"
        title="The two research models, layer by layer"
        subtitle="Rotate, select and trace the actual CREMA-D and UTD-MHAD networks. Every layer, tensor shape and parameter count is generated from the model code and checked against the saved baseline checkpoints."
      />

      {/* ── toolbar ─────────────────────────────────────────────── */}
      <div className="card flex flex-wrap items-end gap-x-6 gap-y-4 p-4" data-testid="arch-toolbar">
        <Seg<ArchDataset>
          label="Model"
          value={dataset}
          onChange={switchDataset}
          options={[
            { id: "cremad", label: "CREMA-D" },
            { id: "utd_mhad", label: "UTD-MHAD" },
          ]}
        />
        <Seg<Mode>
          label="Mode"
          value={mode}
          onChange={(m) => {
            setMode(m);
            setPlaying(false);
          }}
          options={[
            { id: "explore", label: "Explore" },
            { id: "flow", label: "Data flow" },
            { id: "compare", label: "Compare" },
          ]}
        />
        {mode !== "compare" && (
          <Seg<View>
            label="View"
            value={glAvailable ? view : "2d"}
            onChange={setView}
            options={[
              { id: "3d", label: "3D", disabled: !glAvailable },
              { id: "2d", label: "2D" },
            ]}
          />
        )}
        {mode !== "compare" && (
          <div>
            <div className="label mb-1.5">Research overlay</div>
            <button
              type="button"
              role="switch"
              aria-checked={overlay}
              onClick={() => setOverlay((o) => !o)}
              className="seg border border-line"
              data-active={overlay}
              data-testid="overlay-toggle"
            >
              {overlay ? "Conceptual pathways: on" : "Conceptual pathways: off"}
            </button>
          </div>
        )}
        {mode !== "compare" && view === "3d" && glAvailable && (
          <div className="flex gap-1.5">
            <button type="button" className="btn-ghost !px-2.5 !py-1.5 text-2xs" onClick={() => viewport.current?.resetCamera()}>
              <Crosshair size={13} /> Reset view
            </button>
            <button type="button" className="btn-ghost !px-2.5 !py-1.5 text-2xs" onClick={() => viewport.current?.fitView()}>
              <Maximize2 size={13} /> Fit to view
            </button>
          </div>
        )}
        <Link
          to={`/studio?dataset=${dataset}`}
          className="btn-primary ml-auto !py-2 text-2xs"
          data-testid="try-in-studio"
        >
          <ScanSearch size={14} /> Try this model in Prediction Studio
        </Link>
      </div>

      <p className="-mt-4 font-mono text-2xs leading-relaxed text-muted" data-testid="arch-provenance">
        {graph.modelClass} · {graph.params.total.toLocaleString("en-US")} parameters · {graph.checkpoint.path} (sha256{" "}
        {graph.checkpoint.sha256.slice(0, 12)}…) loaded with strict=True · {graph.preprocessingVersion} · generated by
        scripts/export_architecture.py · nothing is executed on this page
      </p>

      {mode === "compare" ? (
        <CompareTable />
      ) : (
        <div className="space-y-6">
          <div className="min-w-0 space-y-4">
            <StageNav graph={graph} active={activeStage} onPick={goStage} />
            <div
              className={`card relative overflow-hidden ${view === "3d" && glAvailable ? "arch-stage" : ""}`}
              style={view === "3d" && glAvailable ? { height: "min(78vh, 760px)", minHeight: 460, width: "100%" } : undefined}
            >
              {glError && (
                <div className="absolute inset-x-0 top-0 z-10 border-b border-warning/30 bg-warning/[0.08] px-4 py-1.5 text-2xs text-warning" role="status">
                  {glError}
                </div>
              )}
              {view === "3d" && glAvailable ? (
                <ViewportBoundary onError={(m) => setGlError(`The 3D view could not start (${m}); showing the 2D view.`)}>
                  <Suspense
                    fallback={
                      <div className="flex h-full items-center justify-center gap-2 text-2xs text-muted" data-testid="viewport-loading">
                        <Loader2 size={15} className="animate-spin" /> Loading the 3D viewer…
                      </div>
                    }
                  >
                    <Viewport3D
                      ref={viewport}
                      graph={graph}
                      state={sceneState}
                      onHover={onHover}
                      onSelect={select}
                      onError={(m) => setGlError(`The 3D view could not start (${m}); showing the 2D view.`)}
                      onOrbitChange={setOrbit}
                    />
                  </Suspense>
                </ViewportBoundary>
              ) : (
                <div className={glError ? "pt-8" : ""}>
                  <Diagram2D graph={graph} state={sceneState} onSelect={select} onHover={onHover} />
                </div>
              )}

              <HoverCard graph={graph} concepts={concepts} hover={hover} raised={sceneState.raised} view3d={view === "3d" && !!glAvailable} />

              {view === "3d" && glAvailable && (
                <>
                  <div className="pointer-events-none absolute left-3 top-3 rounded-md border border-line/70 bg-surface/80 px-2.5 py-1.5 font-mono text-[0.62rem] leading-relaxed text-muted backdrop-blur-sm">
                    <span className="font-semibold text-ink">360° workflow</span> · drag to orbit · right-drag to pan · wheel to zoom
                    <br />
                    click a section to raise it · hover its parts to see what they do · double-click to fly there
                  </div>
                  <div className="absolute bottom-3 right-3 flex flex-wrap items-center justify-end gap-1" data-testid="orbit-controls">
                    <button
                      type="button"
                      className="btn-ghost !bg-surface/90 !px-2.5 !py-1.5 text-2xs backdrop-blur-sm"
                      aria-pressed={orbit}
                      onClick={() => viewport.current?.setAutoRotate(!orbit)}
                    >
                      {orbit ? <Pause size={12} /> : <Play size={12} />} {orbit ? "Stop orbit" : "Orbit 360°"}
                    </button>
                    {(["front", "side", "back", "top"] as ViewPreset[]).map((v) => (
                      <button
                        key={v}
                        type="button"
                        className="btn-ghost !bg-surface/90 !px-2.5 !py-1.5 text-2xs capitalize backdrop-blur-sm"
                        onClick={() => viewport.current?.setView(v)}
                      >
                        {v}
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>
            <WorkflowLegend view={view === "3d" && glAvailable ? "3d" : "2d"} />
            <p className="font-mono text-[0.62rem] leading-relaxed text-muted">
              The moving data shows the direction of the forward pass; it is an animation, not live inference. Input thumbnails are real
              preprocessing outputs (TTS speech, a public-domain photo, UTD-MHAD test trial a3_s4_t2). Parts tagged "experiments only" or
              "synthetic exp." are not layers of this checkpoint.
            </p>
            <ResearchStrip graph={graph} active={selected} onPick={pickAndFocus} />

            {mode === "flow" && (
              <div className="flex items-center gap-2 font-mono text-2xs text-muted">
                {view === "3d" ? <Box size={13} /> : <Square size={13} />}
                Stage {step + 1}: {steps[step].node.label} · {fmtShape(steps[step].node.out_shape)}
              </div>
            )}
            {overlay && <OverlayExplainer graph={graph} />}
          </div>

          <aside className="grid min-w-0 gap-4 lg:grid-cols-2">
            {mode === "flow" ? (
              <FlowPanel graph={graph} steps={steps} step={step} playing={playing} setStep={setStep} setPlaying={startPlaying} />
            ) : selected && concepts[selected] ? (
              <ConceptInspector concept={concepts[selected]} />
            ) : (
              <Inspector graph={graph} node={nodeById(graph, selected)} onSelect={(id) => setSelected(id)} />
            )}
            <LayerList
              graph={graph}
              activeId={activeId}
              onSelect={(id) => pickAndFocus(id)}
              extra={Object.values(concepts).map((c) => ({ id: c.id, label: c.label }))}
            />
          </aside>
        </div>
      )}

      <section className="grid gap-3 sm:grid-cols-3">
        {[
          { to: "/disentanglement", label: "Disentanglement", desc: "What the scale operator at the fusion boundary does to training, with measured effects." },
          { to: `/studio?dataset=${dataset}`, label: "Prediction Studio", desc: `Run this ${graph.name} checkpoint on your own files.` },
          { to: "/methodology?section=trace", label: "Measured layer trace", desc: "Per-layer activation statistics from a real Studio prediction." },
        ].map((l) => (
          <Link key={l.label} to={l.to} className="group card p-4 transition-colors hover:border-ink/25">
            <div className="flex items-center justify-between text-sm font-semibold text-ink">
              {l.label} <ArrowRight size={13} className="text-muted group-hover:text-ink" />
            </div>
            <p className="mt-1 text-2xs leading-relaxed text-muted">{l.desc}</p>
          </Link>
        ))}
      </section>
    </div>
  );
}
