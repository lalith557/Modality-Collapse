import { useMemo, useState } from "react";
import { Play, Info } from "lucide-react";
import { SectionHeader, MetricBlock, IllustrativeNote, Card } from "../components/primitives";
import { Field, Select, SeedPicker, Slider } from "../components/controls";
import { ModalityPipeline } from "../components/diagrams";
import { TrainingLineChart } from "../charts/TrainingLineChart";
import { decomposition } from "../data/research";
import { signed, times } from "../lib/format";

type Arm = "forward" | "gradient" | "ordinary";

// Reported decomposition endpoints per (dataset, optimizer) — from Table I.
const CONFIGS = {
  "synthetic-adam": { data: decomposition.adamSynthetic, updateRatio: "≈ 1", measured: true },
  "cremad-adam": { data: decomposition.adamCremad, updateRatio: times(decomposition.ratios.adam.rTheta, 2), measured: true },
  "synthetic-sgd": { data: decomposition.sgdSynthetic, updateRatio: times(decomposition.ratios.sgd.rTheta, 1), measured: true },
} as const;

const datasetOptions = [
  { value: "synthetic", label: "Synthetic" },
  { value: "cremad", label: "CREMA-D" },
];
const interventionOptions = [
  { value: "forward", label: "Forward scaling" },
  { value: "gradient", label: "Gradient scaling" },
  { value: "ordinary", label: "Ordinary (both)" },
];
const fusionOptions = [
  { value: "sum", label: "Sum" },
  { value: "concat", label: "Concatenation" },
  { value: "gated", label: "Gated" },
];

export function ExperimentLab() {
  const [dataset, setDataset] = useState("synthetic");
  const [optimizer, setOptimizer] = useState("adam");
  const [intervention, setIntervention] = useState<Arm>("forward");
  const [alpha, setAlpha] = useState(100);
  const [fusion, setFusion] = useState("sum");
  const [seed, setSeed] = useState(1);

  // CREMA-D was evaluated under Adam only; keep configs to measured ones.
  const optimizerOptions =
    dataset === "cremad"
      ? [{ value: "adam", label: "Adam" }]
      : [
          { value: "adam", label: "Adam" },
          { value: "sgd", label: "SGD" },
        ];
  const effectiveOptimizer = dataset === "cremad" ? "adam" : optimizer;
  const key = `${dataset}-${effectiveOptimizer}` as keyof typeof CONFIGS;
  const cfg = CONFIGS[key];

  const forwardVal = cfg.data.forward.dDom;
  const gradientVal = cfg.data.gradient.dDom;
  const activeVal = useMemo(() => cfg.data[intervention].dDom, [cfg, intervention]);
  const fmt = (x: number) => signed(x, Math.abs(x) > 0 && Math.abs(x) < 0.001 ? 4 : 3);

  return (
    <div className="space-y-10">
      <SectionHeader
        eyebrow="Experiment lab"
        title="Configure a controlled scaling intervention"
        subtitle="Select a dataset, optimizer, and intervention arm to inspect how the forward and gradient pathways carry a training-time feature-scale change."
      />

      <div className="flex items-center gap-2 rounded-md border border-warning/30 bg-warning/[0.06] px-4 py-3">
        <Info size={15} className="shrink-0 text-warning" />
        <p className="text-2xs leading-relaxed text-ink/80">
          <span className="font-semibold">Interactive visualization.</span> No live training runs here.
          Result endpoints are the values reported in the manuscript for the selected dataset/optimizer;
          the α slider and training curve are schematic.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[300px_1fr]">
        {/* Configuration */}
        <Card className="h-fit p-5">
          <div className="label mb-5">Configuration</div>
          <div className="space-y-5">
            <Field label="Dataset">
              <Select value={dataset} onChange={(v) => { setDataset(v); if (v === "cremad") setOptimizer("adam"); }} options={datasetOptions} />
            </Field>
            <Field label="Optimizer">
              <Select value={effectiveOptimizer} onChange={setOptimizer} options={optimizerOptions} />
            </Field>
            <Field label="Intervention">
              <Select value={intervention} onChange={(v) => setIntervention(v as Arm)} options={interventionOptions} />
            </Field>
            <div>
              <div className="label mb-2">Scale α · <span className="text-warning">schematic</span></div>
              <Slider value={alpha} onChange={setAlpha} min={1} max={100} step={1} format={(v) => `${v}×`} />
            </div>
            <Field label="Fusion mechanism">
              <Select value={fusion} onChange={setFusion} options={fusionOptions} />
            </Field>
            <div>
              <div className="label mb-2">Seed</div>
              <SeedPicker value={seed} onChange={setSeed} />
            </div>
            <button className="btn-primary w-full justify-center" onClick={() => setAlpha((a) => a)}>
              <Play size={14} />
              Run visualization
            </button>
          </div>
        </Card>

        {/* Pipeline + results */}
        <div className="space-y-6">
          <Card className="p-5">
            <div className="mb-5 flex items-center justify-between">
              <div className="label">Model pipeline</div>
              <span className="font-mono text-2xs text-muted">{fusion} fusion · seed {seed}</span>
            </div>
            <div className="grid grid-cols-2 gap-8">
              <ModalityPipeline
                modality="Modality A"
                signal={dataset === "cremad" ? "Audio" : "Signal A"}
                symbol="hₐ"
                scaled={intervention !== "gradient"}
                alpha={intervention !== "gradient" ? alpha : undefined}
              />
              <ModalityPipeline
                modality="Modality B"
                signal={dataset === "cremad" ? "Video" : "Signal B"}
                symbol="h_b"
              />
            </div>
            <p className="mt-5 border-t border-line pt-3 text-2xs leading-relaxed text-muted">
              The operator acts at the fusion boundary on modality A only.{" "}
              {intervention === "forward" && "Forward-only: forward scaled by α, gradient unscaled."}
              {intervention === "gradient" && "Gradient-only: forward unchanged (identity), only the encoder gradient scaled by α."}
              {intervention === "ordinary" && "Ordinary: both forward and gradient scaled by α."}
            </p>
          </Card>

          <div>
            <div className="mb-4 flex items-center justify-between">
              <div className="label">Results · reported endpoints ({cfg.data.label})</div>
            </div>
            <div className="grid grid-cols-3 gap-px overflow-hidden rounded-md border border-line bg-line">
              <div className="bg-surface p-5">
                <MetricBlock value={fmt(forwardVal)} label="Forward contribution Δ𝒟" accent="forward" size="sm" />
              </div>
              <div className="bg-surface p-5">
                <MetricBlock value={fmt(gradientVal)} label="Gradient contribution Δ𝒟" accent="gradient" size="sm" />
              </div>
              <div className="bg-surface p-5">
                <MetricBlock value={cfg.updateRatio} label="Update ratio r_Δθ" accent="neutral" size="sm" />
              </div>
            </div>
          </div>

          <Card className="p-5">
            <div className="mb-3 flex items-center justify-between">
              <div className="label">Training progress · {intervention} arm</div>
              <IllustrativeNote>Schematic — endpoint is reported</IllustrativeNote>
            </div>
            <TrainingLineChart target={activeVal} />
          </Card>
        </div>
      </div>
    </div>
  );
}
