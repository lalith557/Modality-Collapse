import { useState } from "react";
import { SectionHeader, MetricBlock, Callout, Tag, SpecRow } from "../components/primitives";
import { ScientificFigure } from "../components/ScientificFigure";
import { Tabs, TabPanel } from "../components/Tabs";
import { CremadScaleChart } from "../charts/CremadScaleChart";
import { UtdAccuracyChart } from "../charts/UtdChart";
import { PathwayBars, type BarRow } from "../components/PathwayBars";
import { cremad, utd } from "../data/research";
import { signed, fixed } from "../lib/format";

const TABS = [
  { id: "cremad", label: "CREMA-D" },
  { id: "utd", label: "UTD-MHAD" },
];

function CremadPanel() {
  return (
    <TabPanel className="space-y-10">
      <div className="grid gap-8 lg:grid-cols-[1fr_300px]">
        <div>
          <h2 className="font-display text-xl font-semibold text-ink">Audio scaling effect</h2>
          <p className="mt-2 max-w-xl text-sm leading-relaxed text-muted">
            Scaling one modality during training monotonically reshapes contribution: amplifying audio raises
            audio dominance and erodes multimodal gain, while amplifying video does the opposite.
          </p>
        </div>
        <div className="rounded-md border border-line bg-surface p-5">
          <div className="label mb-3">Dataset</div>
          <SpecRow field="Modalities">Audio + Video</SpecRow>
          <SpecRow field="Classes">{cremad.classes}</SpecRow>
          <SpecRow field="Evaluation">Actor-disjoint</SpecRow>
          <SpecRow field="Joint accuracy">{cremad.jointAccuracy.toFixed(3)}</SpecRow>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        <div className="rounded-md border border-line bg-surface p-5">
          <MetricBlock value={fixed(cremad.spearman, 3)} label="Spearman ρ (scale, dominance)" accent="forward" />
        </div>
        <div className="rounded-md border border-line bg-surface p-5">
          <MetricBlock value={fixed(cremad.scale[2].gain, 3)} label="Gain at audio ×10" accent="critical" sub="from 0.248 at baseline" />
        </div>
        <div className="rounded-md border border-line bg-surface p-5">
          <MetricBlock value={fixed(cremad.normalization.domAfter, 3)} label="Dominance after z-score (audio ×10)" accent="positive" sub="from 0.328 before" />
        </div>
      </div>

      <div className="grid gap-8 lg:grid-cols-2">
        <ScientificFigure index="Fig. 1a" caption="Modality contribution (dominance 𝒟 = audio − video) across training-time scale conditions.">
          <CremadScaleChart metric="dom" />
        </ScientificFigure>
        <ScientificFigure index="Fig. 1b" caption="Multimodal gain 𝒢 across the same conditions — amplifying either modality far enough erodes the benefit of fusion.">
          <CremadScaleChart metric="gain" />
        </ScientificFigure>
      </div>

      <Callout title="Reading the table" tone="forward">
        Imposed scale and learned dominance are rank-correlated (Spearman {fixed(cremad.spearman, 3)}). Per-modality
        z-scoring largely suppresses the imposed difference (audio ×10: dominance {fixed(cremad.normalization.domBefore, 3)} → {fixed(cremad.normalization.domAfter, 3)}),
        a control on the scale channel rather than a general cure.
      </Callout>
    </TabPanel>
  );
}

function UtdPanel() {
  const inertialRows: BarRow[] = [
    { label: "Baseline", magnitude: Math.abs(utd.base.dominance), value: signed(utd.base.dominance), color: "neutral" },
    { label: "Inertial ×10", magnitude: Math.abs(utd.inertialX10.dominance), value: signed(utd.inertialX10.dominance), color: "forward" },
    { label: "Depth ×10", magnitude: Math.abs(utd.depthX10.dominance), value: signed(utd.depthX10.dominance), color: "gradient" },
  ];
  return (
    <TabPanel className="space-y-10">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="font-display text-xl font-semibold text-ink">Modality-utility boundary</h2>
          <p className="mt-2 max-w-xl text-sm leading-relaxed text-muted">
            Depth and inertial sensing over {utd.actions} actions ({utd.samples} usable samples), subject-disjoint.
            The two modalities have strongly asymmetric predictive utility.
          </p>
        </div>
        <Tag tone="warning">Boundary case · not a universal law</Tag>
      </div>

      <div className="grid gap-3 sm:grid-cols-4">
        <div className="rounded-md border border-line bg-surface p-5">
          <MetricBlock value={fixed(utd.base.depthAcc, 3)} label="Depth accuracy" accent="critical" sub={`≈ ${utd.actions}-class chance (${utd.chanceLevel})`} />
        </div>
        <div className="rounded-md border border-line bg-surface p-5">
          <MetricBlock value={fixed(utd.base.inertialAcc, 3)} label="Inertial accuracy" accent="forward" />
        </div>
        <div className="rounded-md border border-line bg-surface p-5">
          <MetricBlock value={fixed(utd.base.multimodalAcc, 3)} label="Multimodal accuracy" accent="ink" />
        </div>
        <div className="rounded-md border border-line bg-surface p-5">
          <MetricBlock value={fixed(utd.base.gain, 3)} label="Multimodal gain 𝒢" accent="positive" />
        </div>
      </div>

      <div className="grid gap-8 lg:grid-cols-2">
        <ScientificFigure index="Fig. 2" caption={`Masked unimodal and multimodal accuracy. Depth sits near the ${utd.actions}-class chance level; inertial is informative.`} source="Source: manuscript §V, Table III">
          <UtdAccuracyChart />
        </ScientificFigure>
        <div>
          <div className="label mb-4">Effect of amplifying each modality ×10</div>
          <div className="rounded-md border border-line bg-surface p-6">
            <PathwayBars rows={inertialRows} unit="dominance 𝒟 (magnitude)" />
          </div>
          <p className="mt-4 text-2xs leading-relaxed text-muted">
            Amplifying the informative inertial modality pushes dominance to {signed(utd.inertialX10.dominance)} and
            collapses gain to {fixed(utd.inertialX10.gain, 3)}; amplifying the near-chance depth modality barely moves
            contribution (gain ≈ {fixed(utd.depthX10.gain, 2)}).
          </p>
        </div>
      </div>

      <Callout title="Boundary condition" tone="warning">
        A boundary case with strongly asymmetric predictive utility. The dominance metric is influenced by the
        large difference in unimodal utility — scaling a near-chance modality cannot manufacture contribution.
        Reported as per-subject consistency over four fixed test subjects, not population inference.
      </Callout>
    </TabPanel>
  );
}

export function RealWorld() {
  const [tab, setTab] = useState("cremad");
  return (
    <div className="space-y-10">
      <SectionHeader
        eyebrow="Real-world evidence"
        title="Does the pattern hold on real datasets?"
        subtitle="Two audio-visual and multimodal-sensing benchmarks: a genuine multimodal baseline (CREMA-D) and a utility-asymmetric boundary case (UTD-MHAD)."
      />
      <Tabs tabs={TABS} active={tab} onChange={setTab} />
      {tab === "cremad" ? <CremadPanel /> : <UtdPanel />}
    </div>
  );
}
