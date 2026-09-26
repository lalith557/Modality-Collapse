import { useState } from "react";
import { SectionHeader, Callout, Tag } from "../components/primitives";
import { PathwayDiagram } from "../components/PathwayDiagram";
import { PathwayBars, type BarRow } from "../components/PathwayBars";
import { Tabs } from "../components/Tabs";
import { ScientificFigure } from "../components/ScientificFigure";
import { decomposition } from "../data/research";
import { signed, fixed, pValue } from "../lib/format";

const armRows = (
  d: { forward: { d: number }; gradient: { d: number } }
): BarRow[] => [
  { label: "Forward pathway", magnitude: Math.abs(d.forward.d), value: `d = ${fixed(d.forward.d, 2)}`, color: "forward" },
  { label: "Gradient pathway", magnitude: Math.abs(d.gradient.d), value: `d = ${fixed(d.gradient.d, 2)}`, color: "gradient" },
];

const TABS = [
  { id: "adam", label: "Adam" },
  { id: "sgd", label: "SGD" },
];

export function Disentanglement() {
  const [tab, setTab] = useState("adam");
  const set = tab === "adam" ? decomposition.adamSynthetic : decomposition.sgdSynthetic;
  const maxD = 4;

  return (
    <div className="space-y-12">
      <SectionHeader
        eyebrow="Disentanglement"
        title="Where does the intervention travel?"
        subtitle="Separating the forward and gradient pathways to understand how feature scaling affects modality contribution. Custom autograd operators scale the forward pass, the backward pass, or both — under one unchanged objective."
      />

      <section>
        <ScientificFigure
          index="Fig. 1"
          caption="Conceptual decomposition of a training-time feature-scale intervention into a forward representation-magnitude pathway and a gradient / parameter-update pathway. This is a conceptual diagram, not an identified causal graph."
        >
          <div className="mx-auto max-w-xl">
            <PathwayDiagram alpha="100" />
          </div>
        </ScientificFigure>
      </section>

      <section>
        <div className="mb-1 flex flex-wrap items-end justify-between gap-3">
          <div className="label">Pathway strength comparison</div>
          <Tag tone="neutral">synthetic · α = 100 · five seeds</Tag>
        </div>
        <p className="mb-6 max-w-2xl text-sm leading-relaxed text-muted">
          Bar length is the effect size (|Cohen's d|) of the isolated arm on modality dominance.
          Switch optimizer to see which pathway carries the intervention.
        </p>

        <Tabs tabs={TABS} active={tab} onChange={setTab} className="mb-7" />

        <div className="grid gap-8 lg:grid-cols-[1fr_1fr]">
          <div className="rounded-md border border-line bg-surface p-6">
            <PathwayBars rows={armRows(set)} max={maxD} unit="effect size · |Cohen's d|" />
          </div>

          <div className="space-y-3">
            {(["forward", "gradient", "ordinary"] as const).map((arm) => {
              const a = set[arm];
              return (
                <div key={arm} className="flex items-baseline justify-between gap-4 border-b border-line py-2.5">
                  <span className="text-sm capitalize text-ink">{arm}-only{arm === "ordinary" ? " (both)" : ""}</span>
                  <span className="flex items-baseline gap-4 font-mono text-sm">
                    <span className="tabular text-ink">Δ𝒟 {signed(a.dDom)}</span>
                    <span className="tabular text-muted">d {fixed(a.d, 2)}</span>
                    <span className="tabular text-muted">p {pValue(a.p)}</span>
                  </span>
                </div>
              );
            })}
            <p className="pt-2 text-2xs leading-relaxed text-muted">
              {tab === "adam"
                ? "Under Adam, gradient-only leaves the parameter-update ratio at r_Δθ ≈ 1 even with a gradient-norm ratio in the hundreds; forward-only closely matches the ordinary intervention."
                : "Under SGD, gradient-only does shift dominance (r_Δθ ≈ 95.9); both pathways contribute and the ordinary shift is sub-additive (0.322 < 0.244 + 0.200)."}
            </p>
          </div>
        </div>
      </section>

      <Callout title="Key insight" tone={tab === "adam" ? "forward" : "neutral"}>
        Under tested Adam settings, the intervention primarily travels through the forward pathway, with
        minimal transmission through the gradient pathway. Under SGD the same gradient rescale is
        transmitted into the parameter update, so both pathways contribute — which pathway carries the
        effect is optimizer-dependent, not a universal law.
      </Callout>
    </div>
  );
}
