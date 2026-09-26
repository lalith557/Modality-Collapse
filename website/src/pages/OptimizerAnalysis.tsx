import { ArrowDown } from "lucide-react";
import { SectionHeader, MetricBlock, Callout } from "../components/primitives";
import { ScientificFigure, LegendSwatch } from "../components/ScientificFigure";
import { OptimizerRatioChart } from "../charts/OptimizerRatioChart";
import { C } from "../charts/theme";
import { decomposition } from "../data/research";
import { times } from "../lib/format";

const { adam, sgd } = decomposition.ratios;

function OptimizerPanel({
  name,
  rG,
  rTheta,
  note,
}: {
  name: string;
  rG: number;
  rTheta: number;
  note: string;
}) {
  return (
    <div className="rounded-md border border-line bg-surface p-6">
      <div className="mb-6 flex items-center justify-between">
        <h3 className="font-display text-xl font-semibold text-ink">{name}</h3>
        <span className="font-mono text-2xs uppercase tracking-label text-muted">gradient-only arm</span>
      </div>
      <div className="grid grid-cols-2 gap-6">
        <MetricBlock value={times(rG, 1)} label="Gradient-norm ratio r_g" accent="gradient" size="lg" />
        <MetricBlock value={times(rTheta, rTheta >= 100 ? 0 : rTheta >= 10 ? 1 : 2)} label="Update ratio r_Δθ" accent="forward" size="lg" />
      </div>
      <p className="mt-6 border-t border-line pt-4 text-2xs leading-relaxed text-muted">{note}</p>
    </div>
  );
}

export function OptimizerAnalysis() {
  return (
    <div className="space-y-12">
      <SectionHeader
        eyebrow="Optimizer analysis"
        title="Adam vs. SGD"
        subtitle="Why does the observed pathway transmission differ between the tested optimizers? The same gradient-only intervention meets two different update rules."
      />

      <section className="grid gap-6 lg:grid-cols-2">
        <OptimizerPanel
          name="Adam"
          rG={adam.rG}
          rTheta={adam.rTheta}
          note="A gradient-norm ratio in the hundreds collapses to an update ratio near 1 — consistent with Adam's normalized update attenuating an approximately constant per-parameter gradient rescale, within the tested configuration."
        />
        <OptimizerPanel
          name="SGD"
          rG={sgd.rG}
          rTheta={sgd.rTheta}
          note="A comparable gradient-norm ratio passes through to an update ratio of ~96× — the gradient scale behaves like a substantial effective per-modality learning-rate difference in this setting."
        />
      </section>

      <section>
        <div className="label mb-4">Gradient-norm ratio vs. parameter-update ratio</div>
        <ScientificFigure
          index="Fig. 1"
          caption="For the gradient-only arm, the gradient-norm ratio (r_g) is comparable across optimizers, but the resulting parameter-update ratio (r_Δθ) is not: Adam attenuates it to ≈ 1×, SGD passes it through to ≈ 96×. Log scale."
          source="Source: manuscript Table I"
          legend={
            <>
              <LegendSwatch color={C.gradient} label="gradient-norm ratio r_g" />
              <LegendSwatch color={C.forward} label="update ratio r_Δθ" />
            </>
          }
        >
          <OptimizerRatioChart />
        </ScientificFigure>
      </section>

      <section>
        <div className="label mb-5">From gradient to contribution</div>
        <div className="mx-auto flex max-w-xs flex-col items-center">
          {[
            { t: "Gradient magnitude", tone: "gradient" },
            { t: "Optimizer", tone: "ink" },
            { t: "Parameter update", tone: "forward" },
            { t: "Learned contribution", tone: "ink" },
          ].map((n, i, arr) => (
            <div key={n.t} className="flex w-full flex-col items-center">
              <div
                className={`w-full rounded border px-4 py-2.5 text-center font-mono text-sm ${
                  n.tone === "gradient"
                    ? "border-gradient/40 bg-gradient/[0.08] text-gradient"
                    : n.tone === "forward"
                    ? "border-forward/40 bg-forward/[0.08] text-forward"
                    : "border-ink bg-ink text-canvas"
                }`}
              >
                {n.t}
              </div>
              {i < arr.length - 1 && <ArrowDown size={15} className="my-1.5 text-muted/50" />}
            </div>
          ))}
        </div>
      </section>

      <Callout title="Key insight" tone="gradient">
        A large gradient-norm ratio does not necessarily produce a proportional update ratio. The optimizer
        sits between gradient magnitude and the parameter update, so whether an amplified gradient reaches
        the encoder depends on the update rule — attenuated under Adam, transmitted under SGD, within the
        tested configurations. We do not claim Adam is universally invariant to gradient scaling.
      </Callout>
    </div>
  );
}
