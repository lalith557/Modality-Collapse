import { Link } from "react-router-dom";
import { ArrowUpRight } from "lucide-react";
import { SectionHeader, MetricBlock } from "../components/primitives";
import { ScientificFigure, LegendSwatch } from "../components/ScientificFigure";
import { ContributionScatter } from "../charts/ContributionScatter";
import { C } from "../charts/theme";
import { headline, cremad } from "../data/research";

const keyFindings = [
  { n: "01", text: "The forward pathway dominates under the tested Adam settings.", to: "/disentanglement" },
  { n: "02", text: "The gradient pathway is strongly attenuated under tested Adam.", to: "/optimizer" },
  { n: "03", text: "Under SGD, both pathways contribute.", to: "/disentanglement" },
  { n: "04", text: "Modality utility changes observed behavior on UTD-MHAD.", to: "/real-world" },
];

export function Overview() {
  return (
    <div className="space-y-12">
      <SectionHeader
        eyebrow="Research overview"
        title="Understanding how feature scaling affects modality contribution during training"
        subtitle="A controlled autograd intervention separates the forward-representation and encoder-gradient consequences of scaling one modality, across a synthetic generator and two real datasets."
      />

      <section className="grid grid-cols-2 gap-px overflow-hidden rounded-md border border-line bg-line sm:grid-cols-4">
        {[
          { value: headline.conditions, label: "Experimental conditions" },
          { value: headline.datasets, label: "Datasets" },
          { value: headline.seeds, label: "Seeds per condition" },
          { value: headline.fusionMechanisms, label: "Fusion mechanisms" },
        ].map((m) => (
          <div key={m.label} className="bg-surface p-5">
            <MetricBlock value={m.value} label={m.label} />
          </div>
        ))}
      </section>

      <section className="grid gap-10 lg:grid-cols-[1.35fr_1fr]">
        <div>
          <div className="label mb-4">Contribution landscape</div>
          <ScientificFigure
            index="Fig. 1"
            caption="Learned audio-video dominance on CREMA-D as one modality's features are scaled during training. Imposed scale and dominance are rank-correlated (Spearman ρ = 0.922)."
            source="Source: manuscript Table II"
            legend={
              <>
                <LegendSwatch color={C.forward} label="audio scaled" />
                <LegendSwatch color={C.gradient} label="video scaled" />
              </>
            }
          >
            <ContributionScatter />
          </ScientificFigure>
        </div>

        <div>
          <div className="label mb-4">Key findings</div>
          <ul className="divide-y divide-line rounded-md border border-line bg-surface">
            {keyFindings.map((f) => (
              <li key={f.n}>
                <Link
                  to={f.to}
                  className="group flex items-start gap-4 p-5 transition-colors hover:bg-black/[0.015]"
                >
                  <span className="measure text-lg font-semibold text-line">{f.n}</span>
                  <span className="flex-1 text-[0.95rem] leading-relaxed text-ink">{f.text}</span>
                  <ArrowUpRight size={16} className="mt-0.5 text-muted/50 transition-colors group-hover:text-forward" />
                </Link>
              </li>
            ))}
          </ul>
          <div className="mt-4 rounded-md border border-line bg-black/[0.015] p-4 text-2xs leading-relaxed text-muted">
            Baseline CREMA-D is genuinely multimodal — joint accuracy{" "}
            <span className="measure text-ink">{cremad.jointAccuracy.toFixed(3)}</span>, gain{" "}
            <span className="measure text-ink">{cremad.scale[0].gain.toFixed(3)}</span>, dominance{" "}
            <span className="measure text-ink">{cremad.scale[0].dom.toFixed(3)}</span>.
          </div>
        </div>
      </section>
    </div>
  );
}
