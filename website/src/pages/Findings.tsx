import { useNavigate } from "react-router-dom";
import { TrendingUp, Activity, GitCompare, Scale, ArrowUpRight } from "lucide-react";
import { SectionHeader } from "../components/primitives";
import { findings } from "../data/research";

const META = [
  { icon: TrendingUp, to: "/disentanglement" },
  { icon: Activity, to: "/optimizer" },
  { icon: GitCompare, to: "/disentanglement" },
  { icon: Scale, to: "/real-world" },
];

export function Findings() {
  const nav = useNavigate();
  return (
    <div className="space-y-8">
      <SectionHeader
        eyebrow="Findings"
        title="What did we learn?"
        subtitle="Key findings, evidence, and interpretations from our study. Select a finding to open the relevant experiment."
      />

      <section className="grid gap-5 md:grid-cols-2">
        {findings.map((f, i) => {
          const Icon = META[i].icon;
          return (
            <button
              key={f.n}
              onClick={() => nav(META[i].to)}
              className="selectable group flex flex-col p-6 text-left"
            >
              <div className="mb-3 flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <span className="measure text-2xl font-semibold text-forward">{f.n}</span>
                  <Icon size={18} className="text-muted" strokeWidth={1.75} />
                </div>
                <ArrowUpRight size={16} className="text-muted/40 transition-colors group-hover:text-forward" />
              </div>
              <h3 className="mb-4 max-w-md text-base font-semibold leading-snug text-ink">{f.title}</h3>
              <dl className="mt-auto space-y-3">
                <div>
                  <dt className="mb-1 flex items-center gap-1.5"><span className="h-2.5 w-0.5 bg-forward" /><span className="label text-forward">Evidence</span></dt>
                  <dd className="text-2xs leading-relaxed text-ink/80">{f.evidence}</dd>
                </div>
                <div>
                  <dt className="mb-1 flex items-center gap-1.5"><span className="h-2.5 w-0.5 bg-positive" /><span className="label text-positive">Interpretation</span></dt>
                  <dd className="text-2xs leading-relaxed text-ink/80">{f.interpretation}</dd>
                </div>
              </dl>
            </button>
          );
        })}
      </section>

      <div className="rounded-md border border-line bg-black/[0.015] p-6">
        <div className="label mb-3">What we do not claim</div>
        <p className="max-w-3xl text-sm leading-relaxed text-ink/80">
          We do not claim that feature scale is a sufficient cause of persistent collapse, that gradient imbalance is
          universally irrelevant, that normalization cures imbalance, that Adam is universally invariant to gradient
          scaling, or that the phenomenon generalizes across architectures and optimizers. We identify experimentally
          separable pathways within the evaluated regimes.
        </p>
      </div>
    </div>
  );
}
