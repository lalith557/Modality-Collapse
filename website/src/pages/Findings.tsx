import { SectionHeader } from "../components/primitives";
import { FindingBlock } from "../components/FindingBlock";
import { findings } from "../data/research";

export function Findings() {
  return (
    <div className="space-y-12">
      <SectionHeader
        eyebrow="Findings"
        title="What did we learn?"
        subtitle="Each finding is stated as an observation with its evidence, a scoped interpretation, and its limitation — not as a marketing claim."
      />

      <div className="space-y-10">
        {findings.map((f) => (
          <FindingBlock key={f.n} finding={f} />
        ))}
      </div>

      <div className="rounded-md border border-line bg-black/[0.015] p-6">
        <div className="label mb-3">What we do not claim</div>
        <p className="max-w-3xl text-sm leading-relaxed text-ink/80">
          We do not claim that feature scale is a sufficient cause of persistent collapse, that gradient
          imbalance is universally irrelevant, that normalization cures imbalance, that Adam is universally
          invariant to gradient scaling, or that the phenomenon generalizes across architectures and
          optimizers. We identify experimentally separable pathways within the evaluated regimes.
        </p>
      </div>
    </div>
  );
}
