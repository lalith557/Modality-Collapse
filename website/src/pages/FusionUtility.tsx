import { SectionHeader, Callout, Tag } from "../components/primitives";
import { FusionDiagram } from "../components/diagrams";
import { fusion } from "../data/research";
import { fixed, pValue } from "../lib/format";

export function FusionUtility() {
  return (
    <div className="space-y-12">
      <SectionHeader
        eyebrow="Fusion & utility"
        title="Fusion mechanisms & modality utility"
        subtitle="Repeating the forward-vs-gradient decomposition under three fusion mechanisms (synthetic, α = 100). The gradient-only arm stays approximately null throughout; the forward effect persists but its magnitude varies."
      />

      <section className="grid gap-5 md:grid-cols-3">
        {fusion.mechanisms.map((m) => (
          <div key={m.key} className="flex flex-col rounded-md border border-line bg-surface p-5">
            <div className="mb-1 flex items-center justify-between">
              <h3 className="font-display text-base font-semibold text-ink">{m.name} fusion</h3>
              {m.significant ? (
                <Tag tone="forward">significant</Tag>
              ) : (
                <Tag tone="warning">n.s.</Tag>
              )}
            </div>
            <div className="my-4 rounded border border-line bg-canvas/50 p-3">
              <FusionDiagram kind={m.key as "sum" | "concat" | "gated"} />
            </div>
            <dl className="mt-auto space-y-2.5">
              <div className="flex items-baseline justify-between border-b border-line pb-2">
                <dt className="text-2xs uppercase tracking-wider text-muted">Forward effect</dt>
                <dd className="measure text-sm font-medium text-forward">
                  d = {fixed(m.forwardD, 2)}
                </dd>
              </div>
              <div className="flex items-baseline justify-between border-b border-line pb-2">
                <dt className="text-2xs uppercase tracking-wider text-muted">Significance</dt>
                <dd className="measure text-sm text-muted">p = {pValue(m.p)}</dd>
              </div>
              <div className="flex items-baseline justify-between">
                <dt className="text-2xs uppercase tracking-wider text-muted">Gradient effect</dt>
                <dd className="measure text-sm font-medium text-gradient">{fusion.gradientNull}</dd>
              </div>
            </dl>
            <p className="mt-3 text-2xs leading-relaxed text-muted">{m.note}</p>
          </div>
        ))}
      </section>

      <Callout title="Key takeaway" tone="forward">
        The forward-pathway effect persists across the tested fusion mechanisms, although its magnitude varies —
        strong under sum and concatenation, attenuated and not individually significant under gated fusion,
        consistent with the possibility that learned per-modality gates can partly absorb the imposed forward
        magnitude. The gradient-only arm stays approximately null ({fusion.gradientNull}) across all three.
        This is variation across the evaluated mechanisms, not a claim of architecture-general behavior.
      </Callout>
    </div>
  );
}
