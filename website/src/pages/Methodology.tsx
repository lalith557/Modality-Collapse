import { Code2, FileText, Download, Settings2 } from "lucide-react";
import { SectionHeader } from "../components/primitives";
import { protocol, metrics, paper } from "../data/research";

function Block({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="grid gap-4 border-t border-line pt-6 lg:grid-cols-[200px_1fr]">
      <h2 className="label pt-1">{title}</h2>
      <div>{children}</div>
    </section>
  );
}

export function Methodology() {
  return (
    <div className="space-y-10">
      <SectionHeader
        eyebrow="Methodology"
        title="Experimental specification"
        subtitle="A controlled four-arm autograd intervention (baseline / forward-only / gradient-only / ordinary) under one unchanged objective, across a synthetic generator and two real datasets."
      />

      <Block title="Datasets & protocol">
        <div className="overflow-x-auto rounded-md border border-line bg-surface">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-line">
                <th className="p-3 text-left font-mono text-2xs uppercase tracking-wider text-muted"> </th>
                {protocol.columns.map((c) => (
                  <th key={c} className="p-3 text-left font-display text-sm font-semibold text-ink">
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {protocol.rows.map((r) => (
                <tr key={r.field} className="border-b border-line last:border-0">
                  <td className="p-3 text-muted">{r.field}</td>
                  {r.values.map((v, i) => (
                    <td key={i} className="measure p-3 text-ink">
                      {v}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-2xs leading-relaxed text-muted">{protocol.optimizerNotes}</p>
      </Block>

      <Block title="Architecture">
        <p className="text-sm leading-relaxed text-ink/85">{protocol.architectureNote}</p>
        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          {[
            { k: "Modality encoders", v: "MLP / 2D-CNN / 1D-CNN" },
            { k: "Fusion", v: "sum · concat · gated" },
            { k: "Classifier", v: "linear head" },
          ].map((x) => (
            <div key={x.k} className="rounded border border-line bg-surface p-4">
              <div className="label mb-1.5">{x.k}</div>
              <div className="measure text-sm text-ink">{x.v}</div>
            </div>
          ))}
        </div>
      </Block>

      <Block title="Interventions">
        <div className="grid gap-3 sm:grid-cols-2">
          {[
            { k: "Baseline", v: "h̃ₐ = hₐ  (multipliers 1, 1)" },
            { k: "Forward-only", v: "forward scaled, gradient unscaled  (α, 1)" },
            { k: "Gradient-only", v: "forward identity, gradient scaled  (1, α)" },
            { k: "Ordinary", v: "both scaled  (α, α)" },
          ].map((x) => (
            <div key={x.k} className="rounded border border-line bg-surface p-4">
              <div className="label mb-1.5">{x.k}</div>
              <div className="measure text-sm text-ink">{x.v}</div>
            </div>
          ))}
        </div>
      </Block>

      <Block title="Metrics">
        <dl className="divide-y divide-line rounded-md border border-line bg-surface">
          {metrics.map((m) => (
            <div key={m.symbol} className="flex gap-4 p-4">
              <dt className="measure w-12 shrink-0 text-base font-semibold text-forward">{m.symbol}</dt>
              <dd>
                <div className="text-sm font-medium text-ink">{m.name}</div>
                <div className="mt-0.5 text-2xs leading-relaxed text-muted">{m.def}</div>
              </dd>
            </div>
          ))}
        </dl>
      </Block>

      <Block title="Statistics">
        <p className="text-sm leading-relaxed text-ink/85">
          Two-sided paired t-tests over n = 5 seeds on fixed splits; Cohen's d_z as the effect size; 95%
          confidence intervals as 1.96·sd/√n. No multiplicity correction. Inference is within-split
          reproducibility, not population-level generalization.
        </p>
      </Block>

      <Block title="Reproducibility">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {[
            { icon: Code2, label: "View code", href: "#", note: "add repo URL" },
            { icon: Settings2, label: "View configuration", href: "#", note: "configs/*.yaml" },
            { icon: Download, label: "Download results", href: "#", note: "results/**" },
            { icon: FileText, label: "Read paper", href: `${import.meta.env.BASE_URL}${paper.pdfPath}`, note: "main.pdf" },
          ].map((x) => {
            const Icon = x.icon;
            const external = x.href !== "#";
            return (
              <a
                key={x.label}
                href={x.href}
                {...(external ? { target: "_blank", rel: "noreferrer" } : {})}
                className="group flex items-center gap-3 rounded border border-line bg-surface p-4 transition-colors hover:border-ink/30"
              >
                <Icon size={17} className="text-muted group-hover:text-ink" />
                <span>
                  <span className="block text-sm font-medium text-ink">{x.label}</span>
                  <span className="block font-mono text-[0.6rem] uppercase tracking-wider text-muted">{x.note}</span>
                </span>
              </a>
            );
          })}
        </div>
        <p className="mt-3 text-2xs text-muted">
          Placeholder links — repository URLs will be added on release. Every run records seed, full config,
          git hash, hardware, and metrics.
        </p>
      </Block>
    </div>
  );
}
