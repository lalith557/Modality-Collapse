import { FileText, Download, ExternalLink } from "lucide-react";
import { SectionHeader, Tag } from "../components/primitives";
import { paper } from "../data/research";

const pdfHref = `${import.meta.env.BASE_URL}${paper.pdfPath}`;

const sections = [
  {
    id: "abstract",
    title: "Abstract",
    body: "Multimodal networks frequently under-use one modality, and feature-magnitude imbalance is often invoked as an explanation. We ask a sharper, mechanistic question: how does a training-time, modality-specific feature-scale intervention alter learned modality contribution, and through which pathway — the forward representation magnitude, the encoder-gradient magnitude, or both? Using custom autograd operators that rescale one modality's fused representation in the forward pass only, the backward pass only, or both, we separate the two pathways and record the encoder parameter-update ratio as an optimization-level readout.",
  },
  {
    id: "question",
    title: "Research question",
    body: "When a modality-specific feature scale is imposed during training, does it act through the forward representation-magnitude pathway, the encoder-gradient pathway, or both — and how does this depend on optimizer, fusion mechanism, and modality utility?",
  },
  {
    id: "method",
    title: "Method",
    body: "A four-arm decomposition (baseline / forward-only / gradient-only / ordinary) under one unchanged objective. Custom autograd operators verify their forward/backward multipliers before training; the operator acts at the fusion boundary on modality A only. The parameter-update ratio r_Δθ is recorded as an optimization-level readout (not a formal mediator).",
  },
  {
    id: "experiments",
    title: "Experiments",
    body: "A controlled synthetic generator (matched generative signal-to-noise) plus two real datasets: CREMA-D (audio-video, actor-disjoint) and UTD-MHAD (depth-inertial, subject-disjoint). Optimizers Adam and SGD; fusion by sum, concatenation, and gating; five seeds on fixed splits.",
  },
  {
    id: "results",
    title: "Results",
    body: "Under Adam the intervention travels through the forward pathway (gradient-only leaves r_Δθ ≈ 1 despite 100–282× gradient-norm ratios); under SGD both pathways contribute (r_Δθ ≈ 96×). CREMA-D matches the Adam pattern (forward d ≈ 7.4); the forward effect varies across fusion mechanisms; UTD-MHAD shows a utility-asymmetric boundary. Normalization suppresses the imposed-scale difference but is not a cure.",
  },
  {
    id: "limitations",
    title: "Limitations",
    body: "Bounded by the tested optimizers and hyperparameters; three evaluated fusion mechanisms; one synthetic generator and two real datasets on fixed splits; n = 5 seeds (within-split reproducibility, not population inference); the real-data decomposition evaluated only under Adam and the specified CNN architecture.",
  },
  {
    id: "references",
    title: "References",
    body: "45 references spanning multimodal fusion, modality imbalance and rebalancing, normalization, optimizer dynamics, and the audio-visual / multimodal-sensing benchmarks used. See the compiled PDF for the full list.",
  },
];

export function Paper() {
  return (
    <div className="space-y-10">
      <SectionHeader eyebrow="The paper" title="Forward or Gradient?" subtitle={paper.subtitle} />

      <div className="grid gap-8 lg:grid-cols-[340px_1fr]">
        {/* Paper preview card */}
        <div className="lg:sticky lg:top-8 lg:self-start">
          <div className="overflow-hidden rounded-md border border-line bg-surface shadow-panel">
            <div className="aspect-[1/1.29] border-b border-line bg-white p-6">
              <div className="mx-auto flex h-full flex-col">
                <div className="text-center font-display text-[0.82rem] font-semibold leading-tight text-ink">
                  Forward or Gradient? Disentangling Training-Time Feature-Scale Effects on Modality
                  Contribution
                </div>
                <div className="mt-3 text-center font-mono text-[0.5rem] leading-relaxed text-muted">
                  {paper.authors.join(" · ")}
                  <br />
                  {paper.affiliation}
                </div>
                <div className="mt-4 space-y-1.5">
                  {[100, 92, 96, 88, 94, 70].map((w, i) => (
                    <div key={i} className="h-1 rounded-full bg-line" style={{ width: `${w}%` }} />
                  ))}
                </div>
                <div className="mt-3 grid grid-cols-2 gap-2">
                  <div className="h-16 rounded-sm border border-line bg-canvas" />
                  <div className="h-16 rounded-sm border border-line bg-canvas" />
                </div>
                <div className="mt-3 space-y-1.5">
                  {[96, 90, 93].map((w, i) => (
                    <div key={i} className="h-1 rounded-full bg-line" style={{ width: `${w}%` }} />
                  ))}
                </div>
                <div className="mt-auto text-center font-mono text-[0.5rem] text-muted">6 pages · 45 references</div>
              </div>
            </div>
            <div className="p-4">
              <div className="mb-3 flex flex-wrap gap-1.5">
                <Tag tone="neutral">IEEE-style</Tag>
                <Tag tone="neutral">MLSP format</Tag>
              </div>
              <div className="flex flex-col gap-2">
                <a href={pdfHref} target="_blank" rel="noreferrer" className="btn-primary w-full justify-center">
                  <FileText size={15} /> Read paper
                </a>
                <a href={pdfHref} download className="btn-ghost w-full justify-center">
                  <Download size={15} /> Download PDF
                </a>
              </div>
            </div>
          </div>
        </div>

        {/* Sections */}
        <div>
          <nav className="mb-8 flex flex-wrap gap-2">
            {sections.map((s) => (
              <a
                key={s.id}
                href={`#${s.id}`}
                className="rounded border border-line bg-surface px-3 py-1.5 font-mono text-2xs uppercase tracking-wider text-muted transition-colors hover:border-ink/30 hover:text-ink"
              >
                {s.title}
              </a>
            ))}
          </nav>
          <div className="space-y-8">
            {sections.map((s) => (
              <section key={s.id} id={s.id} className="scroll-mt-24 border-t border-line pt-6">
                <h2 className="mb-2 font-display text-lg font-semibold text-ink">{s.title}</h2>
                <p className="max-w-2xl text-[0.95rem] leading-relaxed text-ink/85">{s.body}</p>
              </section>
            ))}
          </div>
          <a href={pdfHref} target="_blank" rel="noreferrer" className="mt-10 flex items-center gap-2 rounded-md border border-line bg-black/[0.015] p-4 text-2xs text-muted transition-colors hover:border-forward/40 hover:text-forward">
            <ExternalLink size={14} />
            The full manuscript, figures, and 45 references are in the compiled PDF.
          </a>
        </div>
      </div>
    </div>
  );
}
