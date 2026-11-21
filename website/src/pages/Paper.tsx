import { Link } from "react-router-dom";
import { FileText, Download, ExternalLink } from "lucide-react";
import { SectionHeader, Tag } from "../components/primitives";
import { paper } from "../data/research";
import { useSectionScroll } from "../lib/useSectionScroll";

const pdfHref = `${import.meta.env.BASE_URL}${paper.pdfPath}`;

/**
 * Section summaries of the manuscript in website/public/ (CA research paper, 7 pages).
 * Every number quoted here appears in the PDF and traces to a logged run
 * (docs/RESULTS_LOG.md · docs/CLAIMS_LEDGER.md · docs/experiments/EXPERIMENT_LOG.md).
 */
const sections = [
  {
    id: "abstract",
    title: "Abstract",
    body: "Multimodal networks can collapse onto one modality while the other is attenuated. Most explanations blame differences in learning speed or task difficulty; the paper asks whether a feature-magnitude disparity between the encoders causes collapse by itself, and points out that the usual zero-masking accuracy measure mixes classifier out-of-distribution error with genuine representational loss. Forward activation scaling (h′_A = α·h_A) and backward gradient scaling (∇θ_A L ← α·∇θ_A L) are separated experimentally and explained theoretically, and contribution is audited with a frozen-encoder linear probe, mean-ablation dominance and Shapley values on 160 controlled synthetic runs, 520 mechanistic runs and the real audio-visual CREMA-D dataset.",
  },
  {
    id: "question",
    title: "Research questions",
    body: "RQ1 — does a change in feature magnitude alone cause a real collapse of the representation? RQ2 — can an early activation-norm ratio ρ predict which modality will dominate before training converges? RQ3 — how do adaptive (Adam / AdamW) and non-adaptive (SGD) optimizers differ under forward versus gradient scaling? RQ4 — are zero-masking metrics a good measure of representational degradation when compared with linear probes, mean ablation and Shapley values?",
  },
  {
    id: "method",
    title: "Method",
    body: "Two modality encoders map their inputs to h_A and h_B, which a fusion operator (sum, concatenation or gated) combines for a shared classification head. Three custom autograd operators act on modality A: forward-only scales the representation but passes the gradient unchanged, gradient-only leaves the representation unchanged but multiplies the encoder gradient by α, and ordinary does both. A diagnostic battery records the activation-norm ratio ρ, the parameter-update ratio ρ_Δθ, masked dominance 𝒟, a frozen-encoder linear probe and a two-modality Shapley value (excluded modalities mean-ablated). A first-order analysis shows why Adam's second-moment normalisation absorbs a constant gradient gain while SGD transmits it linearly.",
  },
  {
    id: "experiments",
    title: "Experiments",
    body: "Phase 0 audits the logs against the raw telemetry. Phase 1 runs the decoupled protocol on matched synthetic data (160 runs, 10 seeds, forward and gradient interventions, frozen probes and Shapley values). Phase 2 tests the mechanism (520 runs): learning-rate matching, forward-for-k-epochs sweeps, an ε sweep of Adam and collapse curves. Phase 3 validates on CREMA-D with 5 seeds, comparing AdamW with SGD under the decoupled metrics. All runs were executed on one NVIDIA RTX 4070 from git commit c76c5e6.",
  },
  {
    id: "results",
    title: "Results",
    body: "Synthetic, Adam, α = 100: forward scaling lowers modality A on every metric — masked accuracy −0.0525, frozen probe −0.0339, mean-ablation dominance −0.0559, Shapley −0.0432 — so in this low-capacity regime the degradation is real. Early diagnosis: the epoch-5 activation ratio is rank-correlated with dominance (Spearman 0.97), but trainable encoders compensate, shrinking the ratio range from 0.14–7.55 to 0.60–1.68 by convergence. Optimizers: under gradient scaling Adam keeps the update ratio at 1.001 ± 0.010 while SGD passes it through (95.8 ± 4.0); matching modality A's learning rate (η/α) restores dominance to +0.003, and raising Adam's ε from 10⁻⁸ to 10⁻² weakens the cancellation (1.00 → 1.02 → 2.84). CREMA-D: forward scaling moves zero-masked dominance by +0.259, but mean-ablation (+0.012) and Shapley (+0.014) are not significant and the probe is not degraded (+0.023) — the apparent collapse is largely a classifier-head covariate-shift artifact.",
  },
  {
    id: "limitations",
    title: "Limitations",
    body: "Real-world validation is limited to CREMA-D by compute and disk (RTX 4070, 8.6 GB VRAM, about 53 GB free disk); Kinetics-Sounds, AVE and VGGSound need larger clusters. Only two SGD runs survived at the extreme scale in Phase 3, too few for significance testing. The encoders are trained end-to-end from scratch; applying the scale interventions to pretrained multimodal transformers is left as future work. The conclusion that scale does not explain real-world collapse rests on a single real dataset and is evidence rather than proof.",
  },
  {
    id: "references",
    title: "References",
    body: "15 references covering modality competition and gradient-modulation methods (Wang et al., OGM-GE, PMR), the greedy nature of multimodal learning, adaptive optimisers (Adam, AdamW, Wilson et al.), interpretability tools (linear probes, ablation, SHAP, Shapley values), the CREMA-D dataset and layer normalisation. See the PDF for the full list.",
  },
];

export function Paper() {
  useSectionScroll();
  return (
    <div className="space-y-10">
      <SectionHeader eyebrow="The paper" title={paper.title} subtitle={paper.subtitle} />

      <div className="grid gap-8 lg:grid-cols-[340px_1fr]">
        {/* Paper preview card */}
        <div className="lg:sticky lg:top-8 lg:self-start">
          <div className="overflow-hidden rounded-md border border-line bg-surface shadow-panel">
            <div className="aspect-[1/1.29] border-b border-line bg-white p-6">
              <div className="mx-auto flex h-full flex-col">
                <div className="text-center font-display text-[0.82rem] font-semibold leading-tight text-ink" data-testid="paper-preview-title">
                  {paper.fullTitle}
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
                <div className="mt-auto text-center font-mono text-[0.5rem] text-muted">
                  {paper.pages} pages · {paper.references} references
                </div>
              </div>
            </div>
            <div className="p-4">
              <div className="mb-3 flex flex-wrap gap-1.5">
                <Tag tone="neutral">{paper.venue}</Tag>
                <Tag tone="neutral">{paper.authors.length} authors</Tag>
              </div>
              <div className="flex flex-col gap-2">
                <a href={pdfHref} target="_blank" rel="noreferrer" className="btn-primary w-full justify-center" data-testid="paper-pdf">
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
              <Link
                key={s.id}
                to={`/paper?section=${s.id}`}
                className="rounded border border-line bg-surface px-3 py-1.5 font-mono text-2xs uppercase tracking-wider text-muted transition-colors hover:border-ink/30 hover:text-ink"
              >
                {s.title}
              </Link>
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
            The full manuscript, its figures and {paper.references} references are in the PDF.
          </a>
        </div>
      </div>
    </div>
  );
}
