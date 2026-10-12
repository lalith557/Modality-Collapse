import { useEffect } from "react";
import { Link } from "react-router-dom";
import { X, ArrowRight } from "lucide-react";
import { paper, headline } from "../data/research";
import { nav } from "../data/nav";

const NEXT_PATHS = ["/overview", "/lab", "/disentanglement", "/findings", "/paper"];

interface Props {
  onClose: () => void;
}

export function AboutPanel({ onClose }: Props) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onClose]);

  return (
    <>
      {/* Backdrop — stronger blur */}
      <div
        className="fixed inset-0 z-40 bg-ink/30 backdrop-blur-[5px]"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Panel — centered in content area, wide 2-column layout */}
      <div
        className="fixed inset-y-0 left-[240px] right-0 z-50 flex items-start justify-center
                   px-6 pt-[58px] pointer-events-none"
      >
        <div
          role="dialog"
          aria-modal="true"
          aria-label="About this project"
          className="animate-fade-in pointer-events-auto w-full max-w-[760px] overflow-hidden
                     rounded-xl border border-line bg-surface
                     shadow-[0_8px_48px_rgba(23,35,43,0.18)]"
        >

          {/* ── Header ── */}
          <div className="flex items-start justify-between border-b border-line px-6 py-5">
            <div className="min-w-0 pr-4">
              <div className="text-[1rem] font-semibold leading-snug text-ink">
                {paper.title}
              </div>
              <div className="mt-0.5 text-2xs leading-snug text-muted">{paper.subtitle}</div>
            </div>
            <button
              onClick={onClose}
              aria-label="Close about panel"
              className="mt-0.5 shrink-0 rounded p-1 text-muted transition-colors hover:bg-black/[0.05] hover:text-ink"
            >
              <X size={15} />
            </button>
          </div>

          {/* ── Body — two columns ── */}
          <div className="grid grid-cols-2 divide-x divide-line">

            {/* Left column: narrative sections */}
            <div className="divide-y divide-line">

              {/* What + Research question */}
              <div className="px-6 py-4 space-y-3.5">
                <div>
                  <div className="label mb-1">What is this?</div>
                  <p className="text-[0.82rem] leading-relaxed text-ink/75">
                    An interactive research explorer for the paper above. Browse experimental
                    results, compare pathway effects, and understand the evidence — no
                    re-running required.
                  </p>
                </div>
                <div>
                  <div className="label mb-1">Research question</div>
                  <p className="text-[0.82rem] leading-relaxed text-ink/75">
                    When feature-scale α is applied to a modality encoder at training time,
                    which mechanism drives the change in learned modality contribution — the{" "}
                    <span className="font-medium text-forward">
                      forward representation-magnitude pathway
                    </span>
                    , the{" "}
                    <span className="font-medium text-gradient">
                      gradient / parameter-update pathway
                    </span>
                    , or both?
                  </p>
                </div>
              </div>

              {/* Core experimental idea */}
              <div className="px-6 py-4">
                <div className="label mb-1">Core experimental idea</div>
                <p className="text-[0.82rem] leading-relaxed text-ink/75">
                  Custom autograd operators independently isolate the two pathways — scaling
                  only the forward representation, or only the gradient — under a single
                  unchanged objective. This lets the study measure each pathway's contribution
                  separately across Adam and SGD.
                </p>
              </div>

            </div>

            {/* Right column: evidence + produced by */}
            <div className="divide-y divide-line">

              {/* Evidence stats */}
              <div className="px-6 py-4">
                <div className="label mb-3">Evidence</div>
                <div className="grid grid-cols-2 gap-x-5 gap-y-3">
                  {[
                    { value: String(headline.datasets),       detail: "datasets",    sub: "Synthetic · CREMA-D · UTD-MHAD" },
                    { value: headline.conditions,             detail: "conditions",  sub: "across optimizers & fusion" },
                    { value: String(headline.seeds),          detail: "seeds each",  sub: "within-split reproducibility" },
                    { value: String(headline.fusionMechanisms), detail: "fusion types", sub: "sum · concat · gated" },
                  ].map(({ value, detail, sub }) => (
                    <div key={detail}>
                      <span className="font-mono text-[1.05rem] font-semibold text-ink">{value}</span>
                      <span className="ml-1.5 font-mono text-2xs text-muted">{detail}</span>
                      <div className="mt-0.5 text-[0.6rem] leading-snug text-muted/65">{sub}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Produced by */}
              <div className="px-6 py-4">
                <div className="label mb-2">Produced by</div>
                <p className="text-2xs text-muted">{paper.authors.join(", ")}</p>
                <p className="mt-0.5 text-2xs text-muted/80">{paper.affiliation}</p>
                <p className="mt-0.5 text-2xs text-muted/60">{paper.venue}</p>
              </div>

            </div>
          </div>

          {/* ── Footer: where next (full width, horizontal) ── */}
          <div className="border-t border-line bg-canvas/50 px-6 py-4">
            <div className="label mb-2.5">Where next?</div>
            <div className="flex flex-wrap gap-2">
              {NEXT_PATHS.map((path) => {
                const item = nav.find((n) => n.path === path);
                if (!item) return null;
                const Icon = item.icon;
                return (
                  <Link
                    key={path}
                    to={path}
                    onClick={onClose}
                    className="inline-flex items-center gap-1.5 rounded border border-line bg-surface
                               px-3 py-1.5 font-mono text-2xs text-muted transition-colors
                               hover:border-ink/25 hover:text-ink"
                  >
                    <Icon size={10} strokeWidth={2} />
                    {item.label}
                    <ArrowRight size={9} className="text-muted/50" />
                  </Link>
                );
              })}
            </div>
          </div>

        </div>
      </div>
    </>
  );
}
