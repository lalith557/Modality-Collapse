import { Link } from "react-router-dom";
import { ArrowRight, FileText } from "lucide-react";
import { PathwayDiagram } from "../components/PathwayDiagram";
import { Contour } from "../components/Contour";
import { MetricBlock } from "../components/primitives";
import { headline } from "../data/research";

export function Home() {
  return (
    <div>
      {/* Hero */}
      <section className="relative">
        <Contour />
        <div className="relative grid items-center gap-12 lg:grid-cols-[1.05fr_0.95fr]">
          <div className="animate-fade-up">
            <div className="label mb-5">Forward or Gradient?</div>
            <h1 className="font-display text-[2.9rem] font-semibold leading-[1.02] tracking-tight text-ink sm:text-[3.6rem]">
              Forward or
              <br />
              Gradient?
            </h1>
            <p className="mt-5 max-w-md font-display text-lg font-medium leading-snug text-ink/80">
              Disentangling training-time feature-scale effects on modality contribution.
            </p>
            <p className="mt-4 max-w-md text-[0.95rem] leading-relaxed text-muted">
              An experimental study of how modality-specific feature scaling propagates through
              representation magnitude, gradients, and parameter updates.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link to="/disentanglement" className="btn-primary">
                Explore experiments
                <ArrowRight size={16} />
              </Link>
              <Link to="/paper" className="btn-ghost">
                <FileText size={16} />
                Read the paper
              </Link>
            </div>
          </div>

          <div className="animate-fade-in rounded-lg border border-line bg-surface/70 p-6 sm:p-8">
            <PathwayDiagram />
          </div>
        </div>
      </section>

      {/* Headline metrics */}
      <section className="mt-16 grid grid-cols-2 gap-px overflow-hidden rounded-md border border-line bg-line sm:grid-cols-4">
        {[
          { value: headline.conditions, label: "Experimental conditions" },
          { value: headline.datasets, label: "Datasets" },
          { value: headline.seeds, label: "Seeds per condition" },
          { value: headline.fusionMechanisms, label: "Fusion mechanisms" },
        ].map((m) => (
          <div key={m.label} className="bg-surface p-6">
            <MetricBlock value={m.value} label={m.label} size="lg" />
          </div>
        ))}
      </section>

      {/* The story, in one line */}
      <section className="mt-14">
        <div className="label mb-4">The story in one line</div>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2 font-display text-[0.95rem] font-medium text-ink/75">
          {[
            "Feature scaling",
            "Forward representation magnitude",
            "Gradient / optimizer pathway",
            "Parameter updates",
            "Modality contribution",
            "Multimodal performance",
          ].map((step, i, arr) => (
            <span key={step} className="flex items-center gap-3">
              <span>{step}</span>
              {i < arr.length - 1 && <ArrowRight size={14} className="text-muted/60" />}
            </span>
          ))}
        </div>
      </section>
    </div>
  );
}
