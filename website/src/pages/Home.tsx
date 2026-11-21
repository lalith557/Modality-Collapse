import { Link } from "react-router-dom";
import { ArrowRight, FileText } from "lucide-react";
import { PathwayDiagram } from "../components/PathwayDiagram";
import { MetricBlock } from "../components/primitives";
import { BlurFade } from "../components/ui/blur-fade";
import { ShaderBackground } from "../components/ui/shader-background";
import { MorphingPopover } from "../components/ui/morphing-popover";
import { AnimatedBorder } from "../components/ui/animated-border";
import { useTheme } from "../context/theme-context";
import { headline } from "../data/research";

const STORY_TIPS: Record<string, string> = {
  "Feature scaling":
    "Multiplying a modality encoder's output by a scalar α (e.g. ×10, ×100) before fusion. This is the root perturbation that propagates through both pathways.",
  "Forward representation magnitude":
    "The L2 norm of the feature vector at the fusion boundary. Scaling α inflates this directly, giving that modality a proportionally louder signal in the fused representation.",
  "Gradient / optimizer pathway":
    "The gradient back-propagating through fusion to the encoder. Under Adam it is heavily normalized by per-parameter adaptive scaling; under SGD it passes at full strength.",
  "Parameter updates":
    "The weight changes written to encoder parameters after an optimizer step. Adam nearly equalizes update magnitudes across modalities regardless of the gradient-norm ratio.",
  "Modality contribution":
    "The fraction of the model's output attributable to each modality, measured via dominance (𝒟 = Acc_A − Acc_B) and multimodal gain 𝒢.",
  "Multimodal performance":
    "Joint classification accuracy of the combined model. Contribution imbalance from scaling can degrade performance when one modality is less reliable.",
};

export function Home() {
  const { theme } = useTheme();
  return (
    <div>
      {/* Hero */}
      <section className="relative overflow-hidden rounded-xl border border-line">
        <ShaderBackground scale={0.4} />
        {/* Theme overlay — JS-controlled so it responds to toggle, not OS preference */}
        <div
          className="absolute inset-0 rounded-xl"
          style={{
            background: theme === "dark"
              ? "rgba(12,17,22,0.88)"
              : "rgba(245,243,238,0.93)",
          }}
        />

        <div className="relative z-10 grid items-center gap-12 px-8 py-14 lg:grid-cols-[1.05fr_0.95fr]">
          <div>
            <BlurFade delay={0} inView>
              <div className="label mb-5">Feature scale as a hidden learning rate</div>
            </BlurFade>
            <BlurFade delay={0.08} inView>
              <h1 className="font-display text-[2.9rem] font-semibold leading-[1.02] tracking-tight text-ink sm:text-[3.6rem]">
                A Hidden Per-Modality
                <br />
                Learning Rate
              </h1>
            </BlurFade>
            <BlurFade delay={0.16} inView>
              <p className="mt-5 max-w-md font-display text-lg font-medium leading-snug text-ink/80">
                A training-time feature scale acts as an effective per-modality learning rate — and
                masked-accuracy dominance can mislead.
              </p>
            </BlurFade>
            <BlurFade delay={0.22} inView>
              <p className="mt-4 max-w-md text-[0.95rem] leading-relaxed text-muted">
                A controlled study that splits a feature-scale intervention into its forward-magnitude
                and gradient pathways, then tests whether a shift in masked-accuracy dominance reflects
                a genuine change in modality contribution — or an artifact of the measurement.
              </p>
            </BlurFade>
            <BlurFade delay={0.28} inView>
              <div className="mt-8 flex flex-wrap gap-3">
                <AnimatedBorder>
                  <Link to="/disentanglement" className="btn-primary rounded-[5px]">
                    Explore experiments
                    <ArrowRight size={16} />
                  </Link>
                </AnimatedBorder>
                <Link to="/paper" className="btn-ghost">
                  <FileText size={16} />
                  Read the paper
                </Link>
              </div>
            </BlurFade>
          </div>

          {/* Diagram — solid background so it's always legible in both modes */}
          <BlurFade delay={0.2} inView>
            <div
              className="rounded-lg border border-line p-6 sm:p-8"
              style={{ background: theme === "dark" ? "#141b22" : "var(--surface)" }}
            >
              <PathwayDiagram />
            </div>
          </BlurFade>
        </div>
      </section>

      {/* Headline metrics */}
      <section className="mt-16 grid grid-cols-2 gap-px overflow-hidden rounded-md border border-line bg-line sm:grid-cols-4">
        {[
          { value: headline.conditions, label: "Experimental conditions" },
          { value: headline.datasets, label: "Datasets" },
          { value: headline.seeds, label: "Seeds per condition" },
          { value: headline.fusionMechanisms, label: "Fusion mechanisms" },
        ].map((m, i) => (
          <BlurFade key={m.label} delay={0.05 * i} inView>
            <div className="bg-surface p-6 h-full transition-colors duration-200 hover:bg-black/[0.015] dark:hover:bg-white/[0.02]">
              <MetricBlock value={m.value} label={m.label} size="lg" />
            </div>
          </BlurFade>
        ))}
      </section>

      {/* The story in one line */}
      <section className="mt-14">
        <BlurFade delay={0} inView>
          <div className="label mb-4">The story in one line</div>
        </BlurFade>
        <BlurFade delay={0.1} inView>
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
                <MorphingPopover
                  trigger={
                    <span className="cursor-default underline decoration-muted/30 underline-offset-2 transition-colors hover:text-ink">
                      {step}
                    </span>
                  }
                >
                  {STORY_TIPS[step]}
                </MorphingPopover>
                {i < arr.length - 1 && <ArrowRight size={14} className="text-muted/60" />}
              </span>
            ))}
          </div>
        </BlurFade>
      </section>
    </div>
  );
}
