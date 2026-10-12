import { useState, useEffect, useRef } from "react";
import { ArrowDown } from "lucide-react";
import { useResearch } from "../state/research-context";
import { SectionHeader, Card, Callout, Tag } from "../components/primitives";
import { PathwayDiagram, type Emphasis } from "../components/PathwayDiagram";
import { PathwayBars, type BarRow } from "../components/PathwayBars";
import { AnimatedNumber } from "../components/AnimatedNumber";
import { ScientificFigure } from "../components/ScientificFigure";
import { decomposition } from "../data/research";
import { fixed, pValue } from "../lib/format";
import type { OptimizerId } from "../data/research";

/* ── Constants ───────────────────────────────────────────────────────── */

const MAX_BAR_D = 8; // common scale — covers Adam |d|≈3.76, SGD |d|≈7.75

const PATHWAY_TIPS: Record<"forward" | "gradient", { title: string; body: string }> = {
  forward: {
    title: "Forward pathway",
    body:
      "Feature scaling directly changes the forward representation magnitude entering fusion. Under the tested Adam setting, this pathway carries almost the entire ordinary effect.",
  },
  gradient: {
    title: "Gradient pathway",
    body:
      "The intervention can also alter the backward signal and resulting parameter updates. Under Adam, a 280×+ gradient-norm ratio still leaves the update ratio ≈ 1 in this setting.",
  },
};

/* ── FlowChain (282.7× → 1.61× visual) ──────────────────────────────── */

type FlowStyle = "grad" | "dark" | "fwd" | "muted";

interface FlowStep {
  value: string;
  top: string;
  sub: string;
  style: FlowStyle;
}

function FlowChain({
  opt,
  ratios,
  gradDDom,
}: {
  opt: OptimizerId;
  ratios: { rG: number; rTheta: number };
  gradDDom: number;
}) {
  const isAdam = opt === "adam";
  const gradDomStr =
    Math.abs(gradDDom) < 0.001 ? fixed(gradDDom, 4) : fixed(gradDDom, 3);

  const steps: FlowStep[] = [
    {
      value: `≈ ${ratios.rG}×`,
      top: "Gradient-norm ratio r_g",
      sub: "‖g_A‖ / ‖g_B‖",
      style: "grad",
    },
    {
      value: "",
      top: isAdam ? "Adam — adaptive update" : "SGD — momentum",
      sub: isAdam ? "per-parameter normalization" : "momentum-based update",
      style: "dark",
    },
    {
      value: `≈ ${ratios.rTheta}×`,
      top: "Parameter-update ratio r_Δθ",
      sub: "‖Δθ_A‖ / ‖Δθ_B‖",
      style: (isAdam ? "fwd" : "grad") as FlowStyle,
    },
    {
      value: gradDomStr,
      top: "Gradient-path contribution Δ𝒟",
      sub: isAdam ? "≈ zero — not individually significant" : "measurable shift",
      style: "muted",
    },
  ];

  const cls: Record<FlowStyle, string> = {
    grad: "border-gradient/40 bg-gradient/[0.07] text-gradient",
    dark: "border-ink bg-ink text-canvas",
    fwd: "border-forward/40 bg-forward/[0.07] text-forward",
    muted: "border-line bg-canvas/50 text-muted",
  };

  return (
    <div>
      <div className="label mb-3">From gradient magnitude to parameter update</div>
      <Card className="p-5">
        <div className="mx-auto max-w-[260px]">
          {steps.map((step, i) => (
            <div key={i}>
              <div
                className={`rounded border px-3 py-2.5 text-center transition-colors duration-300 ${cls[step.style]}`}
              >
                {step.value && (
                  <div
                    className={`font-mono text-xl font-semibold tabular-nums leading-snug ${
                      step.style === "dark" ? "text-canvas" : ""
                    }`}
                  >
                    {step.value}
                  </div>
                )}
                <div
                  className={`font-mono text-[0.62rem] uppercase tracking-wider leading-tight ${
                    step.style === "dark" ? "text-canvas/60" : "opacity-60"
                  }`}
                >
                  {step.top}
                </div>
                {step.sub && (
                  <div
                    className={`mt-0.5 font-mono text-[0.6rem] leading-tight ${
                      step.style === "dark" ? "text-canvas/40" : "opacity-40"
                    }`}
                  >
                    {step.sub}
                  </div>
                )}
              </div>
              {i < steps.length - 1 && (
                <div className="flex justify-center py-1">
                  <ArrowDown size={12} className="text-muted/40" />
                </div>
              )}
            </div>
          ))}
        </div>
        <p className="mt-4 border-t border-line pt-3 text-2xs leading-relaxed text-muted">
          {isAdam
            ? `A ≈${ratios.rG}× gradient-norm ratio under Adam leaves the parameter-update ratio ≈${ratios.rTheta}×, consistent with Adam normalizing per-parameter gradient magnitudes. The resulting gradient-path contribution is approximately ${gradDomStr}.`
            : `Under SGD, a ≈${ratios.rG}× gradient-norm ratio translates into a ≈${ratios.rTheta}× parameter-update ratio — the gradient channel measurably shifts contribution, unlike the Adam case.`}
        </p>
      </Card>
    </div>
  );
}

/* ── ArmCard ─────────────────────────────────────────────────────────── */

interface ArmCardProps {
  label: string;
  tone: "forward" | "gradient" | "neutral";
  dDom: number;
  d: number;
  p: number;
  opt: OptimizerId;
  highlighted: boolean;
  faded: boolean;
}

function ArmCard({ label, tone, dDom, d, p, opt, highlighted, faded }: ArmCardProps) {
  const textColor: Record<typeof tone, string> = {
    forward: "text-forward",
    gradient: "text-gradient",
    neutral: "text-ink",
  };

  return (
    <div
      className={`rounded-md border p-4 transition-all duration-300 ${
        highlighted
          ? "border-ink/25 bg-surface shadow-card"
          : faded
          ? "border-line bg-surface opacity-60"
          : "border-line bg-surface"
      }`}
    >
      <div className="mb-3 flex items-center justify-between gap-2">
        <span className="text-sm font-medium text-ink">{label}</span>
        <Tag tone={tone === "neutral" ? "neutral" : tone}>
          {tone === "forward" ? "fwd" : tone === "gradient" ? "grad" : "both"}
        </Tag>
      </div>

      <AnimatedNumber
        key={`${opt}-${tone}-dom`}
        value={dDom}
        digits={Math.abs(dDom) < 0.001 ? 4 : 3}
        signed
        className={`text-[1.75rem] font-semibold leading-none ${textColor[tone]}`}
      />
      <div className="mt-0.5 font-mono text-[0.6rem] uppercase tracking-wider text-muted/60">
        Δ𝒟 dominance shift
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-0.5 border-t border-line pt-2.5 font-mono text-2xs text-muted">
        <span>
          d_z{" "}
          <AnimatedNumber
            key={`${opt}-${tone}-d`}
            value={d}
            digits={2}
            signed
            className="text-ink"
          />
        </span>
        <span>p {pValue(p)}</span>
      </div>
    </div>
  );
}

/* ── Main page ───────────────────────────────────────────────────────── */

export function Disentanglement() {
  const s = useResearch();
  const opt = s.optimizer as OptimizerId;
  const armSectionRef = useRef<HTMLElement>(null);

  const [hoveredPath, setHoveredPath] = useState<"forward" | "gradient" | null>(null);

  // Reset hover state whenever optimizer changes
  useEffect(() => {
    setHoveredPath(null);
  }, [opt]);

  const activeSet =
    opt === "adam" ? decomposition.adamSynthetic : decomposition.sgdSynthetic;
  const ratios =
    opt === "adam" ? decomposition.ratios.adam : decomposition.ratios.sgd;
  const isAdam = opt === "adam";

  // Diagram emphasis: hover overrides the optimizer-driven default
  const defaultEmphasis: Emphasis = isAdam ? "forward" : "both";
  const diagramEmphasis: Emphasis = hoveredPath ?? defaultEmphasis;

  const bars: BarRow[] = [
    {
      label: "Forward pathway",
      magnitude: Math.abs(activeSet.forward.d),
      value: `d_z ${fixed(activeSet.forward.d, 2)}`,
      color: "forward",
    },
    {
      label: "Gradient pathway",
      magnitude: Math.abs(activeSet.gradient.d),
      value: `d_z ${fixed(activeSet.gradient.d, 2)}`,
      color: "gradient",
      note: activeSet.gradient.p >= 0.05 ? "n.s." : undefined,
    },
  ];

  const setOpt = (o: OptimizerId) =>
    s.setScenario({ dataset: "synthetic", optimizer: o });

  return (
    <div className="space-y-12">

      {/* ── SECTION 1 — HERO ─────────────────────────────────────────── */}
      <section>
        <div className="flex flex-wrap items-start gap-y-6">
          <div className="flex-1 min-w-[260px]">
            <SectionHeader
              eyebrow="Disentanglement"
              title="Where does the intervention travel?"
              subtitle="A controlled decomposition of the feature-scale intervention into forward representation magnitude and gradient/update pathways."
            />
          </div>

          {/* Optimizer toggle + regime */}
          <div className="flex shrink-0 flex-col items-end gap-3">
            <div>
              <div className="label mb-1.5 text-right">Optimizer</div>
              <div
                className="inline-flex rounded border border-line bg-surface p-0.5"
                role="group"
                aria-label="Select optimizer"
              >
                {(["adam", "sgd"] as OptimizerId[]).map((o) => (
                  <button
                    key={o}
                    onClick={() => setOpt(o)}
                    data-active={opt === o}
                    aria-pressed={opt === o}
                    className="seg data-[active=true]:bg-ink data-[active=true]:text-canvas"
                  >
                    {o === "adam" ? "Adam" : "SGD"}
                  </button>
                ))}
              </div>
            </div>

            <div className="text-right">
              <div className="label">Pathway regime</div>
              <div
                className={`font-display text-base font-semibold transition-colors duration-300 ${
                  isAdam ? "text-forward" : "text-ink"
                }`}
              >
                {isAdam ? "Forward-dominant" : "Both pathways contribute"}
              </div>
              <div className="font-mono text-2xs text-muted">
                Under tested {isAdam ? "Adam" : "SGD"} configuration
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── SECTION 2 — PATHWAY MAP + PATHWAY STRENGTH ───────────────── */}
      <section className="grid gap-8 lg:grid-cols-[3fr_2fr]">

        {/* Pathway diagram */}
        <ScientificFigure
          index="Fig. 1"
          caption="Conceptual intervention-pathway decomposition; not an identified causal graph."
          source={`Synthetic · α=100× · ${isAdam ? "Adam" : "SGD"} · 5 seeds`}
        >
          {/*
            Outer div owns hover clearing — only fires when mouse leaves the
            entire figure area (SVG + description). Node mouseLeave no longer
            clears hoveredPath, so the mouse can travel to the tooltip or
            description div without the state resetting.
          */}
          <div onMouseLeave={() => setHoveredPath(null)}>
            <PathwayDiagram
              emphasis={diagramEmphasis}
              alpha="100"
              animated={true}
              onHover={(e) => {
                if (e === "forward" || e === "gradient") setHoveredPath(e);
              }}
              onSelect={() => {
                armSectionRef.current?.scrollIntoView({
                  behavior: "smooth",
                  block: "start",
                });
              }}
            />

            {/* Hover description below SVG */}
            <div
              aria-live="polite"
              className={`mt-3 rounded border px-3 py-2 text-2xs leading-relaxed transition-colors duration-200 ${
                hoveredPath === "forward"
                  ? "border-forward/30 bg-forward/[0.05] text-forward"
                  : hoveredPath === "gradient"
                  ? "border-gradient/30 bg-gradient/[0.05] text-gradient"
                  : "invisible border-transparent"
              }`}
            >
              {hoveredPath ? (
                <>
                  <span className="font-medium">
                    {PATHWAY_TIPS[hoveredPath].title}:
                  </span>{" "}
                  {PATHWAY_TIPS[hoveredPath].body}
                </>
              ) : (
                <span aria-hidden className="select-none">
                  {PATHWAY_TIPS.forward.title}: {PATHWAY_TIPS.forward.body}
                </span>
              )}
            </div>
          </div>
        </ScientificFigure>

        {/* Pathway bars + Gradient → Update chain */}
        <div className="space-y-6">
          <div>
            <div className="label mb-1">Pathway response</div>
            <p className="mb-3 font-mono text-2xs text-muted">
              Synthetic · α=100× · {isAdam ? "Adam" : "SGD"} · n=5 seeds
            </p>
            <Card className="p-5">
              <PathwayBars
                rows={bars}
                max={MAX_BAR_D}
                unit="Observed effect size · Cohen's d_z"
              />
            </Card>
          </div>

          <FlowChain
            opt={opt}
            ratios={ratios}
            gradDDom={activeSet.gradient.dDom}
          />
        </div>
      </section>

      {/* ── SECTION 3 — OBSERVED EFFECT BY ARM ───────────────────────── */}
      <section ref={armSectionRef}>
        <div className="mb-4 flex flex-wrap items-baseline justify-between gap-3">
          <div className="label">Observed effect by arm</div>
          <span className="font-mono text-2xs text-muted">
            Synthetic · α=100× · {isAdam ? "Adam" : "SGD"} · n=5 seeds
          </span>
        </div>

        <div className="grid gap-4 sm:grid-cols-3">
          <ArmCard
            label="Forward-only"
            tone="forward"
            dDom={activeSet.forward.dDom}
            d={activeSet.forward.d}
            p={activeSet.forward.p}
            opt={opt}
            highlighted={true}
            faded={false}
          />
          <ArmCard
            label="Gradient-only"
            tone="gradient"
            dDom={activeSet.gradient.dDom}
            d={activeSet.gradient.d}
            p={activeSet.gradient.p}
            opt={opt}
            highlighted={true}
            faded={false}
          />
          <ArmCard
            label="Ordinary (both)"
            tone="neutral"
            dDom={activeSet.ordinary.dDom}
            d={activeSet.ordinary.d}
            p={activeSet.ordinary.p}
            opt={opt}
            highlighted={true}
            faded={false}
          />
        </div>

        {/* Relationship annotation */}
        <div className="mt-4 rounded-md border border-line bg-surface/50 px-4 py-3">
          {isAdam ? (
            <p className="text-2xs leading-relaxed text-muted">
              <span className="font-semibold text-ink">
                Relationship (tested Adam):
              </span>{" "}
              Forward-only (Δ𝒟 = {fixed(activeSet.forward.dDom, 3)}) closely
              tracks Ordinary (Δ𝒟 = {fixed(activeSet.ordinary.dDom, 3)}),
              while Gradient-only (Δ𝒟 ≈{" "}
              {fixed(activeSet.gradient.dDom, 4)}) is not individually
              significant (p = {pValue(activeSet.gradient.p)}).
            </p>
          ) : (
            <p className="text-2xs leading-relaxed text-muted">
              <span className="font-semibold text-ink">
                Relationship (tested SGD):
              </span>{" "}
              Both Forward-only (Δ𝒟 = {fixed(activeSet.forward.dDom, 3)}) and
              Gradient-only (Δ𝒟 = {fixed(activeSet.gradient.dDom, 3)}) produce
              individually significant shifts. The Ordinary total (
              {fixed(activeSet.ordinary.dDom, 3)}) is sub-additive relative to
              their sum (
              {fixed(
                activeSet.forward.dDom + activeSet.gradient.dDom,
                3,
              )}
              ), consistent with pathway interaction.
            </p>
          )}
          <p className="mt-2 font-mono text-[0.6rem] uppercase tracking-wider text-muted/50">
            Effect sizes summarize cross-seed paired comparisons on the fixed
            experimental split
          </p>
        </div>
      </section>

      {/* ── SECTION 4 — KEY INSIGHT ───────────────────────────────────── */}
      <Callout title="Key insight" tone={isAdam ? "forward" : "neutral"}>
        {isAdam ? (
          <>
            Under the tested Adam setting, the intervention is primarily carried
            through the forward pathway: forward-only scaling reproduces almost
            the entire ordinary effect (Δ𝒟 ≈ {fixed(activeSet.forward.dDom, 3)}
            ), while gradient-only scaling produces negligible contribution
            change (Δ𝒟 ≈ {fixed(activeSet.gradient.dDom, 4)}) despite a
            gradient-norm ratio above {ratios.rG}×.{" "}
            <span className="text-[0.9em] text-muted/75">
              Scoped to the tested configuration and fixed experimental protocol.
            </span>
          </>
        ) : (
          <>
            Under the tested SGD setting, both pathways contribute: the
            gradient-only arm produces a dominance shift of{" "}
            {fixed(activeSet.gradient.dDom, 3)} (d_z ={" "}
            {fixed(activeSet.gradient.d, 2)}, p ={" "}
            {pValue(activeSet.gradient.p)}), alongside a forward effect of{" "}
            {fixed(activeSet.forward.dDom, 3)}. With SGD the {ratios.rG}×
            gradient-norm ratio translates into a {ratios.rTheta}× update ratio
            — which pathway carries the intervention is optimizer-dependent.{" "}
            <span className="text-[0.9em] text-muted/75">
              Scoped to the Synthetic dataset and fixed experimental protocol.
            </span>
          </>
        )}
      </Callout>
    </div>
  );
}
