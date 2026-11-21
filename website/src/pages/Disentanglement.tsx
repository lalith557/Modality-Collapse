import { useState, useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import { ArrowDown, ArrowRight } from "lucide-react";
import { SectionHeader, Card, Callout, Tag } from "../components/primitives";
import { PathwayDiagram, type Emphasis } from "../components/PathwayDiagram";
import { PathwayBars, type BarRow } from "../components/PathwayBars";
import { AnimatedNumber } from "../components/AnimatedNumber";
import { ScientificFigure } from "../components/ScientificFigure";
import {
  CremadRatioNote,
  DualChannelDiagram,
  GradientOnlyDetail,
  MeaningCard,
  OptimizerComparison,
  TransformationPanel,
  fmtRatio,
} from "../components/disentanglement/OptimizerSection";
import {
  FusionDetail,
  FusionSelector,
  ForwardMagnitudeChart,
  GradientOnlyTable,
} from "../components/disentanglement/FusionSection";
import { decomposition, fusion, fusionByKey, type FusionId, type OptimizerId } from "../data/research";
import { fixed, pValue } from "../lib/format";
import { usePageContext } from "../state/page-context";
import { useSectionScroll } from "../lib/useSectionScroll";

/* ── Constants ───────────────────────────────────────────────────────── */

const MAX_BAR_D = 8; // common scale — covers Adam |d|≈3.76, SGD |d|≈7.75

function SectionMark({ n, label, id }: { n: string; label: string; id: string }) {
  return (
    <div id={id} className="mb-5 flex scroll-mt-24 items-center gap-3">
      <span className="shrink-0 font-mono text-[0.56rem] uppercase tracking-[0.2em] text-muted/40">{n}</span>
      <span className="h-px w-5 shrink-0 bg-line" />
      <span className="label">{label}</span>
    </div>
  );
}

function OptimizerToggle({ opt, setOpt }: { opt: OptimizerId; setOpt: (o: OptimizerId) => void }) {
  return (
    <div className="inline-flex rounded border border-line bg-surface p-0.5" role="radiogroup" aria-label="Select optimizer">
      {(["adam", "sgd"] as OptimizerId[]).map((o) => (
        <button
          key={o}
          type="button"
          role="radio"
          onClick={() => setOpt(o)}
          data-active={opt === o}
          aria-checked={opt === o}
          className="seg data-[active=true]:bg-ink data-[active=true]:text-canvas"
        >
          {o === "adam" ? "Adam" : "SGD"}
        </button>
      ))}
    </div>
  );
}

function ScopeChip({ children }: { children: string }) {
  return (
    <span className="inline-flex items-center gap-2 rounded border border-line bg-canvas/80 px-3 py-1.5 font-mono text-2xs uppercase tracking-wider text-muted" data-testid="scope-chip">
      <span className="inline-block h-1.5 w-1.5 rounded-full bg-muted/50" />
      {children}
    </span>
  );
}

/* ── FlowChain (gradient ratio → optimizer → update ratio) ──────────── */

type FlowStyle = "grad" | "dark" | "fwd" | "muted";

function FlowChain({ opt, ratios, gradDDom }: { opt: OptimizerId; ratios: { rG: number; rTheta: number }; gradDDom: number }) {
  const isAdam = opt === "adam";
  const gradDomStr = Math.abs(gradDDom) < 0.001 ? fixed(gradDDom, 4) : fixed(gradDDom, 3);
  const steps: { value: string; top: string; sub: string; style: FlowStyle }[] = [
    { value: `≈ ${fmtRatio(ratios.rG)}×`, top: "Gradient-norm ratio r_g", sub: "‖g_A‖ / ‖g_B‖", style: "grad" },
    { value: "", top: isAdam ? "Adam — adaptive update" : "SGD — momentum", sub: isAdam ? "per-parameter normalization" : "step ∝ gradient", style: "dark" },
    { value: `≈ ${fmtRatio(ratios.rTheta)}×`, top: "Parameter-update ratio r_Δθ", sub: "‖Δθ_A‖ / ‖Δθ_B‖", style: isAdam ? "fwd" : "grad" },
    { value: gradDomStr, top: "Gradient-only Δ𝒟", sub: isAdam ? "≈ zero — not individually significant" : "significant shift", style: "muted" },
  ];
  const cls: Record<FlowStyle, string> = {
    grad: "border-gradient/40 bg-gradient/[0.07] text-gradient",
    dark: "border-ink bg-ink text-canvas",
    fwd: "border-forward/40 bg-forward/[0.07] text-forward",
    muted: "border-line bg-canvas/50 text-muted",
  };
  return (
    <div data-testid="flow-chain">
      <div className="label mb-3">From gradient magnitude to parameter update</div>
      <Card className="p-5">
        <div className="mx-auto max-w-[260px]">
          {steps.map((step, i) => (
            <div key={i}>
              <div className={`rounded border px-3 py-2.5 text-center transition-colors duration-300 ${cls[step.style]}`}>
                {step.value && (
                  <div className={`font-mono text-xl font-semibold tabular-nums leading-snug ${step.style === "dark" ? "text-canvas" : ""}`}>{step.value}</div>
                )}
                <div className={`font-mono text-[0.62rem] uppercase leading-tight tracking-wider ${step.style === "dark" ? "text-canvas/60" : "opacity-60"}`}>
                  {step.top}
                </div>
                <div className={`mt-0.5 font-mono text-[0.6rem] leading-tight ${step.style === "dark" ? "text-canvas/40" : "opacity-40"}`}>{step.sub}</div>
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
            ? `Under the tested Adam configuration a ≈${fmtRatio(ratios.rG)}× gradient-norm ratio leaves the parameter-update ratio ≈${fmtRatio(ratios.rTheta)}×, consistent with Adam normalizing per-parameter gradient magnitudes; the gradient-only shift is ${gradDomStr}.`
            : `Under the tested SGD configuration a ≈${fmtRatio(ratios.rG)}× gradient-norm ratio becomes a ≈${fmtRatio(ratios.rTheta)}× parameter-update ratio — the gradient channel measurably shifts contribution, unlike the Adam case.`}
        </p>
      </Card>
    </div>
  );
}

/* ── ArmCard ─────────────────────────────────────────────────────────── */

function ArmCard({ label, tone, dDom, d, p, opt }: { label: string; tone: "forward" | "gradient" | "neutral"; dDom: number; d: number; p: number; opt: OptimizerId }) {
  const textColor = { forward: "text-forward", gradient: "text-gradient", neutral: "text-ink" }[tone];
  return (
    <div className="rounded-md border border-ink/25 bg-surface p-4 shadow-card transition-all duration-300" data-testid={`arm-${tone}`}>
      <div className="mb-3 flex items-center justify-between gap-2">
        <span className="text-sm font-medium text-ink">{label}</span>
        <Tag tone={tone === "neutral" ? "neutral" : tone}>{tone === "forward" ? "fwd" : tone === "gradient" ? "grad" : "both"}</Tag>
      </div>
      <AnimatedNumber key={`${opt}-${tone}-dom`} value={dDom} digits={Math.abs(dDom) < 0.001 ? 4 : 3} signed className={`text-[1.75rem] font-semibold leading-none ${textColor}`} />
      <div className="mt-0.5 font-mono text-[0.6rem] uppercase tracking-wider text-muted/60">Δ𝒟 dominance shift</div>
      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-0.5 border-t border-line pt-2.5 font-mono text-2xs text-muted">
        <span>
          d_z <AnimatedNumber key={`${opt}-${tone}-d`} value={d} digits={2} signed className="text-ink" />
        </span>
        <span data-testid={`arm-${tone}-p`}>p {pValue(p)}</span>
      </div>
    </div>
  );
}

/* ── Main page ───────────────────────────────────────────────────────── */

export function Disentanglement() {
  useSectionScroll();
  const [opt, setOpt] = useState<OptimizerId>("adam");
  const [fusionKey, setFusionKey] = useState<FusionId>("sum");
  const armSectionRef = useRef<HTMLElement>(null);
  const [hoveredPath, setHoveredPath] = useState<"forward" | "gradient" | null>(null);

  useEffect(() => setHoveredPath(null), [opt]);

  const isAdam = opt === "adam";
  const activeSet = isAdam ? decomposition.adamSynthetic : decomposition.sgdSynthetic;
  const ratios = isAdam ? decomposition.ratios.adam : decomposition.ratios.sgd;
  const mech = fusionByKey(fusionKey);

  usePageContext({
    title: "Analysis context",
    chips: [
      { k: "Dataset", v: "Synthetic" },
      { k: "α", v: "100×" },
      { k: "Seeds", v: "5" },
      { k: "Pathway sections", v: isAdam ? "Adam" : "SGD", tone: isAdam ? "forward" : "gradient" },
      { k: "Fusion section", v: `Adam · ${mech.name}` },
    ],
    note: "Frozen experimental results",
  });

  const tips: Record<"forward" | "gradient", { title: string; body: string }> = {
    forward: {
      title: "Forward pathway",
      body: "Feature scaling changes the magnitude of the representation entering fusion. Under the tested Adam setting, this pathway carries almost the entire ordinary effect.",
    },
    gradient: {
      title: "Gradient pathway",
      body: `The intervention can also multiply the encoder's gradient and so its parameter updates. Under the tested Adam setting a ≈${fmtRatio(decomposition.ratios.adam.rG)}× gradient-norm ratio still leaves the update ratio ≈${fmtRatio(decomposition.ratios.adam.rTheta)}×.`,
    },
  };

  const bars: BarRow[] = [
    { label: "Forward pathway", magnitude: Math.abs(activeSet.forward.d), value: `d_z ${fixed(activeSet.forward.d, 2)}`, color: "forward" },
    {
      label: "Gradient pathway",
      magnitude: Math.abs(activeSet.gradient.d),
      value: `d_z ${fixed(activeSet.gradient.d, 2)}`,
      color: "gradient",
      note: activeSet.gradient.p >= 0.05 ? "n.s." : undefined,
    },
  ];
  const defaultEmphasis: Emphasis = isAdam ? "forward" : "both";
  const scope = `Synthetic · α=100× · ${isAdam ? "Adam" : "SGD"} · 5 seeds`;

  return (
    <div className="space-y-14">
      {/* ── 01 INTERVENTION OVERVIEW ───────────────────────────────── */}
      <section>
        <SectionHeader
          eyebrow="Disentanglement"
          title="Where does the intervention travel?"
          subtitle="One page for the whole mechanism: the feature-scale intervention, its forward and gradient pathways, how the optimizer decides whether the gradient pathway matters, and whether the fusion mechanism changes the picture."
        />
        <div className="mt-8">
          <SectionMark n="01" label="Intervention overview" id="overview" />
          <div className="grid gap-8 lg:grid-cols-[3fr_2fr]">
            <ScientificFigure index="Fig. 1" caption="Conceptual intervention-pathway decomposition; not an identified causal graph and not the model's computational graph." source={scope}>
              <div onMouseLeave={() => setHoveredPath(null)}>
                <PathwayDiagram
                  emphasis={hoveredPath ?? defaultEmphasis}
                  alpha="100"
                  animated
                  onHover={(e) => {
                    if (e === "forward" || e === "gradient") setHoveredPath(e);
                  }}
                  onSelect={() => armSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })}
                />
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
                      <span className="font-medium">{tips[hoveredPath].title}:</span> {tips[hoveredPath].body}
                    </>
                  ) : (
                    <span aria-hidden className="select-none">
                      {tips.forward.title}: {tips.forward.body}
                    </span>
                  )}
                </div>
              </div>
            </ScientificFigure>
            <div className="space-y-4">
              <p className="text-[0.95rem] leading-relaxed text-ink">
                The intervention multiplies one modality's representation by a constant α just before fusion. Custom autograd
                operators split it into three arms that leave the training objective unchanged:
              </p>
              <dl className="space-y-2 text-2xs" data-testid="arm-definitions">
                {[
                  { k: "Forward-only", v: "value α·h; gradient passed through unchanged", c: "text-forward" },
                  { k: "Gradient-only", v: "value h unchanged; encoder gradient multiplied by α", c: "text-gradient" },
                  { k: "Ordinary", v: "value α·h and gradient ×α (plain multiplication)", c: "text-ink" },
                ].map((r) => (
                  <div key={r.k} className="grid grid-cols-[110px_1fr] gap-2 rounded border border-line bg-surface px-3 py-2">
                    <dt className={`font-mono font-semibold ${r.c}`}>{r.k}</dt>
                    <dd className="text-muted">{r.v}</dd>
                  </div>
                ))}
              </dl>
              <p className="text-2xs leading-relaxed text-muted">
                The outcome is the modality contribution, measured as masked-accuracy dominance Δ𝒟 (shift versus baseline). Real-data
                checks with probes, Shapley and mean-ablation are on{" "}
                <Link to="/real-world" className="text-forward hover:underline">Real-World Evidence</Link>.
              </p>
              <Link to="/architecture" className="inline-flex items-center gap-1 text-2xs font-medium text-forward hover:underline">
                See where α sits in the real CREMA-D and UTD-MHAD models <ArrowRight size={12} />
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* ── 02 FORWARD VS GRADIENT ─────────────────────────────────── */}
      <section ref={armSectionRef}>
        <SectionMark n="02" label="Forward versus gradient disentanglement" id="forward-vs-gradient" />
        <div className="mb-5 flex flex-wrap items-end justify-between gap-4 rounded-md border border-line bg-surface/60 px-4 py-3">
          <div>
            <div className="label mb-1.5">Optimizer (drives sections 02–04)</div>
            <OptimizerToggle opt={opt} setOpt={setOpt} />
          </div>
          <div className="text-right">
            <div className="label">Pathway regime</div>
            <div className={`font-display text-base font-semibold transition-colors duration-300 ${isAdam ? "text-forward" : "text-ink"}`} data-testid="regime">
              {isAdam ? "Forward-dominant" : "Both pathways contribute"}
            </div>
            <div className="font-mono text-2xs text-muted">Under the tested {isAdam ? "Adam" : "SGD"} configuration</div>
          </div>
        </div>

        <div className="mb-4 flex flex-wrap items-baseline justify-between gap-3">
          <div className="label">Observed effect by arm</div>
          <ScopeChip>{scope}</ScopeChip>
        </div>
        <div className="grid gap-4 sm:grid-cols-3">
          <ArmCard label="Forward-only" tone="forward" {...activeSet.forward} opt={opt} />
          <ArmCard label="Gradient-only" tone="gradient" {...activeSet.gradient} opt={opt} />
          <ArmCard label="Ordinary (both)" tone="neutral" {...activeSet.ordinary} opt={opt} />
        </div>

        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          <div>
            <div className="label mb-3">Pathway response · Cohen's d_z</div>
            <Card className="p-5">
              <PathwayBars rows={bars} max={MAX_BAR_D} unit="Observed effect size · Cohen's d_z" />
            </Card>
          </div>
          <div className="rounded-md border border-line bg-surface/50 px-4 py-3">
            <div className="label mb-2">Pathway comparison</div>
            {isAdam ? (
              <p className="text-2xs leading-relaxed text-muted">
                <span className="font-semibold text-ink">Tested Adam:</span> forward-only (Δ𝒟 = {fixed(activeSet.forward.dDom, 3)}) closely
                tracks ordinary (Δ𝒟 = {fixed(activeSet.ordinary.dDom, 3)}), while gradient-only (Δ𝒟 ≈ {fixed(activeSet.gradient.dDom, 4)}) is not
                individually significant (p = {pValue(activeSet.gradient.p)}).
              </p>
            ) : (
              <p className="text-2xs leading-relaxed text-muted">
                <span className="font-semibold text-ink">Tested SGD:</span> forward-only (Δ𝒟 = {fixed(activeSet.forward.dDom, 3)}) and gradient-only
                (Δ𝒟 = {fixed(activeSet.gradient.dDom, 3)}) are both individually significant. Ordinary ({fixed(activeSet.ordinary.dDom, 3)}) exceeds
                each arm but is below their sum ({fixed(activeSet.forward.dDom + activeSet.gradient.dDom, 3)}); ordinary vs forward d_z ={" "}
                {decomposition.ordVsFwd.synthSgd.d.toFixed(2)}, p = {pValue(decomposition.ordVsFwd.synthSgd.p)}.
              </p>
            )}
            <p className="mt-3 border-t border-line pt-2 font-mono text-[0.6rem] uppercase tracking-wider text-muted/60">
              Paired t-tests across 5 seeds on the fixed split · within-split reproducibility
            </p>
            <div className="mt-3 text-2xs text-muted">
              Same decomposition on real data (CREMA-D, Adam): forward Δ𝒟 = {fixed(decomposition.adamCremad.forward.dDom, 3)} (d_z{" "}
              {decomposition.adamCremad.forward.d.toFixed(2)}), gradient-only Δ𝒟 = {fixed(decomposition.adamCremad.gradient.dDom, 3)} (p ={" "}
              {pValue(decomposition.adamCremad.gradient.p)}).{" "}
              <Link to="/real-world" className="text-forward hover:underline">Real-World Evidence →</Link>
            </div>
          </div>
        </div>
      </section>

      {/* ── 03 ADAM VS SGD ─────────────────────────────────────────── */}
      <section>
        <SectionMark n="03" label="Adam versus SGD" id="adam-vs-sgd" />
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <p className="max-w-2xl text-[0.95rem] leading-relaxed text-muted">
            The gradient-only arm isolates the backward pathway. The optimizer then decides how much of the imposed gradient-scale
            difference reaches the parameter update.
          </p>
          <div className="flex items-center gap-2">
            <span className="font-mono text-2xs text-muted/70">Optimizer</span>
            <OptimizerToggle opt={opt} setOpt={setOpt} />
          </div>
        </div>
        <div className="grid gap-8 lg:grid-cols-[3fr_2fr]">
          <ScientificFigure
            index="Fig. 2"
            caption="The same gradient-only intervention meets two update rules. Dashed thin connector (Adam): attenuation; solid thick connector (SGD): pass-through. Line weight is illustrative, not proportional."
            source="Synthetic · α=100× · gradient-only · 5 seeds · frozen experimental results"
          >
            <div className="mx-auto max-w-xl" data-testid="dual-channel">
              <DualChannelDiagram focused={opt} />
            </div>
          </ScientificFigure>
          <FlowChain opt={opt} ratios={ratios} gradDDom={activeSet.gradient.dDom} />
        </div>
        <div className="mt-6 grid gap-5 lg:grid-cols-2">
          <TransformationPanel
            name="Adam"
            kind="Adaptive optimization"
            tone="adam"
            rG={decomposition.ratios.adam.rG}
            rTheta={decomposition.ratios.adam.rTheta}
            equation="m̂, v̂ → Δθ"
            equationNote="adaptive moment normalization"
            focused={isAdam}
            onClick={() => setOpt("adam")}
            note={`Under the tested Adam configuration the ${fmtRatio(decomposition.ratios.adam.rG)}× gradient-norm ratio is strongly attenuated, yielding a ${fmtRatio(decomposition.ratios.adam.rTheta)}× parameter-update ratio.`}
          />
          <TransformationPanel
            name="SGD"
            kind="Non-adaptive optimization"
            tone="sgd"
            rG={decomposition.ratios.sgd.rG}
            rTheta={decomposition.ratios.sgd.rTheta}
            equation="Δθ ∝ −ηg"
            equationNote="gradient magnitude directly enters update"
            focused={!isAdam}
            onClick={() => setOpt("sgd")}
            note={`Under the tested SGD configuration the ${fmtRatio(decomposition.ratios.sgd.rG)}× gradient-norm ratio largely carries through to a ${fmtRatio(decomposition.ratios.sgd.rTheta)}× parameter-update ratio.`}
          />
        </div>
        <div className="mt-5">
          <CremadRatioNote />
        </div>
      </section>

      {/* ── 04 OPTIMIZER COMPARISON ────────────────────────────────── */}
      <section>
        <SectionMark n="04" label="Optimizer comparison" id="optimizer-comparison" />
        <div className="grid gap-6 lg:grid-cols-[3fr_2fr]">
          <OptimizerComparison focused={opt} onFocus={setOpt} />
          <MeaningCard opt={opt} />
        </div>
        <div className="mt-5">
          <GradientOnlyDetail />
        </div>
        <div className="mt-5">
          <Callout title="Key insight" tone={isAdam ? "forward" : "neutral"}>
            {isAdam ? (
              <>
                Under the tested Adam setting the intervention is carried by the forward pathway: forward-only scaling reproduces almost the
                entire ordinary effect (Δ𝒟 ≈ {fixed(activeSet.forward.dDom, 3)}), while gradient-only scaling produces a negligible change
                (Δ𝒟 ≈ {fixed(activeSet.gradient.dDom, 4)}) despite a ≈{fmtRatio(ratios.rG)}× gradient-norm ratio.{" "}
                <span className="text-[0.9em] text-muted/75">Scoped to the tested configuration; not a claim about all Adam variants or schedules.</span>
              </>
            ) : (
              <>
                Under the tested SGD setting both pathways contribute: gradient-only gives Δ𝒟 = {fixed(activeSet.gradient.dDom, 3)} (d_z ={" "}
                {fixed(activeSet.gradient.d, 2)}, p = {pValue(activeSet.gradient.p)}) alongside a forward effect of {fixed(activeSet.forward.dDom, 3)}, because
                the ≈{fmtRatio(ratios.rG)}× gradient ratio becomes a ≈{fmtRatio(ratios.rTheta)}× update ratio. Which pathway carries the intervention is
                optimizer-dependent.{" "}
                <span className="text-[0.9em] text-muted/75">Scoped to the synthetic data and the tested SGD configuration.</span>
              </>
            )}
          </Callout>
        </div>
      </section>

      {/* ── 05 FUSION ──────────────────────────────────────────────── */}
      <section>
        <SectionMark n="05" label="Does fusion change the pathway?" id="fusion" />
        <div className="mb-5 space-y-3">
          <p className="max-w-2xl text-[0.95rem] leading-relaxed text-muted">
            The forward-only and gradient-only interventions are repeated under sum, concatenation and gated fusion. Optimizer, α,
            data and seeds are fixed; only the fusion mechanism varies.
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <ScopeChip>{fusion.scope}</ScopeChip>
            <span className="font-mono text-2xs text-muted">This section always uses Adam, independent of the optimizer selector above.</span>
          </div>
        </div>
        <FusionSelector active={fusionKey} onChange={setFusionKey} />
        <div className="mt-6">
          <div className="mb-4 flex items-center justify-between">
            <div className="label">Selected mechanism — {mech.name}</div>
            {mech.significant ? <Tag tone="forward">Significant · strong effect</Tag> : <Tag tone="warning">Not individually significant</Tag>}
          </div>
          <FusionDetail mechanism={mech} />
        </div>
        <div className="mt-6 grid gap-6 lg:grid-cols-[3fr_2fr]">
          <ScientificFigure
            index="Fig. 3"
            caption="Forward-only effect magnitude |d_z| per fusion mechanism (click a row to select it). Bar length encodes |d_z|."
            source={fusion.scope}
          >
            <ForwardMagnitudeChart active={fusionKey} onSelect={setFusionKey} />
          </ScientificFigure>
          <div>
            <div className="label mb-1">Gradient-only intervention</div>
            <p className="mb-3 text-2xs text-muted">Approximately null across all three mechanisms</p>
            <GradientOnlyTable active={fusionKey} />
          </div>
        </div>
        <div className="mt-6">
          <Callout title="What this test establishes" tone="forward">
            The gradient-only intervention stays approximately null under every tested fusion mechanism. The forward effect is strong under
            sum and concatenation (d_z = {fusionByKey("sum").forwardD.toFixed(2)} and {fusionByKey("concat").forwardD.toFixed(2)}) and attenuated, not
            individually significant, under gated fusion (d_z = {fusionByKey("gated").forwardD.toFixed(2)}, p = {pValue(fusionByKey("gated").p)}). Gate
            values were not logged, so learned-gate compensation is a possibility, not a finding.{" "}
            <span className="text-muted">Synthetic data, tested model configuration, Adam, α = 100, five seeds.</span>
          </Callout>
        </div>
      </section>
    </div>
  );
}
