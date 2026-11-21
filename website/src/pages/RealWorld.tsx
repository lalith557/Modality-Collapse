import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useSectionScroll } from "../lib/useSectionScroll";
import { SectionHeader, Callout, Tag, Card } from "../components/primitives";
import { ScientificFigure } from "../components/ScientificFigure";
import { Tabs, TabPanel } from "../components/Tabs";
import { CremadScaleChart } from "../charts/CremadScaleChart";
import { UtdAccuracyChart } from "../charts/UtdChart";
import { AnimatedNumber } from "../components/AnimatedNumber";
import { cremad, utd, decomposition, utdConditions, decoupled, sgdStable, baselines } from "../data/research";
import { fixed, pValue, signed } from "../lib/format";

const TABS = [
  { id: "cremad", label: "CREMA-D (Audio-Video)" },
  { id: "utd", label: "UTD-MHAD (Depth-Inertial)" },
];

type Modality = "audio" | "video";
type Scale = 1 | 5 | 10;

function getRow(modality: Modality, scale: Scale) {
  if (scale === 1) return cremad.scale[0];
  return cremad.scale.find((r) => r.condition === `${modality} ×${scale}`) ?? cremad.scale[0];
}

/* ─── CREMA-D Panel ─────────────────────────────────────────────── */
function CremadPanel() {
  const [modality, setModality] = useState<Modality>("audio");
  const [scale, setScale] = useState<Scale>(10);
  const row = getRow(modality, scale);
  const activeName = scale === 1 ? "baseline" : `${modality} ×${scale}`;
  const fwd = decomposition.adamCremad.forward;
  const grad = decomposition.adamCremad.gradient;

  return (
    <TabPanel className="space-y-6">
      {/* Panel header */}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="label mb-1.5">CREMA-D · Audio-Video</div>
          <p className="max-w-xl text-sm leading-relaxed text-muted">
            Scale sweep under Adam (weight decay 0) across 5 seeds. Masked-accuracy dominance tracks feature
            magnitude at the fusion boundary (ρ_f) — but the decoupled metrics below show that
            masked shift overstates the real change in audio's contribution.
          </p>
        </div>
        <Tag tone="forward">5 seeds · Adam</Tag>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-end gap-x-8 gap-y-4">
        <div>
          <div className="label mb-2">Scaled modality</div>
          <div className="inline-flex gap-1.5">
            {(["audio", "video"] as Modality[]).map((m) => (
              <button
                key={m}
                onClick={() => setModality(m)}
                data-active={modality === m}
                className="seg border border-line data-[active=true]:border-ink capitalize"
              >
                {m}
              </button>
            ))}
          </div>
        </div>
        <div>
          <div className="label mb-2">Feature scale α</div>
          <div className="inline-flex gap-1.5">
            {([1, 5, 10] as Scale[]).map((s) => (
              <button
                key={s}
                onClick={() => setScale(s)}
                data-active={scale === s}
                className="seg border border-line data-[active=true]:border-forward"
              >
                {s}×
              </button>
            ))}
          </div>
        </div>
        <div className="font-mono text-2xs text-muted">
          Condition:{" "}
          <span className="text-ink">{scale === 1 ? "baseline (1×)" : `${modality} ×${scale}`}</span>
        </div>
      </div>

      {/* Metric tiles */}
      <div className="grid grid-cols-3 gap-3">
        <Card className="p-4">
          <div className="text-2xs uppercase tracking-wider text-muted">Dominance 𝒟</div>
          <AnimatedNumber
            key={activeName + "d"}
            value={row.dom}
            digits={3}
            className="measure text-[1.6rem] font-semibold text-forward"
          />
          <div className="mt-0.5 text-2xs text-muted">audio − video</div>
        </Card>
        <Card className="p-4">
          <div className="text-2xs uppercase tracking-wider text-muted">Gain 𝒢</div>
          <AnimatedNumber
            key={activeName + "g"}
            value={row.gain}
            digits={3}
            className="measure text-[1.6rem] font-semibold text-positive"
          />
          <div className="mt-0.5 text-2xs text-muted">multimodal − best unimodal</div>
        </Card>
        <Card className="p-4">
          <div className="text-2xs uppercase tracking-wider text-muted">ρ_f (post-scale)</div>
          <AnimatedNumber
            key={activeName + "r"}
            value={row.rhoFus}
            digits={2}
            className="measure text-[1.6rem] font-semibold text-ink"
          />
          <div className="mt-0.5 text-2xs text-muted">fusion norm ratio</div>
        </Card>
      </div>

      {/* Charts + disentanglement side-by-side */}
      <div className="grid gap-6 lg:grid-cols-[3fr_2fr]">
        {/* Charts */}
        <div className="grid gap-4 sm:grid-cols-2">
          <ScientificFigure
            index="Dominance 𝒟"
            caption={`Audio − video contribution by condition. ${activeName !== "baseline" ? `${activeName.charAt(0).toUpperCase() + activeName.slice(1)} highlighted.` : "Baseline highlighted."}`}
          >
            <CremadScaleChart metric="dom" active={activeName} />
          </ScientificFigure>
          <ScientificFigure
            index="Gain 𝒢"
            caption="Multimodal gain above the stronger unimodal baseline, by condition."
          >
            <CremadScaleChart metric="gain" active={activeName} />
          </ScientificFigure>
        </div>

        {/* Disentanglement card */}
        <Card className="p-5">
          <div className="label mb-1">Disentanglement · α=100×</div>
          <p className="mb-4 text-xs leading-relaxed text-muted">
            Which pathway carries the scale effect on CREMA-D? Three-arm comparison under Adam.
          </p>
          <div className="space-y-3">
            {/* Forward arm */}
            <div className="rounded border border-forward/30 bg-forward/[0.05] p-3.5">
              <div className="mb-2 font-mono text-2xs uppercase tracking-wider text-forward">
                Forward-only
              </div>
              <div className="flex items-baseline gap-3">
                <span className="measure text-2xl font-semibold text-forward">
                  {signed(fwd.dDom)}
                </span>
                <span className="font-mono text-2xs text-muted">Δ𝒟</span>
              </div>
              <div className="mt-1 font-mono text-2xs text-muted">
                d = {fwd.d.toFixed(2)} · p = {pValue(fwd.p)}
              </div>
              <div className="mt-2 text-2xs text-ink/70">
                Replicates ordinary on masked dominance only
              </div>
            </div>

            {/* Gradient arm */}
            <div className="rounded border border-line bg-canvas/40 p-3.5">
              <div className="mb-2 font-mono text-2xs uppercase tracking-wider text-gradient">
                Gradient-only
              </div>
              <div className="flex items-baseline gap-3">
                <span className="measure text-2xl font-semibold text-ink">
                  {signed(grad.dDom)}
                </span>
                <span className="font-mono text-2xs text-muted">Δ𝒟</span>
              </div>
              <div className="mt-1 font-mono text-2xs text-muted">
                d = {grad.d.toFixed(2)} · p = {pValue(grad.p)}
              </div>
              <div className="mt-2 text-2xs text-muted/80">
                Negligible — consistent with synthetic result
              </div>
            </div>

            {/* Ordinary arm */}
            <div className="rounded border border-line bg-surface p-3.5">
              <div className="mb-2 font-mono text-2xs uppercase tracking-wider text-muted">
                Ordinary (both)
              </div>
              <div className="flex items-baseline gap-3">
                <span className="measure text-2xl font-semibold text-ink">
                  {signed(decomposition.adamCremad.ordinary.dDom)}
                </span>
                <span className="font-mono text-2xs text-muted">Δ𝒟</span>
              </div>
              <div className="mt-1 font-mono text-2xs text-muted">
                d = {decomposition.adamCremad.ordinary.d.toFixed(2)} · p = {pValue(decomposition.adamCremad.ordinary.p)}
              </div>
            </div>
          </div>
        </Card>
      </div>

      <div id="masked-mislead" className="scroll-mt-24">
      {/* Decoupled-metric check — the post-UI experiment (W2) */}
      <Callout title="Masked accuracy can mislead here" tone="warning">
        Forward scaling (α=100, Adam, weight decay 0) moves <span className="measure text-ink">masked</span>{" "}
        dominance by <span className="measure text-ink">{signed(decoupled.cremadAdamW.masked)}</span>,
        but the decoupled metrics barely move: mean-ablation{" "}
        <span className="measure text-ink">{signed(decoupled.cremadAdamW.meanAbl)}</span> (n.s.),
        Shapley <span className="measure text-ink">{signed(decoupled.cremadAdamW.shapley)}</span>{" "}
        (n.s.), and the frozen-encoder probe only{" "}
        <span className="measure text-ink">{signed(decoupled.cremadAdamW.probe)}</span>. A positive
        control — adding audio input noise — does lower the probe{" "}
        ({signed(decoupled.positiveControl.probe[0])} then{" "}
        {signed(decoupled.positiveControl.probe[1])}, n = 3 seeds) and Shapley — evidence that
        the diagnostics can detect this tested form of input-induced degradation, though not
        proof of sensitivity to every representational change. On CREMA-D the masked shift is
        not accompanied by a comparable change in these diagnostics.
      </Callout>
      </div>

      {/* Normalization callout */}
      <Callout title="Normalization control" tone="forward">
        Imposed scale and learned dominance are rank-correlated (Spearman{" "}
        {fixed(cremad.spearman, 3)}). Per-modality z-scoring largely suppresses the imposed
        difference — for audio ×10, dominance{" "}
        <span className="measure text-ink">{fixed(cremad.normalization.domBefore, 3)}</span>
        {" → "}
        <span className="measure text-ink">{fixed(cremad.normalization.domAfter, 3)}</span>
        {" and ρ_f "}
        <span className="measure text-ink">{fixed(cremad.normalization.rhoFusBefore, 2)}</span>
        {" → "}
        <span className="measure text-ink">{fixed(cremad.normalization.rhoFusAfter, 2)}</span>
        . Normalization acts as a control on the scale channel, not a general cure.
      </Callout>

      <div id="sgd-stable" className="scroll-mt-24">
      {/* Stable SGD (Phase 4) */}
      <Callout title="Stable SGD, by clipping" tone="gradient">
        Under SGD the scaled arms diverge on real data (gradient arm{" "}
        <span className="measure text-ink">{sgdStable.divergedGradientArm.noClip}</span> seeds at
        α=100), but standard gradient-norm clipping removes it entirely —{" "}
        <span className="measure text-ink">{sgdStable.divergedGradientArm.clip}</span> with clip = 1.
        The mismatch survives: the gradient arm moves masked dominance{" "}
        <span className="measure text-ink">{signed(sgdStable.gradientArmMasked)}</span> with
        mean-ablation, Shapley, and probe all null; the forward arm moves masked{" "}
        <span className="measure text-ink">{signed(sgdStable.forwardArmMasked)}</span> while the
        probe moves <span className="measure text-ink">{signed(sgdStable.forwardArmProbe)}</span>{" "}
        — opposite in sign. So the dissociation is not Adam-specific.
      </Callout>
      </div>

      <div id="baselines" className="scroll-mt-24">
      {/* Payoff baseline (Phase 4) */}
      <Callout title="Adapted payoff baseline: a published rebalancing method" tone="neutral">
        Applied to {baselines.ogmGE.label} ({baselines.ogmGE.cite}) rather than our own operator —
        adapted to our shared sum-fusion setting (per-modality score from the shared head;
        generalization term approximated with Gaussian noise), it attenuates audio (k_audio ={" "}
        <span className="measure text-ink">{fixed(baselines.ogmGE.kAudio, 2)}</span>). Relative to
        vanilla training it shifts masked dominance{" "}
        <span className="measure text-ink">{signed(baselines.ogmGE.masked)}</span>, yet its
        decoupled metrics are mostly not statistically resolved: mean-ablation{" "}
        <span className="measure text-ink">{signed(baselines.ogmGE.meanAbl)}</span> and Shapley{" "}
        <span className="measure text-ink">{signed(baselines.ogmGE.shapley)}</span> include zero
        under Student-t, and only the tiny probe{" "}
        <span className="measure text-ink">{signed(baselines.ogmGE.probe)}</span> is resolved. Judging the
        method by masked-accuracy dominance would read a large rebalancing effect the
        representation-level metrics do not support — a caution for methods in the literature, not a
        claim about OGM-GE's accuracy benefit.
      </Callout>
      </div>

      {/* Dataset specimen */}
      <Card className="p-5">
        <div className="label mb-3">Dataset · CREMA-D</div>
        <div className="grid grid-cols-2 gap-x-8 gap-y-3 text-sm sm:grid-cols-3">
          {[
            ["Modalities", "Audio + Video"],
            ["Classes", `${cremad.classes} emotions`],
            ["Split", "64 / 9 / 18 actors"],
            ["Joint acc.", cremad.jointAccuracy.toFixed(3)],
            ["Audio", cremad.audioSpec],
            ["Video", cremad.videoSpec],
          ].map(([k, v]) => (
            <div key={k}>
              <div className="text-2xs uppercase tracking-wider text-muted">{k}</div>
              <div className="mt-0.5 font-mono text-xs text-ink">{v}</div>
            </div>
          ))}
        </div>
      </Card>
    </TabPanel>
  );
}

/* ─── UTD-MHAD Panel ────────────────────────────────────────────── */
function UtdPanel() {
  const [cond, setCond] = useState("base");
  const active = utdConditions.find((c) => c.id === cond)!;

  return (
    <TabPanel className="space-y-6">
      {/* Panel header */}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="label mb-1.5">UTD-MHAD · Depth-Inertial</div>
          <p className="max-w-xl text-sm leading-relaxed text-muted">
            {utd.actions}-action recognition, {utd.samples} usable samples, subject-disjoint
            split. Strongly asymmetric predictive utility — depth sits near the{" "}
            {utd.actions}-class chance level.
          </p>
        </div>
        <Tag tone="warning">Boundary case · 4 test subjects</Tag>
      </div>

      {/* Topline metrics */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          {
            k: "Depth acc.",
            v: utd.base.depthAcc,
            tone: "critical" as const,
            sub: `≈ chance (${utd.chanceLevel})`,
          },
          {
            k: "Inertial acc.",
            v: utd.base.inertialAcc,
            tone: "forward" as const,
            sub: "informative",
          },
          { k: "Multimodal acc.", v: utd.base.multimodalAcc, tone: "ink" as const, sub: "" },
          { k: "Gain 𝒢 (baseline)", v: utd.base.gain, tone: "positive" as const, sub: "" },
        ].map((m) => (
          <Card key={m.k} className="p-4">
            <div className="text-2xs uppercase tracking-wider text-muted">{m.k}</div>
            <div
              className={`measure text-2xl font-semibold ${
                m.tone === "critical"
                  ? "text-critical"
                  : m.tone === "forward"
                  ? "text-forward"
                  : m.tone === "positive"
                  ? "text-positive"
                  : "text-ink"
              }`}
            >
              {fixed(m.v, 3)}
            </div>
            {m.sub && <div className="mt-1 text-2xs text-muted">{m.sub}</div>}
          </Card>
        ))}
      </div>

      {/* Chart + condition selector */}
      <div className="grid gap-6 lg:grid-cols-2">
        <ScientificFigure
          index="Masked unimodal + multimodal accuracy"
          caption={`Depth sits near the ${utd.actions}-class chance level (${utd.chanceLevel}). Inertial is informative. Multimodal combines both.`}
        >
          <UtdAccuracyChart />
        </ScientificFigure>

        <div className="space-y-4">
          <div>
            <div className="label mb-2">Amplify a modality ×10</div>
            <div className="inline-flex flex-wrap gap-1.5">
              {utdConditions.map((c) => (
                <button
                  key={c.id}
                  onClick={() => setCond(c.id)}
                  data-active={cond === c.id}
                  className="seg border border-line data-[active=true]:border-ink"
                >
                  {c.label}
                </button>
              ))}
            </div>
          </div>

          <Card className="p-5">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-2xs uppercase tracking-wider text-muted">Dominance 𝒟</div>
                <AnimatedNumber
                  key={cond + "d"}
                  value={active.dominance}
                  digits={3}
                  signed
                  className="measure text-2xl font-semibold text-ink"
                />
                <div className="mt-0.5 text-2xs text-muted">inertial − depth</div>
              </div>
              <div>
                <div className="text-2xs uppercase tracking-wider text-muted">Gain 𝒢</div>
                <AnimatedNumber
                  key={cond + "g"}
                  value={active.gain}
                  digits={3}
                  className="measure text-2xl font-semibold text-positive"
                />
              </div>
            </div>

            {/* Dominance bar comparison */}
            <div className="mt-4 border-t border-line pt-4">
              <div className="mb-2.5 font-mono text-2xs uppercase tracking-wider text-muted">
                All conditions · Dominance |𝒟|
              </div>
              <div className="space-y-2.5">
                {utdConditions.map((c) => (
                  <div key={c.id} className="flex items-center gap-3">
                    <span className="w-24 shrink-0 font-mono text-2xs text-muted">
                      {c.label}
                    </span>
                    <div className="flex-1 h-1.5 rounded-full bg-line overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          c.emphasis === "forward"
                            ? "bg-forward"
                            : c.emphasis === "gradient"
                            ? "bg-gradient"
                            : "bg-muted/50"
                        }`}
                        style={{ width: `${(Math.abs(c.dominance) / 0.65) * 100}%` }}
                      />
                    </div>
                    <span className="w-14 shrink-0 text-right font-mono text-2xs text-ink">
                      {c.dominance >= 0 ? "+" : "−"}
                      {Math.abs(c.dominance).toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </Card>
        </div>
      </div>

      <Callout title="Boundary condition" tone="warning">
        Amplifying the informative inertial modality further increases its relative contribution
        and collapses gain to {fixed(utd.inertialX10.gain, 3)}; amplifying the near-chance depth
        modality barely moves contribution (gain ≈ {fixed(utd.depthX10.gain, 2)}). This is a
        boundary case — four fixed test subjects — showing that a near-chance modality cannot
        manufacture contribution under scaling. Not a population inference.
      </Callout>

      {/* Decoupled check on the second dataset (Phase 4, α=100) */}
      <Callout title="Masked overstates the probe here too" tone="forward">
        Forward scaling the informative inertial branch (α=100, Adam, weight decay 0, 5 seeds) moves masked
        dominance <span className="measure text-ink">{signed(decoupled.utdAdamW.masked)}</span>{" "}
        toward inertial, but the inertial probe rises only{" "}
        <span className="measure text-ink">{signed(decoupled.utdAdamW.probe)}</span> — masked
        overstates the representational change ~5×. Unlike CREMA-D the decoupled metrics do move
        here (mean-ablation <span className="measure text-ink">{signed(decoupled.utdAdamW.meanAbl)}</span>,
        Shapley <span className="measure text-ink">{signed(decoupled.utdAdamW.shapley)}</span>),
        because the scaled modality's probe has headroom. The gradient-only arm equals the baseline
        on every metric — the Adam cancellation, replicated on a second real dataset.
      </Callout>
    </TabPanel>
  );
}

/* ─── Page ──────────────────────────────────────────────────────── */
export function RealWorld() {
  const [params] = useSearchParams();
  const [tab, setTab] = useState(params.get("tab") === "utd" ? "utd" : "cremad");
  useSectionScroll();
  const fwd = decomposition.adamCremad.forward;
  const grad = decomposition.adamCremad.gradient;

  return (
    <div className="space-y-8">
      <SectionHeader
        eyebrow="Real-World Evidence"
        title="Tracing the observed mechanisms into real multimodal data"
        subtitle="On CREMA-D, forward scaling moves masked-accuracy dominance strongly while the decoupled metrics stay flat — a masking artifact. On a second dataset (UTD-MHAD) the masked shift overstates the real probe change ~5×, and the gradient arm equals baseline on every metric. The same dissociation holds under SGD stabilized by gradient clipping, and for a published rebalancing method (OGM-GE). All findings are conditional on the tested configurations."
      />

      {/* Cross-dataset synthesis */}
      <div className="grid gap-4 sm:grid-cols-3">
        <Card className="p-5">
          <div className="label mb-1.5 text-forward">Masked dominance shifts</div>
          <div className="measure text-2xl font-semibold text-forward">
            {signed(fwd.dDom)}
          </div>
          <div className="mt-0.5 font-mono text-2xs text-muted">
            masked Δ𝒟 · CREMA-D (Adam)
          </div>
          <p className="mt-3 text-xs leading-relaxed text-muted/90">
            Forward scaling moves masked-accuracy dominance strongly — but no decoupled metric
            follows, so on CREMA-D this is a masking artifact, not a genuine contribution change.
          </p>
        </Card>

        <Card className="p-5">
          <div className="label mb-1.5">Gradient arm negligible</div>
          <div className="measure text-2xl font-semibold text-ink">{signed(grad.dDom)}</div>
          <div className="mt-0.5 font-mono text-2xs text-muted">
            gradient Δ𝒟 · p = {pValue(grad.p)}
          </div>
          <p className="mt-3 text-xs leading-relaxed text-muted/90">
            The gradient pathway remains near zero on CREMA-D, consistent with Adam's adaptive
            moment normalization attenuating the gradient rescale.
          </p>
        </Card>

        <Card className="p-5">
          <div className="label mb-1.5 text-warning">UTD-MHAD · masked overstates</div>
          <div className="measure text-2xl font-semibold text-warning">
            {signed(decoupled.utdAdamW.masked)} vs {signed(decoupled.utdAdamW.probe)}
          </div>
          <div className="mt-0.5 font-mono text-2xs text-muted">
            masked Δ𝒟 vs probe · UTD-MHAD (Adam)
          </div>
          <p className="mt-3 text-xs leading-relaxed text-muted/90">
            On a second real dataset, forward scaling moves masked dominance ~5× more than the
            frozen-encoder probe — and the gradient arm equals baseline on every metric, as Adam
            predicts.
          </p>
        </Card>
      </div>

      {/* Dataset tabs */}
      <div id="datasets" className="scroll-mt-24" />
      <Tabs tabs={TABS} active={tab} onChange={setTab} />
      {tab === "cremad" ? <CremadPanel /> : <UtdPanel />}
    </div>
  );
}
