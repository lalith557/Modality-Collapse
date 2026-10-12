import { useState } from "react";
import { SectionHeader, Callout, Tag, Card } from "../components/primitives";
import { ScientificFigure } from "../components/ScientificFigure";
import { Tabs, TabPanel } from "../components/Tabs";
import { CremadScaleChart } from "../charts/CremadScaleChart";
import { UtdAccuracyChart } from "../charts/UtdChart";
import { AnimatedNumber } from "../components/AnimatedNumber";
import { cremad, utd, decomposition, utdConditions } from "../data/research";
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
            Scale sweep under Adam across 5 seeds. Modality contribution tracks feature
            magnitude at the fusion boundary (ρ_f), and the forward pathway carries the effect.
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
                Replicates most of the ordinary effect
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
    </TabPanel>
  );
}

/* ─── Page ──────────────────────────────────────────────────────── */
export function RealWorld() {
  const [tab, setTab] = useState("cremad");
  const fwd = decomposition.adamCremad.forward;
  const grad = decomposition.adamCremad.gradient;

  return (
    <div className="space-y-8">
      <SectionHeader
        eyebrow="Real-World Evidence"
        title="Tracing the observed mechanisms into real multimodal data"
        subtitle="Scale effects observed in synthetic data recur in CREMA-D under Adam. UTD-MHAD introduces a boundary condition: strongly asymmetric modality utility. All findings are conditional on the tested configurations."
      />

      {/* Cross-dataset synthesis */}
      <div className="grid gap-4 sm:grid-cols-3">
        <Card className="p-5">
          <div className="label mb-1.5 text-forward">Forward pathway active</div>
          <div className="measure text-2xl font-semibold text-forward">
            {signed(fwd.dDom)}
          </div>
          <div className="mt-0.5 font-mono text-2xs text-muted">
            forward Δ𝒟 · CREMA-D · d = {fwd.d.toFixed(2)}
          </div>
          <p className="mt-3 text-xs leading-relaxed text-muted/90">
            Under Adam, the forward arm replicates the ordinary scale effect on real audio-video
            data, consistent with the synthetic result.
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
          <div className="label mb-1.5 text-warning">Utility shapes response</div>
          <div className="measure text-2xl font-semibold text-warning">
            {fixed(utd.base.depthAcc, 3)} ≈ chance
          </div>
          <div className="mt-0.5 font-mono text-2xs text-muted">
            depth acc. · UTD-MHAD · boundary case
          </div>
          <p className="mt-3 text-xs leading-relaxed text-muted/90">
            Scaling a near-chance modality cannot manufacture contribution. This defines a
            boundary on the mechanism: utility must be present for scale to matter.
          </p>
        </Card>
      </div>

      {/* Dataset tabs */}
      <Tabs tabs={TABS} active={tab} onChange={setTab} />
      {tab === "cremad" ? <CremadPanel /> : <UtdPanel />}
    </div>
  );
}
