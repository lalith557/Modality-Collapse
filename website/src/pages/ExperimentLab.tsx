import { useState } from "react";
import { ArrowDown, RotateCcw, ArrowRight, ChevronRight } from "lucide-react";
import { useResearch } from "../state/research-context";
import { SectionHeader, Card, Tag, Callout } from "../components/primitives";
import { Field, Select, SeedPicker } from "../components/controls";
import { AnimatedNumber } from "../components/AnimatedNumber";
import {
  OPTIONS, LABELS, armEndpoint, getDecomposition,
  type InterventionId, type ScaleId, type FusionId, type OptimizerId, type DatasetId,
} from "../data/research";
import { pValue, fixed } from "../lib/format";

const opt = (a: readonly { id: string; label: string }[]) => a.map((o) => ({ value: o.id, label: o.label }));

/* ---------- Modality config per dataset ---------- */
const MOD_LABELS: Record<DatasetId, { a: string; b: string; aShort: string; bShort: string }> = {
  synthetic: { a: "Modality A", b: "Modality B", aShort: "Sig A", bShort: "Sig B" },
  cremad:    { a: "Audio",      b: "Video",      aShort: "Audio", bShort: "Video" },
  utd:       { a: "Inertial",   b: "Depth",      aShort: "Inert", bShort: "Depth" },
};

const EPOCHS: Record<DatasetId, number> = { synthetic: 40, cremad: 30, utd: 60 };

const FUSION_EXPR: Record<FusionId, string> = {
  sum:    "h_A + h_B",
  concat: "[h_A ; h_B]",
  gated:  "gate(h_A, h_B)",
};

/* ---------- small signal glyphs ---------- */
function SignalGlyphA() {
  const bars = [6, 12, 20, 14, 24, 10, 18, 8, 22, 12, 16, 9];
  return (
    <svg viewBox="0 0 96 28" className="h-6 w-full" aria-hidden="true">
      {bars.map((h, i) => (
        <rect key={i} x={i * 8 + 2} y={14 - h / 2} width="4" height={h} rx="1" fill="#557A95" opacity={0.7} />
      ))}
    </svg>
  );
}
function SignalGlyphB() {
  return (
    <svg viewBox="0 0 96 28" className="h-6 w-full" aria-hidden="true">
      {[0, 1, 2, 3].map((i) => (
        <rect key={i} x={i * 24 + 2} y="3" width="20" height="22" rx="2" fill="#9A7358" opacity={0.13} stroke="#9A7358" strokeWidth="0.8" />
      ))}
    </svg>
  );
}

/* ---------- Pipeline Diagram ---------- */
function PipelineDiagram({
  dataset, intervention, fusion, scale,
}: {
  dataset: DatasetId;
  intervention: InterventionId;
  fusion: FusionId;
  scale: number;
}) {
  const mod = MOD_LABELS[dataset];
  const fwdActive = intervention === "forward" || intervention === "ordinary";
  const gradActive = intervention === "gradient" || intervention === "ordinary";
  const fusionExpr = FUSION_EXPR[fusion];

  const fwdClass = fwdActive
    ? "border-forward bg-forward/[0.08] text-forward"
    : "border-line bg-surface text-muted";

  return (
    <div className="mx-auto w-full max-w-[320px] space-y-1.5">
      {/* Modality input row */}
      <div className="grid grid-cols-2 gap-3">
        <div className="rounded border border-line bg-canvas/40 p-2.5">
          <div className="mb-1 flex items-baseline justify-between">
            <span className="font-mono text-2xs font-semibold text-ink">{mod.a}</span>
            <span className="font-mono text-[0.55rem] uppercase tracking-wider text-muted">{mod.aShort}</span>
          </div>
          <SignalGlyphA />
          <div className="mt-1 text-center font-mono text-xs text-muted">h_A</div>
        </div>
        <div className="rounded border border-line bg-canvas/40 p-2.5">
          <div className="mb-1 flex items-baseline justify-between">
            <span className="font-mono text-2xs font-semibold text-ink">{mod.b}</span>
            <span className="font-mono text-[0.55rem] uppercase tracking-wider text-muted">{mod.bShort}</span>
          </div>
          <SignalGlyphB />
          <div className="mt-1 text-center font-mono text-xs text-muted">h_B</div>
        </div>
      </div>

      {/* Scale operator */}
      <div className="grid grid-cols-2 gap-3">
        <div className={`rounded border px-2 py-1.5 text-center font-mono text-2xs transition-all duration-200 ${fwdClass}`}>
          {fwdActive ? `α·h_A = ${scale}·h_A` : "h_A (unscaled)"}
        </div>
        <div className={`rounded border px-2 py-1.5 text-center font-mono text-2xs transition-all duration-200 ${gradActive ? "border-gradient/40 text-gradient" : "border-line text-muted"} bg-transparent`}>
          {gradActive ? `∂L·α (grad ×${scale})` : "h_B (no grad scale)"}
        </div>
      </div>

      <div className="flex justify-center py-0.5">
        <ArrowDown size={13} className="text-muted/40" />
      </div>

      {/* Fusion block */}
      <div className="rounded border border-ink bg-ink px-3 py-2 text-center">
        <div className="font-mono text-2xs text-canvas/60 uppercase tracking-wider">Fusion</div>
        <div className="font-mono text-xs text-canvas">{fusionExpr}</div>
      </div>

      <div className="flex justify-center py-0.5">
        <ArrowDown size={13} className="text-muted/40" />
      </div>

      {/* Prediction */}
      <div className="rounded border border-line bg-surface px-3 py-2 text-center font-mono text-sm text-ink">
        Prediction ŷ
      </div>

      {/* Gradient path annotation */}
      {gradActive && (
        <div className="mt-1 flex items-center justify-center gap-1.5 rounded border border-gradient/30 bg-gradient/[0.05] px-2 py-1">
          <span className="font-mono text-[0.6rem] text-gradient">↑ gradient × α flows back through h_A encoder</span>
        </div>
      )}
    </div>
  );
}

/* ---------- Arm summary row for ALL SEEDS mode ---------- */
function ArmRow({ label, arm, tone }: {
  label: string;
  arm: { dDom: number; d: number; p: number };
  tone: "forward" | "gradient" | "neutral";
}) {
  const colors = {
    forward: "text-forward",
    gradient: "text-gradient",
    neutral: "text-ink",
  }[tone];
  return (
    <div className="flex items-center justify-between gap-2 border-b border-line py-2 last:border-0">
      <span className="text-sm text-muted">{label}</span>
      <div className="flex items-center gap-4">
        <span className={`measure text-sm font-semibold ${colors}`}>
          {arm.dDom >= 0 ? "+" : ""}{fixed(arm.dDom, 3)}
        </span>
        <span className="font-mono text-2xs text-muted/80 tabular-nums">
          d {fixed(arm.d, 2)} · p {pValue(arm.p)}
        </span>
      </div>
    </div>
  );
}

/* ================================================================ */
/*  Main component                                                   */
/* ================================================================ */

export function ExperimentLab() {
  const s = useResearch();
  const [showAllSeeds, setShowAllSeeds] = useState(false);

  const dec = getDecomposition(s.dataset, s.optimizer);
  const hasResult = dec !== null;

  const { endpoint, measured } = armEndpoint(s.dataset, s.optimizer, s.intervention);
  const optimizerOptions =
    s.dataset === "synthetic"
      ? opt(OPTIONS.optimizer)
      : [{ value: "adam", label: "Adam (only run)" }];

  const epochs = EPOCHS[s.dataset];

  return (
    <div className="space-y-6">
      {/* ---- Header ---- */}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <SectionHeader
          eyebrow="Experiment lab"
          title="Experiment Lab"
          subtitle="Explore the controlled experiments behind the study. All results are reproduced from frozen artifacts — no model is trained in the browser."
        />
        <div className="mt-1 flex shrink-0 items-center gap-1.5 rounded border border-line bg-black/[0.02] px-3 py-1.5">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-forward/70" />
          <span className="font-mono text-[0.6rem] uppercase tracking-[0.12em] text-muted">
            Frozen results · Experiment artifacts
          </span>
        </div>
      </div>

      <div className="grid gap-5 xl:grid-cols-[250px_1fr_310px]">

        {/* ============ Column 1: Configuration ============ */}
        <Card className="h-fit p-5">
          <div className="mb-5 flex items-center justify-between">
            <div className="label">Configuration</div>
            <button
              onClick={s.reset}
              className="flex items-center gap-1 rounded border border-line px-2 py-1 font-mono text-[0.6rem] uppercase tracking-wider text-muted transition-colors hover:border-ink/30 hover:text-ink"
              title="Reset to defaults"
            >
              <RotateCcw size={10} /> Reset
            </button>
          </div>
          <div className="space-y-4">
            <Field label="Dataset">
              <Select
                value={s.dataset}
                onChange={(v) => s.set("dataset", v as DatasetId)}
                options={opt(OPTIONS.dataset)}
              />
            </Field>
            <Field label="Optimizer">
              <Select
                value={s.optimizer}
                onChange={(v) => s.set("optimizer", v as OptimizerId)}
                options={optimizerOptions}
                disabled={s.dataset !== "synthetic"}
              />
            </Field>
            <Field label="Intervention">
              <Select
                value={s.intervention}
                onChange={(v) => s.set("intervention", v as InterventionId)}
                options={opt(OPTIONS.intervention)}
              />
            </Field>
            <div>
              <div className="label mb-2 flex justify-between">
                <span>Scale α</span>
                <span className="measure text-forward">{s.scale}×</span>
              </div>
              <input
                type="range" min={0} max={3} step={1}
                value={OPTIONS.scale.indexOf(s.scale)}
                onChange={(e) => s.set("scale", OPTIONS.scale[Number(e.target.value)] as ScaleId)}
                className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-line accent-forward"
              />
              <div className="mt-1 flex justify-between font-mono text-[0.6rem] text-muted">
                {OPTIONS.scale.map((v) => <span key={v}>{v}×</span>)}
              </div>
            </div>
            <Field label="Fusion">
              <Select
                value={s.fusion}
                onChange={(v) => s.set("fusion", v as FusionId)}
                options={opt(OPTIONS.fusion)}
              />
            </Field>
            <div>
              <div className="label mb-2">Seed</div>
              <SeedPicker value={s.seed} onChange={(v) => s.set("seed", v)} />
            </div>

            {/* Honest status indicator */}
            <div className={`rounded border px-3 py-2 text-center font-mono text-2xs ${hasResult ? "border-forward/30 bg-forward/[0.06] text-forward" : "border-warning/40 bg-warning/[0.06] text-warning"}`}>
              {hasResult ? "Result available" : "No recorded result"}
            </div>
          </div>
        </Card>

        {/* ============ Column 2: Pipeline Diagram ============ */}
        <Card className="p-5">
          <div className="mb-4 flex items-center justify-between">
            <div className="label">Experiment Pipeline</div>
            <span className="font-mono text-2xs text-muted">
              {LABELS.fusion[s.fusion]} · α = {s.scale}
            </span>
          </div>

          <PipelineDiagram
            dataset={s.dataset}
            intervention={s.intervention}
            fusion={s.fusion}
            scale={s.scale}
          />

          {/* Arm description */}
          <div className="mt-4 rounded border border-line bg-canvas/30 p-3 text-2xs leading-relaxed text-muted">
            {s.intervention === "forward" && (
              <><span className="font-semibold text-forward">Forward-only:</span> The autograd operator scales h_A by α in the forward pass. Gradients flow unmodified. This isolates the representation-magnitude channel.</>
            )}
            {s.intervention === "gradient" && (
              <><span className="font-semibold text-gradient">Gradient-only:</span> The autograd operator scales the gradient at h_A by α during backprop. The forward representation is unchanged. This isolates the gradient / parameter-update channel.</>
            )}
            {s.intervention === "ordinary" && (
              <><span className="font-semibold text-ink">Ordinary (both):</span> Both the forward representation and the gradient carry α. This reproduces a standard feature-scale intervention with no separation.</>
            )}
          </div>

          {/* Protocol mini-table */}
          <div className="mt-4 border-t border-line pt-4">
            <div className="label mb-2">Protocol snapshot</div>
            <div className="space-y-1">
              {[
                ["Dataset",   LABELS.dataset[s.dataset]],
                ["Optimizer", s.dataset === "synthetic" ? LABELS.optimizer[s.optimizer] : "Adam (only run)"],
                ["Epochs",    String(epochs)],
                ["Seeds",     "5"],
                ["α",         `${s.scale}×`],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between gap-2 font-mono text-2xs">
                  <span className="text-muted">{k}</span>
                  <span className="text-ink">{v}</span>
                </div>
              ))}
            </div>
          </div>
        </Card>

        {/* ============ Column 3: Evidence ============ */}
        <div className="space-y-4">
          {/* Header row */}
          <div className="flex items-center justify-between gap-2">
            <div className="label">Evidence</div>
            {hasResult && (
              <div className="inline-flex rounded border border-line bg-surface p-0.5">
                <button
                  onClick={() => setShowAllSeeds(false)}
                  data-active={!showAllSeeds}
                  className="seg data-[active=true]:bg-ink data-[active=true]:text-canvas"
                >
                  Selected arm
                </button>
                <button
                  onClick={() => setShowAllSeeds(true)}
                  data-active={showAllSeeds}
                  className="seg data-[active=true]:bg-ink data-[active=true]:text-canvas"
                >
                  All arms
                </button>
              </div>
            )}
          </div>

          {/* ---- RESULT NOT AVAILABLE ---- */}
          {!hasResult ? (
            <>
              <Card className="p-5">
                <Tag tone="warning">Result not available</Tag>
                <p className="mt-3 text-sm leading-relaxed text-ink/80">
                  The forward-vs-gradient decomposition was not run for{" "}
                  <span className="font-medium">{LABELS.dataset[s.dataset]} · {LABELS.optimizer[s.optimizer]}</span>.
                </p>
                <p className="mt-2 text-2xs leading-relaxed text-muted">
                  Measured combinations: Synthetic + Adam, Synthetic + SGD, CREMA-D + Adam.
                  UTD-MHAD was evaluated without the decomposition protocol.
                </p>
                <div className="mt-4 flex flex-col gap-2">
                  <button
                    onClick={() => s.setScenario({ dataset: "synthetic", optimizer: "adam" })}
                    className="btn-ghost justify-between text-2xs"
                  >
                    Synthetic · Adam <ArrowRight size={13} />
                  </button>
                  <button
                    onClick={() => s.setScenario({ dataset: "cremad", optimizer: "adam" })}
                    className="btn-ghost justify-between text-2xs"
                  >
                    CREMA-D · Adam <ArrowRight size={13} />
                  </button>
                  <button
                    onClick={() => s.setScenario({ dataset: "synthetic", optimizer: "sgd" })}
                    className="btn-ghost justify-between text-2xs"
                  >
                    Synthetic · SGD <ArrowRight size={13} />
                  </button>
                </div>
              </Card>
            </>
          ) : showAllSeeds ? (
            /* ---- ALL ARMS view ---- */
            <>
              <Card className="p-5">
                <div className="mb-1 flex items-center justify-between">
                  <div className="label">All arms · Δ𝒟 (dominance shift)</div>
                  <Tag tone="neutral">n = 5 seeds</Tag>
                </div>
                <p className="mb-3 text-2xs text-muted">
                  Aggregated across {epochs}-epoch runs, α = 100. Values are paired t-test results.
                </p>
                <ArmRow label="Forward-only" arm={dec!.data.forward} tone="forward" />
                <ArmRow label="Gradient-only" arm={dec!.data.gradient} tone="gradient" />
                <ArmRow label="Ordinary (both)" arm={dec!.data.ordinary} tone="neutral" />
              </Card>

              <Card className="p-5">
                <div className="label mb-3">Optimizer dynamics</div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <div className="text-2xs uppercase tracking-wider text-muted">Grad-norm ratio r_g</div>
                    <div className="measure text-2xl font-semibold text-gradient">{dec!.ratios.rG}×</div>
                  </div>
                  <div>
                    <div className="text-2xs uppercase tracking-wider text-muted">Update ratio r_Δθ</div>
                    <div className="measure text-2xl font-semibold text-forward">
                      {s.optimizer === "adam" ? "≈ 1.6×" : "≈ 95.9×"}
                    </div>
                  </div>
                </div>
                <p className="mt-3 text-2xs leading-relaxed text-muted">
                  {s.optimizer === "adam"
                    ? "Under Adam, a ~283× gradient ratio leaves the update ratio ≈ 1.6, consistent with Adam normalizing per-parameter gradient magnitudes."
                    : "Under SGD, the ~319× gradient ratio passes through to a ~96× update ratio — the gradient channel contributes."}
                </p>
              </Card>
            </>
          ) : (
            /* ---- SELECTED ARM view ---- */
            <>
              {/* Key metrics */}
              <Card className="p-5">
                <div className="mb-4 flex items-center justify-between">
                  <div className="label">Decomposition summary</div>
                  <Tag tone="neutral">n = 5 seeds · α = 100</Tag>
                </div>
                <div className="space-y-3">
                  <div>
                    <div className="text-2xs uppercase tracking-wider text-muted">Forward Δ𝒟</div>
                    <AnimatedNumber
                      key={`fwd-${s.dataset}-${s.optimizer}`}
                      value={dec!.data.forward.dDom}
                      digits={3} signed
                      className="text-2xl font-semibold text-forward"
                    />
                    <div className="font-mono text-2xs text-muted">
                      d = {fixed(dec!.data.forward.d, 2)} · p = {pValue(dec!.data.forward.p)}
                    </div>
                  </div>
                  <div>
                    <div className="text-2xs uppercase tracking-wider text-muted">Gradient Δ𝒟</div>
                    <AnimatedNumber
                      key={`grad-${s.dataset}-${s.optimizer}`}
                      value={dec!.data.gradient.dDom}
                      digits={4} signed
                      className="text-2xl font-semibold text-gradient"
                    />
                    <div className="font-mono text-2xs text-muted">
                      d = {fixed(dec!.data.gradient.d, 2)} · p = {pValue(dec!.data.gradient.p)}
                    </div>
                  </div>
                  <div>
                    <div className="text-2xs uppercase tracking-wider text-muted">Update ratio r_Δθ</div>
                    <div className="measure text-xl font-semibold text-ink">
                      {s.optimizer === "adam" ? "≈ 1.6×" : "≈ 95.9×"}
                    </div>
                  </div>
                </div>
              </Card>

              {/* Selected arm detail */}
              {measured && endpoint ? (
                <Card className="p-5">
                  <div className="mb-3 flex items-center justify-between">
                    <div className="label">
                      Selected arm
                    </div>
                    <Tag tone={s.intervention === "forward" ? "forward" : s.intervention === "gradient" ? "gradient" : "neutral"}>
                      {LABELS.intervention[s.intervention]}
                    </Tag>
                  </div>
                  <div className="text-2xs uppercase tracking-wider text-muted">Δ𝒟 dominance shift</div>
                  <AnimatedNumber
                    key={`sel-${s.dataset}-${s.optimizer}-${s.intervention}`}
                    value={endpoint.dDom}
                    digits={Math.abs(endpoint.dDom) < 0.001 ? 4 : 3}
                    signed
                    className="text-[2rem] font-semibold text-ink"
                  />
                  <div className="mt-1 flex gap-4 font-mono text-2xs text-muted">
                    <span>Cohen's d = {fixed(endpoint.d, 2)}</span>
                    <span>p = {pValue(endpoint.p)}</span>
                  </div>
                </Card>
              ) : null}

              {/* Training Outcome */}
              <Card className="p-4">
                <div className="label mb-2">Training outcome</div>
                <p className="text-2xs leading-relaxed text-muted">
                  End-of-training endpoints only. Epoch-level logs are not included in this explorer.
                  Reported values are from {epochs}-epoch runs across 5 seeds (α = 100 decomposition).
                  No model trains in the browser.
                </p>
                <div className="mt-2 flex items-center gap-1.5 font-mono text-[0.6rem] text-muted/70">
                  <ChevronRight size={10} />
                  Source: manuscript Table I
                </div>
              </Card>

              {/* Evidence provenance */}
              <div className="rounded border border-line bg-black/[0.015] px-4 py-2.5 space-y-1">
                {[
                  ["Source", "manuscript Table I"],
                  ["Seeds", "5"],
                  ["α", "100 (decomposition)"],
                  ["Epochs", String(epochs)],
                ].map(([k, v]) => (
                  <div key={k} className="flex gap-2 font-mono text-[0.6rem] text-muted">
                    <span className="w-16 shrink-0 text-muted/60">{k}</span>
                    <span>{v}</span>
                  </div>
                ))}
              </div>
            </>
          )}

          {/* Context note about SGD */}
          {hasResult && s.optimizer === "sgd" && (
            <Callout title="SGD note" tone="neutral">
              Under SGD the gradient-only arm also shifts dominance (r_Δθ ≈ 96×), unlike Adam where Adam's
              normalization keeps r_Δθ ≈ 1.6. Which pathway carries the effect is optimizer-dependent.
            </Callout>
          )}
        </div>
      </div>
    </div>
  );
}
