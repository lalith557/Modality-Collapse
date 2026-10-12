import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { SectionHeader, Card, Callout, Tag } from "../components/primitives";
import { ScientificFigure } from "../components/ScientificFigure";
import { decomposition } from "../data/research";
import { pValue } from "../lib/format";

const adamR = decomposition.ratios.adam;
const sgdR = decomposition.ratios.sgd;
const adamGrad = decomposition.adamSynthetic.gradient;
const sgdGrad = decomposition.sgdSynthetic.gradient;

/* ── Dual-channel SVG mechanism diagram ─────────────────────────── */

function DualChannelDiagram({ focused }: { focused: "adam" | "sgd" }) {
  const aFocused = focused === "adam";
  const aOp = aFocused ? 1 : 0.32;
  const sOp = aFocused ? 0.32 : 1;
  const T: React.CSSProperties = { transition: "opacity 280ms ease" };

  return (
    <svg
      viewBox="0 0 560 410"
      className="h-auto w-full"
      role="img"
      aria-label={`Dual-channel mechanism diagram. Same gradient-only intervention (α=100×) splits through Adam (${adamR.rG}× gradient → ${adamR.rTheta}× update) and SGD (${sgdR.rG}× gradient → ${sgdR.rTheta}× update), then merges into learned modality contribution.`}
    >
      {/* ── SAME INTERVENTION ─────────────────────────────── */}
      <rect x="148" y="10" width="264" height="48" rx="6" fill="#17232B" />
      <text x="280" y="28" textAnchor="middle" fill="rgba(255,255,255,0.5)" fontSize={9}
        fontFamily="'IBM Plex Mono', monospace" letterSpacing={2}>SAME INTERVENTION</text>
      <text x="280" y="48" textAnchor="middle" fill="#F5F3EE" fontSize={13} fontWeight={600}>
        Gradient-only · α=100×
      </text>

      {/* ── Pre-split connector ───────────────────────────── */}
      <path d="M280,58 V72" fill="none" stroke="#DCD9D1" strokeWidth={1.5} />
      <path d="M280,58 V72" fill="none" stroke="#7A8589" strokeWidth={1.75} className="pw-flow" />

      {/* ── Split connectors ──────────────────────────────── */}
      <path d="M280,72 H120 V90" fill="none" stroke="#557A95" strokeWidth={1.5}
        style={{ ...T, opacity: aOp }} />
      <path d="M280,72 H440 V90" fill="none" stroke="#9A7358" strokeWidth={1.5}
        style={{ ...T, opacity: sOp }} />

      {/* ── ADAM column ───────────────────────────────────── */}
      <g style={{ ...T, opacity: aOp }}>
        {/* Gradient box */}
        <rect x="20" y="90" width="200" height="48" rx="5" fill="#FCFBF8" stroke="#557A95" strokeWidth={1.3} />
        <text x="120" y="107" textAnchor="middle" fill="#687278" fontSize={8}
          fontFamily="'IBM Plex Mono', monospace" letterSpacing={1}>GRADIENT-NORM RATIO r_g</text>
        <text x="120" y="128" textAnchor="middle" fill="#557A95" fontSize={20} fontWeight={700}
          fontFamily="'IBM Plex Mono', monospace">{adamR.rG.toFixed(1)}×</text>

        {/* Gradient → optimizer connector */}
        <path d="M120,138 V158" fill="none" stroke="#557A95" strokeWidth={1.5} />

        {/* Optimizer box */}
        <rect x="20" y="158" width="200" height="72" rx="5"
          fill={aFocused ? "#EEF3F7" : "#FCFBF8"} stroke="#557A95" strokeWidth={aFocused ? 1.8 : 1.2} />
        <text x="120" y="177" textAnchor="middle" fill="#557A95" fontSize={11} fontWeight={700}
          fontFamily="'IBM Plex Mono', monospace" letterSpacing={2}>ADAM</text>
        <text x="120" y="193" textAnchor="middle" fill="#687278" fontSize={9.5}>Adaptive optimization</text>
        <text x="120" y="211" textAnchor="middle" fill="#17232B" fontSize={10.5} fontStyle="italic">
          m̂, v̂ → Δθ
        </text>
        <text x="120" y="224" textAnchor="middle" fill="#687278" fontSize={8}
          fontFamily="'IBM Plex Mono', monospace">adaptive moment normalization</text>

        {/* KEY VISUAL: dashed thin connector (attenuated) */}
        <path d="M120,230 V252" fill="none" stroke="#557A95" strokeWidth={1.5} strokeDasharray="5 3" />

        {/* Update box */}
        <rect x="20" y="252" width="200" height="66" rx="5" fill="#FCFBF8" stroke="#557A95"
          strokeWidth={aFocused ? 1.8 : 1.2} />
        <text x="120" y="269" textAnchor="middle" fill="#687278" fontSize={8}
          fontFamily="'IBM Plex Mono', monospace" letterSpacing={1}>PARAMETER-UPDATE RATIO r_Δθ</text>
        <text x="120" y="299" textAnchor="middle" fill="#557A95" fontSize={26} fontWeight={700}
          fontFamily="'IBM Plex Mono', monospace">{adamR.rTheta.toFixed(2)}×</text>
        <text x="120" y="313" textAnchor="middle" fill="#557A95" fontSize={8}
          fontFamily="'IBM Plex Mono', monospace" opacity={0.7}>strong attenuation</text>

        {/* Merge connector from Adam */}
        <path d="M120,318 V332 H280" fill="none" stroke="#557A95" strokeWidth={1.5} />
      </g>

      {/* ── SGD column ────────────────────────────────────── */}
      <g style={{ ...T, opacity: sOp }}>
        {/* Gradient box */}
        <rect x="340" y="90" width="200" height="48" rx="5" fill="#FCFBF8" stroke="#9A7358" strokeWidth={1.3} />
        <text x="440" y="107" textAnchor="middle" fill="#687278" fontSize={8}
          fontFamily="'IBM Plex Mono', monospace" letterSpacing={1}>GRADIENT-NORM RATIO r_g</text>
        <text x="440" y="128" textAnchor="middle" fill="#9A7358" fontSize={20} fontWeight={700}
          fontFamily="'IBM Plex Mono', monospace">{sgdR.rG.toFixed(1)}×</text>

        {/* Gradient → optimizer connector */}
        <path d="M440,138 V158" fill="none" stroke="#9A7358" strokeWidth={1.5} />

        {/* Optimizer box */}
        <rect x="340" y="158" width="200" height="72" rx="5"
          fill={!aFocused ? "#F5EFEA" : "#FCFBF8"} stroke="#9A7358" strokeWidth={!aFocused ? 1.8 : 1.2} />
        <text x="440" y="177" textAnchor="middle" fill="#9A7358" fontSize={11} fontWeight={700}
          fontFamily="'IBM Plex Mono', monospace" letterSpacing={2}>SGD</text>
        <text x="440" y="193" textAnchor="middle" fill="#687278" fontSize={9.5}>Non-adaptive optimization</text>
        <text x="440" y="211" textAnchor="middle" fill="#17232B" fontSize={10.5} fontStyle="italic">
          Δθ ∝ −ηg
        </text>
        <text x="440" y="224" textAnchor="middle" fill="#687278" fontSize={8}
          fontFamily="'IBM Plex Mono', monospace">direct gradient update</text>

        {/* KEY VISUAL: thick solid connector (pass-through) */}
        <path d="M440,230 V252" fill="none" stroke="#9A7358" strokeWidth={3.5} />

        {/* Update box */}
        <rect x="340" y="252" width="200" height="66" rx="5" fill="#FCFBF8" stroke="#9A7358"
          strokeWidth={!aFocused ? 1.8 : 1.2} />
        <text x="440" y="269" textAnchor="middle" fill="#687278" fontSize={8}
          fontFamily="'IBM Plex Mono', monospace" letterSpacing={1}>PARAMETER-UPDATE RATIO r_Δθ</text>
        <text x="440" y="299" textAnchor="middle" fill="#9A7358" fontSize={26} fontWeight={700}
          fontFamily="'IBM Plex Mono', monospace">{sgdR.rTheta.toFixed(1)}×</text>
        <text x="440" y="313" textAnchor="middle" fill="#9A7358" fontSize={8}
          fontFamily="'IBM Plex Mono', monospace" opacity={0.7}>large update response</text>

        {/* Merge connector from SGD */}
        <path d="M440,318 V332 H280" fill="none" stroke="#9A7358" strokeWidth={1.5} />
      </g>

      {/* ── Post-merge connector ──────────────────────────── */}
      <path d="M280,332 V348" fill="none" stroke="#DCD9D1" strokeWidth={1.5} />
      <path d="M280,332 V348" fill="none" stroke="#7A8589" strokeWidth={1.75} className="pw-flow" />

      {/* ── LEARNED CONTRIBUTION ─────────────────────────── */}
      <rect x="148" y="348" width="264" height="54" rx="6" fill="#17232B" />
      <text x="280" y="368" textAnchor="middle" fill="rgba(255,255,255,0.5)" fontSize={9.5}
        fontFamily="'IBM Plex Mono', monospace" letterSpacing={1.5}>OUTCOME</text>
      <text x="280" y="390" textAnchor="middle" fill="#F5F3EE" fontSize={14} fontWeight={600}>
        Learned modality contribution
      </text>
    </svg>
  );
}

/* ── Transformation panel ────────────────────────────────────────── */

function TransformationPanel({
  name, kind, tone, rG, rTheta, equation, equationNote, focused, note, onClick,
}: {
  name: string;
  kind: string;
  tone: "adam" | "sgd";
  rG: number;
  rTheta: number;
  equation: string;
  equationNote: string;
  focused: boolean;
  note: string;
  onClick: () => void;
}) {
  const isAdam = tone === "adam";
  const accent = isAdam ? "text-forward" : "text-gradient";
  const borderGrad = isAdam ? "border-forward/40 bg-forward/[0.07]" : "border-gradient/40 bg-gradient/[0.07]";
  const lineColor = isAdam ? "#557A95" : "#9A7358";

  return (
    <button
      onClick={onClick}
      data-active={focused}
      aria-pressed={focused}
      className="selectable w-full p-6 text-left"
    >
      <div className="mb-5 flex items-baseline justify-between gap-2">
        <span className="font-display text-xl font-semibold text-ink">{name}</span>
        <span className="font-mono text-2xs uppercase tracking-label text-muted">{kind}</span>
      </div>

      {/* Transformation flow */}
      <div className="mx-auto max-w-[220px]">
        {/* Gradient box */}
        <div className={`rounded border px-3 py-2.5 text-center ${borderGrad}`}>
          <div className="font-mono text-[0.6rem] uppercase tracking-wider text-muted/70">
            Gradient-norm ratio r_g
          </div>
          <div className={`font-mono text-2xl font-semibold tabular-nums ${accent}`}>
            {rG.toFixed(1)}×
          </div>
        </div>

        {/* Arrow + optimizer label */}
        <div className="flex flex-col items-center gap-1 py-2.5">
          <div
            className={`font-mono text-[0.6rem] uppercase tracking-wider ${isAdam ? "text-forward" : "text-gradient"}`}
          >
            {name}
          </div>
          <div style={{ width: 1, height: 18, background: lineColor, opacity: 0.35 }} />
          <div className="font-mono text-[0.7rem] italic text-ink/80">{equation}</div>
          <div className="font-mono text-[0.58rem] uppercase tracking-wider text-muted/60">
            {equationNote}
          </div>
          <div style={{ width: 1, height: 18, background: lineColor, opacity: 0.35 }} />
        </div>

        {/* Update box */}
        <div className={`rounded border px-3 py-2.5 text-center ${borderGrad}`}>
          <div className="font-mono text-[0.6rem] uppercase tracking-wider text-muted/70">
            Parameter-update ratio r_Δθ
          </div>
          <div className={`font-mono text-[2.25rem] font-semibold leading-none tabular-nums ${accent}`}>
            {rTheta >= 10 ? rTheta.toFixed(1) : rTheta.toFixed(2)}×
          </div>
        </div>
      </div>

      <p className="mt-4 text-2xs leading-relaxed text-muted">{note}</p>
    </button>
  );
}

/* ── Gradient-only arm detail (expandable) ───────────────────────── */

function GradientOnlyDetail() {
  const [open, setOpen] = useState(false);

  return (
    <section>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between rounded-md border border-line bg-surface px-5 py-3 transition-colors hover:border-ink/25"
        aria-expanded={open}
      >
        <span className="label">Gradient-only arm — intervention detail</span>
        <ChevronDown
          size={14}
          className={`text-muted transition-transform duration-200 ${open ? "rotate-180" : ""}`}
        />
      </button>

      {open && (
        <div className="mt-2 animate-fade-in rounded-md border border-line bg-surface p-5">
          {/* Intervention structure */}
          <div className="mb-5 grid gap-3 sm:grid-cols-3">
            {[
              { field: "Arm", value: "Gradient-only" },
              { field: "Forward representation", value: "Unchanged — α = 1 for forward path" },
              { field: "Backward gradient", value: "Scaled by α = 100×" },
            ].map(({ field, value }) => (
              <div key={field} className="rounded border border-line bg-canvas/50 px-3 py-2.5">
                <div className="mb-1 font-mono text-[0.6rem] uppercase tracking-wider text-muted/70">
                  {field}
                </div>
                <div className="text-2xs font-medium text-ink">{value}</div>
              </div>
            ))}
          </div>

          {/* Observed ratios: Adam vs SGD side by side */}
          <div className="label mb-3">Observed ratios — Adam vs. SGD</div>
          <div className="grid gap-4 sm:grid-cols-2">
            {/* Adam */}
            <div className="rounded border border-forward/25 bg-forward/[0.04] p-4">
              <div className="mb-3 font-mono text-xs font-semibold text-forward">Adam</div>
              <div className="space-y-2 font-mono text-2xs text-muted">
                <div className="flex justify-between">
                  <span>Gradient-norm ratio r_g</span>
                  <span className="text-ink">{adamR.rG.toFixed(1)}×</span>
                </div>
                <div className="flex justify-between">
                  <span>Parameter-update ratio r_Δθ</span>
                  <span className="text-ink">{adamR.rTheta.toFixed(2)}×</span>
                </div>
                <div className="border-t border-line pt-2">
                  <div className="flex justify-between">
                    <span>Gradient-only Δ𝒟</span>
                    <span className="text-ink">
                      {adamGrad.dDom >= 0 ? "+" : "−"}{Math.abs(adamGrad.dDom).toFixed(4)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Cohen's d_z</span>
                    <span className="text-ink">{adamGrad.d.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>p-value</span>
                    <span className="text-ink">{pValue(adamGrad.p)}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* SGD */}
            <div className="rounded border border-gradient/25 bg-gradient/[0.04] p-4">
              <div className="mb-3 font-mono text-xs font-semibold text-gradient">SGD</div>
              <div className="space-y-2 font-mono text-2xs text-muted">
                <div className="flex justify-between">
                  <span>Gradient-norm ratio r_g</span>
                  <span className="text-ink">{sgdR.rG.toFixed(1)}×</span>
                </div>
                <div className="flex justify-between">
                  <span>Parameter-update ratio r_Δθ</span>
                  <span className="text-ink">{sgdR.rTheta.toFixed(1)}×</span>
                </div>
                <div className="border-t border-line pt-2">
                  <div className="flex justify-between">
                    <span>Gradient-only Δ𝒟</span>
                    <span className="text-ink">
                      {sgdGrad.dDom >= 0 ? "+" : "−"}{Math.abs(sgdGrad.dDom).toFixed(3)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Cohen's d_z</span>
                    <span className="text-ink">{sgdGrad.d.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>p-value</span>
                    <span className="text-ink">{pValue(sgdGrad.p)}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <p className="mt-4 border-t border-line pt-3 text-2xs leading-relaxed text-muted">
            In the gradient-only arm, only the backward pathway is intervened upon. The forward
            representation entering fusion is identical to the baseline — only the gradient signal and
            resulting parameter updates differ. The optimizer then determines how much of the imposed
            gradient-scale difference reaches the parameter update.
          </p>
          <p className="mt-1.5 font-mono text-[0.6rem] uppercase tracking-wider text-muted/50">
            Synthetic dataset · α=100× · 5 seeds · frozen experimental results
          </p>
        </div>
      )}
    </section>
  );
}

/* ── Main page ───────────────────────────────────────────────────── */

export function OptimizerAnalysis() {
  const [focused, setFocused] = useState<"adam" | "sgd">("adam");

  return (
    <div className="space-y-12">

      {/* ── SECTION 1 — HERO ──────────────────────────────── */}
      <section>
        <SectionHeader
          eyebrow="Optimizer analysis"
          title="Adam vs. SGD"
          subtitle="Why does the same gradient-only intervention produce different parameter updates?"
        />
        <p className="mt-2 max-w-2xl text-[0.925rem] leading-relaxed text-muted">
          The optimizer determines how strongly an imposed gradient-scale difference reaches the
          parameter update. The gradient-only intervention isolates the backward pathway; the
          optimizer then determines how much of that imposed gradient difference carries through.
        </p>
        <div className="mt-5 inline-flex items-center gap-2.5 rounded border border-navy bg-navy px-4 py-2">
          <span className="font-mono text-[0.6rem] uppercase tracking-wider text-white/50">
            Same intervention
          </span>
          <span className="h-1 w-1 rounded-full bg-white/25" />
          <span className="font-mono text-xs font-medium text-white">Gradient-only · α=100×</span>
          <span className="h-1 w-1 rounded-full bg-white/25" />
          <span className="font-mono text-[0.6rem] text-white/40">5 seeds · Synthetic</span>
        </div>
      </section>

      {/* ── SECTION 2 — MECHANISM DIAGRAM ─────────────────── */}
      <section>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <div className="label">Gradient → update mechanism</div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-2xs text-muted/70">Focus</span>
            <div
              className="inline-flex rounded border border-line bg-surface p-0.5"
              role="group"
              aria-label="Select optimizer focus"
            >
              {(["adam", "sgd"] as const).map((o) => (
                <button
                  key={o}
                  onClick={() => setFocused(o)}
                  data-active={focused === o}
                  aria-pressed={focused === o}
                  className="seg data-[active=true]:bg-ink data-[active=true]:text-canvas"
                >
                  {o === "adam" ? "Adam" : "SGD"}
                </button>
              ))}
            </div>
          </div>
        </div>
        <ScientificFigure
          index="Fig. 1"
          caption="Dual optimizer mechanism: same gradient-only intervention meets two different update rules. Dashed thin connector (Adam) indicates attenuation; solid thick connector (SGD) indicates relative pass-through. Line weight is illustrative, not quantitatively proportional."
          source="Synthetic · α=100× · Gradient-only · 5 seeds · frozen experimental results"
        >
          <div className="mx-auto max-w-xl">
            <DualChannelDiagram focused={focused} />
          </div>
        </ScientificFigure>
      </section>

      {/* ── SECTION 3 — TRANSFORMATION PANELS ─────────────── */}
      <section>
        <div className="label mb-4">Gradient-norm vs. parameter-update ratio</div>
        <div className="grid gap-5 lg:grid-cols-2">
          <TransformationPanel
            name="Adam"
            kind="Adaptive optimization"
            tone="adam"
            rG={adamR.rG}
            rTheta={adamR.rTheta}
            equation="m̂, v̂ → Δθ"
            equationNote="adaptive moment normalization"
            focused={focused === "adam"}
            onClick={() => setFocused("adam")}
            note={`Under the tested Adam configuration, the ${adamR.rG.toFixed(1)}× gradient-norm ratio is strongly attenuated, yielding only a ${adamR.rTheta.toFixed(2)}× parameter-update ratio. The adaptive update rule normalizes per-parameter gradient magnitudes.`}
          />
          <TransformationPanel
            name="SGD"
            kind="Non-adaptive optimization"
            tone="sgd"
            rG={sgdR.rG}
            rTheta={sgdR.rTheta}
            equation="Δθ ∝ −ηg"
            equationNote="gradient magnitude directly enters update"
            focused={focused === "sgd"}
            onClick={() => setFocused("sgd")}
            note={`Under the tested SGD configuration, the ${sgdR.rG.toFixed(1)}× gradient-norm ratio largely carries through to a ${sgdR.rTheta.toFixed(1)}× parameter-update ratio. The non-adaptive update passes the gradient-scale difference into the parameter update.`}
          />
        </div>
      </section>

      {/* ── SECTION 4 — WHAT THIS MEANS ────────────────────── */}
      <section>
        <div className="label mb-4">What this means for the gradient pathway</div>
        <div className="grid gap-5 sm:grid-cols-2">
          <Card className="p-5">
            <div className="mb-4 flex flex-wrap items-center gap-2">
              <span className="h-2 w-2 flex-shrink-0 rounded-full bg-forward" />
              <span className="font-display text-base font-semibold text-ink">Adam</span>
              <Tag tone="forward">strong attenuation</Tag>
            </div>
            <div className="space-y-2.5 text-2xs leading-relaxed">
              <div className="text-ink">
                Large gradient-norm ratio (≈{adamR.rG.toFixed(0)}×)
              </div>
              <div className="text-muted">
                ↓ Strong update attenuation under the tested configuration
              </div>
              <div className="font-medium text-muted">
                ↓ Gradient-only pathway weakly expressed in learned contribution
              </div>
            </div>
          </Card>
          <Card className="p-5">
            <div className="mb-4 flex flex-wrap items-center gap-2">
              <span className="h-2 w-2 flex-shrink-0 rounded-full bg-gradient" />
              <span className="font-display text-base font-semibold text-ink">SGD</span>
              <Tag tone="gradient">large update response</Tag>
            </div>
            <div className="space-y-2.5 text-2xs leading-relaxed">
              <div className="text-ink">
                Large gradient-norm ratio (≈{sgdR.rG.toFixed(0)}×)
              </div>
              <div className="text-muted">
                ↓ Much larger update response under the tested configuration
              </div>
              <div className="font-medium text-muted">
                ↓ Gradient pathway can materially contribute to learned contribution
              </div>
            </div>
          </Card>
        </div>
      </section>

      {/* ── SECTION 5 — GRADIENT-ONLY DETAIL ──────────────── */}
      <GradientOnlyDetail />

      {/* ── SECTION 6 — KEY INSIGHT ────────────────────────── */}
      <Callout title="Key insight" tone="neutral">
        A large gradient-norm ratio does not necessarily produce a proportional parameter-update
        ratio. Under the tested Adam configuration, the {adamR.rG.toFixed(1)}× gradient difference
        corresponds to only a {adamR.rTheta.toFixed(2)}× update ratio; under the tested SGD
        configuration, the observed update ratio is {sgdR.rTheta.toFixed(1)}×.{" "}
        <span className="text-muted/80">
          This supports an optimizer-dependent interpretation of the gradient pathway under the
          tested configurations.
        </span>
        <br />
        <span className="mt-1 inline-block font-mono text-[0.6rem] uppercase tracking-wider text-muted/60">
          Scoped to the frozen experimental protocol · not a universal optimizer claim
        </span>
      </Callout>

    </div>
  );
}
