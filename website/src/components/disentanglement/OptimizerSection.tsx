/**
 * Optimizer pathway components (formerly the Optimizer Analysis page), now a section of
 * Disentanglement. All values come from data/research.ts (gradient-only arm, synthetic,
 * α = 100, 5 seeds); the CREMA-D ratios are shown only under a CREMA-D label.
 */
import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { Tag } from "../primitives";
import { decomposition } from "../../data/research";
import { fixed, pValue } from "../../lib/format";

const adamR = decomposition.ratios.adam;
const sgdR = decomposition.ratios.sgd;
const adamGrad = decomposition.adamSynthetic.gradient;
const sgdGrad = decomposition.sgdSynthetic.gradient;
const cremadR = decomposition.ratiosCremad;

export const fmtRatio = (x: number) => (x >= 10 ? x.toFixed(1) : x.toFixed(2));

/* ── Dual-channel SVG mechanism diagram ─────────────────────────── */

export function DualChannelDiagram({ focused }: { focused: "adam" | "sgd" }) {
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
      <rect x="148" y="10" width="264" height="48" rx="6" fill="var(--ink)" />
      <text x="280" y="28" textAnchor="middle" fill="var(--canvas)" fontSize={9}
        fontFamily="'IBM Plex Mono', monospace" letterSpacing={2}>SAME INTERVENTION</text>
      <text x="280" y="48" textAnchor="middle" fill="var(--canvas)" fontSize={13} fontWeight={600}>
        Gradient-only · α=100×
      </text>

      {/* ── Pre-split connector ───────────────────────────── */}
      <path d="M280,58 V72" fill="none" stroke="var(--line)" strokeWidth={1.5} />
      <path d="M280,58 V72" fill="none" stroke="var(--neutral)" strokeWidth={1.75} className="pw-flow" />

      {/* ── Split connectors ──────────────────────────────── */}
      <path d="M280,72 H120 V90" fill="none" stroke="var(--forward)" strokeWidth={1.5}
        style={{ ...T, opacity: aOp }} />
      <path d="M280,72 H440 V90" fill="none" stroke="var(--gradient)" strokeWidth={1.5}
        style={{ ...T, opacity: sOp }} />

      {/* ── ADAM column ───────────────────────────────────── */}
      <g style={{ ...T, opacity: aOp }}>
        {/* Gradient box */}
        <rect x="20" y="90" width="200" height="48" rx="5" fill="var(--surface)" stroke="var(--forward)" strokeWidth={1.3} />
        <text x="120" y="107" textAnchor="middle" fill="var(--muted)" fontSize={8}
          fontFamily="'IBM Plex Mono', monospace" letterSpacing={1}>GRADIENT-NORM RATIO r_g</text>
        <text x="120" y="128" textAnchor="middle" fill="var(--forward)" fontSize={20} fontWeight={700}
          fontFamily="'IBM Plex Mono', monospace">{adamR.rG.toFixed(1)}×</text>

        {/* Gradient → optimizer connector */}
        <path d="M120,138 V158" fill="none" stroke="var(--forward)" strokeWidth={1.5} />

        {/* Optimizer box */}
        <rect x="20" y="158" width="200" height="72" rx="5"
          fill={aFocused ? "rgb(var(--forward-rgb) / 0.10)" : "var(--surface)"} stroke="var(--forward)" strokeWidth={aFocused ? 1.8 : 1.2} />
        <text x="120" y="177" textAnchor="middle" fill="var(--forward)" fontSize={11} fontWeight={700}
          fontFamily="'IBM Plex Mono', monospace" letterSpacing={2}>ADAM</text>
        <text x="120" y="193" textAnchor="middle" fill="var(--muted)" fontSize={9.5}>Adaptive optimization</text>
        <text x="120" y="211" textAnchor="middle" fill="var(--ink)" fontSize={10.5} fontStyle="italic">
          m̂, v̂ → Δθ
        </text>
        <text x="120" y="224" textAnchor="middle" fill="var(--muted)" fontSize={8}
          fontFamily="'IBM Plex Mono', monospace">adaptive moment normalization</text>

        {/* KEY VISUAL: dashed thin connector (attenuated) */}
        <path d="M120,230 V252" fill="none" stroke="var(--forward)" strokeWidth={1.5} strokeDasharray="5 3" />

        {/* Update box */}
        <rect x="20" y="252" width="200" height="66" rx="5" fill="var(--surface)" stroke="var(--forward)"
          strokeWidth={aFocused ? 1.8 : 1.2} />
        <text x="120" y="269" textAnchor="middle" fill="var(--muted)" fontSize={8}
          fontFamily="'IBM Plex Mono', monospace" letterSpacing={1}>PARAMETER-UPDATE RATIO r_Δθ</text>
        <text x="120" y="299" textAnchor="middle" fill="var(--forward)" fontSize={26} fontWeight={700}
          fontFamily="'IBM Plex Mono', monospace">{adamR.rTheta.toFixed(2)}×</text>
        <text x="120" y="313" textAnchor="middle" fill="var(--forward)" fontSize={8}
          fontFamily="'IBM Plex Mono', monospace" opacity={0.7}>strong attenuation</text>

        {/* Merge connector from Adam */}
        <path d="M120,318 V332 H280" fill="none" stroke="var(--forward)" strokeWidth={1.5} />
      </g>

      {/* ── SGD column ────────────────────────────────────── */}
      <g style={{ ...T, opacity: sOp }}>
        {/* Gradient box */}
        <rect x="340" y="90" width="200" height="48" rx="5" fill="var(--surface)" stroke="var(--gradient)" strokeWidth={1.3} />
        <text x="440" y="107" textAnchor="middle" fill="var(--muted)" fontSize={8}
          fontFamily="'IBM Plex Mono', monospace" letterSpacing={1}>GRADIENT-NORM RATIO r_g</text>
        <text x="440" y="128" textAnchor="middle" fill="var(--gradient)" fontSize={20} fontWeight={700}
          fontFamily="'IBM Plex Mono', monospace">{sgdR.rG.toFixed(1)}×</text>

        {/* Gradient → optimizer connector */}
        <path d="M440,138 V158" fill="none" stroke="var(--gradient)" strokeWidth={1.5} />

        {/* Optimizer box */}
        <rect x="340" y="158" width="200" height="72" rx="5"
          fill={!aFocused ? "rgb(var(--gradient-rgb) / 0.10)" : "var(--surface)"} stroke="var(--gradient)" strokeWidth={!aFocused ? 1.8 : 1.2} />
        <text x="440" y="177" textAnchor="middle" fill="var(--gradient)" fontSize={11} fontWeight={700}
          fontFamily="'IBM Plex Mono', monospace" letterSpacing={2}>SGD</text>
        <text x="440" y="193" textAnchor="middle" fill="var(--muted)" fontSize={9.5}>Non-adaptive optimization</text>
        <text x="440" y="211" textAnchor="middle" fill="var(--ink)" fontSize={10.5} fontStyle="italic">
          Δθ ∝ −ηg
        </text>
        <text x="440" y="224" textAnchor="middle" fill="var(--muted)" fontSize={8}
          fontFamily="'IBM Plex Mono', monospace">direct gradient update</text>

        {/* KEY VISUAL: thick solid connector (pass-through) */}
        <path d="M440,230 V252" fill="none" stroke="var(--gradient)" strokeWidth={3.5} />

        {/* Update box */}
        <rect x="340" y="252" width="200" height="66" rx="5" fill="var(--surface)" stroke="var(--gradient)"
          strokeWidth={!aFocused ? 1.8 : 1.2} />
        <text x="440" y="269" textAnchor="middle" fill="var(--muted)" fontSize={8}
          fontFamily="'IBM Plex Mono', monospace" letterSpacing={1}>PARAMETER-UPDATE RATIO r_Δθ</text>
        <text x="440" y="299" textAnchor="middle" fill="var(--gradient)" fontSize={26} fontWeight={700}
          fontFamily="'IBM Plex Mono', monospace">{sgdR.rTheta.toFixed(1)}×</text>
        <text x="440" y="313" textAnchor="middle" fill="var(--gradient)" fontSize={8}
          fontFamily="'IBM Plex Mono', monospace" opacity={0.7}>large update response</text>

        {/* Merge connector from SGD */}
        <path d="M440,318 V332 H280" fill="none" stroke="var(--gradient)" strokeWidth={1.5} />
      </g>

      {/* ── Post-merge connector ──────────────────────────── */}
      <path d="M280,332 V348" fill="none" stroke="var(--line)" strokeWidth={1.5} />
      <path d="M280,332 V348" fill="none" stroke="var(--neutral)" strokeWidth={1.75} className="pw-flow" />

      {/* ── LEARNED CONTRIBUTION ─────────────────────────── */}
      <rect x="148" y="348" width="264" height="54" rx="6" fill="var(--ink)" />
      <text x="280" y="368" textAnchor="middle" fill="var(--canvas)" fontSize={9.5}
        fontFamily="'IBM Plex Mono', monospace" letterSpacing={1.5}>OUTCOME</text>
      <text x="280" y="390" textAnchor="middle" fill="var(--canvas)" fontSize={14} fontWeight={600}>
        Learned modality contribution
      </text>
    </svg>
  );
}

/* ── Transformation panel ────────────────────────────────────────── */

export function TransformationPanel({
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
  const lineColor = isAdam ? "var(--forward)" : "var(--gradient)";

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

export function GradientOnlyDetail() {
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
                    <span className="text-ink">{fixed(adamGrad.d, 2)}</span>
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
                    <span className="text-ink">{fixed(sgdGrad.d, 2)}</span>
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

/* ── Optimizer comparison table (selected column follows the page's optimizer) ── */

export function OptimizerComparison({ focused, onFocus }: { focused: "adam" | "sgd"; onFocus: (o: "adam" | "sgd") => void }) {
  const cols = {
    adam: { dec: decomposition.adamSynthetic, r: adamR, regime: "Forward-dominant" },
    sgd: { dec: decomposition.sgdSynthetic, r: sgdR, regime: "Both pathways contribute" },
  } as const;
  const sgn = (x: number, dg = 3) => (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(dg);
  const rows: { k: string; v: (o: "adam" | "sgd") => string }[] = [
    { k: "Gradient-norm ratio r_g (gradient-only)", v: (o) => `${fmtRatio(cols[o].r.rG)}×` },
    { k: "Parameter-update ratio r_Δθ (gradient-only)", v: (o) => `${fmtRatio(cols[o].r.rTheta)}×` },
    { k: "Gradient-only Δ𝒟", v: (o) => sgn(cols[o].dec.gradient.dDom, Math.abs(cols[o].dec.gradient.dDom) < 0.001 ? 4 : 3) },
    { k: "Gradient-only d_z · p", v: (o) => `${fixed(cols[o].dec.gradient.d, 2)} · ${pValue(cols[o].dec.gradient.p)}` },
    { k: "Forward-only Δ𝒟 · d_z", v: (o) => `${sgn(cols[o].dec.forward.dDom)} · ${fixed(cols[o].dec.forward.d, 2)}` },
    { k: "Ordinary Δ𝒟 · d_z", v: (o) => `${sgn(cols[o].dec.ordinary.dDom)} · ${fixed(cols[o].dec.ordinary.d, 2)}` },
    { k: "Pathway regime (tested configuration)", v: (o) => cols[o].regime },
  ];
  return (
    <div className="card overflow-hidden" data-testid="optimizer-comparison">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[520px] text-left text-2xs">
          <thead>
            <tr className="border-b border-line bg-canvas/50">
              <th className="px-4 py-2 font-mono font-normal uppercase tracking-wider text-muted">Synthetic · α = 100 · 5 seeds</th>
              {(["adam", "sgd"] as const).map((o) => (
                <th key={o} className="px-4 py-2">
                  <button
                    type="button"
                    onClick={() => onFocus(o)}
                    aria-pressed={focused === o}
                    className={`rounded px-2 py-0.5 font-mono text-2xs uppercase tracking-wider ${
                      focused === o ? "bg-ink text-canvas" : "text-muted hover:text-ink"
                    }`}
                  >
                    {o === "adam" ? "Adam" : "SGD"}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.k} className="border-b border-line last:border-0">
                <th scope="row" className="px-4 py-2 font-normal text-muted">{r.k}</th>
                {(["adam", "sgd"] as const).map((o) => (
                  <td
                    key={o}
                    data-col={o}
                    className={`px-4 py-2 font-mono transition-colors ${focused === o ? "bg-ink/[0.04] font-semibold text-ink" : "text-muted"}`}
                  >
                    {r.v(o)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="border-t border-line px-4 py-2 font-mono text-[0.6rem] uppercase tracking-wider text-muted/70">
        Sources: results/disentangle, results/sgd_disentangle, results/fusion_robustness (Adam r_Δθ) · paired t-tests
      </p>
    </div>
  );
}

export function MeaningCard({ opt }: { opt: "adam" | "sgd" }) {
  const isAdam = opt === "adam";
  const r = isAdam ? adamR : sgdR;
  return (
    <div className="card p-5" data-testid="meaning-card">
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <span className={`h-2 w-2 flex-shrink-0 rounded-full ${isAdam ? "bg-forward" : "bg-gradient"}`} />
        <span className="font-display text-base font-semibold text-ink">{isAdam ? "Adam" : "SGD"}</span>
        <Tag tone={isAdam ? "forward" : "gradient"}>{isAdam ? "strong attenuation" : "large update response"}</Tag>
      </div>
      <div className="space-y-2.5 text-2xs leading-relaxed">
        <div className="text-ink">Large gradient-norm ratio (≈{fmtRatio(r.rG)}×)</div>
        <div className="text-muted">
          ↓ {isAdam ? `Update ratio stays ≈${fmtRatio(r.rTheta)}× under the tested configuration` : `Update ratio ≈${fmtRatio(r.rTheta)}× under the tested configuration`}
        </div>
        <div className="font-medium text-muted">
          ↓ {isAdam ? "Gradient-only arm moves no contribution metric (not individually significant)" : "Gradient-only arm produces a significant dominance shift"}
        </div>
      </div>
      <p className="mt-3 border-t border-line pt-2 text-2xs leading-relaxed text-muted">
        {isAdam
          ? "Why (tested Adam, ε = 10⁻⁸): Adam divides each parameter's step by a running estimate of its own gradient magnitude, so multiplying the encoder's gradient by a roughly constant factor largely cancels out of the update."
          : "Why (tested SGD, momentum 0.9): the step is proportional to the gradient itself, so the multiplied gradient reaches the weights almost in full."}
      </p>
    </div>
  );
}

export function CremadRatioNote() {
  return (
    <p className="rounded border border-line bg-surface/60 px-4 py-3 text-2xs leading-relaxed text-muted" data-testid="cremad-ratio-note">
      <span className="font-semibold text-ink">Real-data replication (CREMA-D · Adam · α = 100 · 5 seeds):</span> the gradient-only arm
      gives a {fmtRatio(cremadR.rG)}× gradient-norm ratio and a {fmtRatio(cremadR.rTheta)}× update ratio. The CREMA-D baseline's own update ratio
      is {fmtRatio(cremadR.baselineRTheta)}×, so the imposed gradient scale leaves the updates unchanged here too (results/cremad_disentangle).
    </p>
  );
}
