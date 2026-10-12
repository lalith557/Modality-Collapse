import { useState } from "react";
import { useResearch } from "../state/research-context";
import { SectionHeader, Callout, Card, Tag } from "../components/primitives";
import { ScientificFigure } from "../components/ScientificFigure";
import { AnimatedNumber } from "../components/AnimatedNumber";
import { fusion, fusionByKey, type FusionId } from "../data/research";
import { pValue } from "../lib/format";
import { C } from "../charts/theme";

/* ── Mechanism SVG diagram ───────────────────────────────────────── */
function MechanismSVG({ kind }: { kind: "sum" | "concat" | "gated" }) {
  const opSymbol = { sum: "⊕", concat: "‖", gated: "◉" }[kind];
  const opColor = { sum: C.forward, concat: C.neutral, gated: C.gradient }[kind];
  // Key geometry: inputs at right edge x=50, op circle at cx=91 cy=44 r=14
  // Diagonal connectors terminate exactly at the circle's left tangent point
  const opCx = 91, opCy = 44, opR = 14;
  const circleLeft = opCx - opR;   // 77
  const circleRight = opCx + opR;  // 105
  const haY = 23, hbY = 65, inputRight = 50;
  return (
    <svg
      viewBox="0 0 180 88"
      className="h-auto w-full"
      role="img"
      aria-label={`${kind} fusion: hₐ and hᵦ combined via ${opSymbol} to produce ŷ`}
    >
      <defs>
        <marker id={`arr-${kind}`} markerWidth="5" markerHeight="4" refX="4" refY="2" orient="auto">
          <polygon points="0 0, 5 2, 0 4" fill="#B9B5AA" />
        </marker>
      </defs>

      {/* hₐ input box */}
      <rect x="8" y="10" width="42" height="26" rx="4" fill={C.forward} fillOpacity="0.08" stroke={C.forward} strokeWidth="1.25" />
      <text x="29" y="27" textAnchor="middle" fontFamily="'IBM Plex Mono', monospace" fontSize="12" fill={C.forward}>hₐ</text>

      {/* hᵦ input box */}
      <rect x="8" y="52" width="42" height="26" rx="4" fill={C.gradient} fillOpacity="0.08" stroke={C.gradient} strokeWidth="1.25" />
      <text x="29" y="69" textAnchor="middle" fontFamily="'IBM Plex Mono', monospace" fontSize="12" fill={C.gradient}>hᵦ</text>

      {/* diagonal connector: hₐ → op circle */}
      <line
        x1={inputRight} y1={haY} x2={circleLeft} y2={opCy}
        stroke={C.forward} strokeWidth="1.3" strokeOpacity="0.55"
      />
      {/* diagonal connector: hᵦ → op circle */}
      <line
        x1={inputRight} y1={hbY} x2={circleLeft} y2={opCy}
        stroke={C.gradient} strokeWidth="1.3" strokeOpacity="0.55"
      />

      {/* output line: op circle → ŷ */}
      <line
        x1={circleRight} y1={opCy} x2={133} y2={opCy}
        stroke="#B9B5AA" strokeWidth="1.3"
        markerEnd={`url(#arr-${kind})`}
      />

      {/* op node */}
      <circle cx={opCx} cy={opCy} r={opR} fill={C.surface} stroke={opColor} strokeWidth="1.6" />
      <text x={opCx} y={opCy + 5.5} textAnchor="middle" fontFamily="'IBM Plex Mono', monospace" fontSize="15" fill={opColor}>{opSymbol}</text>

      {/* prediction output box */}
      <rect x="136" y="31" width="36" height="26" rx="4" fill="#17232B" />
      <text x="154" y="48" textAnchor="middle" fontFamily="'IBM Plex Mono', monospace" fontSize="12" fill="#F5F3EE">ŷ</text>
    </svg>
  );
}

/* ── Forward-effect magnitude chart ─────────────────────────────── */
const MAX_MAG = Math.max(...fusion.mechanisms.map((m) => Math.abs(m.forwardD)));

function ForwardMagnitudeChart({ active }: { active: FusionId }) {
  const [hovered, setHovered] = useState<string | null>(null);
  const gridTicks = [1, 2, 3, 4, 5];

  return (
    <div className="select-none py-3">
      {/* X-axis tick labels */}
      <div className="relative mb-2 flex items-center">
        <div className="w-20 shrink-0" />
        <div className="relative h-4 flex-1">
          {gridTicks.map((t) => (
            <span
              key={t}
              className="absolute -translate-x-1/2 font-mono text-2xs text-muted/50"
              style={{ left: `${(t / MAX_MAG) * 100}%` }}
            >
              {t}
            </span>
          ))}
        </div>
        <div className="w-24 shrink-0" />
      </div>

      {/* Rows */}
      <div className="space-y-5">
        {fusion.mechanisms.map((m) => {
          const isActive = m.key === active;
          const isHovered = hovered === m.key;
          const mag = Math.abs(m.forwardD);
          const pct = (mag / MAX_MAG) * 100;

          return (
            <div key={m.key} className="relative">
              <div
                className="flex items-center"
                onMouseEnter={() => setHovered(m.key)}
                onMouseLeave={() => setHovered(null)}
              >
                {/* Name */}
                <div
                  className={`w-20 shrink-0 pr-3 text-right font-mono text-xs transition-colors duration-200 ${
                    isActive ? "font-semibold text-ink" : "text-muted/60"
                  }`}
                >
                  {m.name}
                </div>

                {/* Track */}
                <div className="relative h-1.5 flex-1 rounded-full bg-line/50">
                  {/* Grid tick marks */}
                  {gridTicks.map((t) => (
                    <div
                      key={t}
                      className="absolute top-1/2 h-3 w-px -translate-y-1/2 bg-line"
                      style={{ left: `${(t / MAX_MAG) * 100}%` }}
                    />
                  ))}
                  {/* Bar fill */}
                  <div
                    className="absolute inset-y-0 left-0 rounded-full"
                    style={{
                      width: `${pct}%`,
                      backgroundColor: isActive ? C.forward : "#C2D3DE",
                      opacity: isActive ? 1 : 0.35,
                      transition: "width 400ms ease-out, background-color 200ms ease-out, opacity 200ms ease-out",
                    }}
                  />
                  {/* Lollipop dot */}
                  <div
                    className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full transition-all duration-300"
                    style={{
                      left: `${pct}%`,
                      width: isActive ? "14px" : "10px",
                      height: isActive ? "14px" : "10px",
                      backgroundColor: isActive ? C.forward : "#C2D3DE",
                      opacity: isActive ? 1 : 0.35,
                      boxShadow: isActive ? `0 0 0 3px ${C.forward}30` : undefined,
                    }}
                  />
                </div>

                {/* Values */}
                <div className="w-24 shrink-0 pl-4">
                  <div
                    className={`font-mono text-xs font-semibold transition-colors duration-200 ${
                      isActive && m.significant
                        ? "text-forward"
                        : isActive
                        ? "text-warning"
                        : "text-muted/40"
                    }`}
                  >
                    |d| = {mag.toFixed(2)}
                  </div>
                  <div
                    className={`font-mono text-2xs transition-colors duration-200 ${
                      isActive ? "text-muted" : "text-muted/30"
                    }`}
                  >
                    d = {m.forwardD.toFixed(2)}
                  </div>
                </div>
              </div>

              {/* Hover tooltip */}
              {isHovered && (
                <div className="absolute left-20 top-full z-10 mt-1.5 rounded border border-line bg-surface px-3 py-2 shadow-sm">
                  <div className="mb-0.5 font-mono text-xs font-semibold text-ink">{m.name} fusion</div>
                  <div className="font-mono text-2xs text-muted">|dz| = {mag.toFixed(2)}</div>
                  <div className="font-mono text-2xs text-muted">dz = {m.forwardD.toFixed(2)}</div>
                  <div className="font-mono text-2xs text-muted">p = {pValue(m.p)}</div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* X-axis label */}
      <div className="mt-5 flex items-center">
        <div className="w-20 shrink-0" />
        <div className="flex-1 text-center font-mono text-2xs text-muted/50">
          Effect magnitude |dz|
        </div>
      </div>
    </div>
  );
}

/* ── Gradient-null bar visual ────────────────────────────────────── */
function GradientNullBar({ name, active }: { name: string; active: boolean }) {
  const [val, setVal] = useState(0);
  // map [-1, 1] → [0%, 100%] for thumb position
  const thumbPct = ((val + 1) / 2) * 100;
  // fill runs from 50% toward the thumb
  const fillLeft = val >= 0 ? 50 : thumbPct;
  const fillWidth = Math.abs(val) * 50;

  return (
    <div className="flex items-center gap-3">
      <span
        className={`w-20 shrink-0 font-mono text-2xs transition-colors duration-200 ${
          active ? "font-medium text-ink" : "text-muted"
        }`}
      >
        {name}
      </span>

      {/* Track */}
      <div className="relative h-1.5 flex-1 rounded-full bg-line">
        {/* Center marker */}
        <div className="absolute left-1/2 top-1/2 h-2.5 w-px -translate-x-1/2 -translate-y-1/2 bg-line" />
        {/* Fill from center toward thumb */}
        <div
          className="absolute top-0 h-full rounded-full bg-gradient/40 transition-all duration-75"
          style={{ left: `${fillLeft}%`, width: `${fillWidth}%` }}
        />
        {/* Invisible native range input — drives interaction */}
        <input
          type="range"
          min={-1}
          max={1}
          step={0.001}
          value={val}
          onChange={(e) => setVal(Number(e.target.value))}
          aria-label={`${name} gradient effect (drag to explore)`}
          className="absolute inset-0 h-full w-full cursor-grab opacity-0 active:cursor-grabbing"
          style={{ margin: 0 }}
        />
        {/* Custom thumb dot */}
        <div
          className="pointer-events-none absolute top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 bg-surface transition-all duration-75"
          style={{
            left: `${thumbPct}%`,
            width: active ? "14px" : "12px",
            height: active ? "14px" : "12px",
            borderColor: active ? C.gradient : "#B9B5AA",
            boxShadow: active ? `0 0 0 2px ${C.gradient}28` : undefined,
          }}
        />
      </div>

      {/* Value display */}
      <span
        className={`w-16 shrink-0 text-right font-mono text-2xs transition-colors duration-200 ${
          active ? "text-ink" : "text-muted"
        }`}
      >
        {Math.abs(val) < 0.0015
          ? "≈ 0.000"
          : val > 0
          ? `+${val.toFixed(3)}`
          : val.toFixed(3)}
      </span>
    </div>
  );
}

/* ── Selected mechanism detail ───────────────────────────────────── */
function DetailPanel({ mechanism }: { mechanism: (typeof fusion.mechanisms)[number] }) {
  const isGated = mechanism.key === "gated";
  return (
    <div className="animate-fade-in space-y-4">
      <div className="grid gap-4 sm:grid-cols-2">
        {/* Diagram card */}
        <Card className="p-5">
          <div className="label mb-3">{mechanism.name} fusion — architecture</div>
          <div className="mx-auto max-w-[210px] py-2">
            <MechanismSVG kind={mechanism.key as "sum" | "concat" | "gated"} />
          </div>
          <p className="mt-3 text-center font-mono text-2xs text-muted">{mechanism.diagram}</p>
        </Card>

        {/* Stats grid */}
        <div className="grid grid-rows-3 gap-3">
          <Card className="p-4">
            <div className="text-2xs uppercase tracking-wider text-muted">Forward effect (Cohen's dz)</div>
            <AnimatedNumber
              key={mechanism.key + "_d"}
              value={mechanism.forwardD}
              digits={2}
              signed
              className={`measure text-[1.9rem] font-semibold leading-none mt-1 ${mechanism.significant ? "text-forward" : "text-muted"}`}
            />
          </Card>
          <Card className="p-4">
            <div className="text-2xs uppercase tracking-wider text-muted">Two-sided p-value</div>
            <div className={`measure text-[1.9rem] font-semibold leading-none mt-1 ${mechanism.significant ? "text-ink" : "text-warning"}`}>
              p = {pValue(mechanism.p)}
            </div>
          </Card>
          <Card className="p-4">
            <div className="text-2xs uppercase tracking-wider text-muted">Gradient-only arm</div>
            <div className="measure text-xl font-semibold leading-none mt-1 text-gradient">{fusion.gradientNull}</div>
            <div className="mt-1 text-2xs text-muted">No measurable shift</div>
          </Card>
        </div>
      </div>

      {/* Interpretation callout */}
      <Callout
        tone={isGated ? "warning" : "forward"}
        title={isGated ? "Attenuated effect — boundary case" : "Interpretation"}
      >
        {isGated ? (
          <>
            The forward effect is attenuated and not individually significant under gated fusion (d = −0.70,
            p = 0.19). This is consistent with the possibility that learned per-modality gates can partly
            compensate for the imposed forward magnitude within the tested configuration.{" "}
            <em className="not-italic text-muted">
              Gate values were not directly analyzed in this experiment.
            </em>
          </>
        ) : mechanism.key === "sum" ? (
          "Strong forward-pathway effect under sum fusion (d = −3.76, p ≈ 0.001). The additive combination does not attenuate the imposed magnitude difference."
        ) : (
          "The strongest forward-pathway effect across the three tested mechanisms appears under concatenation (d = −5.16, p ≈ 0.0003). Forward-only scaling has a large, statistically significant contribution shift."
        )}
      </Callout>
    </div>
  );
}

/* ── Main page ───────────────────────────────────────────────────── */
export function FusionUtility() {
  const s = useResearch();
  const active = fusionByKey(s.fusion);
  const [gradientReset, setGradientReset] = useState(0);

  return (
    <div className="space-y-10">

      {/* ── 1–5: Header ───────────────────────────────── */}
      <SectionHeader
        eyebrow="Fusion & Utility"
        title="Does Fusion Change the Pathway?"
        subtitle="Same intervention. Different fusion."
      />

      <div className="space-y-3 -mt-4">
        <p className="max-w-2xl text-[0.95rem] leading-relaxed text-muted">
          The matched forward-only and gradient-only interventions are repeated under sum, concatenation,
          and gated fusion to determine whether the observed contribution shift depends on how modalities
          are combined.
        </p>
        <div className="inline-flex items-center gap-2 rounded border border-line bg-canvas/80 px-3 py-1.5">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-muted/50" />
          <span className="font-mono text-2xs uppercase tracking-wider text-muted">
            Synthetic · α = 100 · Adam · 5 seeds
          </span>
        </div>
      </div>

      {/* ── 6: Fixed / Varied strip ───────────────────── */}
      <div className="flex flex-wrap items-stretch gap-px overflow-hidden rounded-md border border-line bg-line">
        <div className="flex flex-1 flex-col gap-1 bg-canvas/60 px-5 py-3.5 min-w-[160px]">
          <span className="label">Fixed</span>
          <span className="font-mono text-xs text-ink">Adam · α=100 · 5 seeds · Synthetic</span>
        </div>
        <div className="flex flex-1 flex-col gap-1 bg-canvas/60 px-5 py-3.5 min-w-[160px]">
          <span className="label">Varied</span>
          <span className="font-mono text-xs text-ink">Sum ↔ Concat ↔ Gated</span>
        </div>
      </div>

      {/* ── 7–8: Mechanism cards ─────────────────────── */}
      <section>
        <div className="label mb-3">Select Fusion Mechanism</div>
        <div className="grid gap-4 md:grid-cols-3">
          {fusion.mechanisms.map((m) => (
            <button
              key={m.key}
              onClick={() => s.set("fusion", m.key as FusionId)}
              data-active={s.fusion === m.key}
              aria-pressed={s.fusion === m.key}
              aria-label={`${m.name} fusion, forward effect d = ${m.forwardD.toFixed(2)}`}
              className="selectable p-5 text-left focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-forward"
            >
              <div className="mb-3 flex items-start justify-between gap-2">
                <div>
                  <h3 className="font-display text-[1.05rem] font-semibold text-ink">{m.name}</h3>
                  <div className="mt-0.5 font-mono text-2xs text-muted">
                    {m.key === "sum" ? "hₐ + hᵦ → ŷ" : m.key === "concat" ? "[hₐ ‖ hᵦ] → ŷ" : "gate(hₐ, hᵦ) → ŷ"}
                  </div>
                </div>
                {m.significant ? (
                  <Tag tone="forward">Strong</Tag>
                ) : (
                  <Tag tone="warning">Attenuated · N.S.</Tag>
                )}
              </div>

              <div className="my-3 rounded border border-line bg-canvas/50 px-3 py-2">
                <MechanismSVG kind={m.key as "sum" | "concat" | "gated"} />
              </div>

              <div className="flex items-baseline justify-between border-t border-line pt-3 mt-1">
                <span className="text-2xs uppercase tracking-wider text-muted">Forward effect</span>
                <span
                  className={`measure text-sm font-semibold ${
                    m.significant ? "text-forward" : "text-muted"
                  }`}
                >
                  d = {m.forwardD.toFixed(2)}
                </span>
              </div>
            </button>
          ))}
        </div>
      </section>

      {/* ── 9–12: Comparison visualization ───────────── */}
      <section className="space-y-4">
        <div>
          <div className="label mb-0.5">Forward-Pathway Effect Strength</div>
          <p className="text-2xs text-muted">
            Forward-only intervention · Effect magnitude |d<sub>z</sub>| · Active mechanism highlighted
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-[3fr_2fr]">
          {/* Forward-effect chart */}
          <ScientificFigure
            index="Forward-only effect magnitude"
            caption="Effect magnitude |dz| under forward-only intervention per fusion mechanism. Bar length encodes |dz|; signed Cohen's dz retained alongside. Active mechanism highlighted."
            source="Synthetic · α = 100 · Adam · 5 seeds"
          >
            <ForwardMagnitudeChart active={s.fusion} />
            {/* Exact values row */}
            <div className="mt-2 grid grid-cols-3 divide-x divide-line border-t border-line">
              {fusion.mechanisms.map((m) => (
                <div
                  key={m.key}
                  className={`px-3 py-2.5 text-center transition-colors duration-200 ${
                    s.fusion === m.key ? "bg-forward/[0.04]" : ""
                  }`}
                >
                  <div className="font-mono text-2xs text-muted">{m.name}</div>
                  <div
                    className={`measure text-sm font-semibold ${
                      m.significant
                        ? s.fusion === m.key
                          ? "text-forward"
                          : "text-ink/60"
                        : s.fusion === m.key
                        ? "text-warning"
                        : "text-muted/60"
                    }`}
                  >
                    d = {m.forwardD.toFixed(2)}
                  </div>
                  <div className="font-mono text-2xs text-muted/70">p = {pValue(m.p)}</div>
                </div>
              ))}
            </div>
          </ScientificFigure>

          {/* Gradient-only null */}
          <div className="space-y-3">
            <div className="flex items-start justify-between gap-2">
              <div>
                <div className="label mb-0.5">Gradient-only Intervention</div>
                <p className="text-2xs text-muted">Approximately null across all three mechanisms</p>
              </div>
              <button
                onClick={() => setGradientReset((k) => k + 1)}
                className="shrink-0 rounded border border-line bg-canvas/80 px-2.5 py-1 font-mono text-2xs text-muted transition-colors hover:border-forward/40 hover:text-forward active:scale-95"
              >
                Reset
              </button>
            </div>
            <Card className="p-5">
              <div className="measure text-2xl font-semibold text-gradient mb-1">
                {fusion.gradientNull}
              </div>
              <p className="text-xs leading-relaxed text-muted mb-4">
                No measurable contribution shift is observed across the three tested fusion mechanisms
                under the gradient-only intervention.
              </p>
              <div className="space-y-3 border-t border-line pt-4">
                {fusion.mechanisms.map((m) => (
                  <GradientNullBar
                    key={`${m.key}-${gradientReset}`}
                    name={m.name}
                    active={s.fusion === m.key}
                  />
                ))}
              </div>
              <p className="mt-4 font-mono text-2xs text-muted/70 border-t border-line pt-3">
                Update ratio r<sub>Δθ</sub> ≈ 1 across all mechanisms
              </p>
            </Card>
          </div>
        </div>
      </section>

      {/* ── 13–16: Selected mechanism detail ─────────── */}
      <section>
        <div className="mb-4 flex items-center justify-between">
          <div className="label">
            Selected Mechanism — {active.name}
          </div>
          {active.significant ? (
            <Tag tone="forward">Significant · strong effect</Tag>
          ) : (
            <Tag tone="warning">Not individually significant</Tag>
          )}
        </div>
        <DetailPanel mechanism={active} />
      </section>

      {/* ── 17–18: What this test establishes ────────── */}
      <Callout title="What this test establishes" tone="forward">
        The forward-pathway effect persists strongly under sum and concatenation (d = −3.76 and d = −5.16),
        while becoming attenuated under gated fusion (d = −0.70, p = 0.19). The gradient-only intervention
        remains approximately null across all three tested mechanisms.{" "}
        <span className="text-muted">
          This is evidence of variation across the evaluated fusion mechanisms within the tested configuration,
          not a claim of architecture-general behavior.
        </span>
      </Callout>

      {/* ── 19: Why this matters ──────────────────────── */}
      <section>
        <div className="label mb-3">Why This Matters</div>
        <Card className="p-5">
          <p className="max-w-2xl text-[0.95rem] leading-relaxed text-ink">
            Changing fusion does not erase the forward-pathway signal, but it can change its magnitude.
            Gated fusion provides a useful boundary case because its learned per-modality weighting can be
            consistent with attenuating the observed effect — though gate values were not directly measured.
            The result is informative precisely because it is not null.
          </p>
        </Card>
      </section>
    </div>
  );
}
