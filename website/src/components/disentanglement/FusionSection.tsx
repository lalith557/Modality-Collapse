/**
 * Fusion-mechanism components (formerly the Fusion & Utility page), now the
 * "Does fusion change the pathway?" section of Disentanglement. Every value is from
 * data/research.ts `fusion` (results/fusion_robustness: synthetic, Adam, α = 100, 5 seeds).
 * Gate activations were not logged, so none are shown.
 */
import { useState } from "react";
import { Callout, Card, Tag } from "../primitives";
import { AnimatedNumber } from "../AnimatedNumber";
import { fusion, type FusionId } from "../../data/research";
import { fixed, pValue } from "../../lib/format";
import { C } from "../../charts/theme";

type Mechanism = (typeof fusion.mechanisms)[number];

const sgn = (x: number, dg = 3) => (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(dg);

/* ── Mechanism SVG diagram ───────────────────────────────────────── */
export function MechanismSVG({ kind }: { kind: FusionId }) {
  const opSymbol = { sum: "⊕", concat: "‖", gated: "◉" }[kind];
  const opColor = { sum: C.forward, concat: C.neutral, gated: C.gradient }[kind];
  const opCx = 91, opCy = 44, opR = 14;
  const circleLeft = opCx - opR;
  const circleRight = opCx + opR;
  const haY = 23, hbY = 65, inputRight = 50;
  return (
    <svg viewBox="0 0 180 88" className="h-auto w-full" role="img" aria-label={`${kind} fusion: hₐ and hᵦ combined via ${opSymbol} to produce ŷ`}>
      <defs>
        <marker id={`arr-${kind}`} markerWidth="5" markerHeight="4" refX="4" refY="2" orient="auto">
          <polygon points="0 0, 5 2, 0 4" fill="#B9B5AA" />
        </marker>
      </defs>
      <rect x="8" y="10" width="42" height="26" rx="4" fill={C.forward} fillOpacity="0.08" stroke={C.forward} strokeWidth="1.25" />
      <text x="29" y="27" textAnchor="middle" fontFamily="'IBM Plex Mono', monospace" fontSize="12" fill={C.forward}>hₐ</text>
      <rect x="8" y="52" width="42" height="26" rx="4" fill={C.gradient} fillOpacity="0.08" stroke={C.gradient} strokeWidth="1.25" />
      <text x="29" y="69" textAnchor="middle" fontFamily="'IBM Plex Mono', monospace" fontSize="12" fill={C.gradient}>hᵦ</text>
      <line x1={inputRight} y1={haY} x2={circleLeft} y2={opCy} stroke={C.forward} strokeWidth="1.3" strokeOpacity="0.55" />
      <line x1={inputRight} y1={hbY} x2={circleLeft} y2={opCy} stroke={C.gradient} strokeWidth="1.3" strokeOpacity="0.55" />
      <line x1={circleRight} y1={opCy} x2={133} y2={opCy} stroke="#B9B5AA" strokeWidth="1.3" markerEnd={`url(#arr-${kind})`} />
      <circle cx={opCx} cy={opCy} r={opR} fill={C.surface} stroke={opColor} strokeWidth="1.6" />
      <text x={opCx} y={opCy + 5.5} textAnchor="middle" fontFamily="'IBM Plex Mono', monospace" fontSize="15" fill={opColor}>
        {opSymbol}
      </text>
      <rect x="136" y="31" width="36" height="26" rx="4" fill="var(--ink)" />
      <text x="154" y="48" textAnchor="middle" fontFamily="'IBM Plex Mono', monospace" fontSize="12" fill="var(--canvas)">ŷ</text>
    </svg>
  );
}

/* ── Forward-effect magnitude chart ─────────────────────────────── */
const MAX_MAG = Math.max(...fusion.mechanisms.map((m) => Math.abs(m.forwardD)));

export function ForwardMagnitudeChart({ active, onSelect }: { active: FusionId; onSelect: (k: FusionId) => void }) {
  const [hovered, setHovered] = useState<string | null>(null);
  const gridTicks = [1, 2, 3, 4, 5];
  return (
    <div className="select-none py-3" data-testid="forward-magnitude-chart">
      <div className="relative mb-2 flex items-center">
        <div className="w-24 shrink-0" />
        <div className="relative h-4 flex-1">
          {gridTicks.map((t) => (
            <span key={t} className="absolute -translate-x-1/2 font-mono text-2xs text-muted/50" style={{ left: `${(t / MAX_MAG) * 100}%` }}>
              {t}
            </span>
          ))}
        </div>
        <div className="w-24 shrink-0" />
      </div>
      <div className="space-y-5">
        {fusion.mechanisms.map((m) => {
          const isActive = m.key === active;
          const mag = Math.abs(m.forwardD);
          const pct = (mag / MAX_MAG) * 100;
          return (
            <div key={m.key} className="relative">
              <button
                type="button"
                className="flex w-full items-center text-left"
                onClick={() => onSelect(m.key)}
                onMouseEnter={() => setHovered(m.key)}
                onMouseLeave={() => setHovered(null)}
                aria-label={`${m.name}: forward d_z ${fixed(m.forwardD, 2)}, p ${pValue(m.p)}`}
              >
                <div className={`w-24 shrink-0 pr-3 text-right font-mono text-xs transition-colors duration-200 ${isActive ? "font-semibold text-ink" : "text-muted/60"}`}>
                  {m.name}
                </div>
                <div className="relative h-1.5 flex-1 rounded-full bg-line/50">
                  {gridTicks.map((t) => (
                    <div key={t} className="absolute top-1/2 h-3 w-px -translate-y-1/2 bg-line" style={{ left: `${(t / MAX_MAG) * 100}%` }} />
                  ))}
                  <div
                    className="absolute inset-y-0 left-0 rounded-full"
                    style={{
                      width: `${pct}%`,
                      backgroundColor: isActive ? C.forward : "#C2D3DE",
                      opacity: isActive ? 1 : 0.35,
                      transition: "width 400ms ease-out, background-color 200ms ease-out, opacity 200ms ease-out",
                    }}
                  />
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
                <div className="w-24 shrink-0 pl-4">
                  <div className={`font-mono text-xs font-semibold ${isActive && m.significant ? "text-forward" : isActive ? "text-warning" : "text-muted/40"}`}>
                    |d| = {mag.toFixed(2)}
                  </div>
                  <div className={`font-mono text-2xs ${isActive ? "text-muted" : "text-muted/30"}`}>p = {pValue(m.p)}</div>
                </div>
              </button>
              {hovered === m.key && (
                <div className="absolute left-24 top-full z-10 mt-1.5 rounded border border-line bg-surface px-3 py-2 shadow-sm">
                  <div className="mb-0.5 font-mono text-xs font-semibold text-ink">{m.name} fusion</div>
                  <div className="font-mono text-2xs text-muted">Δ𝒟 = {sgn(m.forwardDDom)}</div>
                  <div className="font-mono text-2xs text-muted">d_z = {fixed(m.forwardD, 2)} · p = {pValue(m.p)}</div>
                </div>
              )}
            </div>
          );
        })}
      </div>
      <div className="mt-5 flex items-center">
        <div className="w-24 shrink-0" />
        <div className="flex-1 text-center font-mono text-2xs text-muted/50">Forward-only effect magnitude |d_z|</div>
      </div>
    </div>
  );
}

/* ── Gradient-only arm, measured per mechanism (replaces the former draggable demo) ── */
export function GradientOnlyTable({ active }: { active: FusionId }) {
  return (
    <Card className="p-5">
      <div data-testid="gradient-only-table">
        <div className="measure mb-1 text-2xl font-semibold text-gradient">{fusion.gradientNull}</div>
        <p className="mb-4 text-xs leading-relaxed text-muted">
          Gradient-only arm minus baseline, per mechanism. No mechanism shows a measurable contribution shift.
        </p>
        <table className="w-full text-left text-2xs">
          <thead>
            <tr className="border-b border-line font-mono uppercase tracking-wider text-muted">
              <th className="py-1.5 font-normal">Fusion</th>
              <th className="py-1.5 font-normal">Δ𝒟</th>
              <th className="py-1.5 font-normal">d_z</th>
              <th className="py-1.5 font-normal">p</th>
              <th className="py-1.5 font-normal">r_g → r_Δθ</th>
            </tr>
          </thead>
          <tbody>
            {fusion.mechanisms.map((m) => (
              <tr
                key={m.key}
                data-fusion={m.key}
                className={`border-b border-line last:border-0 ${m.key === active ? "font-semibold text-ink" : "text-muted"}`}
              >
                <td className="py-1.5">{m.name}</td>
                <td className="py-1.5 font-mono">{sgn(m.gradient.dDom, 4)}</td>
                <td className="py-1.5 font-mono">{fixed(m.gradient.d, 2)}</td>
                <td className="py-1.5 font-mono">{pValue(m.gradient.p)}</td>
                <td className="py-1.5 font-mono">
                  {m.gradientArm.rG.toFixed(1)}× → {m.gradientArm.rTheta.toFixed(2)}×
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-3 border-t border-line pt-2 font-mono text-2xs text-muted/70">
          Update ratio r<sub>Δθ</sub> ≈ 1 across all mechanisms (Adam)
        </p>
      </div>
    </Card>
  );
}

/* ── Selected mechanism detail ─────────────────────────────────────── */
export function FusionDetail({ mechanism }: { mechanism: Mechanism }) {
  const isGated = mechanism.key === "gated";
  return (
    <div className="space-y-4" data-testid="fusion-detail">
      <div className="grid gap-4 sm:grid-cols-2">
        <Card className="p-5">
          <div className="label mb-3">{mechanism.name} fusion — as implemented</div>
          <div className="mx-auto max-w-[210px] py-2">
            <MechanismSVG kind={mechanism.key} />
          </div>
          <p className="mt-3 text-center font-mono text-2xs text-ink" data-testid="fusion-formula">
            {mechanism.formula}
          </p>
          <p className="mt-1 text-center text-2xs text-muted">
            s(·) = the arm's scale operator on h_A
            {isGated ? "; each gate is computed from the unscaled representation" : ""} (experiments/fusion_robustness.py)
          </p>
        </Card>
        <div className="grid grid-cols-2 gap-3">
          <Card className="p-4">
            <div className="text-2xs uppercase tracking-wider text-muted">Forward-only Δ𝒟</div>
            <div className={`measure mt-1 text-xl font-semibold ${mechanism.significant ? "text-forward" : "text-muted"}`} data-testid="fusion-forward-ddom">
              {sgn(mechanism.forwardDDom)}
            </div>
          </Card>
          <Card className="p-4">
            <div className="text-2xs uppercase tracking-wider text-muted">Forward d_z · p</div>
            <AnimatedNumber
              key={mechanism.key + "_d"}
              value={mechanism.forwardD}
              digits={2}
              signed
              className={`measure mt-1 text-xl font-semibold leading-none ${mechanism.significant ? "text-forward" : "text-muted"}`}
            />
            <div className={`mt-1 font-mono text-2xs ${mechanism.significant ? "text-ink" : "text-warning"}`} data-testid="fusion-forward-p">
              p = {pValue(mechanism.p)}
            </div>
          </Card>
          <Card className="p-4">
            <div className="text-2xs uppercase tracking-wider text-muted">Gradient-only Δ𝒟 · p</div>
            <div className="measure mt-1 text-xl font-semibold text-gradient" data-testid="fusion-gradient-ddom">
              {sgn(mechanism.gradient.dDom, 4)}
            </div>
            <div className="mt-1 font-mono text-2xs text-muted">
              d_z {fixed(mechanism.gradient.d, 2)} · p = {pValue(mechanism.gradient.p)}
            </div>
          </Card>
          <Card className="p-4">
            <div className="text-2xs uppercase tracking-wider text-muted">Ordinary Δ𝒟</div>
            <div className="measure mt-1 text-xl font-semibold text-ink">{sgn(mechanism.ordinaryDDom)}</div>
            <div className="mt-1 font-mono text-2xs text-muted">
              vs forward: d_z {fixed(mechanism.ordinaryVsForward.d, 2)}, p = {pValue(mechanism.ordinaryVsForward.p)}
            </div>
          </Card>
        </div>
      </div>
      <Callout tone={isGated ? "warning" : "forward"} title={isGated ? "Attenuated effect — boundary case" : "Interpretation"}>
        {isGated ? (
          <>
            Under gated fusion the forward effect is attenuated and not individually significant (d_z = {fixed(mechanism.forwardD, 2)},
            p = {pValue(mechanism.p)}). One possible reading is that the learned gates partly offset the imposed magnitude, but gate
            activations were not logged in this experiment, so this is not established. The gradient-only arm stays null.
          </>
        ) : mechanism.key === "sum" ? (
          <>
            Strong forward-pathway effect under sum fusion (d_z = {fixed(mechanism.forwardD, 2)}, p = {pValue(mechanism.p)}): the
            unweighted sum passes the imposed magnitude difference straight to the head. The gradient-only arm stays null.
          </>
        ) : (
          <>
            The largest forward effect size of the three tested mechanisms is under concatenation (d_z = {fixed(mechanism.forwardD, 2)},
            p = {pValue(mechanism.p)}), although its Δ𝒟 ({sgn(mechanism.forwardDDom)}) is smaller than under sum; d_z also reflects
            seed-to-seed spread. The gradient-only arm stays null.
          </>
        )}
      </Callout>
    </div>
  );
}

export function FusionSelector({ active, onChange }: { active: FusionId; onChange: (k: FusionId) => void }) {
  return (
    <div className="grid gap-4 md:grid-cols-3" role="radiogroup" aria-label="Fusion mechanism">
      {fusion.mechanisms.map((m) => (
        <button
          key={m.key}
          type="button"
          role="radio"
          aria-checked={active === m.key}
          onClick={() => onChange(m.key)}
          data-active={active === m.key}
          data-testid={`fusion-${m.key}`}
          className="selectable p-5 text-left focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-forward"
        >
          <div className="mb-3 flex items-start justify-between gap-2">
            <div>
              <h3 className="font-display text-[1.05rem] font-semibold text-ink">{m.name}</h3>
              <div className="mt-0.5 font-mono text-[0.62rem] text-muted">{m.formula}</div>
            </div>
            {m.significant ? <Tag tone="forward">Strong</Tag> : <Tag tone="warning">Attenuated · n.s.</Tag>}
          </div>
          <div className="my-3 rounded border border-line bg-canvas/50 px-3 py-2">
            <MechanismSVG kind={m.key} />
          </div>
          <div className="mt-1 flex items-baseline justify-between border-t border-line pt-3">
            <span className="text-2xs uppercase tracking-wider text-muted">Forward effect</span>
            <span className={`measure text-sm font-semibold ${m.significant ? "text-forward" : "text-muted"}`}>d = {fixed(m.forwardD, 2)}</span>
          </div>
        </button>
      ))}
    </div>
  );
}
