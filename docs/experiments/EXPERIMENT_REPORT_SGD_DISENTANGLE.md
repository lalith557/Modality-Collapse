# SGD forward-scale vs gradient-scale disentangling — final report

_Preregistration: `PREREGISTRATION_SGD_DISENTANGLE.md` (frozen before results). Start commit `4465ebc`._
_Only change vs the Adam disentangle experiment: Adam → SGD. Same generator/architecture/seeds/α/arms;
validated `ForwardScale`/`GradScale` reused. Adam runs NOT rerun. All prior experiments immutable._
_Labels: **[OBS]** observed · **[STAT]** statistical · **[MECH]** mechanistic · **[SPEC]** speculation._

## Verdict: **CASE C — mixed (both pathways shift contribution under SGD)** — the Adam "forward-dominated" result is **optimizer-specific**
Under SGD, isolated **gradient-only** scaling is **NOT inert** — it produces a large, persistent
contribution shift (dominance +0.200, d=+3.90, p=0.0010), because SGD translates the constant gradient
scaling into a **~α× parameter-update imbalance** (update ratio ≈ 96 ≈ α at α=100), i.e. an effective
per-modality learning-rate boost. **Forward-only also shifts contribution** (+0.244, d=+3.93). Both
pathways contribute; ordinary exceeds either (+0.322). This directly contradicts the Adam result
(gradient-only inert, forward-dominated) and confirms **H_SGD**. 60 runs, **0 diverged**.

## 1. Purpose
Test whether the forward-dominated pathway found under Adam is optimizer-dependent, and whether isolated
gradient scaling becomes an effective-learning-rate effect under SGD (connecting to E4/E4-EXTREME).

## 2. Exact SGD configuration (fixed; NOT tuned)
`torch.optim.SGD(lr=1e-3, momentum=0.9, weight_decay=0)` — the project's existing SGD default at the
**same numeric lr as Adam's base_lr**; no scheduler, no gradient clipping, per-encoder LR equal. All else
identical to the Adam run (8-class equal-info generator, dim 32+32, margin 1.5, σ=1, tiny MLP encoders,
sum fusion, linear head, batch 256, 40 epochs, seeds [0–4], α∈{1,10,100}, modality A scaled). **No
hyperparameter sweep; no result-driven adjustment; 0 diverged (no stability fix needed).** SGD **learned
adequately** (baseline test acc 0.516, gain 0.140, both modalities ≈ equal) — the config is not crippled.

## 3. Autograd sanity — PASSED (reused validated ops; `sgd_disentangle_sanity.json`)
Forward/backward multipliers exact (baseline 1/1, forward α/1, gradient 1/α, ordinary α/α); gradient-only
forward bit-identical to baseline. (Autograd behaviour is optimizer-independent; rerun for the record.)

## 4. Per-seed results
`results/sgd_disentangle/sgd_disentangle_primary.csv`. α=1 control: all four arms identical (+0.003).
Seed SD of dominance at α=100: baseline 0.047, forward 0.027, gradient 0.052, ordinary 0.020 — stable.

## 5. α=10 results (dose-response) [OBS]
baseline dom +0.003 (gain 0.140); forward +0.169 (shift +0.166, gain 0.042); gradient +0.217 (shift
+0.214, gain 0.043, update ratio 4.83 ≈ α/2); ordinary +0.224 (gain 0.030). Both isolated arms already
shift at α=10; ordinary ≈ largest.

## 6. α=100 results [OBS/STAT] (test, 5 seeds, paired)
| arm | dominance | shift vs baseline | Cohen d | p | gain | ρ_enc | grad ratio | **update ratio** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | +0.003 | — | — | — | +0.140 | 1.02 | 1.0 | 0.99 |
| **forward** | +0.247 | **+0.244 ±0.054** | **+3.93** | **0.0009** | +0.002 | 0.99 | 0.86 | **0.41** |
| **gradient** | +0.203 | **+0.200 ±0.045** | **+3.90** | **0.0010** | +0.021 | 6.89 | 318.5 | **95.9** |
| ordinary | +0.325 | +0.322 ±0.036 | +7.75 | 6.5e-5 | +0.003 | 0.58 | 65.1 | 18.3 |

**Direct comparisons:** ordinary vs forward +0.078 (d=2.86, p=0.003); ordinary vs gradient +0.122 (d=2.37,
p=0.006) → **ordinary ≠ either isolated arm** (both pathways contribute; sub-additive). **Multimodal gain
collapses to ≈0** in all scaled arms (0.140 → ~0), a much larger effect than under Adam (0.195→0.157).

## 7. Gradient-ratio results [OBS]
Encoder gradient ratio ‖g_A‖/‖g_V‖ (α=100): baseline ~1, forward 0.86, **gradient 318**, ordinary 65.
As designed, gradient-only imposes a huge gradient imbalance (the ×100 amplifies, compounded by learning
dynamics).

## 8. Parameter-update-ratio results (the decisive metric) [OBS/MECH]
Update ratio ‖Δθ_A‖/‖Δθ_V‖ (α=100): baseline ~1, forward **0.41**, **gradient 95.9 (≈ α)**, ordinary 18.3.
**[MECH]** Under SGD the constant gradient scaling **passes through to the parameter update** (update ≈ α×),
i.e. gradient-only ≈ a per-modality effective-LR boost of ~α. This is exactly what Adam neutralizes: the
Adam gradient-only arm had a ~100× *gradient* ratio but (by Adam's moment normalization) a ~1× *update*
and a null outcome. **The update ratio, not the gradient ratio, is the correct mediator**, and it
dissociates the two optimizers.

## 9. Forward-only vs gradient-only [OBS]
Both shift contribution (forward +0.244, gradient +0.200; comparable magnitude). Their *mechanisms differ*:
forward-only entrenches A via the **large forward representation at fusion** (ρ_enc stays ~1, no
compensation; update ratio 0.41 — enc_A actually updates *less*); gradient-only entrenches A via a
**faster-learning encoder** (update ratio 96 → ρ_enc grows to 6.89, A's representation inflates). Ordinary
combines both (ρ_enc 0.58, update 18) and yields the largest shift.

## 10. Adam vs SGD (α=100; Adam NOT rerun, loaded from `results/disentangle/`)
| arm | Adam dom shift | SGD dom shift | Adam grad ratio | SGD grad ratio | SGD update ratio | Adam gain | SGD gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | +0.000 | +0.000 | 1.0 | 1.0 | 0.99 | 0.195 | 0.140 |
| forward | **−0.070** | **+0.244** | 0.3 | 0.86 | 0.41 | 0.157 | 0.002 |
| gradient | **−0.000 (inert)** | **+0.200 (active)** | 100.9 | 318.5 | 95.9 | 0.195 | 0.021 |
| ordinary | −0.069 | +0.322 | 27.0 | 65.1 | 18.3 | 0.157 | 0.003 |

**Two optimizer-dependent differences:** (i) **gradient pathway** — inert under Adam, **active under SGD**
(the central result); (ii) **direction** — the forward/ordinary effect is **negative under Adam**
(encoder *over-compensates*, ρ_enc→0.09, A ends sub-dominant) but **positive under SGD** (encoder does not
compensate; A entrenches), and the SGD gain collapse is far larger (→0 vs →0.157).

## 11. Persistence analysis (α=100 trajectories) [OBS]
- forward: dom rises to +0.24 by epoch ~3 and **holds through epoch 40**; gain collapses to ~0 immediately and stays. **Persistent** (through the tested horizon).
- gradient: dom peaks +0.38 (epoch 3–5) then **relaxes to +0.216** by epoch 40; gain ~0 then **creeps up to 0.023** (slow partial recovery). **Persistent with partial slow recovery** — the signature of the E4-EXTREME optimization regime.
- ordinary: dom +0.33 held; gain ~0 held. **Persistent.**
**Caveat:** SGD is not fully converged at 40 epochs (baseline gain still rising, 0.149). Permanence beyond
40 epochs is **untested** here; E4-EXTREME separately showed the LR/optimization collapse persists to 200
epochs, which is consistent with the gradient arm's slow relaxation. Do **not** over-read the transient peak.

## 12. Mechanistic interpretation [MECH]
The training-time scale intervention couples a forward-magnitude change and a gradient-scaling change. The
**forward** change shifts contribution under both optimizers (net direction set by encoder compensation:
Adam over-compensates → negative; SGD does not → positive). The **gradient** change is mediated by the
**parameter update**: under **Adam** it is normalized away (update ≈1 → inert); under **SGD** it becomes a
~α× effective per-modality learning rate (update ≈ α → large, persistent shift + gain collapse). So the
"pathway decomposition" is **optimizer-dependent**: Adam → forward-dominated (CASE A); SGD → mixed (CASE C),
with the gradient arm behaving as an effective-LR / optimization effect.

## 13. What this changes about E4 / E4-EXTREME
**Strengthens and connects.** E4/E4-EXTREME found that per-modality **learning-rate** asymmetry drives
persistent collapse (scale alone is compensable). This experiment shows that, **under SGD**, isolated
**gradient scaling** *becomes* such a per-modality effective-LR effect (update ratio ≈ α) and produces a
persistent dominance shift + multimodal-gain collapse with a peak-then-slow-relaxation profile — the same
qualitative signature as E4-EXTREME's optimization regime. This is **consistent with / connects to** E4's
optimization channel; it is **not** a direct re-proof (different induction: single-modality gradient scale
vs reciprocal scale+LR), so it should be reported as convergent evidence, not as "proof of E4."

## 14. What this changes about the Adam disentanglement result
The Adam "forward-dominated" conclusion is **confirmed as correct but optimizer-specific.** The audit's
insistence on the scoped wording ("under the tested Adam configuration, isolated gradient scaling did not
alter learned contribution") was necessary: the result does **not** generalize to SGD. The broader claim
"feature magnitude is the operative mechanism" must be **downgraded to Adam-specific**; under SGD the
optimization/effective-LR pathway is co-equal to the forward pathway.

## 15. Updated SUPPORTED claims
- Forward feature magnitude shifts learned contribution under **both** Adam and SGD (direction optimizer-
  dependent via encoder compensation).
- Under **SGD**, isolated gradient scaling shifts learned contribution via a **~α× parameter-update**
  (effective per-modality LR); the **parameter-update ratio**, not the gradient-norm ratio, is the mediator.
- The forward-vs-gradient **pathway decomposition is optimizer-dependent**: Adam → forward-dominated; SGD →
  mixed (forward + effective-LR).
- The SGD gradient pathway is **consistent with** E4/E4-EXTREME's optimization-driven collapse channel.
- All effects reproducible (0/60 diverged, α=1 control identical, stable across seeds).

## 16. Updated UNSUPPORTED claims
- "Feature magnitude is *the* operative mechanism" **in general** (holds only under Adam; SGD is mixed).
- "The gradient pathway is inert" **in general** (Adam-specific).
- Permanence of the SGD effect beyond 40 epochs (untested here).
- "Proves the E4 mechanism" (convergent, not identical setup).
- Any single fixed **direction** of the forward effect (sign flips Adam↔SGD).

## 17. Remaining limitations
- Two optimizers only (Adam, SGD momentum 0.9), one fixed LR each; other optimizers/LRs untested.
- Synthetic equal-information, tiny MLP, sum fusion; real-data decomposition still not done.
- SGD not fully converged at 40 epochs (persistence-horizon caveat).
- Effect direction and magnitude depend on encoder-compensation dynamics, which are architecture/optimizer-specific.

## 18. Does the SGD result materially strengthen the paper? **Yes.**
It resolves the #1 audit limitation (optimizer scope), converts "under Adam" into a genuine **cross-
optimizer mechanistic finding**, and — via the parameter-update-ratio metric — cleanly unifies the study's
two channels: a **forward feature-scale channel** (present under both optimizers, compensable) and an
**optimization / effective-LR channel** (Adam-normalized, SGD-active, connecting to E4/E4-EXTREME).

## 19. Is another experiment scientifically necessary?
**No** (see decision below). The optimizer limitation is now characterized on both sides; remaining gaps
are disclosed limitations, not validity gaps for the scoped claims.

---
_Reproducibility: `results/sgd_disentangle/sgd_disentangle_{primary,trajectories,agg}.csv`, `_analysis.json`,
`_sanity.json`, `_config.json`, `_adam_vs_sgd.csv`; figures `sgd_{dominance,grad_update_ratio,adam_vs_sgd,
trajectories}.png`. Not committed pending review. Verdict chosen from evidence, not before analysis._
