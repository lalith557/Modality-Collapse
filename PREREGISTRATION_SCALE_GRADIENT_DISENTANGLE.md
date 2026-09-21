# Pre-registration — forward-scale vs gradient-scale disentangling control

Written **before** running. Targeted mechanism experiment authorized by the hostile-reviewer audit
(Audit 1/14): the ordinary training-time scale intervention `h_A' = α·h_A` changes BOTH the forward
magnitude seen by fusion AND the gradient flowing into encoder A. This experiment separates the two
pathways. **All prior experiments (E1–E5, CREMA-D R1–R3/H6, UTD-MHAD) are immutable and untouched.**
Start commit `4465ebc`. Date: 2026-09-20.

## Objective
Determine whether the learned modality-contribution shift produced by training-time scaling is
associated primarily with **(A) forward magnitude at fusion**, **(B) the encoder-gradient scaling
induced by that scaling**, or **(C) both**. A null or contradictory result is acceptable and will be
reported as such. The result is not assumed.

## Four arms (scaling applied to modality A's fusion-boundary representation, IN-GRAPH during training)
Let `h_A` = enc_A(x_A) (normalization='none', so rep = encoder output). Each arm replaces `h_A` by:
| arm | forward value | gradient into `h_A` (d out/d h_A) |
|---|---|---|
| **baseline** | `h_A` | ×1 |
| **forward-only** | `α·h_A` | **×1** (α removed from the backward path) |
| **gradient-only** | `h_A` (unchanged) | **×α** |
| **ordinary** | `α·h_A` | ×α (standard multiply) |

Implementation (verified by autograd, below):
- `ForwardScale.apply(h,α)`: forward returns `α·h`; backward returns `grad_output` (×1).
- `GradScale.apply(h,α)`: forward returns `h`; backward returns `grad_output·α`.
- ordinary = `α*h` (plain multiply); baseline = `h`.
Only modality A is scaled (the equal-information generator is A/V-symmetric, so the V direction is
redundant by construction and is not separately run). Fusion of V is unchanged in every arm.

## Autograd sanity gate (MUST pass before the main run)
For each arm, with a scalar `h` (requires_grad) and `loss = f(arm(h))`, verify the multiplicative
forward and backward behaviour:
| arm | forward mult | backward mult |
|---|---:|---:|
| baseline | 1 | 1 |
| forward-only | α | 1 |
| gradient-only | 1 | α |
| ordinary | α | α |
Also numerically verify: (i) gradient-only forward activations entering fusion are **bit-identical** to
baseline; (ii) forward-only encoder gradients do **not** carry the α factor; (iii) ordinary carries
both. Save results as JSON. **If the sanity gate fails, STOP — do not run the main experiment.**

## Scale conditions (small, mechanism-focused — not a dose study)
`α ∈ {1, 10, 100}` on modality A. (α=1 is a degenerate control: all four arms are identity → must
give identical results, an internal consistency check.) 4 arms × 3 α × 5 seeds = **60 runs**.

## Training protocol (identical to E4/E4-EXTREME/E5; frozen)
Equal-information generator (`configs/boundary.yaml` data block: 8 classes, dim 32+32, margin 1.5,
σ=1, n_train 4000/val 1000/test 2000, prototype_seed 12345), **data α_a=α_v=1** (no data scaling; the
scale is imposed at the representation during training). Tiny MLP encoders (hidden 64, rep 32), sum
fusion, linear head. **Adam, base_lr 1e-3, batch 256, 40 epochs, weight_decay 0, 5 seeds [0–4].** No
per-encoder LR asymmetry (lr_a=lr_v=base). Normalization='none'. No hyperparameter changes.

**Optimizer caveat (pre-stated, not a design change):** under Adam, multiplying a parameter's gradient
by a constant is largely normalized away (first/second moments both rescale). The gradient-only arm is
a faithful implementation of "scale the encoder gradient by α"; its *downstream* effect under Adam is
part of what the experiment measures and will be reported honestly, including if it is near-null.

## Metrics (per run; reuse frozen helpers where possible)
multimodal accuracy `acc_av`; contributions `Δ_A, Δ_V`; dominance `Acc_A−Acc_V`; multimodal gain;
`ρ_fusion` = E‖arm(h_A)‖/E‖h_V‖ (post-arm, what fusion sees); `ρ_encoder` = E‖h_A‖/E‖h_V‖ (pre-arm,
compensation); per-encoder gradient norms ‖g_A‖, ‖g_V‖ and ratio; compensation `1−ln(ρ_fusion)/ln(α)`
(interpretable only for forward-scaled arms); and **contribution shift vs baseline** = dominance(arm,α)
− dominance(baseline). Evaluation uses each arm's own forward transform (grad-only & baseline present
`h_A`; forward-only & ordinary present `α·h_A`), i.e. the forward path each model was trained under.

## Mechanistic trace (α=10 and α=100)
Per-epoch `ρ_fusion(t)`, `ρ_encoder(t)`, gradient_ratio(t), val dominance(t), val gain(t) — to see
whether a forward imbalance or a gradient imbalance appears first, or both together. Used for
mechanistic tracing only; no temporal-causality claim from correlation.

## Statistics
5 seeds; per-seed values, mean ± SD, 95% CI, Cohen's d and paired t for: forward-only vs baseline,
gradient-only vs baseline, ordinary vs baseline, ordinary vs forward-only, ordinary vs gradient-only.
Plot per-seed distributions, not only means. Report effect sizes, not only p-values.

## Decision framework (choose from evidence)
- **CASE A (forward-dominated):** forward-only ≈ ordinary AND gradient-only ≈ baseline → strengthens a
  magnitude-specific mechanism.
- **CASE B (gradient-dominated):** gradient-only ≈ ordinary AND forward-only ≈ baseline → weakens the
  feature-magnitude mechanism.
- **CASE C (mixed):** both isolated arms produce partial effects → quantify relative contributions.
- **CASE D (interactive):** neither isolated arm reproduces ordinary → forward×gradient interaction.
- **CASE E (unresolved):** unstable across seeds → no mechanism claimed.
Do not force the result into a case. "≈" judged by overlapping 95% CIs / |Cohen d|<~0.5 vs the
reference; "reproduces ordinary" judged by CI overlap with the ordinary effect.

## Failure handling / STOP conditions
Non-finite loss → record as diverged (don't drop). STOP and report if: the autograd sanity gate fails;
gradient-only changes the forward activations; forward-only's encoder gradient carries α; ordinary does
not carry both; NaNs/divergence; the intervention requires changing the frozen architecture. Do **not**
"fix" a failed intervention by redefining the science.

## Outputs (new directory `results/disentangle/`, no overwrite of existing artifacts)
`PREREGISTRATION_SCALE_GRADIENT_DISENTANGLE.md` (this), `EXPERIMENT_REPORT_SCALE_GRADIENT_DISENTANGLE.md`,
`results/disentangle/disentangle_{primary,trajectories,agg}.csv`, `_analysis.json`, `_sanity.json`,
`_config.json`; figures `disentangle_{arms_contribution,arms_distribution,trajectories,compensation}.png`.
Commit only after review. Then STOP (no other experiment, no paper).
