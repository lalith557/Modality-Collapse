# Pre-registration — CREMA-D forward-scale vs gradient-scale disentangling (real-data mechanism)

Written **before** running. Authorized Phase 2: reproduce the validated synthetic forward-vs-gradient
disentanglement on **real CREMA-D**, to test whether the pathway decomposition (forward-magnitude vs
encoder-gradient) found synthetically holds on real audio-video data under the **same Adam** used by H6.
**All prior experiments (E1–E5, CREMA-D R1–R3/H6, UTD-MHAD, synthetic disentangle Adam/SGD) are immutable
and untouched.** No CREMA-D re-download; frozen actor-disjoint split and cached encoder inputs reused
unchanged. Start commit `4465ebc`. Date: 2026-09-21.

## Scientific question (not assumed)
Does real CREMA-D training-time feature-scaling separate into a **forward-magnitude pathway** vs an
**encoder-gradient pathway**, as it did synthetically under Adam (forward-dominated; gradient-only inert
because Adam normalizes constant gradient rescaling)? A result that **differs** from the synthetic Adam
finding is scientifically important and will be reported as-is. Not tuned toward replication.

## Design (only new thing vs H6: the four arms replace H6's plain scale)
Modality **A = audio** (enc_a), **B = video** (enc_v). **Scale audio** (the more-informative modality;
parallels the synthetic "scale A" and H6's audio×10, which entrenched audio). Four arms applied to the
audio fusion-boundary representation `h_A`, in-graph during training (validated `ForwardScale`/`GradScale`
reused, NOT redesigned):
| arm | forward → fusion | gradient into h_A |
|---|---|---|
| baseline | `h_A` | ×1 |
| forward-only | `α·h_A` (ForwardScale) | ×1 |
| gradient-only | `h_A` (bit-identical fwd) | ×α |
| ordinary | `α·h_A` | ×α |

## Conditions (frozen)
α ∈ {10, 100}; 5 seeds [0,1,2,3,4]. 4 arms × 2 α × 5 = **40 runs** (baseline is α-invariant → the two
baseline sets are an internal consistency check).

## Architecture & optimizer (identical to CREMA-D H6 / `configs/cremad.yaml`; frozen)
`CremadMultimodalModel`, matched CNN encoders (16160=16160), sum fusion, linear head, rep_dim 64,
**normalization='none'**, scale-injector disabled (the arm replaces it). **Adam lr 1e-3, batch 64, 30
epochs, weight_decay 0.** Cached encoder inputs (log-mel `[1,64,188]`, frames `[8,1,64,64]`); encoders
trainable. Test = the 18 frozen held-out actors only (1470 clips). Batched eval (chunk 512) to avoid OOM.

## Autograd sanity gate (on the CREMA-D model + real cached features; MUST pass)
Scalar multipliers baseline (1,1)/forward (α,1)/gradient (1,α)/ordinary (α,α); gradient-only forward
activations **bit-identical to baseline**; forward-only encoder gradient **free of the α multiplier**
(ratio ≪ α); ordinary carries both; record encoder gradient AND parameter-update norms. **Fail ⇒ STOP.**

## Metrics (per run)
unimodal (masking) audio/video accuracy; multimodal accuracy; multimodal gain; dominance = Acc_audio −
Acc_video; Δ_audio, Δ_video; ρ_fusion = E‖arm(h_A)‖/E‖h_V‖ (post-arm); ρ_encoder = E‖h_A‖/E‖h_V‖
(pre-arm, compensation); encoder gradient ratio ‖g_A‖/‖g_V‖; **parameter-update ratio ‖Δθ_A‖/‖Δθ_V‖**;
compensation `1 − ln(ρ_fusion)/ln(α)`. Evaluation uses each arm's own forward transform.

## Statistics
5 seeds: mean ± SD, 95% CI, paired Cohen's d + paired t (same seeds → paired). Primary comparisons
(α=100): forward vs baseline, gradient vs baseline, ordinary vs baseline, ordinary vs forward, ordinary
vs gradient. **Seeds are init/data-order variance, NOT population samples** (test = 18 fixed actors); do
not read seed p-values as population significance. Report effect sizes alongside.

## Decision framework (choose from evidence; masking-video degeneracy caveat inherited from H6/R1)
- **CASE A (forward-dominated):** forward ≈ ordinary, gradient ≈ baseline → matches synthetic Adam.
- **CASE B (gradient-dominated):** gradient ≈ ordinary, forward ≈ baseline.
- **CASE C (mixed):** both isolated arms shift; ordinary ≠ either.
- **CASE D (interactive):** neither isolated arm reproduces ordinary.
- **CASE E (unresolved):** unstable across seeds.
Caveat: CREMA-D masking-based unimodal video is degenerate (→ majority class, R1); dominance/gain are the
cleaner signals — the pathway decomposition is read primarily off dominance shift and ρ dynamics.

## Failure handling / STOP conditions
Non-finite loss → record diverged (don't drop). STOP and report if: sanity fails; gradient-only alters
forward activations; forward-only encoder gradient carries α; ordinary misbehaves; OOM/instability; the
intervention needs architecture changes. Do not tune to a result.

## Outputs (new dir `results/cremad_disentangle/`, no overwrite)
`PREREGISTRATION_CREMAD_DISENTANGLE.md` (this), `EXPERIMENT_REPORT_CREMAD_DISENTANGLE.md`,
`results/cremad_disentangle/cremad_disentangle_{primary,trajectories,agg}.csv`, `_analysis.json`,
`_sanity.json`, `_config.json`; figures `cremad_dis_{dominance,grad_update_ratio,trajectories,compensation}.png`.
Do not commit pending review. Then proceed to Phase 3.
