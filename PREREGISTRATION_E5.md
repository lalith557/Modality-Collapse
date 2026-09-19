# Pre-registration — E5 (OGM-GE mechanistic comparison)

Written **before** the final E5 run. New experiment; E1–E4-EXTREME frozen and
unmodified. Reuses the same generator, architecture, optimizer, frozen collapse
definition. Timestamp: 2026-09-20.

## Scientific question (mechanism, not leaderboard)
Do **representation normalization** (per-modality z-score) and **optimization-level
intervention** (OGM-GE) correct **different components** of modality imbalance?
- **H5-A:** z-score primarily removes the representation-*scale* component.
- **H5-B:** OGM-GE primarily addresses the *optimization/LR* component.
- **H5-C:** if separable, z-score + OGM-GE addresses both more completely than either alone.
Hypotheses — not assumed.

## OGM-GE implementation (fidelity spec; Phase 2)
Method: Peng et al., "Balanced Multimodal Learning via On-the-fly Gradient
Modulation" (CVPR 2022). Adapted to this two-stream sum/concat linear-head model.

**Per-modality logit scores.** With head weight `W` (C×d_fused), bias `b`:
- sum fusion: `s_A = h_A Wᵀ + b`, `s_V = h_V Wᵀ + b` (shared W; `logits = W(h_A+h_V)+b`).
- concat fusion: `s_A = h_A W[:, :d]ᵀ + b`, `s_V = h_V W[:, d:]ᵀ + b`.

**Discrepancy ratio (per batch).** `score_m = Σ_batch softmax(s_m)[y]`;
`ratio = score_A / score_V`. The **dominant** modality is the larger score.

**On-the-fly gradient modulation (OGM), faithful form.** For the dominant modality
`i` (ratio_i = score_i/score_j > 1): `k_i = 1 − tanh(α · ratio_i)`; the other
modality `k_j = 1`. `α = 0.1` (pre-registered; within the paper's reported range;
sensitivity noted, not tuned on test). Coefficients multiply the **uni-modal
encoder** gradients (enc_A, enc_V); head/fusion gradients are **unmodulated**.

**Generalization Enhancement (GE), adapted approximation.** After modulation, add
zero-mean Gaussian noise to the modulated encoder gradients to partly restore the
covariance reduced by modulation: `g += N(0, (γ·(1−k)·std(g))²)`, `γ = 0.5`
(pre-registered). Labelled **adapted** (the paper's GE maintains covariance more
precisely). The balancing mechanism under test is OGM; GE is a generalization add-on.

**Fidelity labelling in the report:** faithful (OGM coefficient + application point)
vs adapted (GE noise form) vs project-specific (per-modality score for sum fusion,
α/γ values). If exact reproduction is impossible it is stated, not hidden.

## Conditions (targeted; both modality directions via the E4 reciprocal convention)
`r_s=α_A/α_V`, `r_l=lr_A/lr_V`; `α_A=√r_s, α_V=1/√r_s`; `lr_A=base·√r_l, lr_V=base/√r_l`.
- **C0 balanced:** (1, 1).
- **C1 scale-only:** (100, 1) and (1/100, 1).
- **C2 LR-only:** (1, 100) and (1, 1/100).
- **C3 aligned moderate:** (20, 20) and (1/20, 1/20).
- **C4 aligned strong:** (100, 100) and (1/100, 1/100).
- **C5 opposed:** (100, 1/100) and (1/100, 100).
11 unique conditions.

## Four intervention arms (mandatory, all four)
- **M0 vanilla** (no norm, no OGM-GE)
- **M1 z-score** (per-modality z-score; E3/E4 implementation)
- **M2 OGM-GE** (gradient modulation; no norm)
- **M3 z-score + OGM-GE** (both)

## Replication & controls
5 seeds; identical generator/architecture (sum fusion, linear head)/optimizer
(Adam)/batch 256/init policy/schedule/40 epochs/information content/frozen collapse
definition. No per-method tuning of architecture/epochs/capacity/seeds. `α=0.1,
γ=0.5` fixed in advance. Primary at 40 epochs; aligned C4 (both directions) also at
100 epochs to check persistence under interventions.

## Metrics (per seed/epoch)
ρ, |log ρ|; Δ_A, Δ_V, weak = min(Δ_A,Δ_V), |Δ_A−Δ_V|; compensation residual
|log ρ_final|; **encoder gradient norms** ‖g_A‖,‖g_V‖ and ratio; OGM modulation
coefficients k_A,k_V; multimodal gain; accuracy; macro-F1; frozen collapse status +
severity. Primary outcome = weak-modality contribution / collapse severity (NOT accuracy).

## Primary comparisons (paired across seeds; report mean, SD, 95% CI, effect size)
1. C1 scale-only: vanilla vs z-score; vanilla vs OGM-GE.
2. C2 LR-only: vanilla vs OGM-GE; vanilla vs z-score (negative control).
3. C4 aligned100: vanilla vs OGM-GE; z-score vs OGM-GE; OGM-GE vs z-score+OGM-GE.
Report effect sizes; do not rely on p-values; correct for multiplicity if formal significance is claimed.

## Decision tree (choose one from results, not publication value)
- **E5-A** mechanistic separation supported (z-score↓scale, OGM-GE↓LR, combined complementary) → GO CREMA-D.
- **E5-B** OGM-GE fixes LR not scale (still supports decomposition) → GO CREMA-D.
- **E5-C** both act on the same phenomenon everywhere → revise framing before real data.
- **E5-D** OGM-GE does not correct LR-driven collapse (after fidelity check) → report negative; reassess.
- **E5-E** implementation uncertain → STOP; resolve fidelity before real data.

## Outputs
`results/e5/e5_primary.csv`, `e5_aggregated.csv`, `e5_config.json`;
figures `results/figures/e5_{mechanism_decomposition,intervention_comparison,aligned100_trajectories,gradient_mechanism,intervention_matrix}.png`;
report `reports/E5_FINAL_REPORT.md`. Commit separately. Then STOP (no CREMA-D/INCLUDE/paper).

## Overclaim guardrails
OGM-GE improving LR-driven collapse = "OGM-GE reduces optimization-driven imbalance
in this controlled synthetic setting", NOT "proves the mechanism". Causal evidence
is from the controlled interventions. z-score is not a universal cure. Complementarity not assumed.
