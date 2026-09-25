# Pre-registration — CREMA-D training-time scale intervention (H6)

Written **before** the final runs. A NEW experiment; **R1–R3 and E1–E5 remain unchanged
and immutable**. This specifically tests the training-time causal pathway that R3 (a
frozen-representation intervention) could not test. Start commit `b7a5295`. Date: 2026-09-20.

## Motivation
R3 scaled modality feature magnitude *after* the representation was trained, so it showed
scale can damage integration but not whether imposing scale disparity *during training*
causes the model to learn persistent modality imbalance. This experiment applies the scale
**in-graph, during every training step, with encoders trainable**.

## H6 (primary hypothesis; not assumed true)
Multiplying one modality's encoder output by a fixed factor at the fusion boundary during
every training step (encoders trainable) alters training dynamics and may produce a
persistent change in learned modality contribution — OR trainable encoders compensate for
the imposed scale (as synthetic E1/E4-EXTREME predict for static scale).

## Conditions (targeted; frozen). (alpha_audio, alpha_video), applied in-graph during training:
- **baseline** (1, 1)
- **audio×5** (5, 1), **audio×10** (10, 1)
- **video×5** (1, 5), **video×10** (1, 10)
Non-reciprocal single-modality scaling (Phase-6 table). 5 conditions.

## Normalization follow-up (frozen)
For the extreme/informative conditions {baseline (1,1), audio×10 (10,1), video×10 (1,10)},
compare **no normalization** vs **per-modality z-score**, with the scale applied BEFORE the
per-modality norm (`scale_first=True`) so normalization can remove the imposed disparity.
No LayerNorm/BatchNorm. → 3 conditions × 2 norms × 5 seeds (baseline/audio10/video10 with
none are shared with the main set) = 15 additional (zscore) runs.

## Architecture & training (identical to R1–R3; frozen)
`CremadMultimodalModel`, sum fusion, rep_dim 64, matched CNN encoders (16160=16160 params),
linear head. **Encoders TRAINABLE.** Adam lr 1e-3, batch 64, weight_decay 0, **30 epochs,
5 seeds** (0–4). Cached features are ENCODER INPUTS (log-mel/frames) — encoders run and
update each step (valid caching; NOT frozen encoder outputs). Frozen actor-disjoint split;
test = 18 held-out actors only.

## Implementation requirement (critical)
`h_audio = alpha_audio * enc_audio(x)` (and video) INSIDE the graph, before fusion; not
detached; not applied only at eval; not on raw input; not on the loss; not on gradients;
not via LR. Uses the existing `ScaleInjector` at the fusion boundary with `scale.enabled=True`
during training and `scale_first=True`.

## Primary outcome (frozen; existing definition)
Learned modality contribution imbalance after training, on TEST, via the existing masking
definition: Δ_audio = Acc_AV − Acc_video, Δ_video = Acc_AV − Acc_audio,
|Δ_audio − Δ_video|, and dominance Acc_audio − Acc_video. Compared across the 5 conditions.

## Secondary outcomes (frozen)
Per epoch: fusion-boundary ρ (post-scale), **learned encoder-output ratio** ρ_enc =
‖enc_audio‖/‖enc_video‖ (pre-scale, for compensation), encoder gradient ratio
‖g_audio‖/‖g_video‖, val contribution, val accuracy. Final: test acc, macro-F1, multimodal
gain, unimodal (masking) accuracy.
**Compensation metric:** compare imposed ratio (alpha_audio/alpha_video) vs final learned
ρ_enc. If a 10× imposition yields near-balanced ρ_enc, that is compensation.

## Statistics
5 seeds; individual values, mean ± SD, 95% CI; paired comparisons where applicable; effect
sizes reported with p-values. Slope of contribution vs log2(imposed ratio).

## Failure handling
Non-finite loss → record (seed/condition/epoch/loss/grad/NaN) as **diverged**; do not
silently drop. If a condition consistently diverges, report divergence as an outcome. If the
intervention implementation is found wrong, STOP, fix, document, do not mix results.

## Test discipline
No test inspection feeding back into epochs/scales/lr/architecture/seed/normalization.

## Verdict (choose one from evidence)
- **SUPPORT**: training-time scale produces the predicted persistent learned imbalance, reproducibly.
- **PARTIAL SUPPORT**: it changes dynamics/contribution but compensation substantially weakens the final effect.
- **NO SUPPORT**: no meaningful change in learned contribution.
- **CONTRADICTORY**: behavior systematically contradicts the mechanism.
- **TECHNICAL FAILURE**: uninterpretable (implementation/data).

## Interpretation rule
Even if supported, claim only: "under the tested CREMA-D architecture/optimization, imposing
modality-specific feature-scale disparity during training can alter learned modality
contributions." Not the broad sufficiency claim.

## Outputs
`results/cremad/cremad_train_scale_primary.csv`, `_trajectories.csv`, `_agg.csv`,
`_analysis.json`, `_config.json`; figures `cremad_TS_{rho,gradratio,contribution,gain,compensation}.png`;
report `reports/CREMAD_TRAINING_SCALE_REPORT.md`. Commit separately. Raw media + feature cache
NOT committed. Then STOP (no INCLUDE-50 / OGM-GE / other datasets / paper).
