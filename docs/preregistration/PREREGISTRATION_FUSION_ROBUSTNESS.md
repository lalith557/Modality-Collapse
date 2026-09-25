# Pre-registration — fusion-mechanism robustness of the forward-vs-gradient decomposition

Written **before** running. Authorized Phase 3: test whether the forward-vs-gradient pathway
decomposition (validated synthetically) **survives different fusion mechanisms** — sum, concat, and
learned gated fusion — under the **same Adam** as the primary synthetic disentangle. This is
**fusion-mechanism robustness**, NOT a claim of general architecture generality. Frozen files
(`models/fusion.py`, `models/multimodal_model.py`, E1–E5, CREMA-D, UTD, disentangle Adam/SGD) are
untouched; a small experiment-local model adds the gated option without editing shared code. Start
commit `4465ebc`. Date: 2026-09-21.

## Scientific question (not assumed)
Does the causal distinction — forward-only reproduces the ordinary contribution shift while gradient-only
(under Adam) is comparatively inert — hold across sum, concat, and gated fusion? A fusion-dependent result
is scientifically informative and will be reported as-is.

## Design
Same synthetic equal-information generator (`configs/boundary.yaml` data: 8 classes, dim 32+32, margin 1.5,
σ=1), same TinyMLPEncoder (hidden 64, rep 32), same validated `ForwardScale`/`GradScale` applied to
modality A. Local model with selectable fusion:
- **sum:** z = mask_a·arm(h_A) + mask_v·h_V (identical to the primary setup).
- **concat:** z = [mask_a·arm(h_A) ; mask_v·h_V], linear head on 2·rep_dim.
- **gated:** per-modality self-gates g_m = sigmoid(Linear_m(h_m)); z = mask_a·(g_A⊙arm(h_A)) +
  mask_v·(g_V⊙h_V) (a learned Gated-Multimodal-Unit-style fusion; masking a modality zeros its whole term).

## Conditions (frozen)
Arms: baseline, forward-only, gradient-only, ordinary. **α = 100.** 3 fusion × 4 arms × 5 seeds = **60 runs.**
**Adam lr 1e-3, batch 256, 40 epochs, wd 0, seeds [0–4]** (identical optimizer/budget to the primary
synthetic Adam disentangle). normalization none.

## Autograd sanity gate
Reuse the validated scalar/numeric sanity (forward α/1, gradient 1/α, ordinary α/α; gradient-only forward
bit-identical to baseline). Fail ⇒ STOP.

## Metrics
dominance = Acc_A − Acc_V; multimodal gain; ρ_fusion; ρ_encoder; encoder gradient ratio; parameter-update
ratio; compensation. Per-seed values; mean ± SD, 95% CI; paired Cohen's d + paired t within each fusion.

## Decision framework (per fusion)
CASE A forward-dominated / CASE B gradient-dominated / CASE C mixed / CASE D interactive / CASE E unstable.
Primary read: does forward-only reproduce ordinary and is gradient-only ≈ baseline, within each fusion?

## Outputs (new dir `results/fusion_robustness/`, no overwrite)
`PREREGISTRATION_FUSION_ROBUSTNESS.md` (this), `EXPERIMENT_REPORT_FUSION_ROBUSTNESS.md`,
`results/fusion_robustness/fusion_robustness_{primary,trajectories,agg}.csv`, `_analysis.json`,
`_config.json`; figures `fusion_{dominance,grad_update_ratio}.png`. Not committed pending review. Then STOP.
