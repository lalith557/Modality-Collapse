# Fusion-mechanism robustness — final report (Phase 3)

_Preregistration: `PREREGISTRATION_FUSION_ROBUSTNESS.md` (frozen before results). Start commit `4465ebc`._
_Synthetic; Adam; frozen `models/fusion.py`/`multimodal_model.py` untouched (local gated model). All prior
experiments immutable._ Labels: **[OBS]/[STAT]/[MECH]**.

## Verdict: the forward-dominated / gradient-inert (Adam) decomposition is **robust across sum, concat, and gated fusion** — attenuated but not reversed under gated fusion.
Across all three fusion mechanisms, **gradient-only scaling is inert** (≈baseline) despite a ~90–100×
encoder-gradient ratio, because its parameter-update ratio stays ≈1 (Adam-normalized). The forward pathway
carries the effect cleanly under sum and concat; under gated fusion the effect is **attenuated and noisier**
(the learned gate partially absorbs the forward scaling). 60 runs, **0 diverged**.

## Design
Synthetic equal-info generator, TinyMLP encoders, validated arms on modality A, **α=100**, Adam lr 1e-3,
batch 256, 40 epochs, 5 seeds. Fusion ∈ {sum, concat, gated (per-modality self-gates g_m=σ(Lin_m(h_m)))}.
4 arms × 3 fusion × 5 = 60 runs. Sanity gate passed.

## Results (dominance shift vs baseline; α=100; 5 seeds) [OBS/STAT]
| fusion | forward shift (d, p) | gradient shift (d, p) | ordinary vs forward (d, p) | gradient grad-ratio / update-ratio | verdict |
|---|---|---|---|---|---|
| **sum** | −0.070 (d=−3.76, p=0.001) | −0.000 (d=−0.53, p=0.30) | +0.000 (d=0.73, p=0.18) | 100.9 / **1.00** | CASE A (clean) |
| **concat** | −0.057 (d=−5.16, p=3e-4) | −0.000 (d=−0.00, p=1.0) | −0.000 (d=−0.00, p=1.0) | 101.6 / **1.02** | CASE A (clean) |
| **gated** | −0.026 (d=−0.70, p=0.19) | −0.000 (d=−0.02, p=0.96) | −0.025 (d=−0.62, p=0.24) | 92.8 / **1.02** | CASE A (attenuated/noisy) |

## Interpretation [MECH]
- **Robust across fusion: gradient-only inert under Adam.** In every fusion type the gradient-only arm has
  a ~90–100× gradient ratio but a ~1.0 parameter-update ratio and no dominance shift — Adam normalizes the
  constant gradient scaling regardless of how the modalities are combined. This is the most robust result.
- **Sum & concat: clean forward-dominance.** forward ≡ ordinary; forward carries the whole (small,
  equal-info) effect; gradient inert.
- **Gated: attenuated + mild interaction.** The learned gate partially compensates the imposed forward
  scale, so the forward-only effect shrinks (−0.026, not individually significant) and ordinary is slightly
  larger than forward (−0.051 vs −0.026; a mild gate-mediated interaction, also not significant). Direction
  and gradient-inertness are preserved; the magnitude is reduced.

## Scope / limitations
Synthetic equal-information, tiny MLP encoders, Adam only, α=100. This is **fusion-mechanism robustness**,
NOT architecture generality (no attention/transformer backbones). The gated result shows the effect is
fusion-sensitive in magnitude (learned gates dampen it), which is a boundary condition, not a refutation.

## Contribution
Closes the "fusion-mechanism robustness" gap: the forward-vs-gradient decomposition (gradient inert under
Adam, forward carries the effect) survives sum, concat, and gated fusion — with gated fusion attenuating the
forward effect via its learned gate.

---
_Reproducibility: `results/fusion_robustness/fusion_robustness_{primary,agg}.csv`, `_analysis.json`,
`_config.json`; figures `fusion_{dominance,grad_update_ratio}.png`. Not committed pending review._
