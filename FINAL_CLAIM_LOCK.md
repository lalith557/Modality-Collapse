# FINAL CLAIM LOCK

_The locked claim boundary for the manuscript. Anything not in STRONGLY / CONDITIONALLY may not be stated
as a result. Source: results/** (see EXPERIMENT_FREEZE_AUDIT.md)._

## STRONGLY SUPPORTED (state directly)
- Static feature scaling is largely compensated by trainable encoders in the synthetic setting; it is not,
  alone, a sufficient cause of persistent collapse (E2, E4-EXTREME).
- Training-time modality-specific feature scaling causally modulates learned modality contribution
  (intervention-level) — synthetic, CREMA-D (Spearman 0.922; disentangle ordinary d=8.5), UTD.
- Under the tested Adam configuration, the contribution shift is reproduced by forward-only scaling and
  isolated gradient scaling is inert despite ~100–282× gradient-norm ratios (synthetic d(grad)=−0.53
  p=0.30; CREMA-D d(grad)=−0.08 p=0.87), with gradient-only parameter-update ratio ≈1.
- Under the tested SGD configuration, isolated gradient scaling contributes (d=3.90) with
  parameter-update ratio ≈α; both pathways act (ordinary ≠ forward, d=2.86).
- Gradient-inertness (under Adam) is robust across sum, concat, and gated fusion (update ratio ≈1 in all).
- On CREMA-D, forward-only reproduces the ordinary/H6 effect and gradient-only does not (five matched seeds,
  fixed actor-disjoint split).
- Per-modality z-score removes the imposed-scale pathway without improving (and sometimes reducing)
  accuracy (H6 0.328→0.110; R2 acc −0.026, p=0.014).

## CONDITIONALLY SUPPORTED (state with the stated scope)
- "The relevant mediator is the parameter-update scale, not the gradient norm" — interpretation supported
  by Adam vs SGD update-ratio contrast; scope: tested optimizers.
- "Forward-magnitude-mediated pathway" on real data — CREMA-D, Adam, matched CNN, sum fusion only.
- Utility-gating (amplifying the useful modality entrenches it; weak modality compensates) — two real
  datasets, one with a degenerate (near-chance) modality.
- Persistent collapse is optimization/LR-driven — synthetic regime; R3 frozen-rep on real data is
  consistent but frozen.
- Forward effect under gated fusion — attenuated, **not individually significant** (d=−0.70, p=0.19).
- Early activation-norm ratio is a transient diagnostic — synthetic; not a predictor of final dominance.

## NOT SUPPORTED (must remain here; never stated as a result)
- Feature scale is a sufficient cause of (persistent) modality collapse.
- Feature scale universally silences modalities.
- Per-modality normalization cures modality collapse.
- Gradient imbalance is irrelevant.
- Adam is (universally) invariant to gradient scaling.
- Universal architecture generality; universal dataset generality.
- UTD-MHAD is a symmetric replication.
- OGM-GE is generally ineffective.
- Any unsupported "first to …" claim.
- Any state-of-the-art / "best" claim (this is an analysis paper).

## UNKNOWN (explicitly open; disclosed in Limitations)
- SGD real-data disentanglement.
- Long-horizon persistence of the exact SGD disentanglement (>40 ep).
- Attention/transformer fusion behavior.
- Whether Adam's gradient inertness is moment normalization vs shared-head absorption.
- Broader optimizer/LR ranges.
