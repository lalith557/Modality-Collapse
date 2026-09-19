# Pre-registration

Written **before** the final E1/E3 runs. Fixed choices are not to be changed
after seeing results. (Timestamp: 2026-09-19, after Milestone-1 / E2.)

## E1 — early-ρ diagnostic (H1)

**Primary early checkpoint:** **epoch 5** (pre-registered; chosen from the E2
observation that scale-driven imbalance is maximal early, before compensation).

**Primary predictor:** `log(rho_epoch5)` where
`rho = E‖h_A‖₂ / E‖h_V‖₂` at the fusion boundary.

**Primary outcomes:**
- Final contribution imbalance `D = Δ_A − Δ_V` (test split).
- Identity of the weaker modality (binary: `A_weaker = 1` iff `Δ_A < Δ_V` at final).
- Contribution imbalance at intermediate checkpoints (epochs 1,3,5,10,final; val split).

**Run population (varied factors):** random seed; scale ratio (symmetric
geometric); moderate information differences (`margin_A ≠ margin_V`); moderate
noise differences (`sigma_A ≠ sigma_V`); encoder initialization (via seed);
fusion (sum vs concat). A clean subset holds information exactly equal.

**Beyond-scale test (decisive for H1).** A correlation driven only by the
injected scale is NOT sufficient. We pre-commit to three controls:
1. **Partial correlation** of `log(rho5)` vs `D` **controlling for `log(injected_scale)`**
   (residual-on-residual Spearman & Pearson).
2. **Balanced-scale subfamily** (`injected_scale = 1`): any `rho5`→`D` association
   here is necessarily beyond injected scale.
3. **Incremental R²**: does `log(rho5)` explain variance in `D` beyond `log(injected_scale)`?

**Statistics:** Spearman & Pearson (with 95% bootstrap CIs), AUROC (95% bootstrap
CI) for `A_weaker`, run counts, seed-level variation. AUROC reported **as a
function of prediction-horizon epoch** (epoch-5 ρ predicting dominance at epoch k).

**Verdict rule (fixed in advance).** Let `r_partial` = partial Spearman(log rho5, D | log scale)
and `r_bal` = Spearman(log rho5, D) in the balanced-scale subfamily.
- **SUPPORTED**: overall |Spearman| > 0.6 AND `|r_partial|` > 0.3 (95% CI excludes 0)
  AND `|r_bal|` > 0.3 (95% CI excludes 0).
- **PARTIALLY SUPPORTED**: overall association strong (|Spearman| > 0.6) but the
  beyond-scale controls are weak/inconsistent (one of r_partial/r_bal fails, or
  early-horizon prediction is strong while final-horizon prediction is not).
- **INCONCLUSIVE**: mixed/underpowered.
- **UNSUPPORTED**: overall |Spearman| ≤ 0.3.

## E3 — per-modality normalization (H3)

**Interventions:** none, LayerNorm, per-modality z-score (train-fit statistics,
no val/test leakage), BatchNorm (only if appropriate — reported with caveat).
**Scale ratios (symmetric geometric):** 0.05, 0.1, 0.2, 1, 5, 10, 20.
**Checkpoints:** epochs 1, 3, 5, 10, final. **Seed policy:** same 5 seeds across
all interventions (paired comparison).
**Primary question:** does normalization prevent the **early** optimization
imbalance (epoch-5 `|D|` and ρ deviation), not merely improve final accuracy?
**Cure rule (fixed):** normalization is called a *cure* only if it reduces
early `|D|` at collapse-inducing ratios by a large, CI-separated margin AND
restores the weak modality's contribution (Δ_weak) toward the balanced level.

## Compensation analysis (new, treated cautiously)
Quantify learned compensation of injected scale by the encoders:
- **compensation ratio** `c(t) = log(rho_t) / log(injected_scale)` (1 = none, 0 = full).
- **time-to-compensation**: first epoch where `|log rho_t| < 0.5·|log injected_scale|`.
- **residual imbalance**: `log(rho_final)`.
- **direction asymmetry**: compensation for scaling A up vs V up.
- Whether normalization removes/reduces the need for learned compensation.
Robustness across seeds AND fusion required before any claim; not asserted as a
discovery in this stage.
