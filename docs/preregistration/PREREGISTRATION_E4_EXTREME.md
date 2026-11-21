# Pre-registration — E4-EXTREME (extreme-ratio + long-horizon resolution)

Written **before** running. A NEW, separately-labelled experiment motivated by E4
(Outcome B). Does **not** modify E1–E4 or their preregistrations. Reuses the E4
generator, architecture, optimizer, metrics, and **frozen collapse definition**.
Timestamp: 2026-09-19.

## Scientific question
Does progressively stronger **aligned** scale × learning-rate asymmetry convert the
graded modality imbalance observed in E4 into **persistent collapse**, or does the
encoder continue to compensate (given enough training)? Neither outcome assumed.

## Held identical to E4 (no new architecture/generator/metric)
Synthetic equal-information generator (`margin_A=margin_V=1.5`, `σ_A=σ_V=1`, 8
classes, 32+32 dims, `prototype_seed=12345`); tiny MLP encoders, **sum** fusion,
rep_dim 32, linear head; Adam, batch 256, base_lr=1e-3; per-encoder LR groups.
Scale `α_A=√r_s, α_V=1/√r_s`; LR `lr_A=base·√r_l, lr_V=base/√r_l`. **5 seeds.**
Interventions: `none`, per-modality `zscore` (E3/E4 implementation).

## Frozen collapse definition (identical to PREREGISTRATION_E4.md)
Final window = last 5 epochs of a run. `weak_window` = mean over that window of
val `min(Δ_A,Δ_V)`. `baseline_weak` = mean over 5 seeds of `weak_window` at the
**balanced cell (scale=1, lr=1, none, 40 epochs)** (cross-check ≈ E4's 0.192).
Severity `S = clip(1 − weak/baseline_weak, 0, 1)`. **Persistent collapse (PC) = 1**
iff `weak ≤ 0.5·baseline_weak` AND per-epoch mean weak `≤ 0.6·baseline_weak`
across every epoch of the final window. Not re-defined after seeing results.

## Conditions (targeted — NOT a grid). Both modality directions (A-favoured and its V-favoured reciprocal).
- **Balanced control:** (1, 1).
- **Scale-only:** (20,1), (100,1), (400,1)  [+ reciprocals (1/20,1),(1/100,1),(1/400,1)].
- **LR-only:** (1,20), (1,100), (1,400)      [+ reciprocals (1,1/20),(1,1/100),(1,1/400)].
- **Aligned:** (20,20), (100,100), (400,400)  [+ reciprocals (1/20,1/20),(1/100,1/100),(1/400,1/400)].
- **Opposed:** (20,1/20), (100,1/100), (400,1/400) [+ reciprocals (1/20,20),(1/100,100),(1/400,400)].

`r_s = α_A/α_V`, `r_l = lr_A/lr_V`. 25 unique conditions (balanced is self-reciprocal).

## Time horizons (frozen priority)
- **Set M (40 epochs):** all 25 conditions × {none, zscore} × 5 seeds.
- **Set L (long horizon, 100 and 200 epochs):** the aligned conditions
  (20/20, 100/100, 400/400 and V-favoured reciprocals) × {none, zscore} × 5 seeds,
  plus (1,1), (400,1), (400,1/400) for trajectory reference.
  If compute is constrained, preserve priority: **1) 400×/400× long, 2) 100×/100×
  long, 3) 20×/20× long.** Not changed after seeing results.
- Divergence guard: if training produces non-finite loss (possible at lr_A up to
  0.02), the run is recorded as **diverged** (not silently dropped).

## Primary measurements (per seed / condition / epoch)
scale ratio, LR ratio, modality direction, epoch, ρ, log ρ, Δ_A, Δ_V,
weak-modality contribution, |Δ_A−Δ_V|, multimodal gain, accuracy, macro-F1,
compensation (residual |log ρ|, comp_ratio, time-to-compensation), collapse status
(frozen def). Primary metric = frozen collapse/severity + residual |log ρ|. No new
primary metric introduced after seeing results.

## Key analyses (frozen)
- **Q1 dose-response:** does residual |log ρ| / severity increase monotonically with
  aligned magnitude (log₂ ∈ {4.32, 6.64, 8.64})? Report effect size, CI, seed
  variability; test monotonicity (Spearman); model saturation/nonlinearity if present.
- **Q2:** does compensation progressively weaken with aligned magnitude?
- **Q3:** does any condition cross the frozen PC criterion?
- **Q4:** if collapse, does it persist at 100 / 200 epochs (compare weak_window across horizons)?
- **Q5:** if no collapse, does the system eventually compensate despite severe early imbalance?
- **Q6:** does z-score remove the scale component (scale-only recovered) while leaving
  the LR component (LR-only unchanged; aligned = scale-part removed, LR-part remains)?

## Classification (choose exactly one, from results not publication value)
- **EXTREME-A — TRUE PERSISTENT REGIME:** stronger aligned asymmetry gives
  reproducible PC=1 that **survives 100–200 epochs**.
- **EXTREME-B — GRADED COMPENSATION FAILURE:** increasingly severe/long-lived
  imbalance, but eventual recovery; PC not robustly crossed.
- **EXTREME-C — COMPENSATION ROBUST:** even extreme asymmetry recovers.
- **EXTREME-D — UNSTABLE/INCONCLUSIVE:** too noisy/divergent/asymmetric to conclude.

## Outputs
`results/processed/e4_extreme_runs.csv`, `e4_extreme_trajectories.csv`,
`results/e4/e4_extreme_analysis.json`, figures
`results/figures/e4_extreme_{dose_response,long_horizon_trajectories,normalization_decomposition}.png`,
report `docs/experiments/E4_EXTREME_FINAL_REPORT.md`. Commit separately. Then STOP (no E5/OGM-GE/CREMA-D/INCLUDE).
