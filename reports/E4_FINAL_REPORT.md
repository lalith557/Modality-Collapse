# E4 FINAL REPORT — Scale × Learning-Rate Boundary

_Project: modality-scale-collapse. Preregistration: `PREREGISTRATION_E4.md` (frozen before the run)._
_Labels used below: **[OBS]** observed data, **[STAT]** statistical result, **[MECH]** mechanistic interpretation, **[SPEC]** speculation._

---

## 1. Executive summary
A pre-registered 9×9 signed grid of **feature-scale ratio × per-encoder learning-rate ratio** (5 seeds, 40 epochs, equal information, {none, z-score}) shows:

- **[STAT] A robust scale×LR interaction.** Collapse severity concentrates where scale and LR disfavour the *same* modality ("aligned"): aligned severity **0.244 [0.220, 0.270]** vs opposed **0.045 [0.036, 0.054]** (non-overlapping CIs); the aligned interaction term is the most significant effect in the response-surface model (p=7×10⁻¹¹⁷).
- **[STAT] LR disparity matters more than scale disparity** as a main effect (coef 0.048 vs 0.013).
- **[MECH] Collapse ⟺ failed compensation.** Residual |log ρ| at convergence is 0.74 (aligned) vs 0.20 (opposed); Spearman(residual, severity)=0.75. An unfavourable LR on the down-scaled modality blocks the learned compensation seen in E1/E3.
- **[OBS] But persistent collapse is RARE.** By the frozen criterion only **2/81 cells** (none) show persistent collapse — the two *extreme aligned corners* (20×/20× and 0.05×/0.05×); **0/81** under z-score; **0/81** with concat fusion. Per-run collapse is 8.7% of aligned runs, 3.5% overall.
- **[STAT] z-score is regime-dependent, not a universal cure.** It removes the scale and alignment terms of collapse (quantitatively cancels them) and partially recovers the weak modality in the collapse corners (weak 0.079→0.131; severity ~halved), but slightly *reduces* accuracy overall (−0.024) and adds sensitivity to the pure-LR axis it cannot fix.

**Classification: OUTCOME B — PARTIAL SUPPORT** (see §19).

## 2. Preregistration compliance
Grid, parameterization, interventions, seeds, epochs, information control, persistent-collapse rule, and QC were taken **verbatim** from `PREREGISTRATION_E4.md`. Nothing was changed after seeing results: the collapse threshold, grid, primary metric, and seed set are as frozen.

**Discrepancy noted (not a change):** a later instruction described the grid as `{1,2,5,10,20,50,100,200,400}`. The pre-registered and actually-run grid is the **signed symmetric** set `{0.05,0.1,0.2,0.5,1,2,5,10,20}` (gaps {1,2,5,10,20} + reciprocals) for both axes — required for the opposed-condition (§8) and symmetry (§12) analyses, which the unsigned set cannot support. The preregistration was honoured; extending to larger gaps would be a **new** preregistered experiment (see §19).

## 3. Experimental configuration
Synthetic equal-information generator (`margin_A=margin_V=1.5`, `σ_A=σ_V=1`, 8 classes, 32+32 dims, `prototype_seed=12345`); tiny MLP encoders, **sum** fusion, rep_dim 32, linear head; Adam, batch 256, **40 epochs**, base_lr=1e-3. Scale: `α_A=√r_s, α_V=1/√r_s`. LR: `lr_A=base·√r_l, lr_V=base/√r_l` on per-encoder optimizer groups (head/norm at base_lr). Grid 9×9 × {none, z-score} × 5 seeds = **810 runs**. Robustness: concat fusion × 3 seeds. All on GPU (RTX 4070).

## 4. QC results
| check | result |
|---|---|
| total runs | **810 / 810** |
| cells × seeds | 162 cells, **exactly 5 seeds each** (0 incomplete) |
| grid coverage | 9 scales × 9 LRs × 2 norms ✓, all reciprocals present |
| LR parameterization error | max \|lr_A − base·√r_l\| = **9.5×10⁻¹⁷** |
| scale parameterization error | max \|α_A − √r_s\| = **5.6×10⁻¹⁷** |
| optimizer param-groups | verified: enc_A=lr_A, enc_V=lr_V, rest=base_lr |
| information equal | margin_A=margin_V, σ_A=σ_V in every cell ✓ |
| non-finite values | 270 = exactly the 3 compensation fields × 90 scale=1 runs (correctly N/A) — no real NaN |
| balanced cell vs E2/E3 | weak **0.192±0.013**, acc **0.699**, \|D\| 0.009 ✓ |
Reproducibility audit passed: only scale and per-encoder LR vary; splits independent; seeds isolated.

## 5. Primary outcome
**[OBS]** Persistent collapse (frozen per-cell rule) occurs in **2/81** cells (none), **0/81** (z-score). Both are the **extreme aligned corners**:
- (α_A/α_V=0.05, lr_A/lr_V=0.05): severity 0.587, weak 0.079, residual |log ρ| 1.61 → **A collapses** (V favoured by both).
- (20, 20): severity 0.548, weak 0.087, residual |log ρ| 1.66 → **V collapses** (A favoured by both).

Run-collapse fraction: none 3.5%, z-score 0.7%. Max severity (none) 0.59; mean 0.10. **[OBS] The interaction produces a graded severity surface; binary persistent collapse is confined to the most extreme aligned cells.**

## 6. Persistent-collapse phase diagram
`results/figures/e4_final_A_severity.png` (severity, none | z-score, shared 0–1 scale) and `e4_final_B_collapse_fraction.png`, `e4_final_C_weak_contribution.png`, `e4_final_D_multimodal_gain.png`. **[OBS]** Under *none*, severity forms a clear diagonal band through the two aligned corners; the opposed anti-diagonal stays near zero. Under *z-score*, the scale-dependence is flattened but a faint LR-driven band remains in the extreme-LR rows.

## 7. Scale × LR interaction  [STAT]
Response-surface OLS on per-run severity (n=810, R²=0.70), predictors in log₂ units:

| term | coef | 95% CI | p |
|---|---:|---|---:|
| abs(log₂ scale) | +0.0131 | [0.008, 0.018] | 1.5e-06 *** |
| **abs(log₂ lr)** | **+0.0476** | [0.042, 0.053] | 7e-59 *** |
| **log₂scale·log₂lr (alignment)** | **+0.0133** | [0.0124, 0.0143] | 7e-117 *** |
| norm(z) | +0.0312 | [0.003, 0.059] | 0.029 * |
| norm : abs(log₂ scale) | −0.0180 | [−0.026, −0.011] | 2.9e-06 *** |
| norm : abs(log₂ lr) | +0.0250 | [0.018, 0.033] | 1.2e-10 *** |
| **norm : alignment** | **−0.0140** | [−0.0154, −0.0127] | 7e-75 *** |

**[STAT]** LR magnitude is the dominant main effect; the **aligned interaction is the most significant term**. **[STAT]** z-score's negative interactions with scale (−0.018) and alignment (−0.014) **cancel** the corresponding base terms (+0.013 each), i.e. z-score removes the scale- and alignment-driven collapse; its positive `norm:abs_ll` (+0.025) shows it does **not** remove — slightly amplifies — the pure-LR component. (**[STAT]** Regression characterises the response surface; causal weight comes from the controlled interventions, not R².)

## 8. Aligned / orthogonal / opposed / balanced  [STAT]
Severity mean [95% bootstrap CI], weak-contribution, collapse fraction:

| class | none severity | none weak | none collapse-frac | z-score severity | z-score weak |
|---|---:|---:|---:|---:|---:|
| balanced (1,1) | 0.031 [0.009,0.052] | 0.192 | 0.00 | 0.057 | 0.185 |
| orthogonal (one axis) | 0.091 [0.072,0.111] | 0.177 | 0.00 | 0.132 | 0.171 |
| **aligned** | **0.244 [0.220,0.270]** | **0.145** | **0.087** | 0.191 | 0.157 |
| opposed | 0.045 [0.036,0.054] | 0.190 | 0.00 | 0.199 | 0.155 |

**[OBS]** Aligned ≫ opposed under *none* (CIs non-overlapping) — the interaction survives 40 epochs and 5 seeds. **[OBS]** Under z-score, aligned (0.191) ≈ opposed (0.199): normalization **equalises** the alignment axis (removes the interaction) while the LR-driven baseline rises — consistent with §7.

## 9. Persistence vs transient (trajectories)  [OBS]
`results/figures/e4_final_trajectories.png` (mean of 5 seeds):
- **pure scale (20,1):** weak starts ≈0, **fully recovers to ≈0.19 by ~epoch 10** (ρ relaxes 2.4→0.5) → **transient** (reproduces E1/E3).
- **opposed (20, 1/20):** ρ **crashes to ≈0 by epoch 5**, weak recovers fast → LR favouring the weak modality *accelerates* compensation.
- **aligned (20,20):** ρ **rises to 3.0 then only to 1.6 by epoch 40**; weak crawls 0→0.095 → **persistent within the 40-epoch budget** (compensation blocked).
- **aligned + z-score:** partial rescue (weak →0.14; ρ held ≈0.5). 

## 10. Compensation mechanism  [MECH]
**[STAT]** Residual |log ρ| at convergence (none): aligned 0.74, orthogonal 0.32, opposed 0.20. Spearman(residual |log ρ|, severity)=**0.75** (p≈0, n=720). `results/figures/e4_final_compensation_vs_severity.png`. **[MECH]** Greater scale/LR alignment → larger residual activation imbalance (learned compensation fails) → higher collapse severity. The data **support** the compensation-failure mechanism; they do not contradict it.

## 11. Boundary estimation  [OBS/STAT]
**[OBS]** Because persistent collapse is confined to 2 extreme cells, a statistically justified 50%-collapse contour does **not** exist within the tested range — the transition is a **smooth response surface**, not a sharp phase boundary. Terminology: **collapse region** (extreme aligned corners) within a **gradual transition**. The minimum ratio producing per-run collapse fraction ≥0.5 is only reached at the ±(20×,20×) corners; no interior boundary is resolvable at ratios ≤20× / 40 epochs.

## 12. Symmetry analysis  [STAT]
Aligned-corner severity: A-up (V collapses) **0.217** vs V-up (A collapses) **0.272**; Mann–Whitney p=**0.039**. **[OBS]** A **mild but significant asymmetry** — collapse of A (when V is favoured) is somewhat more severe. **[SPEC]** Likely from the distinct prototype seed for each modality (`proto_seed` vs `proto_seed+1`) or the A/V ordering in fusion; not hidden, flagged as a limitation/clue.

## 13. Robustness (concat fusion)  [OBS]
Same aligned structure (aligned severity 0.13 vs opposed 0.029) but **weaker**, and **0/81 persistent-collapse cells**. **[MECH]** Concat keeps modality representations separable, so the weak one is not additively swamped as in sum fusion — the phenomenon is architecture-general in *shape* but sum fusion is more collapse-prone.

## 14. Negative findings  [OBS]
- Static scale disparity **alone** (LR=1 column) causes **no** persistent collapse at 40 epochs (compensation wins) — reaffirms E1/E3.
- The scale×LR interaction, though highly significant, is **insufficient** to produce persistent collapse across most of the grid at ratios ≤20× — only the extreme aligned corners cross the threshold.
- z-score does **not** improve final accuracy (−0.024 overall) and even raises severity in some LR-dominated cells — **not** a universal cure.

## 15. Limitations
40-epoch budget defines "persistent" operationally; longer training could recover the aligned corners (untested). Ratios capped at 20× per axis (larger gaps untested). Single synthetic generator, sum/concat only (no cross-attention). z-score uses init-time train stats (E3 caveat). Mild A/V asymmetry (§12).

## 16. Scientific interpretation
**[MECH]** The controlled interventions establish a coherent mechanism: feature-scale disparity creates an early activation-norm imbalance (E2/E1) that a trainable encoder normally compensates (E1/E3); an **aligned unfavourable learning-rate ratio slows the weak modality's encoder enough to block that compensation** within the training budget, converting transient imbalance toward persistent collapse. Normalization neutralises the *scale* channel of this mechanism (structurally fixing ρ) but not the *optimization* channel. **[SPEC]** With larger ratios or longer/greedier optimization, the collapse region would likely expand — but that is not yet demonstrated.

## 17. What E4 establishes
- **[STAT]** A real, highly significant, reproducible **scale×LR aligned interaction** in collapse severity, present in sum and concat fusion.
- **[MECH]** That the interaction operates through **compensation failure** (residual ρ ↔ severity, Spearman 0.75), with LR the dominant lever.
- **[STAT]** That per-modality z-score **removes the scale/alignment component** of collapse and partially recovers the weak modality in the collapse corners — a **regime-dependent** benefit absent in the scale-only regime (E3).

## 18. What E4 does NOT establish
- It does **not** establish that scale×LR produces **widespread or sharp** persistent collapse (only 2/81 extreme cells; smooth surface).
- It does **not** establish z-score as a general cure (net accuracy cost; ineffective on the pure-LR axis).
- It does **not** establish permanence (40-epoch budget) or generality beyond the synthetic equal-information setting.
- No claim of novelty, SOTA, or superiority to OGM-GE is made.

## 19. GO / NO-GO recommendation
**OUTCOME B — PARTIAL SUPPORT.** The scale×LR mechanism is real, significant, mechanistically explained, and reproducible, but persistent collapse and normalization recovery are **weaker and less general** than the strong hypothesis.

**Recommendation: refine the claim before real data (conditional GO).**
1. **Highest priority — a NEW pre-registered "extreme-ratio / long-horizon" grid** to map where persistent collapse becomes common: extend aligned ratios beyond 20× (the earlier `{…,50,100,200,400}` idea, now scientifically motivated) and/or 100–200 epochs, keeping the frozen collapse rule. This decides whether the collapse region is a genuine expanding regime or a synthetic-only edge effect.
2. **Then E5 (OGM-GE) only in the aligned-collapse region**, where a real effect exists to compare against z-score.
3. **Hold** CREMA-D / INCLUDE until (1) clarifies whether persistent collapse is robust enough to be worth testing on real data.

Do **not** proceed to real datasets or OGM-GE on the current evidence alone.

---
_Machine-readable: `results/e4/e4_primary_summary.csv` (per-run), `e4_primary_aggregated.csv` (per-cell mean/std/CI/n), `e4_qc.csv`, `e4_analysis.json`, `e4_normalization_by_cell.csv`. Figures: `results/figures/e4_final_{A_severity,B_collapse_fraction,C_weak_contribution,D_multimodal_gain,compensation_vs_severity,trajectories}.png` and the E4 phase diagrams `e4_phase_diagram_{imbalance,collapse}.png`, `e4_normalization_benefit.png`, `e4_compensation_boundary.png`._
