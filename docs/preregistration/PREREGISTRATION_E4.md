# Pre-registration — E4 (scale × learning-rate boundary)

Written **before** the final E4 grid. Frozen choices are not changed after seeing
results. Builds on E1/E2/E3 (already committed; not modified). Timestamp: 2026-09-19.

## Research question
Does persistent modality collapse emerge when static feature-scale disparity is
combined with **asymmetric optimization dynamics** (per-modality learning-rate
disparity)? Specifically: can an unfavorable learning-rate ratio **prevent the
learned compensation** that (E1/E3) otherwise dissolves scale-driven collapse by
convergence — turning *transient* imbalance into *persistent* collapse?

We do NOT aim to force persistent collapse. We aim to measure whether/where it occurs.

## Design (frozen)
Two axes, both symmetric geometric, **signed with reciprocals** so each modality is
tested as the dominant and the weaker side.

- **Scale ratio** `r_s = alpha_A/alpha_V ∈ {1/20,1/10,1/5,1/2,1,2,5,10,20}`
  applied as `alpha_A=sqrt(r_s)`, `alpha_V=1/sqrt(r_s)`.
  Primary axis: `log2(r_s)` (symmetric about 0).
- **Learning-rate ratio** `r_l = lr_A/lr_V ∈ {1/20,1/10,1/5,1/2,1,2,5,10,20}`
  applied as `lr_A=base_lr*sqrt(r_l)`, `lr_V=base_lr/sqrt(r_l)` on the existing
  per-encoder optimizer parameter groups. `base_lr=0.001`. Head/norm params stay at `base_lr`.
  Primary axis: `log2(r_l)` (symmetric about 0).

Full **9×9 signed grid** (81 cells). Gap magnitudes {1,2,5,10,20} per the spec,
expanded to signed reciprocals for the symmetric phase diagram.

**Interventions (frozen):** `none` and per-modality `zscore` **only** (the E3-validated
z-score). No LayerNorm/BatchNorm, no OGM-GE (OGM-GE is E5, after the phase diagram).

**Held constant (isolation control):** information is **exactly equal** in the
primary grid (`margin_A=margin_V=1.5`, `sigma_A=sigma_V=1.0`), same synthetic
generator, architecture (tiny MLP, sum fusion, rep_dim 32), Adam optimizer, batch
256, **40 epochs**, **5 seeds**, `prototype_seed=12345`, as validated in E2/E3.
Only scale and per-encoder LR vary. Any deviation is documented in the report.

**Robustness (secondary, separately labeled):** repeat the grid with `concat` fusion
(≥3 seeds) to test architecture-specificity. Not part of the primary preregistered claim.

## Primary outcomes (per cell, mean ± std, 95% CI, paired seeds across norms)
`Delta_A`, `Delta_V`, `|Delta_A − Delta_V|` (imbalance), `min(Delta_A,Delta_V)`
(weak-modality contribution), final accuracy, macro-F1, multimodal gain,
fusion-boundary `rho`, and the compensation trajectory (below).

## Persistent-collapse definition (frozen; derived from the balanced baseline)
Let the **final window** = last 5 epochs (epochs 36–40 for a 40-epoch run).
For a run, `weak_window = mean over the final window of val min(Delta_A,Delta_V)`.
- `baseline_weak` = mean over the 5 seeds of `weak_window` at the **balanced cell**
  `(log2 r_s = 0, log2 r_l = 0, none)`. `baseline_std` = its across-seed std.
  (Cross-check: E2/E3 balanced weak-contribution ≈ 0.19.)
- **Continuous collapse severity** `S(cell,norm) = clip(1 − weak_cell/baseline_weak, 0, 1)`
  (0 = no collapse, 1 = full collapse), where `weak_cell` = mean over seeds of `weak_window`.
- **Binary persistent-collapse indicator** `PC(cell,norm) = 1` iff **both**:
  1. `weak_cell ≤ 0.5 · baseline_weak` (weak contribution at least halved), **and**
  2. it does **not recover** at the end: the per-epoch mean weak-contribution stays
     `≤ 0.6 · baseline_weak` across **every** epoch in the final window.

Both S and PC are reported; raw `min(Delta)` is saved so other thresholds can be applied.

## Compensation metric (consistent with E3)
Per cell: initial `log rho` (epoch 0), `rho` at epochs 5 and 10, final `rho`,
`comp_ratio_final = log rho_final / log(r_s)` (1 = none, 0 = full),
`compensation_achieved = 1 − |log rho_final| / |log r_s|`,
`time_to_compensation` (first epoch with `|log rho_t| < 0.5·|log r_s|`),
`residual_abs_logrho = |log rho_final|`. Applicable only where `r_s ≠ 1`.
**Mechanistic test:** does `PC=1` coincide with **failure of compensation**
(large residual `|log rho|`, no/late time-to-compensation)?

## Normalization test (paired, per cell)
`none` vs `zscore` on: early imbalance (epoch 5 `|D|`), final imbalance,
weak-modality contribution, final accuracy, macro-F1, compensation time.
**Normalization benefit** = `weak_cell(zscore) − weak_cell(none)`.
Critical contrast (NOT assumed): in the **scale-only** regime (LR ratio = 1)
normalization should mainly help dynamics with little/no final-accuracy benefit
(as E3 found); in the **scale+optimization** regime does it *recover* a modality
that otherwise stays suppressed?

## Boundary classification (frozen options; do not force)
Classify the phase diagram of `|D|`/`S` as one of:
- **A. no boundary** (flat/no structure),
- **B. gradual transition**,
- **C. clear scale-driven vs optimization-driven boundary**.
Quantify with the measured metrics (e.g., variance of `S` explained by `|log2 r_s|`
vs `|log2 r_l|` vs their interaction; sharpness of the transition). Report a fuzzy
boundary as fuzzy.

## QC (must pass before results are reported)
Assert: scale/LR ratios valid & symmetric (each `r` has its `1/r`); information
equal in every primary cell (`margin_A==margin_V`, `sigma_A==sigma_V`); optimizer
parameter groups carry the intended per-encoder LRs; no train/test leakage
(independent synthetic splits); every cell has exactly the expected seed count; no
NaN/Inf; no incomplete cells.

## Outputs
`results/processed/e4_grid.csv`, `e4_summary.csv`, `e4_statistics.csv`;
`results/figures/e4_phase_diagram_imbalance.png`, `e4_phase_diagram_collapse.png`,
`e4_normalization_benefit.png`, `e4_compensation_boundary.png`;
`results/reports/E4_REPORT.md`.

## Decision tree (report which outcome; do not force)
- **A**: persistent collapse appears AND zscore reliably recovers it → supports scale×optimization mechanism.
- **B**: persistent collapse appears but zscore does not recover it → mechanism plausible, normalization claim weakens.
- **C**: no persistent collapse even with LR disparity → architecture is compensation-capable; reconsider framing before real data.
- **D**: a different mechanism dominates → analyze it rather than forcing the hypothesis.
