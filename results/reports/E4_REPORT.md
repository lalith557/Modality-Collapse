# E4 report — scale × learning-rate boundary (primary)

## 1. Research question
Does persistent modality collapse emerge when static feature-scale disparity is combined with asymmetric per-modality optimization (learning-rate) dynamics — i.e. can an unfavourable LR ratio block the learned compensation that otherwise dissolves scale-driven collapse by convergence?

## 2. Experimental design
Signed geometric grid: scale ratio [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0] × lr ratio [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0] (log2 axes, reciprocals included). Interventions ['none', 'zscore']. Fusion=sum. 5 seeds, 40 epochs. alpha_A=sqrt(r_s),alpha_V=1/sqrt(r_s); lr_A=base*sqrt(r_l),lr_V=base/sqrt(r_l), base=0.001. Same generator/architecture/optimizer/batch as E2/E3.

## 3. Controls
Information held EXACTLY equal (margin_A=margin_V, sigma_A=sigma_V); only scale and per-encoder LR vary. QC: optimizer param-group LRs verified; grid completeness + ratio symmetry checked; no NaN; independent synthetic splits.

## 4. Results
- Baseline weak-contribution (balanced cell, none) = 0.192 ± 0.013.
- Persistent-collapse cells (none): **2/81**; (zscore): 0.
- Max collapse severity (none) = 0.587; mean = 0.123.
- Mean severity by alignment sign(log2 scale)*sign(log2 lr): aligned=0.241, orthogonal=0.077, opposed=0.030.

## 5. Phase-diagram interpretation
Boundary type: **C (clear alignment-driven boundary: collapse when scale & LR disfavour the same modality)**. R²(severity): scale-only=0.02, lr-only=0.26, scale+lr=0.27, +aligned-interaction=0.91 (Δ=0.64).
See results/figures/e4_phase_diagram_imbalance.png and e4_phase_diagram_collapse.png.

## 6. Compensation analysis
Residual |log ρ| (none) ranges 0.02–1.66. Correlation(residual |log ρ|, collapse severity) = 0.93 — tests whether persistent collapse coincides with failed compensation. See e4_compensation_boundary.png.

## 7. Normalization effect
- z-score weak-contribution benefit vs none: -0.011 mean across cells; CI-separated in 39/81 cells.
- Mean recovery within none-collapsed cells: +0.052 (baseline weak 0.192).
- Mean final-accuracy change (zscore − none) across grid: -0.024.
See e4_normalization_benefit.png.

## 8. Failures / caveats
Only sum fusion + z-score in the primary grid; concat is a separate robustness run. 40-epoch budget defines 'persistent' operationally; longer training could shift the boundary. z-score uses init-time train statistics (E3 caveat).

## 9. Statistical uncertainty
5 seeds/cell; per-cell std and paired (norm) 95% CIs in results/processed/e4_statistics.csv. No best-seed reporting.

## 10. Scientific conclusion
**Outcome B — persistent collapse appears but z-score does not reliably recover it.**

## 11. Implications for H1/H2/H3
- H2: whether static scale is *sufficient* for persistent collapse is answered by the LR=1 column vs the off-axis cells.
- H1: early ρ remains a transient signal; residual |log ρ| is the convergence-time analogue used here.
- H3: normalization's value (dynamics-only vs genuine recovery) is decided by the normalization-benefit map.

## 12. Recommended next experiment
If Outcome A/B: E5 — implement OGM-GE faithfully and compare against z-score in the optimization-driven region only. If Outcome C: the architecture is compensation-capable; reconsider whether real-data collapse is scale- or optimization-driven before CREMA-D.
