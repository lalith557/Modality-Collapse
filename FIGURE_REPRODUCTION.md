# FIGURE REPRODUCTION MAP

_Every paper figure → generating script → machine-readable source → existing figure file. All plotted
values are read from `results/**` CSV/JSON by the listed script (no values are hand-typed into plotting
code, except axis labels). Figures regenerate by re-running the script, or by calling its plotting/analyze
function on the saved CSV (no experiment re-run needed)._

| Paper Fig | Content | Script (plotting fn) | Source data | Existing figure file(s) |
|---|---|---|---|---|
| **1** | Conceptual causal framework (scale→forward/gradient→update→contribution; optimizer, utility, norm) | authored from PAPER_DRAFT_FINAL §Causal model (no data) | — | to author (vector) |
| **2** | Scale×LR phase diagram; static scale compensable vs LR/aligned persistent | e4_extreme.py; e4_boundary.py | results/processed/e4_extreme_runs.csv; results/e4/e4_extreme_analysis.json | results/figures/e4_extreme_dose_response.png; e4_extreme_long_horizon_trajectories.png; e4_phase_diagram_collapse.png |
| **3** | Adam forward-vs-gradient decomposition (dominance, grad ratio, update ratio) | disentangle_scale_grad.py (`figures`) | results/disentangle/disentangle_{primary,agg,analysis}.* | results/disentangle/figures/disentangle_{arms_contribution,arms_distribution,trajectories}.png |
| **4** | SGD forward-vs-gradient decomposition + Adam-vs-SGD | sgd_disentangle.py (`figures`) | results/sgd_disentangle/sgd_disentangle_{primary,analysis}.*; sgd_disentangle_adam_vs_sgd.csv | results/sgd_disentangle/figures/sgd_{dominance,grad_update_ratio,adam_vs_sgd,trajectories}.png |
| **5** | CREMA-D real decomposition (dominance, gain, ρ, grad+update ratio) | cremad_disentangle.py (`figures`) | results/cremad_disentangle/cremad_disentangle_{primary,agg,analysis}.* | results/cremad_disentangle/figures/cremad_dis_{dominance,grad_update_ratio,trajectories}.png |
| **6** | CREMA-D dose-response + compensation | cremad_train_scale.py (`_figures`) | results/cremad/cremad_train_scale_{primary,agg,analysis}.* | results/cremad/figures/cremad_TS_{contribution,compensation,dynamics}.png |
| **7** | Fusion robustness (sum/concat/gated × 4 arms; dominance + grad/update ratio) | fusion_robustness.py (`figures`) | results/fusion_robustness/fusion_robustness_{primary,agg,analysis}.* | results/fusion_robustness/figures/fusion_{dominance,grad_update_ratio}.png |
| **8** | Utility-gated cross-dataset (CREMA-D vs UTD; informative vs weak amplification) | compose cremad_train_scale + utd_train_scale contribution panels | results/cremad/…_analysis.json; results/utd_mhad/…_analysis.json | results/cremad/figures/cremad_TS_contribution.png + results/utd_mhad/figures/utd_TS_contribution.png |

## Regeneration commands
- Re-run a script to regenerate its figures from scratch (recomputes + replots): e.g.
  `python experiments/cremad_disentangle.py`.
- Replot from saved results without re-running the experiment: import the script's `analyze`/`figures`
  function and pass the saved `*_primary.csv` (+ trajectories), e.g. as done for the normalization figure
  regeneration. No experiment recomputation occurs; only the saved CSV/JSON are read.
- Figures 1 and 8 are composed for the manuscript from the listed panels/spec; the underlying panels for
  Fig 8 already exist and are data-sourced.

## Provenance guarantee
No figure in this list contains a value that is not present in a committed `results/**` file. The color
scheme, arm ordering, and axis labels are the only hand-authored elements.
