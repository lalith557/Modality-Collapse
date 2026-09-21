# EXPERIMENT FREEZE AUDIT

_Date: 2026-09-21. Purpose: freeze the experiment inventory and verify every headline number in the
manuscript traces to a machine-readable result. **No frozen experiment modified; no result altered.**_
_Audit outcome: **PASS** — all numbers trace to their source JSON/CSV; 0 diverged runs; n=5 seeds
throughout; no missing/stale/overwritten/duplicated results detected._

## Commit state
Committed (immutable): `83714a5` (E2/scaffold), `7490b2d` (E1), `27f2779` (E3), `3449545` (E4),
`e35b2bd` (E4-EXTREME), `e5c1094` (E5), `97a9e04`→`221bc37`→`b7a5295` (CREMA-D R1–R3), `c5069c6`→`c780d60`
(H6), `3f66e72`→`a8575c2`→`4465ebc` (UTD-MHAD). **HEAD = `4465ebc`.**
**Uncommitted (disentanglement program, pending review):** `experiments/{disentangle_scale_grad,
sgd_disentangle,cremad_disentangle,fusion_robustness}.py`; `PREREGISTRATION_{SCALE_GRADIENT_DISENTANGLE,
SGD_DISENTANGLE,CREMAD_DISENTANGLE,FUSION_ROBUSTNESS}.md`; `EXPERIMENT_REPORT_*` for those; `results/
{disentangle,sgd_disentangle,cremad_disentangle,fusion_robustness}/`.

## Inventory + traceability (verified 2026-09-21)
| Experiment | Code | Prereg | Seeds | Conditions | Epochs | Dataset / split | Result artifact | Verified numbers | Status |
|---|---|---|---|---|---|---|---|---|---|
| E1 diagnostic | e1_diagnostic.py | PREREGISTRATION.md | 5 | early-ρ | 40 | synthetic | results/processed/e1_primary_runs.csv | early-ρ AUROC~0.91@ep5; final-dom pred UNSUPPORTED | committed `7490b2d` |
| E2 static scale | e2_causal.py | PREREGISTRATION.md | 5 | scale sweep | 40 | synthetic | results/processed/e2_causal_primary_* | scale→transient; compensated by ep40; probe info-invariant | committed `83714a5` |
| E3 normalization | e3_cure.py | PREREGISTRATION.md | 5 | none/z/LN/BN | 40 | synthetic | results/processed/e3_primary_* | removes early imbalance; ~hurts acc | committed `27f2779` |
| E4 scale×LR | e4_boundary.py | PREREGISTRATION_E4.md | 5 | 9×9 grid | 40 | synthetic | results/processed/e4_grid.csv, e4_statistics.csv | aligned interaction; persistent rare @≤20× | committed `3449545` |
| E4-EXTREME | e4_extreme.py | PREREGISTRATION_E4_EXTREME.md | 5 | 20/100/400× | 40–200 | synthetic | results/processed/e4_extreme_runs.csv; results/e4/e4_extreme_analysis.json | scale compensable ≤400× (S≤0.08); LR/aligned persistent (S→1.0); 0/390 diverged | committed `e35b2bd` |
| E5 OGM-GE | e5_ogm.py | PREREGISTRATION_E5.md | 5 | 4 arms×6 cond | 40 | synthetic | results/e5/e5_{primary,analysis}.* | z-score recovers scale; OGM null in-regime; 0/260 diverged | committed `e5c1094` |
| CREMA-D R1–R3 | cremad_run.py | PREREGISTRATION_CREMAD.md | 5 | none/z; frozen-scale c | 30 | CREMA-D actor-disjoint 64/9/18 | results/cremad/cremad_{primary,scale_intervention,analysis}.* | ρ=1.221; z ρ→0.965 grad 2.74→0.14 acc−0.026; R3 dominance FLAT, gain inverted-U | committed `b7a5295` |
| CREMA-D H6 | cremad_train_scale.py | PREREGISTRATION_CREMAD_TRAINING_SCALE.md | 5 | none{1,5,10 a/v}+z | 30 | CREMA-D (as above) | results/cremad/cremad_train_scale_analysis.json | baseline dom 0.075; audio×10 dom 0.328 gain 0.013; Spearman 0.922; z: 0.328→0.110; 0/40 div | committed `c780d60` |
| UTD-MHAD | utd_train_scale.py | PREREGISTRATION_UTD_MHAD_SCALE.md | 5 | none{1,5,10 d/i}+z | 60 | UTD-MHAD subj 1,3,5/7/2,4,6,8 | results/utd_mhad/utd_train_scale_analysis.json (+persubject.csv) | depth 0.036, inertial 0.227, mm 0.632, gain 0.405; base dom −0.191; inertial×10 −0.615 gain 0.053; Spearman 0.757; 0/40 div; PARTIAL | committed `4465ebc` |
| **Synthetic disentangle (Adam)** | disentangle_scale_grad.py | PREREGISTRATION_SCALE_GRADIENT_DISENTANGLE.md | 5 | 4 arms×α{1,10,100} | 40 | synthetic | results/disentangle/disentangle_analysis.json | CASE A; fwd −0.070 d−3.76; grad −0.0005 d−0.53 p0.30; ordinary≡fwd; 0/60 div | **uncommitted** |
| **Synthetic disentangle (SGD)** | sgd_disentangle.py | PREREGISTRATION_SGD_DISENTANGLE.md | 5 | 4 arms×α{1,10,100} | 40 | synthetic | results/sgd_disentangle/sgd_disentangle_analysis.json | CASE C; fwd +0.244 d3.93; grad +0.200 d3.90; ordinary +0.322; upd-ratio(grad)≈96≈α; 0/60 div | **uncommitted** |
| **CREMA-D disentangle** | cremad_disentangle.py | PREREGISTRATION_CREMAD_DISENTANGLE.md | 5 | 4 arms×α{10,100} | 30 | CREMA-D (as above) | results/cremad_disentangle/cremad_disentangle_analysis.json | CASE A; fwd +0.259 d7.39 p8e-5; grad −0.001 d−0.08 p0.87 (grad_r 282.7, upd_r 1.61); ordinary≡fwd; 0/40 div | **uncommitted** |
| **Fusion robustness** | fusion_robustness.py | PREREGISTRATION_FUSION_ROBUSTNESS.md | 5 | 3 fusion×4 arms | 40 | synthetic | results/fusion_robustness/fusion_robustness_analysis.json | grad inert all 3 (upd_r~1); sum fwd d−3.76 p0.001; concat d−5.16; gated d−0.70 p0.19 (attenuated); 0/60 div | **uncommitted** |

## Inconsistency scan (all NEGATIVE = clean)
- Missing result files: none (every reported metric has a source JSON/CSV).
- Number mismatches vs reports/synthesis: none (spot-checked all headline numbers 2026-09-21).
- Stale reports: none (report numbers match current JSONs).
- Overwritten results: none (each experiment writes to its own `results/<name>/`).
- Seed mismatches: none (all seeds [0–4]; disentangle also α=1 control identical across arms).
- Incorrect n: none (n=5 seeds; UTD test=4 fixed subjects; CREMA-D test=18 fixed actors — statistical unit is the seed, documented).
- Duplicated runs: baseline is α-invariant in CREMA-D disentangle → the two baseline sets (α=10,100) are intentional consistency checks, not accidental duplicates.
- Uncommitted experiment code: the four disentanglement scripts + prereg/report/results (listed above) — intentional, pending review.

## Notes for the manuscript
- **Statistical unit = seed** (initialization + minibatch order) on a **fixed** split. Seeds are NOT
  independent populations; UTD reports per-subject consistency separately (`utd_train_scale_persubject.csv`).
- CREMA-D masking-based unimodal **video** is degenerate (→majority class, R1); dominance/gain and ρ are
  the clean signals (same caveat for UTD depth).
- Runner efficiency note (not a scientific issue): `cremad_disentangle.py` computes per-parameter grad/
  update norms with per-batch host syncs → ~71 min wall-clock for 40 CNN runs; results unaffected.
