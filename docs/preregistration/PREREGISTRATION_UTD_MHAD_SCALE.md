# Pre-registration — UTD-MHAD training-time scale intervention (cross-dataset validation of CREMA-D H6)

Written **before** any full training and **before inspecting any test result**. Start commit `c780d60`.
Date: 2026-09-20. This is an **independent cross-dataset replication** of the CREMA-D H6 training-time
scale mechanism. **E1–E5 and CREMA-D R1–R3 / H6 remain unchanged and immutable.** UTD-MHAD does not
modify, re-run, or reinterpret any frozen experiment.

## 0. Purpose / primary scientific question
Does imposing modality-specific feature-scale disparity **during training** (in-graph, encoders
trainable) alter learned modality contributions in a **substantially different** real multimodal
dataset (action recognition; depth camera + wearable inertial), and does per-modality normalization
suppress the scale-dependent pathway? Result may be replication / partial / null / contradictory —
**not tuned to any outcome.**

## 1. Dataset & provenance
UTD-MHAD (Chen, Jafari, Kehtarnavaz, ICIP 2015). Official source `personal.utdallas.edu/~kehtar/UTD-MHAD.html`.
Verified from the original paper text: **8 subjects (4F/4M), 27 actions × 4 trials = 864; minus 3
author-corrupted = 861 usable sequences.** Four temporally-synchronized modalities per trial (RGB .avi,
depth .mat, skeleton .mat, inertial .mat). Filename `a<i>_s<j>_t<k>_<modality>` (action i, **subject j∈1..8**,
trial k). Official cross-subject split (verbatim): **train = subjects 1,3,5,7 ; test = 2,4,6,8.**

## 2. Modalities (genuinely independent, non-derived)
- **A = Depth** (Kinect optical depth maps). Encoder input: `d_depth` (H×W×T frames).
- **B = Inertial** (separate wearable IMU: 3-axis accel + 3-axis gyro @ 50 Hz). Encoder input: `d_iner` (T×6).
Depth and inertial come from **different physical sensors** — neither derived from the other. (Skeleton IS
depth-derived and RGB is redundant with depth → **both excluded**, as required.) Only `Depth.zip` (120.6 MB)
and `Inertial.zip` (5.1 MB) are downloaded; RGB/skeleton are NOT downloaded.

## 3. Subject-disjoint split (fixed; test unchanged across all seeds/conditions)
- **TRAIN = {1, 3, 5}**  · **VAL = {7}**  · **TEST = {2, 4, 6, 8}** (official test set, unchanged).
- Hard rule: TRAIN ∩ VAL = TRAIN ∩ TEST = VAL ∩ TEST = ∅ (verified in the data audit).
- TEST subjects are **never** used for hyperparameter/architecture/model selection, early stopping,
  normalization fitting, thresholding, or debugging. No k-fold over test subjects.
- VAL {7} is a preregistered carve-out from the official training pool {1,3,5,7}; it is used only for
  trajectory logging, **not** for model selection (fixed-epoch training, see §6), mirroring H6.

## 4. Preprocessing (deterministic; fixed before training; ENCODER INPUTS are cached = valid caching)
- **Depth →** parse `d_depth`; uniformly sample **T=16** frames (`linspace` indices over available
  frames, clamped if <16); resize each frame to **32×32** (area interpolation); per-clip min–max scale to
  [0,1] over the clip (degenerate clip → zeros). Cached tensor per trial: `[16, 1, 32, 32]` float16.
- **Inertial →** parse `d_iner` (T×6); **linearly resample time to L=128**; **per-channel standardize
  using TRAIN-split channel mean/std only** (fit on the 3 train subjects, applied to val/test → no
  leakage; stats saved to config). Cached tensor per trial: `[6, 128]` float16.
- Input normalization above is applied **identically across all scale/normalization conditions** and is
  distinct from the representation-level z-score control (§8). Encoders run and update every step on the
  cached inputs (NOT frozen encoder outputs).

## 5. Architecture (frozen; matched-capacity two-stream; mirrors the H6 model interface)
`UTDMultimodalModel` reuses the H6 `ScaleInjector`, per-modality `ZScoreNorm`, sum `fuse`, linear head,
and `scale_first` ordering **unchanged**; only the two encoders differ:
- **DepthCNN** (enc_a): per-frame 2D CNN `[B,16,1,32,32] → block(1,16)→block(16,32)→block(32,32)→
  GAP→flatten → temporal mean-pool over 16 frames → Linear(32, rep_dim)` (mirrors the H6 VideoCNN).
- **InertialCNN** (enc_v): 1D CNN over `[B,6,128]` → conv1d blocks → GAP → Linear(·, rep_dim).
- rep_dim = 64, fusion = **sum**, head = **linear**. Both encoders **trainable**. **Report both param
  counts**; encoder channels are chosen to keep capacities closely matched (target within ~15%).
- Mapping to the frozen metric names: A=depth→`enc_a`, B=inertial→`enc_v`, so
  `dominance = Acc_depth_only − Acc_inertial_only`, `Δ_depth = Acc_AB − Acc_inertial`, `Δ_inertial = Acc_AB − Acc_depth`.

## 6. Training (frozen)
Adam, **lr 1e-3, weight_decay 0, batch 32, 60 fixed epochs, 5 seeds = [0,1,2,3,4]** (same seed set as H6).
Seeds control initialization + minibatch order (+ any stochastic op); the subject split is **fixed** across
seeds. Final **test** metrics taken at the final epoch (no val/test-based model selection), mirroring H6.

## 7. Scale conditions (primary; frozen). (α_depth, α_inertial), injected **in-graph before fusion**:
- baseline (1, 1); **depth×5** (5, 1); **depth×10** (10, 1); **inertial×5** (1, 5); **inertial×10** (1, 10).
`h_depth = α_depth · enc_depth(x)` and `h_inertial = α_inertial · enc_inertial(x)` inside the graph, before
per-modality norm (scale_first=True) and fusion. Not detached, not eval-only, not on raw input/loss/
gradients/LR. **No additional scale values** unless a preregistered condition is technically impossible.

## 8. Normalization control (frozen)
Per-modality **z-score** (`ZScoreNorm`, fit on TRAIN encoder-outputs only), applied **after** the imposed
scale (`scale_first=True`) so normalization can remove the imposed disparity — identical ordering to H6.
Conditions: {baseline (1,1), depth×10 (10,1), inertial×10 (1,10)} × {none, zscore} × 5 seeds.
No LayerNorm/BatchNorm. Ordering (enc → scale → norm → fuse) is explicit in code.

## 9. Primary outcome (frozen; existing masking definition)
Learned modality-contribution imbalance on TEST: `Δ_depth`, `Δ_inertial`, `|Δ_depth − Δ_inertial|`, and
`dominance = Acc_depth − Acc_inertial`, compared across the 5 scale conditions. Primary statistic:
dominance dose-response vs `log2(α_depth/α_inertial)` — slope + Spearman, and the depth×10−baseline shift.

## 10. Secondary outcomes (frozen), per epoch + final
`ρ_fusion` (post-scale ‖h_depth‖/‖h_inertial‖), learned `ρ_enc` (pre-scale encoder-output ratio, for
compensation), encoder gradient ratio ‖g_depth‖/‖g_inertial‖, val contribution/accuracy; final: test acc,
macro-F1, multimodal gain = Acc_AB − max(Acc_depth, Acc_inertial), unimodal (masking) accuracies.
**Compensation** = `1 − ln(ρ_fusion)/ln(imposed)` (fraction of imposed log-ratio counteracted by encoders).

## 11. Modality utility is NOT assumed
Natural utility (depth-only vs inertial-only accuracy) is established from the **baseline (1,1)** runs
first. Directional-asymmetry interpretation (amplifying the more-informative modality) is applied only
**after** observing baseline utility. The hypothesis is not rewritten after seeing the baseline.

## 12. Statistics
5 seeds: mean ± SD, 95% CI, paired baseline-vs-condition comparisons, effect sizes; Spearman + OLS slope
for dose-response. **Explicitly distinguish seed variance (what 5 seeds estimate) from subject variance
(NOT estimated: test = 4 fixed subjects {2,4,6,8}).** Report **per-test-subject** dominance/gain for
transparency. Do not resample test subjects. Do not claim 5 seeds = 5 subject samples.

## 13. Exclusion / failure handling
- Usable trial = key `a<i>_s<j>_t<k>` present in **both** depth and inertial (intersection). Trials missing
  a modality are logged and excluded (expected: the 3 author-corrupted; any extra discrepancy → **STOP &
  report** before training). No silent deletion.
- Non-finite loss → record (seed/condition/epoch) as **diverged**; do not drop silently. >20% of runs
  diverged → **TECHNICAL FAILURE**.

## 14. Stopping rules (STOP and report before full training if)
missing expected files; depth/inertial keys mismatch beyond the 3 corrupted; unparseable subject IDs;
official test subjects cannot be preserved; synchronization invalid; **scale×10 does not mechanically
change ρ_fusion in the expected direction in the smoke test**; ambiguous normalization ordering; VRAM/
storage problems; any major discrepancy with this preregistration.

## 15. Verdict (chosen from evidence, not before analysis)
- **REPLICATION SUPPORT** — training-time scale reproduces a dose-dependent learned-contribution change
  and z-score suppresses it, as in H6.
- **PARTIAL REPLICATION** — transfers with important differences (state which conditions transferred).
- **NON-REPLICATION** — no meaningful learned-contribution change under this UTD-MHAD setup.
- **CONTRADICTORY** — systematic reversal vs the H6 mechanism.
- **TECHNICAL FAILURE** — uninterpretable (implementation/data/divergence).

## 16. Interpretation rule
Even if supported, claim only: *"the training-time feature-scale intervention altered learned modality
contributions under BOTH tested real-world multimodal settings (CREMA-D audio-video and UTD-MHAD
depth-inertial), within the tested architectures/optimization."* **Not** a universal law from two datasets.

## 17. Outputs
`PREREGISTRATION_UTD_MHAD_SCALE.md` (this file), `DATA_AUDIT_UTD_MHAD.md`, `EXPERIMENT_REPORT_UTD_MHAD.md`,
`results/utd_mhad/utd_train_scale_{primary,trajectories,agg}.csv`, `_analysis.json`, `_config.json`,
per-test-subject CSV; figures `utd_TS_{dynamics,contribution,compensation,normalization}.png` and baseline
unimodal/gain + CREMA-D-vs-UTD comparison. Raw media + encoder-input cache NOT committed. Commit separately.
Then STOP (no third dataset / OGM-GE / new scales / paper) without explicit authorization.
