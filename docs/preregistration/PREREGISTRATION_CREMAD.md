# Pre-registration — CREMA-D real-world validation

Written **before** any CREMA-D experimental results (the media are not yet available;
see the blocker below). New experiment; E1–E5 frozen and unmodified. Reuses the
synthetic fusion model, ρ definition, and modality-contribution definitions.
Timestamp: 2026-09-20.

## Data-availability blocker (must be resolved before R1–R3)
The local `CREMA-D-master` contains only **git-LFS pointer stubs** (all 7442 audio
and 7442 video files are 130 B; each stub declares the real size, e.g. 72862 B).
The folder has **no `.git`**, so `git lfs pull` is impossible locally. Real media
must be provided (e.g. `git clone https://github.com/CheyneyComputerScience/CREMA-D`
+ `git lfs pull`, or another local copy) before preprocessing. The audit, split,
manifest, and this protocol are media-independent and are frozen now.

## Hypotheses (not assumed)
- **Primary:** real audio-video data exhibits a measurable modality feature-scale
  disparity, and per-modality normalization reduces the scale-associated imbalance.
- **Secondary:** the pre-fusion activation-norm ratio ρ is associated with modality
  contribution imbalance.
- **Exploratory:** a controlled scale multiplier on a FIXED trained representation
  predictably changes modality contribution (real analogue of E2).

## Dataset & split (frozen; see data/cremad/cremad_manifest.csv)
CREMA-D: 7442 clips, 91 actors, 6 emotions (ANG/DIS/FEA/HAP/SAD=1271, NEU=1087),
12 sentences. **Actor-disjoint** split stratified by sex, seed 42, 70/10/20:
**train 64 actors / 5234 clips, val 9 / 738, test 18 / 1470** (0 actor overlap —
asserted). Not changed after results. Emotion is the label (6-way).

## Preprocessing (frozen before final training)
- **Audio:** mono, 16 kHz; log-mel spectrogram, 64 mel bins, n_fft=1024, hop=256,
  Hann window; fixed 3.0 s (pad/truncate); per-feature standardization using
  **train statistics only**.
- **Video:** 8 uniformly-sampled frames; center/face crop to 64×64 grayscale;
  pixel scale [0,1]. (Face crop via a lightweight detector if available, else center
  crop — documented at run time.)
- Cached features under `data/cremad/cache/` so extraction is done once.

## Architecture (comparable-capacity encoders; RTX 4070 / 8 GB)
- **Audio encoder:** small 2-D CNN over log-mel (≈3 conv blocks) → global pool → rep_dim=64.
- **Video encoder:** small 2-D CNN per frame (matched conv budget) → temporal mean-pool → rep_dim=64.
- Parameter counts matched within ~20% (Phase-15 capacity control; reported).
- **Fusion:** sum (primary) and concat (secondary), reusing `models/` fusion + linear head.
- **Normalization:** per-modality z-score before fusion (E3/E4 `ZScoreNorm`, train-fit).
- Adam, fixed lr, fixed batch, fixed epochs, ≥5 seeds. All frozen before final runs.

## Three experiments
- **R1 observational:** train vanilla; measure audio/video fusion-boundary norms,
  ρ (mean/median/SD/percentiles/log), modality contributions Δ_audio=Acc_AV−Acc_video,
  Δ_video=Acc_AV−Acc_audio, multimodal gain, gradient norms, trajectories.
- **R2 normalization:** vanilla vs per-modality z-score (before fusion). Measure the
  same + convergence. A reduction in imbalance with unchanged/worse accuracy is a
  valid positive result.
- **R3 controlled scale intervention (primary causal test):** on the FIXED trained
  representation, multiply one modality's rep by c ∈ {0.1,0.2,0.5,1,2,5,10} (and the
  reciprocal on the other), information unchanged, **no retraining**; measure ρ,
  modality contribution, multimodal gain. Also R3×normalization: repeat under z-score
  to test whether the response surface flattens.

## Metrics (per seed/condition; frozen)
audio-norm, video-norm, ρ, |log ρ|, Δ_audio, Δ_video, |Δ_audio−Δ_video|,
multimodal gain, accuracy, macro-F1, encoder gradient norms + ratio, collapse
severity (reuse synthetic definition where applicable). Primary = modality imbalance
& ρ, NOT accuracy.

## Leakage controls (Phase 16)
Actor-disjoint train/val/test (asserted); normalization stats from train only; no
test-derived preprocessing/hyperparameters; R3 scale intervention applied to test
reps only AFTER the model + protocol are frozen.

## Statistics
≥5 seeds; mean/SD/95% CI; R3 slope of contribution vs log(scale) with effect size,
CI, monotonicity; normalization Δρ, Δ|imbalance|, Δgain, Δacc, ΔF1. No method ranking.

## Classification (choose one from results, not publication value)
- **CREMA-A strong transfer:** real scale disparity exists; R3 changes contribution;
  normalization reduces scale sensitivity; reproducible; no major confound → GO INCLUDE-50.
- **CREMA-B partial transfer** → analyze limitations, decide on INCLUDE-50.
- **CREMA-C observational only** (natural disparity but weak/ambiguous R3) → no causal claim.
- **CREMA-D no transfer** (R3 null) → stop, reassess mechanism.
- **CREMA-E invalid** (leakage/preprocessing/data problems) → fix before continuing.
  *(Current state: CREMA-E — media absent; resolve before R1–R3.)*

## Outputs (Phase 23)
`data/cremad/cremad_manifest.csv`, `results/cremad/{cremad_manifest,cremad_primary,
cremad_scale_intervention,cremad_aggregated}.csv`, `cremad_config.json`;
figures per Phase 17; `docs/experiments/CREMAD_FINAL_REPORT.md`. Commit separately. Then STOP
(no INCLUDE-50 / other dataset / architecture / paper auto-run).
