# REPRODUCIBILITY AUDIT

_Every value below was read from the experiment code/configs. Items that could not be verified are
flagged; none were fabricated._

## Synthetic (`data/synthetic.py`, `configs/boundary.yaml`)
- Generator: `x_m = α_m·(μ_y^m + σ_m·ε)`, `ε ~ N(0, I)`.
- Prototypes: orthonormal columns (QR of a Gaussian) scaled by the separation parameter → each
  prototype has norm 1.5; pairwise distance `1.5·√2`. `prototype_seed = 12345` (fixed across α).
- Classes C=8; dims `d_a=d_v=32`; σ=1; both modalities matched geometry + variance (SNR = margin/σ).
- Samples: train/val/test = 4000/1000/2000. Data seed independent of α.
- Encoders: matched 2-layer MLP, hidden 64, ReLU, rep dim 32. Fusion: sum (concat/gated for robustness).
  Head: linear.
- Optimizer: Adam (`torch.optim.Adam`, lr 1e-3, PyTorch defaults β=(0.9,0.999), ε=1e-8, wd 0); SGD
  (momentum 0.9, lr 1e-3, wd 0). No scheduler, no gradient clipping, no gradient accumulation.
- Batch 256, 40 epochs, seeds [0–4]. Scales α ∈ {1,10,100} (E4-EXTREME up to 400×).
- Metrics: masked unimodal accuracy; D = Acc_A−Acc_B; G = Acc_AB−max(Acc_A,Acc_B).

## CREMA-D (`configs/cremad.yaml`, `data/crema_d.py`, `models/cremad_encoders.py`, `cremad_disentangle.py`)
- Classes: 6 emotions (ANG/DIS/FEA/HAP/NEU/SAD). Split: actor-disjoint 64/9/18 actors = 5234/738/1470 clips.
- Audio: WAV → 16 kHz mono, 3.0 s pad/trunc → log-mel (torchaudio, 64 mels, n_fft 1024, hop 256) → [1,64,188].
- Video: .flv → 8 uniformly-sampled frames, grayscale, 64×64 → [8,1,64,64].
- **AudioCNN**: 3 blocks `Conv2d(3×3,pad1)→ReLU→MaxPool2d(2)`, channels 1→16→32→32, `AdaptiveAvgPool2d(1)`,
  `Flatten`, `Linear(32→64)`. **VideoCNN**: same per-frame CNN, temporal mean-pool over 8 frames, `Linear(32→64)`.
- Fusion: sum; head: linear. Optimizer: Adam (lr 1e-3, wd 0). Batch 64, 30 epochs, seeds [0–4].
- Intervention: audio scaled, α ∈ {10,100}; four arms (baseline/forward/gradient/ordinary). 40 runs, 0 diverged.

## UTD-MHAD (`models/utd_encoders.py`, `utd_train_scale.py`, `PREREGISTRATION_UTD_MHAD_SCALE.md`)
- 27 actions, 861 usable samples (depth ∩ inertial). Split: subject-disjoint train {1,3,5} / val {7} /
  test {2,4,6,8}.
- Depth: clip [B,T,1,H,W] (16 frames 32×32). Inertial: 6-channel IMU signal [B,6,L] (L=128).
- **DepthCNN**: same per-frame 2D-CNN as VideoCNN (blocks 1→16→32→32), temporal mean-pool, `Linear(32→64)`
  (~16160 params). **InertialCNN**: 3 blocks `Conv1d(k=5,pad2)→ReLU→MaxPool1d(2)`, channels 6→16→32→64,
  `AdaptiveAvgPool1d(1)`, `Linear(64→64)` (~17552 params).
- Fusion: sum; head: linear. Optimizer: Adam (lr 1e-3, wd 0). Batch 32, 60 epochs, seeds [0–4].
- Intervention: depth/inertial scaled ×5/×10 + z-score control. 40 runs, 0 diverged.

## Intervention definitions (`experiments/disentangle_scale_grad.py`)
- `ForwardScale`: forward `α·x`; backward returns `g` (×1). `GradScale`: forward `x.clone()` (identity);
  backward `α·g`. Applied by `arm_apply(h_A, α, arm)` at the fusion boundary (after encoder + norm, before
  fusion), modality A only.
- Baseline = identity; forward-only = ForwardScale; gradient-only = GradScale; ordinary = `α·h`.
- Verification: a sanity gate checks the forward/backward multipliers (baseline 1,1; forward α,1; gradient
  1,α; ordinary α,α) and that the gradient-only forward pass equals baseline (`GradScale` forward is the
  identity), before each run; divergent runs excluded (none observed).

## Not fully specifiable from artifacts (flagged, not fabricated)
- Exact UTD depth clip length / frame size are set in `utd_extract.py` (16 frames, 32×32 as used); the
  manuscript reports encoder structure and rep dim, deferring pixel-level preprocessing to the code.
