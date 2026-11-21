# Project explanation in depth

**Project:** *Modality scale collapse* — does a difference in feature magnitude between two modalities cause a multimodal network to ignore one of them, and can the usual way of measuring this be trusted?

**Where the numbers come from.** Every number in this document is copied from a logged run. The sources are:

- `docs/RESULTS_LOG.md`, `docs/CLAIMS_LEDGER.md` and `docs/experiments/EXPERIMENT_LOG.md`;
- the CSV/JSON files under `results/`;
- the checkpoint metadata in `checkpoints/*/baseline/metadata.json`;
- the website's verified data file `website/src/data/research.ts`.

Intervals are paired Student-t 95% confidence intervals over seeds unless stated otherwise. Nothing here is estimated or invented.

---

## Contents

1. [The project in one page](#1-the-project-in-one-page)
2. [Key ideas you need first](#2-key-ideas-you-need-first)
3. [Architecture in depth](#3-architecture-in-depth)
4. [Datasets used and why](#4-datasets-used-and-why)
5. [Novelty of the project](#5-novelty-of-the-project)
6. [How efficiently deep learning is used](#6-how-efficiently-deep-learning-is-used)
7. [Major deep-learning components and algorithms](#7-major-deep-learning-components-and-algorithms)
8. [All results](#8-all-results)
9. [Why each metric is used](#9-why-each-metric-is-used)
10. [Negative results, and why they happened](#10-negative-results-and-why-they-happened)
11. [Is it beneficial?](#11-is-it-beneficial)
12. [Improvements that can be made](#12-improvements-that-can-be-made)
13. [Scope and limitations](#13-scope-and-limitations)
14. [The backbone files of the project](#14-the-backbone-files-of-the-project)
15. [Note on the two manuscripts](#15-note-on-the-two-manuscripts)

---

## 1. The project in one page

**The problem.** A multimodal network learns from two inputs at once, for example the voice and the face of a speaker. Such networks often *collapse*: they rely almost entirely on one modality and leave the other under-used. A popular explanation is *feature scale*: if one encoder outputs vectors with a much larger magnitude, the fused representation is dominated by it.

**The question.** When we make one modality's features larger during training, *how* does that change what the network learns? Two routes are possible:

- **forward:** the larger vector changes what the next layer sees;
- **gradient:** the larger gradient changes how fast that encoder learns.

The effect could also depend on the optimizer (Adam or SGD). And when the standard measurement says one modality now "dominates", is the encoder really different, or is that an artifact of how dominance is measured?

**What we did.**

- **Three autograd operators.** We built custom autograd operators that scale only the forward value, only the gradient, or both, while keeping the training objective identical.
- **Many controlled settings.** We trained hundreds of small models across one synthetic dataset and two real datasets (CREMA-D audio-video emotion, UTD-MHAD depth-inertial action).
- **Two ways of measuring contribution:**
  - the usual *masked-accuracy dominance*;
  - three *decoupled* metrics that look at the representation itself: a frozen-encoder linear probe, mean-ablation dominance and a two-modality Shapley value.

**What we found.**

1. **Synthetic data:** forward scaling genuinely degrades the scaled modality. Every metric agrees.
2. **Adam cancels a constant gradient gain:** a ≈100× gradient ratio becomes a ≈1.00× update ratio. **SGD passes it through** (≈96×). The scale therefore acts like a *hidden per-modality learning rate*, and the optimizer decides which route carries it.
3. **Real data (CREMA-D):** forward scaling moves masked dominance by **+0.259**, but the decoupled metrics barely move (mean-ablation +0.012 and Shapley +0.014 are not significant; probe +0.023). The apparent collapse is mostly a *measurement artifact*.
   - On **UTD-MHAD**, masked dominance overstates the real change about **5×**.
   - The same happens under **stable SGD** and for a **published rebalancing method (OGM-GE)**.
4. **The lesson:** do not judge modality dominance by zero-masking accuracy alone. Report decoupled metrics.

---

## 2. Key ideas you need first

| Term | Meaning in this project |
|---|---|
| Modality | One input stream: audio, video, depth, inertial (IMU), or a synthetic vector. |
| Encoder ℰ_A / ℰ_B | The sub-network that turns one modality into a 64-d (real) or 32-d (synthetic) vector h_A / h_B. |
| Fusion | Combines h_A and h_B: **sum** (h_A + h_B), **concatenation** [h_A ; h_B] or **gated** (learned gates). |
| Feature scale α | A constant that multiplies h_A at the fusion boundary. α = 1 means no intervention. |
| Modality collapse | The network relies on one modality and the other contributes little. |
| Masked dominance 𝒟 | Acc(only A) − Acc(only B), where "only A" means B's representation is set to zero at test time. |
| Multimodal gain 𝒢 | Acc(A+B) − max(Acc A, Acc B): how much fusion helps beyond the best single modality. |
| Decoupled metrics | Measurements that do not change just because the numbers were rescaled: frozen probe, mean-ablation, Shapley. |
| r_g / r_Δθ | Ratio of encoder gradient norms, and ratio of actual parameter-update sizes, A vs B. |

---

## 3. Architecture in depth

### 3.1 The data path (same for every model)

```
x_A ─► preprocessing ─► Encoder ℰ_A ─► h_A ─► [scale operator, experiments only] ─┐
                                                                                  ├─► Fusion 𝓕 ─► Linear head ─► softmax ─► ŷ
x_B ─► preprocessing ─► Encoder ℰ_B ─► h_B ──────────────────────(× 1)───────────┘
```

- The scale operator sits **after the encoder and before fusion**. In the saved baseline checkpoints it is disabled (α = 1) and normalization is `nn.Identity`.
- Softmax is applied by the inference code. The network itself outputs logits and is trained with cross-entropy.

### 3.2 Synthetic model (controlled laboratory)

- **Input:** two 32-d vectors per sample from a generator with **8 classes** (`configs/synthetic_causal.yaml`). Each modality is `x_m = α_m (μ_y^m + σ_m ε)`, where `μ_y^m` are fixed orthonormal class prototypes (norm 1.5) and `ε ~ N(0, I)`.
- **Encoders:** `TinyMLPEncoder` = Linear(32→64) → ReLU → Linear(64→32), one per modality.
- **Fusion:** sum, concatenation or gated. **Head:** a linear layer.
- **Splits:** 4,000 train / 1,000 validation / 2,000 test.
- **Training:** Adam (lr 1e-3, β = (0.9, 0.999), ε = 1e-8) or SGD (momentum 0.9); batch 256; 40 epochs; no weight decay, scheduler or clipping.
- **Why synthetic:** the information content of both modalities is identical by construction (checked with an α-invariant linear probe), so any imbalance comes from training dynamics, not from one modality simply being more informative.

### 3.3 CREMA-D model (audio + video) — `CremadMultimodalModel`, 32,710 parameters

Generated from the code by `scripts/export_architecture.py` and checked against the saved checkpoint with a strict state-dict load:

| Stage | Operation | Output shape | Parameters |
|---|---|---|---|
| Audio preprocessing | mono, 16 kHz, 3.0 s pad/crop → 64-bin log-mel (n_fft 1024, hop 256) | [1, 64, 188] | 0 |
| Audio conv block 1 | Conv2d 3×3 → ReLU → MaxPool 2 | [16, 32, 94] | 160 |
| Audio conv block 2 | Conv2d 3×3 → ReLU → MaxPool 2 | [32, 16, 47] | 4,640 |
| Audio conv block 3 | Conv2d 3×3 → ReLU → MaxPool 2 | [32, 8, 23] | 9,248 |
| Global average pool | AdaptiveAvgPool2d(1) | [32] | 0 |
| Linear → h_A | Linear 32 → 64 | [64] | 2,112 |
| Video preprocessing | 8 uniformly sampled frames, grayscale, centre crop, 64×64, /255 | [8, 1, 64, 64] | 0 |
| Video conv blocks 1–3 | same three blocks, applied to **each frame** with shared weights | 8 × [32, 8, 8] | 14,048 |
| GAP + temporal mean | per-frame pool, then mean over the 8 frames | [32] | 0 |
| Linear → h_B | Linear 32 → 64 | [64] | 2,112 |
| Fusion | sum h_A + h_B | [64] | 0 |
| Head | Linear 64 → 6 | [6] | 390 |

Each encoder (AudioCNN, VideoCNN) has 16,160 parameters.

### 3.4 UTD-MHAD model (depth + inertial) — `UTDMultimodalModel`, 35,467 parameters

| Stage | Operation | Output shape | Parameters |
|---|---|---|---|
| Depth preprocessing | 16 frames, resized to 32×32, per-clip min–max to [0, 1] | [16, 1, 32, 32] | 0 |
| DepthCNN blocks 1–3 | Conv2d 3×3 (16/32/32 ch) → ReLU → MaxPool 2, per frame | 16 × [32, 4, 4] | 14,048 |
| GAP + temporal mean + Linear → h_A | mean over 16 frames, Linear 32 → 64 | [64] | 2,112 |
| Inertial preprocessing | 6 channels (3 accel + 3 gyro), resampled to 128 samples, train-set z-score | [6, 128] | 0 |
| InertialCNN block 1 | Conv1d k5 → ReLU → MaxPool 2 | [16, 64] | 496 |
| InertialCNN block 2 | Conv1d k5 → ReLU → MaxPool 2 | [32, 32] | 2,592 |
| InertialCNN block 3 | Conv1d k5 → ReLU → MaxPool 2 | [64, 16] | 10,304 |
| GAP + Linear → h_B | Linear 64 → 64 | [64] | 4,160 |
| Fusion / head | sum, Linear 64 → 27 | [27] | 1,755 |

Encoder parameters: DepthCNN 16,160 and InertialCNN 17,552.

### 3.5 The intervention operators (the heart of the method)

All three are custom `torch.autograd.Function`s in `experiments/disentangle_scale_grad.py`, applied to h_A:

| Arm | Forward pass | Backward pass (gradient into ℰ_A) | What it isolates |
|---|---|---|---|
| Baseline | h_A | ∇ unchanged | reference |
| **Forward-only** (`ForwardScale`) | α · h_A | ∇ unchanged | representation magnitude |
| **Gradient-only** (`GradScale`) | h_A (unchanged) | α · ∇ | learning-signal size |
| **Ordinary** | α · h_A | α · ∇ (automatic chain rule) | the usual "scale the feature" intervention |

- The objective (cross-entropy) is the same in every arm; only the forward value and/or the backward signal of one tensor change.
- A *sanity gate* checks the forward and backward multipliers numerically before any training.

### 3.6 First-order theory (Proposition 1, `docs/THEORY.md`)

Let s_f be the forward gain and s_g the gradient gain on modality A. To first order, the change in A's parameters behaves like a per-modality learning-rate multiplier m:

- **SGD:** m = s_f · s_g. The update is −η·g, so a gradient multiplied by α gives an update multiplied by α.
- **Adam (small ε):** m = s_f. Adam divides by √v̂; if g → αg then v̂ → α²v̂, so the α cancels, and gradient-only scaling becomes inert.

**Testable predictions** (all confirmed, see §8):

- Under Adam, gradient-only ≈ baseline. Under SGD, gradient-only ≈ a learning-rate boost.
- Making Adam's ε large breaks the cancellation.
- Dividing A's learning rate by α under SGD exactly undoes gradient scaling.

### 3.7 The diagnostic battery

| Metric | Formula / procedure | Code |
|---|---|---|
| Activation-norm ratio ρ | E‖h_A‖ / E‖h_B‖ at the fusion boundary (pre- and post-scale) | `interventions/diagnostic.py` |
| Gradient ratio r_g | ‖∇θ_A‖ / ‖∇θ_B‖ per step | experiment scripts |
| Update ratio r_Δθ | ‖θ_A(t+1) − θ_A(t)‖ / ‖θ_B(t+1) − θ_B(t)‖ from parameter snapshots | experiment scripts |
| Masked dominance 𝒟 | Acc(B zeroed) − Acc(A zeroed) | `analysis/metrics.py` |
| Multimodal gain 𝒢 | Acc(A+B) − max unimodal accuracy | `analysis/metrics.py` |
| Frozen-encoder probe | multinomial logistic regression on each frozen, train-standardized encoder output (Adam lr 0.05, 400 steps, L2 1e-4) | `analysis/probes.py` |
| Mean-ablation dominance | v({A}) − v({B}), where a removed modality is replaced by its **training-set mean** (not zero) | phase scripts |
| Shapley value φ_A | ½[v({A}) − v(∅)] + ½[v({A,B}) − v({B})], empty set = both mean-ablated | phase scripts |

### 3.8 The deployed system around the research

- **Prediction Studio** (`inference/`, `backend/`): a FastAPI service that runs the two saved *baseline* checkpoints on files a visitor uploads.
  - It uses the *same* preprocessing as training (checked for parity with the frozen results).
  - It reports per-modality probe outputs and a class-conditional Mahalanobis distribution-shift detector (fitted on training representations, 95th-percentile threshold).
  - Uploads are kept only in a temporary directory, and only the project's own checkpoints can be loaded.
- **Research website** (`website/`, React + TypeScript + Vite):
  - **Pages:** Overview, Experiment Lab, Prediction Studio, a 360° 3D Model Architecture view, Disentanglement, Real-World Evidence, Methodology and Paper.
  - **Data integrity:** every number shown comes from generated files that tests compare against the frozen results.

---

## 4. Datasets used and why

| Dataset | What it is | Split used | Why it was chosen |
|---|---|---|---|
| **Synthetic generator** (`data/synthetic.py`) | 2 modalities × 32-d, 8 classes, Gaussian class clusters | 4k / 1k / 2k | Full control: equal information per modality, so the *only* difference is scale. Cheap enough for hundreds of runs and 10 seeds. |
| **CREMA-D** | 7,442 clips of 91 actors speaking with 6 emotions (anger, disgust, fear, happy, neutral, sad); audio + video | **actor-disjoint**: train 64 actors / 5,234 clips, val 9 / 738, test 18 / 1,470 | A standard audio-visual benchmark where *both* modalities carry information. The actor-disjoint split prevents memorising speakers. |
| **UTD-MHAD** | 27 actions × 8 subjects × 4 trials (864, of which 3 are corrupted, leaving 861 sequences); depth camera + wearable inertial sensor (3-axis accelerometer + 3-axis gyroscope) | **subject-disjoint**: train {1,3,5} = 323, val {7} = 108, test {2,4,6,8} = 430 | A second, fully independent real dataset with two different sensors. Its modalities have **very different usefulness** (depth alone is near chance), which tests the boundary of the finding. Small download (~126 MB for the two modalities). |

**Considered and rejected.**

- **INCLUDE-50** (Indian sign language) was audited. Signer identities are not recorded anywhere, so a signer-disjoint split could not be built without guessing; it was not used.
- **Kinetics-Sounds, AVE and VGGSound** need hundreds of GB and were infeasible on this machine (8 GB GPU, ~50 GB free disk).

---

## 5. Novelty of the project

1. **Separating forward from gradient.** Earlier work rescales *gradients* to rebalance modalities (OGM-GE, PMR, AGM, DGL…) or blames learning speed. This project separates the **forward magnitude** from the **gradient magnitude** of the *same* scale intervention with paired autograd operators, under a fixed objective. The gradient operator is a generalisation of the gradient-reversal layer (cited), so the novelty is the **paired** use to decouple the two pathways, not the operator itself.
2. **Optimizer dependence made explicit for modalities.** Adam's scale-invariance to constant gradient rescaling is known in general ML. Here it is shown to decide *which pathway* carries a modality-imbalance effect, with a first-order account (m_SGD = s_f·s_g, m_Adam = s_f) and falsifiable tests: ε sweep, η/α cancellation and learning-rate matching.
3. **A measurement audit for modality dominance.** The project shows, on two real datasets, under two optimizers and for a published method, that **masked-accuracy dominance can strongly overstate** what the encoders actually learned. It also provides decoupled metrics that detect real degradation (positive control) and are not fooled by rescaling.
4. **Honesty infrastructure.**
   - Pre-registered predictions, including ones that were refuted.
   - A results log and claims ledger.
   - A website whose numbers are generated from, and tested against, the frozen artifacts.
   - A real inference app that refuses to load arbitrary model files.

What is **not** claimed: a new state-of-the-art method, a universal law of collapse, or that per-layer effective learning rates are a new idea (μP is cited).

---

## 6. How efficiently deep learning is used

- **Small, purpose-built models.** The real-data models have only 32,710 and 35,467 parameters, and the synthetic encoders are two-layer MLPs. This is deliberate: small models train in seconds to minutes, so many seeds, arms and scales can be compared statistically on one RTX 4070 (8 GB).
- **Throughput.**

  | Run | Runs | Wall time |
  |---|---|---|
  | Phase 1 | 160 | 376.6 s |
  | Phase 2 | 520 | 1,250 s |
  | Phase 3 CREMA-D study | — | 1,731 s |
  | Phase 4 baselines | — | 1,997 s |
  | Phase 4 stable SGD | — | 2,476 s |
  | Phase 4 UTD-MHAD | — | 100 s |

- **GPU-resident training.** The training loop keeps all tensors on the GPU. Preprocessed CREMA-D and UTD-MHAD inputs are cached once, so experiments do not re-decode media. Mixed precision is unnecessary at this size and is off.
- **Statistical efficiency.** Seeds are *paired* across arms (the same seed for baseline and treatment), so paired t-tests remove seed-to-seed variance. Effects are reported with Student-t intervals rather than over-confident normal approximations.
- **Reuse instead of re-training.** Later phases reuse validated operators, data and training code (for example, Phase 1 only *adds* measurements to the decomposition code), and the Prediction Studio reuses the exact training preprocessing.
- **Efficient deployment.** Inference loads two small checkpoints once at start-up, runs on CUDA and decodes video in a streaming way, keeping only the 8 frames the model needs.

---

## 7. Major deep-learning components and algorithms

| Component | Where it is used |
|---|---|
| **Convolutional neural networks** (Conv2d / Conv1d, ReLU, max-pooling, global average pooling) | Audio, video, depth and inertial encoders |
| **Weight sharing across time** (per-frame CNN + temporal mean pooling) | Video and depth encoders |
| **Multi-layer perceptrons** | Synthetic encoders |
| **Fusion operators**: sum, concatenation, gated (sigmoid gates) | Joining the two modalities |
| **Linear softmax classifier + cross-entropy loss** | All models |
| **Backpropagation with custom autograd functions** (`ForwardScale`, `GradScale`) | The core intervention |
| **Optimizers**: Adam (adaptive, second-moment normalization), SGD with momentum, AdamW with weight decay 0 | Training; the optimizer is an experimental factor |
| **Gradient-norm clipping** (global, max 1.0) | Stabilizing SGD at α = 100 |
| **Per-modality normalization** (LayerNorm, BatchNorm, z-score) | Testing a "cure" (E3) |
| **OGM-GE** (on-the-fly gradient modulation with Gaussian noise enhancement) | Published rebalancing baseline, re-implemented |
| **Linear probing** of frozen representations | Decoupled measurement |
| **Ablation and Shapley values** (cooperative game theory) | Contribution measurement |
| **Mahalanobis distance out-of-distribution detection** | Prediction Studio warnings |
| **Signal processing front-ends**: resampling, STFT and mel filterbank (log-mel), frame sampling | Preprocessing |

---

## 8. All results

### 8.1 Early programme (synthetic, pre-registered experiments E1–E5)

**E1 — early activation ratio as a diagnostic.**

- The epoch-1–7 ratio ρ predicts *early* dominance well (AUROC 0.91), but the prediction fades (0.77 at epoch 10, 0.57 later).
- It does **not** predict *final* dominance (Spearman 0.06, AUROC 0.40).
- **Verdict:** a transient diagnostic only.

**E2 — scale sweep with information held fixed** (ratio 0.05 → 20).

- **After 40 epochs:**
  - joint accuracy stays flat at 0.69–0.70;
  - the encoders *compensate*: the early ratio spans 0.14 → 7.55 but the final ratio only 0.60 → 1.68;
  - Spearman(ratio, dominance) = −0.89, so the verdict is inconclusive.
- **After only 5 epochs:**
  - strong symmetric collapse: joint accuracy 0.67 → 0.56 and gain roughly halves;
  - Spearman(ratio, dominance) = +0.97 (supported).
- **Conclusion:** scale causes *transient* collapse that the encoders compensate.

**E3 — per-modality normalization.**

- **Benefit:** it cuts early imbalance by ≈87% (epoch-5 extreme |𝒟| 0.152 → ~0.02) and speeds convergence about 3× (10.6 → ~3.5 epochs).
- **Cost:** it *lowers* final accuracy by 0.027–0.049.
- **Conclusion:** it fixes dynamics, not final performance.

**E4 / E4-EXTREME — scale versus learning-rate imbalance** (390 runs). Collapse severity at 20× / 100× / 400×:

| Condition | 20× | 100× | 400× |
|---|---|---|---|
| Scale only | 0.05 | 0.04 | 0.08 |
| Learning-rate imbalance | 0.23 | 0.55 | 0.88 |
| Aligned (both) | 0.55 | 0.99 | 1.00 |

- Persistent collapse is **learning-rate-driven**; static scale is compensated.
- z-score normalization removes the scale part only.

**E5 — z-score versus OGM-GE (synthetic, 260 runs).**

- z-score rescues the aligned-100× collapse (weak-modality contribution 0.002 → 0.113, d = −8.8, p = 4e-5).
- OGM-GE (active, k_A → 0.57) does nothing (aligned-100× 0.003; learning-rate-only Δ = −0.000, p = 0.96).

### 8.2 Training-time scale on real data (H6)

**CREMA-D** (Adam, 5 seeds; Spearman(imposed scale, dominance) = 0.922):

| Condition | Dominance | Gain | ρ_fusion |
|---|---|---|---|
| baseline | +0.075 | 0.248 | 1.22 |
| audio × 5 | +0.299 | 0.045 | 2.31 |
| audio × 10 | +0.328 | 0.013 | 3.24 |
| video × 5 | +0.056 | 0.263 | 1.18 |
| video × 10 | +0.009 | 0.295 | 1.02 |
| audio × 10 + z-score | +0.110 | — | 0.97 |

**UTD-MHAD** (Adam, 5 seeds):

- **Baseline:** depth-only 0.036 (chance is 0.037), inertial-only 0.227, both 0.632, gain 0.405, dominance −0.191.
- **Inertial × 10:** dominance −0.615 and gain 0.053.
- **Depth × 10:** dominance −0.198 and gain 0.400. Scaling the weak modality is fully compensated.

### 8.3 The forward-versus-gradient decomposition (α = 100, 5 seeds, paper Table I)

| Setting | Forward-only Δ𝒟 | Gradient-only Δ𝒟 | Ordinary Δ𝒟 | r_g → r_Δθ (gradient arm) |
|---|---|---|---|---|
| Synthetic · Adam | −0.070 (d −3.76, p 1.1e-3) | −0.0005 (d −0.53, p 0.30) | −0.069 | 100.9× → 1.00× |
| CREMA-D · Adam | +0.259 (d 7.39, p 8e-5) | −0.001 (d −0.08, p 0.87) | +0.266 (d 8.51, p 4e-5) | 282.7× → 1.61× (baseline itself 1.65×) |
| Synthetic · SGD | +0.244 (d 3.93, p 9e-4) | +0.200 (d 3.90, p 1e-3) | +0.322 (d 7.75, p 7e-5) | 318.5× → 95.9× |

- **Under Adam:** forward ≈ ordinary and gradient is null.
- **Under SGD:** both pathways contribute.

**Fusion robustness** (synthetic, Adam, α = 100):

| Fusion | Forward Δ𝒟 | Significance |
|---|---|---|
| Sum | −0.070 (d −3.76) | significant |
| Concatenation | −0.057 (d −5.16, p 3e-4) | significant |
| Gated | −0.026 (d −0.70, p 0.19) | not significant |

The gradient-only arm is null for all three (|Δ𝒟| ≤ 0.001).

### 8.4 Phase 1 — decoupled metrics on synthetic data (Adam, α = 100, 10 seeds, 160 runs)

- **Forward arm:** every metric moves the same way.
  - Masked dominance −0.063 ± 0.011 (masked accuracy of A −0.0525).
  - Frozen probe −0.034 ± 0.006.
  - Mean-ablation −0.056 ± 0.005.
  - Shapley −0.043 ± 0.003.
- **Gradient-only arm:** null on every metric (probe shift +0.0001).
- **Same-dataset update ratio** (gradient arm): Adam 1.001 ± 0.011, SGD 95.8 ± 4.6 (Student-t; the older logs quote the normal-approximation ±0.010 / ±4.0).

### 8.5 Phase 2 — mechanism tests (synthetic, 520 runs, 10 seeds)

- **ε sweep:** Adam's update ratio rises 1.00 → 1.02 → 2.84 as ε grows 1e-8 → 1e-4 → 1e-2. The cancellation breaks as predicted.
- **η/α cancellation:** SGD gradient-only with A's learning rate divided by α gives dominance +0.003, exactly baseline (update ratio 0.99).
- **Learning-rate matching:** raising A's learning rate reproduces the forward *direction* but overshoots (paired difference −0.020 ± 0.005 at α = 3, growing to −0.089 ± 0.018 at α = 300). Compensation buffers the forward effect.
- **Forward for only the first k epochs** (k = 1, 5, 10, 20): no lasting effect. The effect needs sustained scaling.
- **Compensation c(α):**
  - synthetic Adam 0.76 / 0.69 / 0.62 / 0.52 / 0.44 for α = 3 → 300;
  - CREMA-D 0.60 / 0.58 / 0.39.
- **Per-modality heads** (4 arms): the forward ≠ gradient difference persists, so a shared-head explanation is rejected.
- **Tuned SGD baseline** (lr 1e-2): forward and gradient *degrade* A's probe (−0.049 ± 0.005 and −0.110 ± 0.009). Adam's forward degradation is robust to tuning (−0.057 ± 0.005).

### 8.6 Phase 3 — CREMA-D with decoupled metrics (Adam, weight decay 0, 5 seeds)

| Metric (forward arm − baseline) | Shift | Verdict |
|---|---|---|
| Masked dominance | **+0.259 ± 0.044** | large |
| Mean-ablation dominance | +0.012 ± 0.027 | not significant |
| Shapley (audio) | +0.014 ± 0.026 | not significant |
| Frozen audio probe | +0.023 ± 0.017 | small and *positive* (no degradation) |

- The gradient arm is null on every metric.
- **Positive control** (audio noise, n = 3): probe −0.021 / −0.064 and Shapley −0.029 / −0.051 at 0.5σ / 1.0σ. The decoupled metrics *do* detect real degradation, so the CREMA-D null is a real no-effect.
- **SGD at α = 10 (stable):** masked +0.150 / +0.156 (forward / gradient), while mean-ablation and Shapley are not significant.
- **SGD at α = 100, learning rate 3e-2:** forward diverged in 3 of 5 seeds and gradient in 2 of 2 (resolved in Phase 4).

### 8.7 Phase 4 — baselines, stable SGD, second dataset (5 seeds each)

**OGM-GE** (re-implemented, no scaling, method − vanilla):

- k_audio = 0.273 and k_video = 0.999.
- Masked dominance **+0.240 ± 0.042**.
- Mean-ablation −0.023 ± 0.024 (n.s.) and Shapley −0.026 ± 0.030 (n.s.); only the probe −0.014 ± 0.012 is resolved.

**Modality-specific learning rate** (audio × 5): masked +0.117 ± 0.110, with all decoupled metrics not significant.

**Stable SGD** (CREMA-D, α = 100, lr 1e-2, gradient clip 1.0):

- The gradient arm diverged in **5 of 5 seeds without clipping** and **0 of 5 with clipping**.
- Gradient arm: masked +0.200 ± 0.050 while mean-ablation, Shapley and probe are all ≤ 0.002.
- Forward arm: masked +0.132 ± 0.079 while the probe moves −0.021 ± 0.020 (opposite sign).
- Clipping keeps the per-modality gradient ratio exactly (1011.3 before and after).

**UTD-MHAD** (scaled = inertial, Adam):

- Forward arm: masked dominance moves **0.525 ± 0.048** toward inertial, while the inertial probe gains only **+0.102 ± 0.048**. Masked overstates the change about 5×. Mean-ablation shifts 0.107 and Shapley 0.113.
- Gradient arm: equal to baseline on every metric (masked +0.003, mean-ablation +0.004, Shapley −0.002).

### 8.8 Deployed baseline checkpoints (Prediction Studio, seed 0, fixed final epoch)

| Model | Test accuracy | Macro-F1 | ECE (15 bins) | Test size | Chance level |
|---|---|---|---|---|---|
| CREMA-D (6 emotions) | 0.508 | 0.494 | 0.031 | 1,470 | 0.167 |
| UTD-MHAD (27 actions) | 0.570 | 0.555 | 0.265 | 430 | 0.037 |

UTD-MHAD is over-confident (mean top probability 0.835 versus accuracy 0.570). That is why the Studio never treats high softmax confidence as reliability.

---

## 9. Why each metric is used

- **Masked dominance 𝒟.** This is the measure most papers use, so it must be reported for comparability. It is cheap: zero one modality and re-test. Its weakness is the reason for this project: zeroing a modality the head was trained to expect at a certain scale moves the input *off the training distribution*, so accuracy can drop because the head is confused, not because the encoder is worse.
- **Frozen-encoder linear probe.** This asks directly how much class information is linearly readable from this encoder, with no fusion head involved. Because features are standardized and the classifier is linear, rescaling h_A by α cannot change it. It measures the *representation*.
- **Mean-ablation dominance.** It replaces a removed modality by its **training mean** instead of zero, so the head stays near its training distribution. It removes most of the off-distribution artifact of masking.
- **Shapley value.** This is a fair, game-theoretic share of the accuracy contributed by each modality. It averages over both orders of "adding" a modality, so interactions are counted fairly.
- **Multimodal gain 𝒢.** It shows whether fusion is still useful. Collapse means gain → 0.
- **Activation ratio ρ.** It measures the scale imbalance itself, and the compensation of it by the encoders, at the fusion boundary.
- **Gradient ratio r_g versus update ratio r_Δθ.** Together they reveal the optimizer's role: a large r_g with r_Δθ ≈ 1 means the optimizer cancelled the gradient gain (Adam). r_Δθ ≈ r_g means it passed it through (SGD).
- **Paired t-tests, Cohen's d_z and Student-t CIs.** Each seed is a matched pair (same initialization and data order), so paired statistics are the correct and most sensitive test with 5–10 seeds. Student-t intervals are honest for small n.
- **Accuracy, macro-F1 and ECE** (Studio). These measure overall quality, quality balanced across classes, and whether confidence matches correctness.

---

## 10. Negative results, and why they happened

Negative or "against-the-hypothesis" results were kept and reported.

1. **The dominance shift is negative on synthetic data but positive on CREMA-D** (−0.070 versus +0.259). This is not a bug: the same code, masking and dominance definition are used in both (audit M2). The synthetic baseline is balanced (𝒟 ≈ +0.003), and under Adam the scaled encoder *over-compensates* (its own ratio falls to ≈0.09), which slightly degrades it, so A *loses*. On CREMA-D audio is already the stronger modality and compensates less, so its masked advantage grows. The sign depends on baseline balance and compensation strength.
2. **Pre-registered prediction "the synthetic drop is only a masking artifact" — refuted (R1).** The probe fell 0.034 with a CI excluding 0. The synthetic effect is real. This is good news: the decoupled metrics are not simply "always null".
3. **CREMA-D forward effect = masked artifact (W2).** The paper's earlier real-data claim of a genuine contribution shift (d ≈ 7.4 on the masked metric) did not survive the decoupled metrics. The audio encoder is at its weak-encoder ceiling (probe ≈ 0.49), so there is no headroom for its representation to change, but the head's calibration to the scaled input makes zero-masking very damaging. This became the central lesson of the project.
4. **SGD direction was an undertrained-baseline artifact (W1).** At lr 1e-3, SGD is undertrained (validation 0.527), so any extra effective learning rate *helps* A. At the tuned lr 1e-2, the same scaling *hurts* A. The update-ratio mechanism (≈96×) is robust; only the *sign* of the contribution change depends on the regime.
5. **SGD diverged on real data (W3).** A 100× gradient with SGD is a 100× learning-rate jump for one encoder, so exploding updates are expected. Global gradient-norm clipping (max 1.0) fixed it completely (0/5 diverged) without changing the per-modality ratio.
6. **Gated fusion not significant** (d −0.70, p 0.19). The learned gates can partly absorb the scale. Gate values were not logged, so this cannot be confirmed directly.
7. **E1 could not predict final dominance** (Spearman 0.06), because the encoders compensate scale during training.
8. **Normalization lowers final accuracy** (E3, −0.027 to −0.049). It removes the scale problem but also discards some useful geometry in a regime where the plain network would have compensated anyway.
9. **OGM-GE did not fix synthetic collapse** (E5), and on CREMA-D it moved masked dominance but not the representation. Our synthetic collapse is driven by imposed learning-rate starvation, which is not the greedy-learning problem OGM-GE was designed for. This is not a test of OGM-GE in general, and nothing is claimed about its accuracy benefits.
10. **Short scaling (first k epochs) had no effect.** The mechanism needs sustained scaling, consistent with a learning-rate interpretation rather than an initialization effect.

---

## 11. Is it beneficial?

**Yes, as a diagnostic and methodological contribution, not as a new accuracy-boosting model.**

- **For researchers.**
  - It warns that a standard metric (zero-masking dominance) can report a large "collapse" or "rebalancing" effect that is several times larger than the representation-level change, or even of opposite sign. On CREMA-D masked dominance moves +0.259 while the probe moves +0.023; on UTD-MHAD the overstatement is about 5×; under stable SGD masked +0.132 meets a probe change of −0.021.
  - It gives a ready protocol (frozen probe, mean-ablation, Shapley, plus a positive control) to check claims.
  - It also explains *why* a rebalancing trick may look different under Adam and SGD.
- **For practitioners.**
  - Under Adam, simply rescaling gradients of one encoder does almost nothing; to rebalance, change the forward scale or the per-modality learning rate.
  - Under SGD, gradient scaling *is* a learning-rate change and needs clipping.
  - Per-modality normalization removes scale-driven imbalance early.
- **For teaching.** The website lets anyone explore the measured results, inspect the real model layer by layer in 3D, and run the actual checkpoints on their own files, with honest warnings (out-of-distribution flags, no claims from softmax confidence).
- **Limits of the benefit.** It does not improve accuracy on CREMA-D or UTD-MHAD (the deployed baselines are modest: 0.508 and 0.570), and the real-data conclusions are scoped to small from-scratch encoders.

---

## 12. Improvements that can be made

1. **More and larger datasets:** Kinetics-Sounds, AVE, VGGSound and CMU-MOSEI on a larger GPU cluster. A signer-disjoint sign-language set if signer labels become available.
2. **Stronger encoders:** ImageNet/AudioSet-pretrained CNNs and multimodal transformers (for example CLIP-style), to test whether masking still misleads when encoders have headroom.
3. **Attention and cross-modal fusion,** and logging of gate values to explain the gated-fusion attenuation.
4. **More rebalancing baselines:** PMR, AGM, MLA and the original OGM-GE codebase on its own setting, all measured with decoupled metrics.
5. **More seeds** (10+ on real data) and pre-registered power analysis. Multiple-comparison correction across the full metric family.
6. **Longer training and learning-rate schedules,** to test whether the effects persist at convergence and with warm-up or decay.
7. **Theory beyond first order:** finite ε, weight decay, momentum, and a separate head per modality.
8. **Better probes:** non-linear (small MLP) probes and representation-similarity measures alongside linear probes.
9. **Product improvements:** better-calibrated UTD-MHAD probabilities (for example temperature scaling fitted on validation), real UTD action names, and test-time augmentation for the Studio.
10. **One consolidated manuscript.** The two paper versions differ in several details (see §15); one corrected version should be prepared.

---

## 13. Scope and limitations

**In scope.**

- Training-time scaling of one modality's representation at the fusion boundary.
- Two-modality classification.
- Small CNN/MLP encoders trained from scratch.
- Adam (weight decay 0) and SGD (momentum 0.9).
- Sum, concatenation and gated fusion (synthetic); sum fusion (real).
- 5–10 seeds; synthetic data plus CREMA-D and UTD-MHAD with fixed splits.

**Limitations.**

- **Theory.** The learning-rate account is *first-order* (local linearity, ε → 0, a shared head) and verified only on matched synthetic data.
- **Data.** Only two real datasets, each with one fixed split; UTD-MHAD has only 4 test subjects, so seeds capture initialization variance rather than population variance.
- **Encoders.** Weak, from-scratch encoders. The CREMA-D audio probe sits near its ceiling, which is part of why masking misleads there. Results may differ with pretrained encoders.
- **Seeds.** Five seeds on real data is small; some effects are tiny relative to their CIs.
- **SGD** needs gradient clipping at α = 100 on real data (disclosed).
- **OGM-GE** is an adapted re-implementation in our shared sum-fusion setting (two documented modifications). Nothing is claimed about its accuracy benefit.
- **Big benchmarks, pretrained models and attention fusion** were not run, for compute and disk reasons.
- **Prediction Studio** is a research prototype, not a production system. High softmax confidence is not treated as reliability; inputs far from the training data are flagged, but the flags are not guarantees.

---

## 14. The backbone files of the project

Only the files that carry the project are listed; each is described in 3–5 lines.

### Data

- **`data/synthetic.py`** — The controlled synthetic generator `x_m = α_m(μ_y^m + σ_m ε)`. It fixes class prototypes and noise so both modalities hold equal information, and lets α change only the magnitude. Every synthetic experiment (E1–E5, decomposition, Phases 1–2) draws its data here.
- **`data/crema_d.py`** — Real CREMA-D loaders. Audio becomes a 16 kHz, 3 s, 64-bin log-mel; video becomes 8 grayscale 64×64 frames. The Prediction Studio reuses exactly this preprocessing, so inference matches training.
- **`data/cremad_manifest.py`** — Parses CREMA-D file names and demographics and builds the frozen, actor-disjoint, sex-stratified train/val/test split (64/9/18 actors). It guarantees no speaker appears in two splits.
- **`data/utd_mhad.py`** — UTD-MHAD depth and inertial loaders: 16 depth frames at 32×32 with min–max scaling, and 6-channel inertial resampled to 128 samples with train-set z-score. It provides the subject-disjoint split used for the second real dataset.

### Models

- **`models/encoders.py`, `models/fusion.py`, `models/heads.py`** — The building blocks: the tiny MLP encoder, the sum/concat fusion operators and the linear (or MLP) classification head. They are shared by every model so that experiments differ only where intended.
- **`models/multimodal_model.py`** — The synthetic two-stream model. It defines the single documented data path encoder → normalization → scale → (diagnostic) → mask → fuse → head, and the masking interface used by all contribution metrics.
- **`models/cremad_encoders.py`, `models/cremad_model.py`** — The CREMA-D `AudioCNN` and `VideoCNN` (three conv blocks, global average pooling, linear to 64-d; video per frame then temporal mean) and the full 32,710-parameter model. They mirror the synthetic model's interface so the same analysis code works unchanged.
- **`models/utd_encoders.py`, `models/utd_model.py`** — The UTD-MHAD `DepthCNN` (2D, per frame) and `InertialCNN` (1D, kernel 5) and the full 35,467-parameter model, with the same interface as the CREMA-D model.

### Interventions and analysis

- **`interventions/scale_injection.py`** — The representation-level scale operator h ← α·h at the fusion boundary (the `ScaleInjector`, disabled in the saved checkpoints). It documents exactly where the scale acts, which is essential for interpreting every result.
- **`interventions/per_modality_norm.py`** — LayerNorm, BatchNorm and train-statistics z-score per modality, the candidate "cure" tested in E3–E5 and as a control in the real-data studies.
- **`interventions/diagnostic.py`** — Measures ρ = E‖h_A‖ / E‖h_B‖ read-only at the fusion boundary. It is the early-warning diagnostic of E1 and the compensation readout everywhere.
- **`analysis/metrics.py`** — Masked unimodal accuracies, dominance 𝒟, multimodal gain and macro-F1: the standard (coupled) contribution metrics every experiment reports.
- **`analysis/probes.py`** — The GPU frozen-encoder linear probe: train-standardized features, multinomial logistic regression with fixed L2. It is the main decoupled metric and is also used by the Studio for per-modality outputs.
- **`analysis/statistics.py`, `analysis/compensation.py`** — Seed aggregation (mean, SD, CIs) and the compensation measure c(α), which quantifies how much of an injected scale the encoder absorbs.

### Experiments

- **`experiments/train.py`** — The reusable GPU training loop. It records ρ every epoch on validation, snapshots the pre-registered epoch-5 value, and computes final test metrics, giving every experiment the same training behaviour.
- **`experiments/disentangle_scale_grad.py`** — Defines the **`ForwardScale` and `GradScale` autograd operators**, the four arms and the sanity gate, and runs the synthetic Adam decomposition. It is the core of the project's method; later phases import its operators.
- **`experiments/sgd_disentangle.py`, `experiments/cremad_disentangle.py`** — The same decomposition under SGD (synthetic) and under Adam on CREMA-D. They add the parameter-update ratio that revealed Adam's cancellation versus SGD's pass-through (Table I).
- **`experiments/fusion_robustness.py`** — Repeats the decomposition with sum, concatenation and gated fusion to test whether the findings depend on the fusion mechanism.
- **`experiments/phase1_decoupled.py`** — Adds the frozen probe, mean-ablation and Shapley values to the decomposition (10 seeds). It decided that the synthetic effect is real and logged Adam's update ratio 1.001.
- **`experiments/phase2_mechanism.py`** (+ `phase2_eps_cancel.py`, `phase2_forwardk.py`, `phase2_lrsweep.py`, …) — The falsifiable tests of the theory: learning-rate matching, η/α cancellation, the ε sweep, forward-for-k-epochs, per-modality heads and the tuned baselines.
- **`experiments/phase3_cremad_scoped.py`, `experiments/phase3_controls.py`** — The CREMA-D decoupled study (finding W2) and the positive control (audio noise) showing the decoupled metrics can detect real damage, plus the stable SGD α = 10 check.
- **`experiments/phase4_baselines.py`, `phase4_sgd_stable.py`, `phase4_utd_decoupled.py`** — The OGM-GE and modality-learning-rate payoff baselines, SGD stabilized by clipping, and the UTD-MHAD second-dataset replication. Together they show the masked-versus-decoupled mismatch generalizes.
- **`baselines/ogm_ge.py`** — The adapted re-implementation of OGM-GE (Peng et al., CVPR 2022), with its fidelity notes. It is the published rebalancing method audited with decoupled metrics.

### Deployment and website

- **`inference/model_loader.py`, `inference/cremad_predictor.py`, `inference/utd_predictor.py`** — Load only the project's own SHA-256-verified checkpoint packages and run audio-only, video-only, depth/inertial or multimodal prediction with per-modality probe outputs. This is the engine of the Prediction Studio.
- **`inference/ood_detection.py`, `inference/quality_checks.py`, `inference/input_validator.py`** — The class-conditional Mahalanobis distribution-shift detector, input quality checks with thresholds calibrated on validation media, and strict file validation. Together they keep the Studio honest about unfamiliar inputs.
- **`backend/main.py`** (+ `backend/routes/`) — The FastAPI service: health, models, prediction and video-preview endpoints, with per-request temporary storage, structured error codes and no content logging.
- **`scripts/export_architecture.py`, `scripts/export_site_evidence.py`, `scripts/export_lab_trajectories.py`, `scripts/export_architecture_media.py`** — Generate the website's architecture, evidence, training-curve and thumbnail data *from the code, checkpoints and frozen results*. Their `--check` modes are run by `tests/test_site_artifacts.py`, so the site cannot drift from the data.
- **`website/src/data/research.ts`** — The single source of truth for every research number shown on the website, tested against the frozen artifacts by `src/data/evidence.test.ts`.
- **`website/src/architecture/ArchitectureScene.ts`, `Diagram2D.tsx`** — The interactive 360° 3D view and the 2D view of the real models: rising sections, hover explanations, animated data flow, and figure components clearly separated from checkpoint layers.

### Records and paper

- **`docs/RESULTS_LOG.md`, `docs/CLAIMS_LEDGER.md`** — What was run (command, commit, seeds, wall time) and every claim tagged MEASURED, HYPOTHESIZED or REFUTED. They are the authority behind every number in the paper and on the site.
- **`docs/THEORY.md`** — The derivation of Proposition 1 (m_SGD = s_f·s_g, m_Adam = s_f), its assumptions and the predictions that were tested.
- **`research_paper/main.tex`** — The IEEE-format manuscript "Feature Scale as a Hidden Per-Modality Learning Rate: When Masked-Accuracy Dominance Misleads", compiled to 6 pages.
- **`research_paper/ca_paper.tex`** — LaTeX source of the 7-page CA manuscript ("When Scale Silences a Modality"); the compiled PDF is served by the website from `website/public/ca_research_paper.pdf`.

---

## 15. Note on the two manuscripts

The repository contains two write-ups of this work.

- **`research_paper/main.tex`** (6 pages) covers everything up to Phase 4, with Student-t intervals.
- **`research_paper/ca_paper.tex`** (7 pages, "When Scale Silences a Modality") covers Phases 0 to 3, plus the Phase 4 clipping result. The PDF on the website's Paper page (`website/public/ca_research_paper.pdf`) was compiled from an earlier revision of this source.

The earlier revision had details that did not match the code, configs and logs. The current `ca_paper.tex` corrects them:

- the CREMA-D encoders are small three-block CNNs (16,160 parameters each), not ResNets;
- the synthetic task has 8 classes, not 2;
- the hyperparameter table now matches `configs/boundary.yaml` and `configs/cremad.yaml` (batch 256 / 64, 40 / 30 epochs, learning rate 1e-3, weight decay 0);
- the SGD gradient ratio 318.47 is labelled as synthetic, not CREMA-D;
- every interval is a paired Student-t 95% CI (for example, ±0.044 rather than ±0.031 on the CREMA-D masked shift);
- the SGD divergence on CREMA-D is reported together with the gradient-clipping fix (0 of 5 runs diverged);
- the early-ratio figure (Fig. 4) plots the measured E2 values, and the citation keys and figure captions resolve.
