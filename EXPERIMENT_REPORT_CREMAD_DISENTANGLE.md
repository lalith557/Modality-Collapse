# CREMA-D forward-scale vs gradient-scale disentangling — final report (Phase 2)

_Preregistration: `PREREGISTRATION_CREMAD_DISENTANGLE.md` (frozen before results). Start commit `4465ebc`._
_Real-data test of the synthetic Adam decomposition. Reuses the frozen CREMA-D actor-disjoint split,
cache, and matched architecture; validated `ForwardScale`/`GradScale` reused. E1–E5, R1–R3/H6, UTD,
synthetic disentangle Adam/SGD all immutable and untouched._
_Labels: **[OBS]** observed · **[STAT]** statistical · **[MECH]** mechanistic._

## Verdict: **CASE A — forward-dominated (matches synthetic Adam).** H6's training-time scale effect is FORWARD-magnitude-mediated on real audio-video data.
On CREMA-D under the same Adam as H6, **forward-only scaling reproduces the entire ordinary (H6) effect**,
while **gradient-only scaling is statistically indistinguishable from baseline — despite imposing a 282×
encoder-gradient ratio** — because Adam normalizes the constant gradient scaling (gradient-only
parameter-update ratio 1.61 ≈ baseline 1.65). 40 runs, **0 diverged**. This closes the "real-world
mechanism" gap: the audio-amplification entrenchment seen in H6 operates through the forward pathway.

## 1. Design
Modality A = audio (scaled), B = video. Four arms (baseline / forward-only / gradient-only / ordinary)
applied to the audio fusion-boundary representation, in-graph during training. `CremadMultimodalModel`,
matched CNN encoders, sum fusion, linear head, normalization none, **Adam lr 1e-3, batch 64, 30 epochs**,
5 seeds. α ∈ {10, 100}. Test = 18 frozen held-out actors (1470 clips). 4 arms × 2 α × 5 = 40 runs.

## 2. Autograd sanity — PASSED (`cremad_disentangle_sanity.json`)
Scalar multipliers exact; gradient-only forward activations bit-identical to baseline. (Diagnostic: at
random init the forward-only encoder-gradient ratio was ~10.6 — the forward magnitude couples to the loss
gradient more strongly on real CNN features than in the synthetic case; this is an *emergent* consequence
of the forward change, not the α being injected into the backward pass, which the construction excludes.)

## 3. Results (test, mean over 5 seeds) [OBS/STAT]
| α | arm | dominance (Acc_audio−Acc_video) | shift vs baseline | Cohen d | p | gain | ρ_enc | ρ_fusion | grad ratio | **update ratio** |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| — | baseline | +0.075 | — | — | — | +0.248 | 1.22 | 1.22 | 2.7 | 1.65 |
| 100 | **forward** | +0.333 | **+0.259** | **+7.39** | **8e-5** | +0.006 | 0.20 | 19.9 | 0.2 | 0.89 |
| 100 | **gradient** | +0.073 | **−0.001** | −0.08 | **0.87** | +0.245 | 1.23 | 1.23 | **282.7** | **1.61** |
| 100 | ordinary | +0.340 | +0.266 | +8.51 | 4e-5 | +0.007 | 0.19 | 19.3 | 25.3 | 0.86 |
| 10 | forward | +0.325 | +0.250 | (large) | | +0.021 | 0.32 | 3.25 | 0.7 | 0.98 |
| 10 | gradient | +0.076 | +0.001 | ~0 | | +0.242 | 1.23 | 1.23 | 28.0 | 1.66 |
| 10 | ordinary | +0.328 | +0.253 | | | +0.013 | 0.32 | 3.24 | 7.2 | 1.07 |

**Direct comparisons (α=100):** ordinary vs forward diff +0.007 (d=0.59, p=0.26) → **indistinguishable**;
gradient vs baseline diff −0.001 (d=−0.08, p=0.87) → **null**; ordinary vs gradient diff +0.267 (d=10.05,
p=2e-5). Seed SD 0.018–0.031 (stable).

## 4. Interpretation [MECH]
- **Ordinary = H6.** The ordinary arm reproduces H6's audio×10 result (dominance ≈0.33, multimodal gain
  collapsing to ≈0.006), so this experiment decomposes H6 itself.
- **Forward carries the whole effect** (forward ≡ ordinary): amplifying audio's forward representation
  entrenches audio's learned contribution and collapses multimodal gain, with the encoder compensating
  (ρ_enc 1.22 → 0.20) but not enough (net ρ_fusion ≈ 20).
- **Gradient scaling is inert under Adam** even at a 282× gradient ratio: the parameter-update ratio stays
  ≈ baseline (1.61 vs 1.65), so Adam normalizes it away — the same mechanism established synthetically.
- **Direction is positive (audio entrenches)** because audio is the *more-informative* modality here
  (utility-gated) — opposite to the synthetic equal-information case, where the encoder over-compensated
  and the sign was negative. This matches H6/UTD and confirms the utility-dependence.

## 5. Statistics / scope
5 seeds = init/data-order variance, **not** population samples (test = 18 fixed actors). Effect sizes lead;
paired p-values are within-split reproducibility. CREMA-D masking-based unimodal video is degenerate (→
majority class, R1), so dominance/gain are the clean signals; the pathway split is read off dominance
shift and ρ — both unambiguous here.

## 6. Relation to synthetic
Real CREMA-D **replicates the synthetic Adam decomposition** (CASE A: forward-dominated, gradient-inert),
now on real audio-video data with a real utility asymmetry. It does **not** by itself establish behaviour
under other optimizers on real data (SGD real-data decomposition untested — a disclosed limitation; the
synthetic SGD result shows the gradient pathway would become active).

## 7. Limitations
- Adam only on CREMA-D (SGD real-data decomposition not run). - Single matched-CNN sum-fusion architecture.
- Masking video degenerate → contribution read via dominance/gain. - 18 fixed test actors (seed variance ≠ population).
- Effect is large but the mechanism (forward-mediated, gradient-inert-under-Adam) is scoped to this architecture/optimizer.

## 8. Contribution
Closes the "real-world mechanism" gap: a controlled forward-vs-gradient decomposition **on real data**
shows the training-time feature-scale effect on modality contribution (H6) is forward-magnitude-mediated
under Adam, with the encoder-gradient side-effect optimizer-normalized — matching the synthetic mechanism.

---
_Reproducibility: `results/cremad_disentangle/cremad_disentangle_{primary,trajectories,agg}.csv`,
`_analysis.json`, `_sanity.json`, `_config.json`; figures `cremad_dis_{dominance,grad_update_ratio,
trajectories}.png`. Raw media + cache not committed. Not committed pending review. Verdict from evidence._
