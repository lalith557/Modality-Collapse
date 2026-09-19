# CREMA-D FINAL REPORT — R1/R2/R3 real-world validation

_Frozen protocol: `PREREGISTRATION_CREMAD.md` + `configs/cremad.yaml`. Media validation:
`reports/CREMAD_MEDIA_VALIDATION.md`. E1–E5 unmodified. Start commit `221bc37`._
_Labels: **[OBS]** observed · **[STAT]** statistical · **[MECH]** mechanistic · **[SPEC]** speculation._

## Executive verdict: **PARTIAL SUPPORT**
The **observational** feature-scale disparity (R1) and the **normalization scale-channel
effect** (R2) transfer to CREMA-D and are consistent with the synthetic mechanism. The
**controlled scale intervention** (R3) shows that magnitude **causally degrades multimodal
gain** (sharp inverted-U at the natural scale) but does **not** reassign relative modality
dominance on the frozen model — consistent with the synthetic finding that *static scale is
compensable* (a trained linear head absorbs constant scale). The specific prediction "scaling
one modality up makes it dominate" does **not** transfer under the frozen-representation R3.

## 1. Dataset & provenance
CREMA-D, official repo `github.com/CheyneyComputerScience/CREMA-D` (HEAD `1658cd34`), via
Git LFS (not over quota). 7442 clips · 91 actors · 6 emotions · 12 sentences; audio 16 kHz WAV,
video .flv. Integrity: 7442/7442 audio+video present, 0 decode failures (see media-validation report).

## 2. Frozen split
Actor-disjoint, seed 42, sex-stratified: **train 64 actors / 5234 clips · val 9 / 738 · test 18 / 1470**;
**0 actor overlap** (asserted). Byte-identical to the pre-media manifest.

## 3. Implementation
Features cached once (deterministic; 0 failures): audio→log-mel `[1,64,188]` (torchaudio, 64 mel,
n_fft 1024, hop 256, 16 kHz, 3 s), video→`[8,1,64,64]` grayscale (cv2/FFmpeg). Model
`CremadMultimodalModel` mirrors the synthetic `MultimodalModel` (sum fusion, rep_dim 64, linear
head) so ρ/masking/z-score/scale-injection are reused unchanged; **capacity-matched encoders
(16160 = 16160 params)**. Adam lr 1e-3, batch 64, **30 epochs, 5 seeds**. Test metrics use ONLY
the frozen test actors. (concat fusion is a preregistered *secondary*, deferred this pass.)

## 4. R1 protocol
Train vanilla (no normalization) × 5 seeds. Measure fusion-boundary audio/video feature norms, ρ,
masking-based unimodal accuracy, Δ_audio = Acc_AV − Acc_video, Δ_video = Acc_AV − Acc_audio,
multimodal gain, encoder gradient norms, training trajectories. (Modality contribution uses the
frozen synthetic *masking* definition.)

## 5. R1 results  [OBS/STAT]
Test, mean ± SD over 5 seeds (individual acc_av: 0.508/0.469/0.486/0.515/0.487):
| metric | value |
|---|---|
| acc_av (audio+video) | **0.493 ± 0.019** |
| acc_audio (video masked) | 0.245 ± 0.031 |
| acc_video (audio masked) | 0.171 ± 0.000 (≈ majority-class 0.171) |
| Δ_audio | 0.322 ± 0.019 |
| Δ_video | 0.248 ± 0.031 |
| \|Δ_audio − Δ_video\| | 0.075 ± 0.031 |
| multimodal gain | 0.248 ± 0.031 |
| **natural ρ = ‖h_audio‖/‖h_video‖** | **1.221 ± 0.043** (ρ per seed 1.20–1.27) |
| encoder grad ratio (audio/video) | 2.74 ± 0.49 |

**[OBS]** A **natural feature-scale disparity exists but is MILD** (ρ≈1.22, i.e. audio activations
~22% larger than video), with an audio-leaning gradient ratio (2.74). Both modalities contribute
(Δ_audio 0.32, Δ_video 0.25); video is uninformative *alone* via zero-masking (0.171 ≈ majority)
but adds a real complementary gain (0.248) in combination.

## 6. R2 protocol
Vanilla vs per-modality z-score applied **before fusion** (E3/E4 `ZScoreNorm`, fit on train reps
only), same seeds/architecture/budget. Paired by seed.

## 7. R2 results  [STAT] (z-score − none; paired, n=5)
| metric | none | z-score | Δ | Cohen d | p |
|---|---:|---:|---:|---:|---:|
| ρ | 1.221 | 0.965 | **−0.257** | −3.60 | **0.001** |
| grad ratio (audio/video) | 2.74 | 0.14 | **−2.60** | −5.19 | **0.0003** |
| \|Δ_audio − Δ_video\| | 0.075 | 0.103 | +0.029 | 0.42 | 0.40 |
| multimodal gain | 0.248 | 0.191 | −0.057 | −0.84 | 0.13 |
| acc_av | 0.493 | 0.467 | **−0.026** | −1.86 | **0.014** |

**[STAT]** z-score **removes the feature-scale disparity** (ρ→0.96) and **strongly rebalances the
encoder gradients** (2.74→0.14) — both CI-separated. **[OBS]** But it does **not** reduce the
modality-*contribution* imbalance (0.075→0.103, ns) and **slightly hurts accuracy** (−2.6 pts,
p=0.014) with a small gain reduction (ns). This is the E3/E4 nuance transferring intact:
normalization is a **scale-channel** intervention, not a universal performance improver.

## 8. R3 protocol (primary causal test)
On the FIXED trained representation (no retraining), multiply audio by c and video by 1/c for
c ∈ {0.1,0.2,0.5,1,2,5,10}; c<1 video-favoured, c>1 audio-favoured (both directions). Evaluate on
the frozen test actors. Also applied to the z-score model. **Information-preservation:** the
intervention is a constant multiply on fixed representations — an invertible linear transform, so
information is preserved by construction. The frozen preregistration prescribes **no** separate
linear-probe check, so none was invented (reported as a design point, per protocol).

## 9. R3 results  [OBS/STAT] (test, mean over 5 seeds)
| c | ρ | dominance (Acc_a−Acc_v) | Δ_audio | Δ_video | gain | acc_av |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.012 | 0.105 | 0.000 | −0.105 | −0.105 | 0.171 |
| 0.5 | 0.305 | 0.077 | 0.060 | −0.017 | −0.017 | 0.231 |
| **1.0** | **1.221** | 0.075 | 0.322 | 0.248 | **0.248** | **0.493** |
| 2.0 | 4.885 | 0.074 | 0.156 | 0.082 | 0.082 | 0.326 |
| 10.0 | 122.1 | 0.073 | 0.077 | 0.003 | 0.003 | 0.247 |

- **[STAT] The intervention works:** Spearman(c, ρ) = **0.99**, ρ spans **0.012 → 122** (10⁴×).
- **[OBS] Relative dominance is FLAT** across the whole range (Spearman(c, dominance) = −0.31,
  slope −0.004; CIs overlapping) → magnitude does **not** reassign which modality dominates.
- **[OBS] Multimodal gain is a sharp inverted-U peaked at the natural scale (c=1)**: 0.248 at c=1,
  collapsing to ≤0 at both extremes; acc_av 0.493 → 0.17–0.25. **Magnitude causally destroys
  multimodal integration when perturbed away from the trained scale.**
- **[STAT]** Under z-score the dominance response is even flatter (slope −0.001; 0.31× the none
  slope), and the gain inverted-U is slightly lower — consistent with z-score removing scale sensitivity.

## 10. Statistical analysis
5 seeds throughout; per-seed values, mean ± SD, paired t + Cohen d (R2), Spearman + OLS slope (R3),
95% bootstrap/CI in `results/cremad/*_agg.csv` and `cremad_analysis.json`. No seed discarded; 0
run failures; 0 divergence. Effect sizes reported with p-values.

## 11. Training dynamics  [OBS]
`results/cremad/figures/cremad_R1_trajectories.png`: ρ(t) is **stable ~1.2 throughout** training
(no large transient scale imbalance like the synthetic extremes); the audio/video grad ratio stays
> 1; val accuracy rises smoothly. `cremad_R3_scale_response.png`, `cremad_R2_comparison.png`.

## 12. Causal interpretation
**[MECH]** R3 provides the controlled intervention. It establishes that **feature magnitude
causally affects multimodal integration** (gain collapses when either modality is scaled away from
natural), but **not** that magnitude reassigns relative dominance on a converged model. **[MECH]**
The reason is the same as the synthetic conclusion: a trained (linear) head expects the natural
scale and largely absorbs a constant multiplier, so post-hoc scaling perturbs the *joint* readout
(brittleness) without changing which modality the model relies on. **[OBS, not causal]** the R1 ρ–
performance association is observational and is **not** claimed causal.

## 13. Comparison with E1–E5
- **Transfers:** natural feature-scale disparity exists (R1 ↔ E2 premise); per-modality
  normalization removes the scale disparity and rebalances gradients but does not improve accuracy
  (R2 ↔ **E3/E4**, precise nuance match); static scale is compensable — post-hoc scaling does not
  reassign dominance (R3 ↔ **E4/E4-EXTREME** "static scale compensable").
- **Does NOT transfer (as tested):** positive causal *scale → dominance* reassignment. The
  synthetic E2 effect arose from **training-time** scale; R3 (frozen-rep, no-retraining) cannot
  reproduce that pathway, and the frozen result is null on dominance.
- **Ambiguous:** magnitude → multimodal-gain inverted-U is a real causal effect on *integration*
  (frozen-head brittleness), not the modality-*competition* mechanism of E2/E4.
- **Not tested:** E1 early-ρ diagnostic (ρ(t) is flat here, little to predict); E5 OGM-GE (not run
  on CREMA-D, per instruction).

## 14. Limitations
- **R3 design (as preregistered) is frozen-representation, no-retraining** — it tests the converged
  model's magnitude sensitivity, **not** the training-time causal mechanism that produced the
  synthetic E2 effect. This is the central limitation.
- Natural ρ disparity is **mild** (1.22) vs the large synthetic ratios; CREMA-D at this
  architecture/scale may simply not exhibit strong scale-driven collapse.
- Masking-based unimodal accuracy is degenerate for video-alone (→ majority class), so the
  dominance metric is noisy; gain/acc_av are the cleaner R3 signals.
- Single lightweight architecture (small matched CNNs), sum fusion, 30 epochs; concat secondary
  deferred. 6-way emotion at ~49% (not optimized for accuracy — by design).
- No separate information-preservation probe (not prescribed; preservation is by construction).

## 15. Reproducibility
Config `configs/cremad.yaml` + `results/cremad/cremad_config.json` (git commit, seeds, feature
shapes, device). Scripts: `data/cremad_manifest.py`, `data/cremad_validate.py`,
`experiments/cremad_extract.py`, `experiments/cremad_run.py`. Results: `cremad_primary.csv`,
`cremad_scale_intervention.csv` (+ `_agg`), `cremad_aggregated.csv`, `cremad_trajectories.csv`,
`cremad_analysis.json`. Raw media (~2.7 GB) NOT committed. GPU RTX 4070; peak well under 8 GB.

## 16. Scientific verdict: **PARTIAL SUPPORT**
- **[OBS/STAT] Transfers:** real feature-scale disparity exists (R1); per-modality normalization is
  a genuine scale-channel intervention (R2: ρ↓, gradients rebalanced), with the synthetic caveat
  intact (no accuracy gain).
- **[STAT] Causal magnitude effect is real but partial:** R3 shows magnitude causally controls
  multimodal gain (inverted-U) but not relative dominance — consistent with "static scale is
  compensable," and therefore **not** positive causal evidence for scale-driven dominance in real
  data under the frozen design.
- **Net:** the *scale-channel* half of the mechanism transfers to CREMA-D; the *training-time causal
  dominance* half is neither confirmed nor refuted here (untestable under frozen-rep R3). Verdict
  **PARTIAL SUPPORT**, chosen from the evidence, not the hypothesis.

## GO/NO-GO for INCLUDE-50: **HOLD** (do not auto-proceed)
The scale channel transfers, but the central causal claim needs a **training-time** scale
intervention on real data (scale applied during training, as in synthetic E2) before INCLUDE-50 is
justified. That would be a new preregistered CREMA-D experiment — not run here. **STOP.**
