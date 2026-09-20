# UTD-MHAD training-time scale intervention — final report (cross-dataset validation of CREMA-D H6)

_Preregistration: `PREREGISTRATION_UTD_MHAD_SCALE.md` (frozen before results). Data audit: `DATA_AUDIT_UTD_MHAD.md`._
_Start commit `c780d60`; prereg `3f66e72`; implementation `a8575c2`. **E1–E5 and CREMA-D R1–R3 / H6 are unchanged.**_
_Modality **A = Depth** (Kinect optical), **B = Inertial** (wearable IMU, 6-channel). Labels: **[OBS]** observed · **[STAT]** statistical · **[MECH]** mechanistic._

## Verdict: **PARTIAL REPLICATION** (the core H6 mechanism reproduces, one-sided / utility-gated)
Imposing a modality-specific feature-scale factor **during training** (in-graph, encoders trainable) on
UTD-MHAD **causally changes learned modality contribution in the same direction and by the same mechanism
as CREMA-D H6** — but the effect is strongly **one-sided**: it appears only when the **more-informative
modality (inertial)** is amplified, while amplifying the weak modality (depth) is fully compensated. Per-
modality z-score **suppresses** the pathway. 40 runs, **0 diverged**.

## 1. What replicated, stated plainly
- **Amplify the more-informative modality (inertial) → entrench it + collapse the multimodal gain**, only
  partially compensated by the encoders. **Amplify the weak modality (depth) → fully compensated, no
  change.** **z-score removes the effect.** This is exactly the H6 finding — with the modality **roles set
  by the data** (here inertial is informative; in CREMA-D audio was), confirming the mechanism is
  **utility-dependent, not tied to a specific modality.**

## 2. Dataset & method (see audit)
861 usable trials (depth ∩ inertial; 0 missing), 27 actions, 8 subjects. **Subject-disjoint split:
train {1,3,5}=323 · val {7}=108 · TEST {2,4,6,8}=430** (official test set unchanged). Matched-capacity
two-stream: DepthCNN (16 160 params, per-frame 2D-CNN + temporal mean-pool over 16×[32×32] frames) +
InertialCNN (17 552 params, 1D-CNN over [6×128]); sum fusion; linear head; rep_dim 64. Adam lr 1e-3,
batch 32, 60 epochs, 5 seeds [0–4]. Scale injected in-graph before fusion (`scale_first=True`), encoders
trainable; cached tensors are encoder **inputs** (valid caching). Metrics via the frozen H6 masking helpers.

## 3. Natural baseline utility (established from baseline, NOT assumed)  [OBS]
| depth-only | inertial-only | multimodal | gain | dominance (depth−inertial) | natural ρ_fusion |
|---:|---:|---:|---:|---:|---:|
| 0.036 (≈ chance 1/27) | 0.227 | 0.632 | **+0.405** | −0.191 | 0.56 |

**The more-informative modality is INERTIAL** (roles flipped vs CREMA-D, where audio was informative).
**Caveat:** masking-based **depth-only accuracy ≈ chance** → the depth unimodal readout is *degenerate*
(as noted for CREMA-D video); depth-side contribution is therefore read only through **gain / dominance**,
not the depth-only number. Multimodal gain is large (+0.405): the fusion genuinely uses both streams.

## 4. Training-time scale — primary result (none, test, mean/5 seeds)  [OBS/STAT]
| condition | imposed log₂(α_d/α_i) | dominance (depth−inertial) | multimodal gain | ρ_fusion | ρ_enc (learned) | compensation |
|---|---:|---:|---:|---:|---:|---:|
| **inertial×10** (1,10) | −3.32 | **−0.615** | **+0.053** | 0.19 | 1.94 | 0.13 (partial) |
| inertial×5 (1,5) | −2.32 | −0.501 | +0.136 | 0.31 | 1.57 | 0.19 (partial) |
| **baseline** (1,1) | 0 | −0.191 | +0.405 | 0.56 | 0.56 | — |
| depth×5 (5,1) | +2.32 | −0.193 | +0.398 | 0.58 | 0.12 | 1.34 (full) |
| depth×10 (10,1) | +3.32 | −0.198 | +0.400 | 0.58 | 0.06 | 1.24 (full) |

- **Dose-response** Spearman(imposed, dominance) = **0.757**, slope 0.064 — but **asymmetric**: flat on the
  depth-amplification side, steep on the inertial-amplification side (see `utd_TS_contribution.png`).
- **inertial×10 vs baseline:** dominance shift **−0.424** (paired t p<0.0001, Cohen's d **−9.47**); gain shift
  **−0.353** (p=0.0007, d **−4.26**). **Consistent across all 4 test subjects** (per-subject dominance
  −0.55…−0.75, gain ≈0.08…−0.02) and all 5 seeds. 0/40 diverged.

**[OBS] Directional asymmetry (central finding, replicates H6):** amplifying the naturally-stronger
(inertial) modality entrenches its dominance and collapses the multimodal gain; amplifying the weaker
(depth) modality is fully absorbed and preserves the gain.

## 5. Compensation  [MECH] (`utd_TS_compensation.png`)
The learned encoder ratio ρ_enc moves **opposite** to the imposed scale (Spearman(imposed, ρ_enc) = −0.886)
— encoders actively counteract the imposition — but **asymmetrically**:
- **Weak modality (depth) amplification: full / over-compensation** — ρ_enc → 0.06, ρ_fusion stays ≈0.58
  (comp ≈1.24–1.34). The optimizer fully undoes a boost to the near-useless modality.
- **Informative modality (inertial) amplification: only ~13–19% compensated** — ρ_enc → 1.94, ρ_fusion
  falls to 0.19 (comp ≈0.13–0.19). Amplifying the *useful* modality lowers the loss, so its dominance is
  "rewarded" and persists. **Same mechanistic logic as H6** (full compensation of the useless modality,
  partial of the useful one), just more extreme here because the utility gap is larger.

## 6. Normalization control  [STAT] (`utd_TS_normalization.png`)
Per-modality z-score (fit on TRAIN reps, applied after the imposed scale) **blocks the pathway**: every
condition converges to the z-score baseline (dominance ≈ **−0.30**):
- **inertial×10:** dominance **−0.615 → −0.299**; ρ_fusion **0.19 → 0.84**.
- depth×10: −0.198 → −0.295; ρ_fusion 0.58 → 0.84 (was already compensated).
As in R2/E3/H6, z-score is a scale-channel fix (it also shifts the baseline dominance −0.19 → −0.30), not
an accuracy tool — but it removes the imposed scale's effect on learned contribution.

## 7. Statistics & the small-N caveat  [STAT]
5 seeds estimate **initialization/optimization variance on the fixed split** — they are **not** 5 subject
samples. The test population is **4 fixed subjects {2,4,6,8}**; population generalization is therefore not
estimated. Mitigating evidence: the effect is **consistent across all 4 test subjects** and all 5 seeds
(§4). Per-seed and per-subject values in `utd_train_scale_{primary,persubject}.csv`.

## 8. Why the automated verdict is PARTIAL (not SUPPORT) — honest accounting
The core mechanism replicates (§1,4,5,6). The classification is conservative for three reasons, reported
transparently rather than tuned away:
1. **One-sidedness.** The contribution effect appears *only* for informative-modality (inertial)
   amplification; weak-modality (depth) amplification is fully compensated — a more extreme asymmetry than
   CREMA-D, because depth is ≈ chance so the utility gap is larger.
2. **Degenerate depth unimodal readout** (depth-only ≈ chance) limits the depth-side contribution signal.
3. **Classifier gate artifact.** The frozen auto-classifier evaluates z-score *suppression* on the
   **depth×10** condition (which was already compensated, so "suppression" is ill-posed there) and returns
   `False`, preventing a "REPLICATION SUPPORT" label — even though z-score clearly suppresses the effect on
   the **inertial×10** side where the effect exists (−0.615 → −0.299). **I report the auto-verdict as-is and
   did not retune the classifier to force SUPPORT.**

## 9. Cross-dataset comparison — CREMA-D H6 vs UTD-MHAD
| axis | CREMA-D H6 | UTD-MHAD |
|---|---|---|
| modalities | audio + video | depth + inertial |
| task | 6-way emotion | 27-way action |
| **more-informative modality** | audio | **inertial (roles flipped)** |
| amplify informative → dominance | entrench (0.075→0.328) | entrench (−0.191→−0.615) |
| amplify informative → gain | collapse (0.248→0.013) | collapse (0.405→0.053) |
| amplify weak modality | full compensation, gain preserved | full compensation, gain preserved |
| compensation of informative | ~49% (partial) | ~13–19% (partial) |
| compensation of weak | ~100% (full) | ~124–134% (full/over) |
| dose Spearman(imposed,dominance) | 0.922 | 0.757 |
| z-score suppresses pathway | yes (0.328→0.110) | yes (−0.615→−0.299) |
| divergence | 0/40 | 0/40 |
| verdict | SUPPORT | PARTIAL REPLICATION |

## 10. Cross-dataset table with synthetic (no ranking)
| setting | causal training-time scale → contribution | compensation | utility-dependent asymmetry | z-score removes it |
|---|---|---|---|---|
| Synthetic (E2/E3/E4) | yes (controlled) | yes (E1/E4-EXTREME) | E4 | yes (E3) |
| CREMA-D H6 | yes | partial (informative), full (weak) | yes | yes |
| UTD-MHAD | yes (informative side) | partial (informative), full/over (weak) | yes (more extreme) | yes |

## 11. Limitations
- **4 test subjects** (seed ≠ subject variance; effect is consistent across them, but generalization N is small).
- **Depth-only masking degenerate** (≈ chance) → depth-side unimodal readout uninformative; depth
  contribution inferred from gain/dominance only.
- **One-sided effect** — driven by the large utility gap (depth ≈ chance vs inertial 0.227); a dataset with
  two comparably-informative modalities might show a two-sided effect (untested).
- Single lightweight matched-CNN two-stream, sum fusion, 27-way (not accuracy-optimized; multimodal ≈0.63).
- Imposed scale is a controlled proxy at the fusion boundary, not a claim about any specific real cause of
  natural scale disparity.

## 12. Scientific interpretation & overall research implication
**[MECH]** On a **second, substantially different** real dataset (action recognition; depth camera +
wearable inertial), imposing modality feature-scale **during training** is again a genuine causal lever on
learned modality contribution: amplifying the more-informative modality entrenches its dominance and
collapses the multimodal gain (partial encoder compensation), amplifying the weak modality is fully
compensated, and per-modality z-score removes the effect — with the modality **roles set by data** (inertial
here vs audio in CREMA-D). Per the pre-registered interpretation rule, the combined evidence now supports:

> *"The training-time feature-scale intervention altered learned modality contributions under **both** tested
> real-world multimodal settings (CREMA-D audio-video and UTD-MHAD depth-inertial), via the same
> utility-dependent directional asymmetry and asymmetric compensation, and per-modality normalization
> suppressed the scale-dependent pathway in both — within the tested architectures and optimization regimes."*

It does **not** establish a universal law. The UTD-MHAD effect is **one-sided/utility-gated** and the depth
unimodal readout is degenerate, so the replication is classified conservatively as **PARTIAL REPLICATION**.
Two real datasets across different modalities/tasks strengthen the mechanism's generality; broader
architectures, fusion types, optimizers, and balanced-utility datasets remain untested.

---
_Reproducibility: `results/utd_mhad/utd_train_scale_{primary,trajectories,persubject,agg}.csv`,
`utd_train_scale_{analysis,config}.json`; figures `utd_TS_{dynamics,contribution,compensation,normalization}.png`.
Raw media + encoder-input cache NOT committed. Verdict chosen from evidence, not before analysis._
