# CREMA-D TRAINING-TIME SCALE INTERVENTION — final report (H6)

_Preregistration: `PREREGISTRATION_CREMAD_TRAINING_SCALE.md` (frozen before results). Start commit `b7a5295`._
_**R1–R3 remain unchanged and immutable. E1–E5 remain unchanged.** This experiment specifically tests_
_the training-time causal pathway that R3 (a frozen-representation intervention) could not test._
_Labels: **[OBS]** observed · **[STAT]** statistical · **[MECH]** mechanistic · **[SPEC]** speculation._

## Verdict: **SUPPORT** (with a directional asymmetry and partial compensation)
Imposing a modality-specific feature-scale factor **during training** (in-graph, encoders trainable)
**causally and persistently changes the learned modality contribution** (Spearman(imposed, dominance)
= 0.922; audio×10 raises learned dominance 0.075→0.328 and collapses multimodal gain 0.248→0.013,
reproducible over 5 seeds). Per-modality z-score **blocks** the pathway. This contrasts sharply with
R3 (frozen-rep), where scaling had **no** effect on dominance — confirming the effect operates through
*learning dynamics*, not the converged readout.

## 1. Motivation
R3 scaled the representation *after* training (linear head absorbs a constant → no dominance change).
H6 applies the scale in-graph during every step so it participates in fusion, loss, gradients, and
parameter updates — the pathway synthetic E2 tested but real-data R3 could not.

## 2. Relation to R1–R3 (unchanged)
R1: natural ρ≈1.221 (audio>video). R2: z-score removes the scale disparity / rebalances gradients,
slight accuracy cost. R3 (frozen-rep, reciprocal scaling): ρ manipulable 0.012→122 but **dominance
FLAT** (slope −0.004). H6 fills the training-time gap R3 left open.

## 3. Preregistered hypothesis (H6)
In-graph training-time modality scaling alters training dynamics and may produce persistent learned
contribution imbalance — OR trainable encoders compensate. Not assumed.

## 4. Conditions
none: baseline (1,1), audio×5 (5,1), audio×10 (10,1), video×5 (1,5), video×10 (1,10);
z-score: {(1,1),(10,1),(1,10)}. 5 seeds. 40 runs, **0 diverged**.

## 5. Implementation
`h = alpha * enc(x)` **in-graph** at the fusion boundary via `ScaleInjector` with `scale.enabled=True`
during training; `scale_first=True` (enc→scale→norm) so z-score can remove the imposed disparity. Not
detached, not eval-only, not on input/loss/gradients/LR. Encoders **trainable**; cached features are
encoder **inputs** (valid caching). Compensation read via `raw_encode` (pre-scale encoder outputs).

## 6. Training protocol
Identical to R1–R3: `CremadMultimodalModel`, sum fusion, rep_dim 64, matched CNN encoders
(16160=16160), linear head, Adam lr 1e-3, batch 64, 30 epochs, frozen actor-disjoint split, test = 18
held-out actors only.

## 7. Results (none, test, mean over 5 seeds)  [OBS/STAT]
| condition | imposed log₂(α_a/α_v) | learned dominance | \|imbalance\| | multimodal gain | ρ_fusion | ρ_enc (learned) | compensation |
|---|---:|---:|---:|---:|---:|---:|---:|
| video×10 (1,10) | −3.32 | +0.009 | 0.009 | **+0.295** | 1.02 | 10.24 | 1.01 (full) |
| video×5 (1,5) | −2.32 | +0.056 | 0.056 | +0.263 | 1.18 | 5.90 | 1.10 (full) |
| **baseline (1,1)** | 0 | +0.075 | 0.075 | +0.248 | 1.22 | 1.22 | — |
| audio×5 (5,1) | +2.32 | +0.299 | 0.299 | +0.045 | 2.31 | 0.46 | 0.49 (partial) |
| audio×10 (10,1) | +3.32 | **+0.328** | 0.328 | **+0.013** | 3.24 | 0.32 | 0.49 (partial) |

Slope(dominance vs log₂ imposed) = **+0.049**; Spearman = **0.922**. Individual seeds consistent
(audio×10 dominance 0.289–0.354).

**[OBS] Directional asymmetry (central finding):**
- **Amplifying the naturally-stronger modality (audio)** during training → audio's learned dominance
  rises sharply and **multimodal gain collapses** (video's complementary contribution is lost).
- **Amplifying the weaker modality (video)** → contributions stay balanced and gain is **preserved/
  slightly improved**.

## 8. Gradient dynamics  [OBS]
`results/cremad/figures/cremad_TS_dynamics.png`. The audio/video encoder gradient ratio tracks the
imposed direction early and relaxes as the encoders adjust; audio-amplified conditions retain an
audio-leaning gradient balance (consistent with persistent audio dominance).

## 9. Representation dynamics  [OBS]
ρ_fusion starts near the imposed ratio (e.g. audio×10 ≈ 16 at epoch 1) and **relaxes over training**
toward a lower plateau (≈3.2), never to 1 for audio amplification — partial relaxation.

## 10. Contribution dynamics  [OBS]
`cremad_TS_contribution.png`: monotone dominance rise and monotone gain collapse with imposed audio
scale; per-seed points shown.

## 11. Compensation analysis  [MECH]
`cremad_TS_compensation.png`. The learned encoder ratio ρ_enc moves **opposite** to the imposed scale
(green vs dashed) — the encoders actively counteract the imposition — but the **net** ρ_fusion (red) is
**asymmetric**: **video amplification is fully compensated** (ρ_fusion≈1, comp≈1.0) while **audio
amplification is only ~49% compensated** (ρ_fusion 2.3–3.2, comp≈0.49). **[MECH]** The optimizer fully
undoes a boost to the *less-useful* modality (letting video dominate would hurt) but only partially
undoes a boost to the *useful* modality (audio genuinely lowers the loss, so its dominance is
"rewarded" and persists). This is *why* the imbalance persists for audio.

## 12. Normalization analysis  [STAT]
Per-modality z-score (applied after the imposed scale) **blocks the pathway**: audio×10 learned
dominance **0.328 (none) → 0.110 (z-score)** and ρ_fusion **3.24 → 0.97**; video×10 dominance
0.009→0.104, ρ_fusion 1.02→0.97. Under z-score every condition converges to the z-score baseline
(dominance ≈0.10) — normalization removes the imposed scale's effect on learned contribution. (As in
R2/E3, z-score is a scale-channel fix, not an accuracy improver.)

## 13. Statistical analysis
5 seeds; per-seed values + mean±SD + 95% CI in `results/cremad/cremad_train_scale_agg.csv`. Primary
effect: dominance dose-response Spearman 0.922, slope 0.049; audio×10 vs baseline dominance shift
+0.253. 0/40 diverged. Effect sizes reported with the trend.

## 14. Comparison with synthetic (E2/E3/E4/E4-EXTREME)
- **E2 (training-time scale → contribution):** **TRANSFERS** — training-time scale causally changes
  learned contribution (audio×10), which frozen R3 could not show.
- **E1/E4-EXTREME (encoders compensate static scale):** **PARTIALLY TRANSFERS** — encoders *do*
  compensate (ρ_enc counteracts), fully for the weak modality but only ~half for the useful one, so
  imbalance persists (a real-data refinement of "static scale is compensable": compensability is
  *modality-utility-dependent*).
- **E3/E4 (normalization = scale channel):** **TRANSFERS** — z-score blocks the training-time pathway.
- **E4/E4-EXTREME (amplifying the dominant side entrenches collapse):** **CONSISTENT** — amplifying the
  naturally-stronger audio entrenches its dominance and collapses gain.

## 15. Limitations
- **Directional asymmetry** means the effect is not a symmetric "scale disparity → collapse"; it is
  "amplifying the *useful/stronger* modality entrenches it." Driven by audio being the more informative
  modality here (dataset/architecture-specific).
- Compensation is only ~49% for audio at 30 epochs; longer training could shift it (untested).
- Single lightweight matched-CNN architecture, sum fusion, 6-way emotion (~49%); not optimized for
  accuracy. Masking-based unimodal video is degenerate (majority class); gain/dominance are the clean signals.
- The scale is imposed at the fusion boundary (post-encoder), a controlled proxy for a modality having
  systematically larger features — not a claim about any specific real cause of scale disparity.

## 16. Scientific interpretation
**[MECH]** On real CREMA-D, feature magnitude imposed *during training* is a genuine causal lever on
learned modality contribution: it works through the fusion/loss/gradient pathway (not the converged
readout, cf. R3), the encoders partially fight it, and the net persistent effect depends on modality
utility. Per the pre-registered interpretation rule, this justifies only the narrow claim: *"under the
tested CREMA-D architecture and optimization regime, imposing modality-specific feature-scale disparity
during training can alter learned modality contributions (strongly when amplifying the more informative
modality), and per-modality normalization removes this effect."* It does **not** establish the broad
sufficiency claim that feature-scale disparity is sufficient for persistent collapse across datasets,
architectures, and optimization regimes.

## 17. Verdict
**SUPPORT** — training-time scale intervention produces the predicted, reproducible, dose-dependent
change in learned modality contribution (with the directional-asymmetry and partial-compensation
qualifications above), and normalization blocks it. Chosen from evidence, not hypothesis.

## 18. Recommendation regarding INCLUDE-50
**Conditional GO (await explicit authorization; do NOT auto-start).** The training-time causal pathway
that R3 could not test is now positively supported on real audio-video data, with normalization control
and a mechanistic compensation account — the strongest real-data evidence to date. INCLUDE-50 (a second
real dataset, different modalities: manual vs non-manual sign streams) would test **cross-dataset /
cross-modality generality**, which is exactly what the narrow claim needs to broaden. Recommended next
step *if authorized*: replicate R1 + H6 (training-time scale) on INCLUDE-50 with the same frozen
discipline. **STOP here pending review.**

---
_Reproducibility: `results/cremad/cremad_train_scale_{primary,trajectories,agg}.csv`,
`cremad_train_scale_analysis.json`, `cremad_train_scale_config.json` (git commit, seeds). Figures
`cremad_TS_{dynamics,contribution,compensation}.png`. Raw media + feature cache NOT committed._
