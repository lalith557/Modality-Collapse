# EXPERIMENT_LOG.md

Running log. Newest first. Each entry: hypothesis / config / result / interpretation / failure modes / next action.

---

## 2026-09-20 — UTD-MHAD training-time scale (cross-dataset validation of H6). Verdict: PARTIAL REPLICATION. Pre-registered (PREREGISTRATION_UTD_MHAD_SCALE.md)

**Question.** Does the H6 training-time feature-scale mechanism transfer to a substantially
different real dataset — UTD-MHAD action recognition, depth camera + wearable inertial?

**Config.** Modality A=depth (16-frame 32×32 clips), B=inertial (6ch×128 @50Hz). Official
cross-subject split train {1,3,5}/val {7}/**test {2,4,6,8}** (subject-disjoint, verified; 861
usable = depth∩inertial, 0 missing). Matched encoders (depth 16160 / inertial 17552 params),
sum fusion, in-graph scale (scale_first=True), trainable encoders, Adam 1e-3, batch 32, 60 ep,
5 seeds. Conditions none {baseline, depth×5/×10, inertial×5/×10} + zscore {baseline, depth×10,
inertial×10}. 40 runs, **0 diverged**. Encoder-input caching (valid).

**Baseline utility (established, not assumed).** depth-only 0.036 (≈chance 1/27), inertial-only
0.227, multimodal 0.632, gain +0.405 → **more-informative modality = INERTIAL (roles flipped vs
CREMA-D)**. Depth-only masking degenerate (≈chance) → depth read via gain/dominance only.

**Result (none, test).** dominance (acc_depth−acc_inertial) & gain by imposed scale:
- baseline: dom −0.191, gain +0.405, ρ_fus 0.56
- depth×5/×10: dom −0.193/−0.198 (flat), gain +0.398/+0.400 (preserved), ρ_enc 0.12/0.06, comp 1.34/1.24 (FULL)
- inertial×5: dom −0.501, gain +0.136, ρ_enc 1.57, comp 0.19
- inertial×10: dom **−0.615**, gain **+0.053**, ρ_fus 0.19, ρ_enc 1.94, comp 0.13 (PARTIAL)
Spearman(imposed,dom)=0.757 (asymmetric); inertial×10−baseline: Δdom −0.424 (p<1e-4, d=−9.47),
Δgain −0.353 (p=7e-4, d=−4.26); consistent across all 4 test subjects & 5 seeds.

**Key findings.** (1) Core H6 mechanism REPLICATES with roles set by data: amplifying the
INFORMATIVE modality (inertial) entrenches its dominance + collapses gain (partial ~13-19%
compensation); amplifying the WEAK modality (depth) is fully/over-compensated (comp ~1.24-1.34),
gain preserved. (2) Compensation asymmetric, Spearman(imposed,ρ_enc)=−0.886 — same logic as H6.
(3) z-score BLOCKS pathway: inertial×10 dom −0.615→−0.299, ρ_fus 0.19→0.84; all conditions
converge to z-score baseline ≈−0.30. (4) Natural ρ_depth/inertial≈0.56.

**Verdict = PARTIAL REPLICATION** (conservative). Core mechanism replicates, but effect is
ONE-SIDED (only informative-modality amplification moves contribution; weak fully compensated) +
depth-only readout degenerate. The frozen auto-classifier's z-score-suppression gate is anchored
to depth×10 (already compensated → ill-posed), so it did not auto-label SUPPORT even though
z-score clearly suppresses the effect on the inertial×10 side. Classifier NOT retuned to force
SUPPORT. Implication: training-time scale altered learned contributions in BOTH real settings
(audio-video AND depth-inertial), utility-dependent, normalization-suppressible — strengthens
generality, not a universal law. E1-E5 / CREMA-D R1-R3/H6 unchanged. **Next: STOP; await review.**

---

## 2026-09-20 — CREMA-D training-time scale (H6). Verdict: SUPPORT. Pre-registered (PREREGISTRATION_CREMAD_TRAINING_SCALE.md)

**Question (H6).** Does imposing modality scale DURING training (in-graph, encoders
trainable) cause persistent learned modality imbalance? (R3 was frozen-rep and couldn't test this.)

**Config.** In-graph ScaleInjector (scale_first=True), conditions none {baseline,
audio×5/×10, video×5/×10} + zscore {baseline, audio×10, video×10}, 5 seeds, 30 ep, same
architecture as R1-R3, cached ENCODER INPUTS (encoders trainable). 40 runs, 0 diverged.

**Result (none, test).** Learned dominance (acc_audio−acc_video) & gain by imposed audio scale:
- baseline: dom 0.075, gain 0.248, ρ_fus 1.22
- audio×5: dom 0.299, gain 0.045, ρ_fus 2.31, ρ_enc 0.46, comp 0.49
- audio×10: dom **0.328**, gain **0.013**, ρ_fus 3.24, ρ_enc 0.32, comp 0.49
- video×5: dom 0.056, gain 0.263, ρ_enc 5.90, comp 1.10
- video×10: dom 0.009, gain 0.295, ρ_enc 10.24, comp 1.01
Slope(dom vs log2 imposed)=0.049, Spearman=**0.922**.

**Key findings.** (1) Training-time scale CAUSALLY changes learned contribution (vs R3 frozen
= null) — RESOLVES the R3 gap. (2) DIRECTIONAL ASYMMETRY: amplifying the stronger modality
(audio) entrenches its dominance + collapses gain (persistent); amplifying the weaker (video)
is fully compensated + preserves gain. (3) COMPENSATION: encoders counteract imposed scale
(ρ_enc opposite slope) — FULL for video (ρ_fus≈1), PARTIAL ~49% for audio (ρ_fus 3.2) because
the loss rewards the useful modality. (4) z-score BLOCKS the pathway: audio×10 dom 0.328→0.110,
ρ_fus 3.24→0.97.

**Transfer.** E2 (training-time causal) TRANSFERS; E1/E4-EXTREME (compensation) PARTIALLY (utility-
dependent); E3/E4 (normalization=scale channel) TRANSFERS; E4 (amplify dominant entrenches) CONSISTENT.

**Verdict → SUPPORT** (narrow claim per interpretation rule; directional-asymmetry + partial-comp
qualifications). GO/NO-GO INCLUDE-50 = conditional GO (await authorization; do not auto-start).
Report: docs/experiments/CREMAD_TRAINING_SCALE_REPORT.md. R1-R3 & E1-E5 immutable.

---

## 2026-09-20 — CREMA-D R1/R2/R3 executed. Verdict: PARTIAL SUPPORT. (frozen protocol)

**Config.** Cached features (0 failures) → 10 models (none/zscore × 5 seeds), 30 ep, sum
fusion, matched CNN encoders. Test = frozen 18 actors. See CREMAD_FINAL_REPORT.md.

**R1 (vanilla, test).** acc_av 0.493±0.019; **natural ρ=1.221±0.043** (MILD audio>video
scale disparity); grad ratio audio/video 2.74; Δ_audio 0.322, Δ_video 0.248, gain 0.248.
→ real feature-scale disparity EXISTS but mild.

**R2 (zscore−none, paired).** ρ 1.221→0.965 (d−3.6,p.001); grad ratio 2.74→0.14
(d−5.2,p.0003) — z-score removes scale disparity + rebalances gradients. BUT
abs_imbalance 0.075→0.103 (ns), gain 0.248→0.191 (ns), acc_av 0.493→0.467 (p.014, HURTS).
→ scale-channel intervention, not a performance improver (E3/E4 nuance transfers exactly).

**R3 (frozen-rep scale c: audio×c, video×1/c; test).** Intervention WORKS: Spearman(c,ρ)
=0.99, ρ 0.012→122. **Relative dominance FLAT** (slope −0.004, spearman −0.31) — magnitude
does NOT reassign dominance. **Multimodal gain = sharp inverted-U peaked at c=1** (0.248 →
≤0 at extremes); acc_av 0.493→0.17-0.25. → magnitude causally destroys INTEGRATION when
perturbed, but doesn't reassign dominance (consistent with "static scale compensable":
linear head absorbs constant scale).

**Transfer.** Transfers: observational scale disparity (R1); normalization scale-channel
effect (R2, ↔E3/E4); static-scale-compensability (R3 flat dominance, ↔E4/E4-EXTREME).
NOT transferred (as tested): positive causal scale→dominance — R3 is frozen-rep/no-retrain,
can't test the training-time E2 pathway. Ambiguous: magnitude→gain inverted-U (integration
brittleness, not competition). E5/OGM not run (per instruction).

**Verdict → PARTIAL SUPPORT.** GO/NO-GO for INCLUDE-50 = **HOLD**: central causal claim needs
a TRAINING-TIME scale intervention on real data (new preregistration) before INCLUDE-50.
0 failures, 0 divergence, GPU peak <8GB. Raw media + feature cache NOT committed.

---

## 2026-09-20 — CREMA-D media UNBLOCKED + pipeline validated (smoke PASSED). No R1-R3 run.

**Unblock.** LFS was accessible (not over quota). Obtained real media from official
repo (github.com/CheyneyComputerScience/CREMA-D, HEAD 1658cd34): shallow no-smudge
clone → `git lfs pull` audio + video into `../CREMA-D-full/` (~2.7 GB, outside repo).
oid matches CREMA-D-master stub (same dataset).

**Validation (results/cremad/cremad_validation.json).** 7442 clips/91 actors/6 emo/
12 sent (IDENTITY_OK). audio present 7442/7442, video 7442/7442; 0 stub/zero/missing/
dup; manifest↔disk consistent. Sample 72: RIFF 72/72, FLV 72/72, decoded 72/72, 0
failures. Audio 16 kHz, 1.4-3.64 s. Video 42-109 frames. MEDIA_OK. Split preserved
byte-identical (train 64/5234, val 9/738, test 18/1470, 0 overlap).

**Pipeline (data/crema_d.py, models/cremad_{encoders,model}.py, experiments/cremad_smoke.py).**
Audio→log-mel[1,64,188] (torchaudio+scipy), video→[8,1,64,64] (cv2/ffmpeg). Model
mirrors MultimodalModel → reuses rho/masking/z-score/scale-injection unchanged.
Encoders capacity-matched (16160=16160 params).

**Smoke PASSED.** 36 clips (actor-disjoint), loss 1.85→1.78, checkpoint reload OK,
GPU peak 245 MB, 4.1 s, 0 leakage. Scale-injection verified: alpha_audio=5 → rho
2.28→11.38 (exactly x5). [descriptive only] natural audio/video scale disparity at
init rho~2.28 (audio 2.07 / video 0.91), grad-ratio ~5.5 — NOT an R1 result.

**Status → READY (media validated; smoke passed). No scientific claim yet.**
R1-R3 technically ready under frozen PREREGISTRATION_CREMAD.md; NOT run (stop after
smoke). Report: docs/experiments/CREMAD_MEDIA_VALIDATION.md. Raw media not committed.

---

## 2026-09-20 — CREMA-D real-world validation — BLOCKED (media absent). Pre-registered (PREREGISTRATION_CREMAD.md)

**Goal.** Test whether the synthetic feature-scale mechanism transfers to real
audio-video data (R1 observational, R2 normalization, R3 controlled scale intervention).

**Blocker (Phase 2 audit).** Local `CREMA-D-master` has **only git-LFS pointer stubs**
(all 7442 audio + 7442 video = 130 B each; stub declares real size ~72 KB). Folder has
**no `.git`** → `git lfs pull` impossible locally. **media_present = 0/0.** No audio/
video to encode → R1–R3 cannot run. Did NOT fabricate results.

**Done (media-independent).** Structural audit: 7442 clips, 91 actors, 6 emotions
(ANG/DIS/FEA/HAP/SAD 1271, NEU 1087), 12 sentences, 43F/48M. Frozen **actor-disjoint**
split (seed 42, sex-stratified, 70/10/20): train 64 actors/5234, val 9/738, test
18/1470, **0 actor overlap (asserted)**. Manifest built (`data/cremad_manifest.py` →
data/cremad/ + results/cremad/). Full protocol pre-registered.

**Classification → CREMA-E (INVALID — data availability). Decision → NO-GO/STOP** before
INCLUDE-50 (same media risk, larger). Resolve media first (clone CREMA-D repo + LFS
pull [may hit LFS quota], or provide a local media copy). Once manifest reports
media_present>0, R1–R3 run under the frozen protocol with no further design choices.
Report: docs/experiments/CREMAD_FINAL_REPORT.md.

---

## 2026-09-20 — E5 (OGM-GE mechanistic comparison). Pre-registered (PREREGISTRATION_E5.md)

**Question.** Do z-score (scale channel) and OGM-GE (optimization channel) correct
SEPARABLE components of modality imbalance? (H5-A/B/C.)

**Config.** 4 arms {vanilla, z-score, OGM-GE, z-score+OGM-GE} × conditions
{balanced, scale-only 100×, LR-only 100×, aligned 20×/100×, opposed 100×, +
difficulty positive control} × 5 seeds = **260 runs**, 40ep, sum fusion. OGM-GE
faithful (baselines/ogm_ge.py): per-modality softmax discrepancy, k=1−tanh(α·ratio),
α=0.1, γ=0.5, applied to encoder grads. Frozen collapse def. 0 diverged.

**Result (weak-modality contribution, baseline 0.192).**
| cond | vanilla | z-score | OGM-GE | both |
|---|---|---|---|---|
| scale-only100 | 0.184 | 0.202 | 0.182 | 0.193 |
| LR-only100 | 0.086 | 0.102 | **0.085** | 0.107 |
| aligned20 | 0.087 | 0.139 | **0.087** | 0.133 |
| aligned100 | 0.002 | **0.113** | **0.003** | 0.112 |
| opposed100 | 0.209 | 0.104(hurts) | 0.202 | 0.094 |
- z-score vs OGM on aligned100: d=−8.8, p=4e-5. OGM on LR-only: Δ=−0.000, p=0.96.
  OGM→both: +0.110 = the z-score component (OGM adds nothing).
- **Mechanism:** aligned |g_A|/|g_V| = 41 (vanilla) → 0.52 (z-score, fixed at source)
  → 42.6 (OGM, unchanged). OGM modulation ACTIVE (k_A→0.57) but ineffective.
- Positive control (difficulty): OGM Δweak ±0.001 (active, no effect — low headroom).

**Interpretation (honest).** z-score corrects the scale/magnitude channel by
rebalancing gradients at the source; OGM-GE (faithful, active) corrects NEITHER the
scale nor the LR channel here — it attenuates the dominant update but can't enlarge
the suppressed rep or unstarve the weak encoder. Decomposition only HALF-supported:
z-score↔scale holds; NO tested method fixes the optimization channel. H5-B/C not
supported. Our LR proxy (imposed starvation) ≠ OGM's greedy-dynamics target — so this
is not a fair test of OGM in general (implementation verified faithful, not E5-E).

**Classification → E5-D** (OGM-GE does not correct the induced collapse; implementation active).

**Decision → HOLD** before real data. Scale channel is clean & real-data-testable;
optimization channel unresolved. Min claim: scale-magnitude disparity → collapse via
rep-magnitude suppression + gradient imbalance; normalization fixes it in the scale
regime (neutral/harmful otherwise). Next: EITHER validate the narrowed scale-channel
claim on CREMA-D, OR induce a natural greedy-dynamics (shared-LR) collapse where OGM
applies and re-run E5. Full report: docs/experiments/E5_FINAL_REPORT.md.

---

## 2026-09-19 — E4-EXTREME (extreme-ratio + long-horizon). Pre-registered (PREREGISTRATION_E4_EXTREME.md)

**Question.** Does stronger aligned scale×LR asymmetry convert graded imbalance into
persistent collapse, or does the encoder keep compensating?

**Config.** 25 targeted conditions (scale-only, LR-only, aligned, opposed @ 20/100/400×,
both directions) × {none, zscore} × 5 seeds @40ep, + aligned/refs @100 & 200ep = **390
runs**. Same generator/architecture/frozen collapse def as E4. GPU, 0 diverged.

**Result (baseline weak 0.192).**
- **Dose-response (severity, none):** scale-only 0.05/0.04/**0.08** (compensated even
  at 400×!); LR-only 0.23/0.55/0.88; aligned 0.55/0.99/**1.00**; opposed 0/0/0.
  Monotone (Spearman 1.0). Residual |logρ| aligned 1.66→4.05→4.81.
- **Long horizon (200 ep):** scale-400 recovers (transient); aligned-20 hovers ~0.08;
  aligned-100 creeps to 0.025; **aligned-400 flat ≈0 through 200ep, ρ GROWS to 6.3
  (runaway)**. None recover. PC survives long for aligned 20/100/400.
- **z-score decomposition (400×):** scale-only→0.208 (full recovery), LR-only→0.083,
  aligned→0.083, opposed→0.084 — z-score maps ALL to the LR-driven floor: removes the
  scale channel, leaves the LR channel.

**Interpretation (honest).** A genuine PERSISTENT-collapse regime exists (survives 200ep,
monotone dose) — but it is **optimization(LR)-driven, NOT scale-driven**: static scale is
compensable at every magnitude; LR asymmetry (learning-speed starvation) drives collapse;
scale alignment amplifies it to totality; z-score fixes only the scale channel.

**Classification → EXTREME-A (TRUE PERSISTENT REGIME)**, with the caveat above.

**Failure modes.** 400× LR (lr_V=5e-5) partly starves the weak modality (extreme, though
graded 20→100 establishes the mechanism); operational "persistence" = 200ep; sum fusion
only; z-score init-time stats. Analysis dose() indexing bug fixed post-hoc (runs
deterministic/unaffected; dose recomputed from CSV).

**Next action (decision tree → EXTREME-A ⇒ E5).** GO to E5 (OGM-GE) targeted at the
aligned-collapse region: z-score fixes scale channel, OGM-GE (gradient modulation) should
address the LR channel z-score cannot — sharp mechanistic test. Hold CREMA-D/INCLUDE.
Full report: docs/experiments/E4_EXTREME_FINAL_REPORT.md.

---

## 2026-09-19 — E4 (scale × learning-rate boundary). Pre-registered (PREREGISTRATION_E4.md)

**Question.** Does asymmetric optimization (per-encoder LR ratio) block the learned
compensation and turn transient scale-driven imbalance into persistent collapse?

**Config.** Signed 9×9 grid: scale ratio × LR ratio ∈ {0.05..20} (geometric,
reciprocals) × {none, z-score} × 5 seeds × 40 epochs = **810 runs**, equal
information, sum fusion. Robustness: concat × 3 seeds. GPU. QC clean (810/810,
5 seeds/cell, LR/scale exact to 1e-16, balanced weak 0.192 matches E2/E3).

**Result.**
- **Aligned interaction robust:** severity aligned **0.244 [0.220,0.270]** vs
  opposed **0.045 [0.036,0.054]**; response-surface OLS (R²=0.70) aligned term
  p=7e-117; LR main effect (0.048) > scale main effect (0.013).
- **Persistent collapse RARE:** frozen rule → **2/81 cells (none), 0/81 (z-score),
  0/81 (concat)** — only the extreme aligned corners (20×/20×, 0.05×/0.05×).
  Run-collapse 8.7% of aligned runs, 3.5% overall.
- **Mechanism = compensation failure:** residual |logρ| aligned 0.74 vs opposed
  0.20; Spearman(residual, severity)=**0.75**. Trajectories: pure-scale recovers
  fully by ep10 (transient); aligned (20,20) ρ stays ~1.6, weak crawls to 0.095
  (persistent); opposed ρ crashes to 0 by ep5 (fast recovery).
- **z-score regime-dependent:** OLS norm:scale=−0.018 and norm:align=−0.014 cancel
  the base terms (removes scale/alignment collapse); norm:LR=+0.025 (can't fix LR).
  Recovers weak +0.045 in collapse cells (0.079→0.131), but −0.024 acc overall.
- **Mild A/V asymmetry:** aligned-corner severity A-up 0.217 vs V-up 0.272 (MWU p=0.039).

**Interpretation (honest).** The scale×LR interaction is real, significant and
mechanistically explained (compensation blocked by aligned unfavourable LR), but at
ratios ≤20× / 40 epochs it mostly yields a **graded severity surface**, not
widespread persistent collapse (2/81). z-score neutralises the scale/alignment
channel (not the LR channel) and partially rescues the collapse corners — a
regime-dependent benefit absent in scale-only E3.

**Verdict → OUTCOME B (PARTIAL SUPPORT).** Not A (persistent collapse rare), not
C/D (interaction is real and normalization alters the regime).

**Failure modes.** 40-epoch operational "persistence"; ratios capped 20×; single
generator; mild A/V asymmetry; z-score init-time stats.

**Next action (GO/NO-GO).** Conditional GO: refine before real data — a NEW
pre-registered extreme-ratio/long-horizon grid (aligned gaps >20×, ~100–200 ep) to
see if the collapse region genuinely expands; then E5 (OGM-GE) only in the
aligned-collapse region. Hold CREMA-D/INCLUDE. Full report: docs/experiments/E4_FINAL_REPORT.md.

---

## 2026-09-19 — E3 (per-modality normalization, H3) + compensation analysis

**Hypothesis (H3).** Per-modality normalization prevents the **early** optimization
imbalance caused by scale disparity (not merely improves final accuracy).

**Config.** `configs/normalization.yaml`. norms {none, layernorm, zscore,
batchnorm} × 7 symmetric geometric scales {0.05..20} × 5 seeds (paired) =
**140 runs**, 40 epochs, headroom regime, sum fusion. Checkpoints 1/3/5/10/final.

**Result (mean over 5 seeds; extreme = scales {0.05,0.1,10,20}).**
| norm | early \|D\|@ext (ep5) | final \|D\|@ext | test_acc | resid\|logρ\| | conv_ep |
|---|---:|---:|---:|---:|---:|
| none | 0.152 | 0.036 | **0.690** | 0.452 | 10.6 |
| layernorm | 0.020 | 0.013 | 0.648 | 0.011 | 3.5 |
| zscore | 0.025 | 0.019 | 0.668 | 0.014 | 3.8 |
| batchnorm | 0.019 | 0.018 | 0.645 | 0.009 | 3.5 |

- **Early-imbalance reduction vs none (ep5, extreme):** layernorm +0.132±0.024,
  zscore +0.127±0.023, batchnorm +0.133±0.024 — all **CI-separated** (n=20). ≈87% cut.
- **Final accuracy vs none:** layernorm −0.047±0.003, zscore −0.027±0.003,
  batchnorm −0.049±0.002 — all **CI-separated NEGATIVE**. Normalization costs 3–5 pts.
- **Convergence:** 10.6 → ~3.5 epochs (≈3× faster).
- **Compensation (scale=20):** none relaxes log ρ from ratio 0.82 → residual 0.17
  over ~20 epochs (slow, incomplete); layernorm/zscore ≈0 from epoch 0 (structural);
  batchnorm 0.82 → 0 by epoch 5 (running-stat warmup). Normalization removes the
  need for *learned* compensation.

**Interpretation (honest).** Normalization **is a cure for the early/transient
imbalance** and a strong training-dynamics accelerant (flat ρ, flat early |D|, ~3×
faster convergence), and it removes the learned-compensation the plain encoder must
otherwise perform. But it is **not a final-performance improver** here — it slightly
**reduces** final accuracy (LayerNorm/BatchNorm worst at −0.05; z-score gentlest at
−0.027), because in this clean equal-information regime the plain encoder already
compensates by convergence and normalization discards representation geometry. So
H3's "minimal cure" claim holds for **transient collapse / dynamics**, not for
converged accuracy.

**Failure modes.** BatchNorm couples batch samples (running-stat eval); z-score is
fit once at init (stats drift as encoder changes) — yet all three cure the early
imbalance. Final-accuracy cost may differ in a genuine-collapse regime (short
training / optimization imbalance) where none stays collapsed — untested here.

**Next action.** E4: does normalization still cure when collapse is
optimization-driven (scale × learning-rate), where the plain encoder *cannot*
compensate? That is the regime where normalization might also help final accuracy.

**Figures.** `results/figures/e3/primary/e3_{norm_vs_scale_contribution,norm_rho_trajectory,norm_convergence,norm_effect_size,compensation}.png`.

---

## 2026-09-19 — E1 (early-ρ diagnostic, H1). Pre-registered (see PREREGISTRATION.md)

**Hypothesis (H1).** `log(rho_epoch5)` predicts later modality dominance
`D = Δ_A − Δ_V`, **beyond** merely measuring the injected scale.

**Config.** `configs/diagnostic.yaml`. n=**150 runs** (5 seeds × factors).
Population varies: scale ratio (equal-info family, geometric), information
difference (margin_A≠margin_V, scale=1), noise difference (σ_A≠σ_V, scale=1),
fusion (sum/concat), seed/init. Primary predictor `log(rho5)`; primary outcome
final `D` (test); AUROC of `A_stronger` at each horizon epoch. All on GPU.

**Result.**
- **AUROC(early ρ → dominance @ epoch k):** 0.91 (ep1–7) → 0.77 (ep10) → 0.57
  (ep15) → **0.40 (ep40, reversed).** Crosses chance ≈ epoch 14.
- **Overall Spearman(log ρ5, D_final) = 0.06 [−0.09, 0.21]** (CI incl. 0).
  → pre-registered PRIMARY outcome (final dominance): **not predicted.**
- **Beyond-scale controls:** partial Spearman | log(scale) = **0.60 [0.49, 0.70]**;
  balanced-scale subfamily = **0.65 [0.51, 0.77]**; incremental R² of ρ5 over
  scale = **+0.425**. → ρ5 *does* carry info beyond injected scale.
- **Per-family Spearman(ρ5, D_final):** equal **−0.825**, info_diff **+0.957**,
  noise_diff **+0.25** — while `trueAsym(snr_A−snr_V)→D_final = 0.97` in BOTH
  info_diff and noise_diff. So ρ5 tracks the *information* asymmetry (aligned) but
  **largely misses the *noise* asymmetry** (0.25 vs 0.97).
- Sanity: `corr(trueAsym, D_final) = 0.94` — final dominance IS predictable, just
  not from ρ5 alone.

**Interpretation (honest).** Early ρ is a **strong predictor of the *transient*
(early-training) dominance** that scale disparity induces (AUROC≈0.91 for the first
~7 epochs) but **not of the *converged* outcome** — the relationship decays to
chance (~ep14) and **reverses** (encoders over-compensate the up-scaled modality,
so high ρ5 → that modality ends slightly *weaker*). ρ is also **confounded by the
source of magnitude**: magnitude from signal/scale/info ⇒ "stronger" (ρ aligned),
but magnitude from noise ⇒ "weaker" (ρ misses it). The near-zero overall Spearman
is a *cancellation* of these opposing family effects, not absence of signal.

**Verdict.** As pre-registered (predict *final* dominance): **UNSUPPORTED**
(Spearman 0.06; AUROC reversed to 0.40). Restricted to the transient regime it was
designed for (early dominance, ep≤7): **SUPPORTED** (AUROC 0.91). Net honest label:
**PARTIALLY SUPPORTED** — a strong *early-warning* diagnostic, not a predictor of
the converged state, and blind to noise-driven collapse. Verdict rule was frozen
before running; not adjusted post-hoc.

**Failure modes.** ρ conflates signal- and noise-magnitude; converged dominance is
compensated/reversed; sum-vs-concat pooled (fusion breakdown in CSV).

**Next action.** E3 (normalization = enforced compensation): does it remove the
early imbalance ρ detects? Later: a *normalized/expected* ρ (ρ minus its
noise-explained part) as a de-confounded diagnostic (H1′).

**Figures.** `results/figures/e1/primary/e1_{auroc_by_epoch,early_rho_vs_final_dominance,rho_trajectory_examples}.png`.

---

## 2026-09-19 — E2 (synthetic causal scale sweep), Milestone 1

**Hypothesis (H2).** With information held fixed (verified by an alpha-invariant
linear probe), increasing feature-magnitude disparity `alpha_A/alpha_V` causes
increasing modality dominance / collapse of the weak modality.

**Common config.** `configs/synthetic_causal.yaml`: 8 classes, 32+32 dims,
`sum` fusion, no normalization, tiny MLP encoders (rep_dim 32), Adam lr 1e-3,
batch 256, 5 seeds, GPU (RTX 4070). Scale sweep `ratios = [0.05 … 20]` in
**geometric** mode (`alpha_A=√r`, `alpha_V=1/√r`; geometric-mean magnitude held
fixed so the *ratio* is the only variable). `prototype_margin = 1.5`
(**headroom regime**: unimodal linear-probe ≈ 0.55, so collapse can visibly hurt).

**QC (all runs).** Linear-probe accuracy varies < 0.04 across the whole sweep
within each seed → **information is genuinely constant**; empirical input-norm
ratio matches the intended ratio; no NaN/Inf.

### Run A — converged (40 epochs), tag `primary`
| ratio | acc_av | Δ_A | Δ_V | gain | final ρ | early ρ (ep5) |
|--:|--:|--:|--:|--:|--:|--:|
| 0.05 | 0.690 | 0.222 | 0.180 | 0.180 | 0.60 | 0.14 |
| 1.00 | 0.699 | 0.201 | 0.198 | 0.195 | 1.00 | 1.01 |
| 20.0 | 0.687 | 0.176 | 0.212 | 0.176 | 1.68 | 7.55 |

- `acc_av` **flat ≈ 0.69–0.70** across the full 400× ratio span; multimodal gain
  preserved. Spearman(ratio, dominance) = **−0.89**. Automated verdict: **INCONCLUSIVE**.
- **Key observation:** `early ρ` spans 0.14 → 7.55 (tracks scale strongly) but
  `final ρ` only 0.60 → 1.68. The encoder **learns to renormalize** the injected
  scale by convergence.

### Run B — undertrained (5 epochs), tag `headroom_ep5`
| ratio | acc_av | Δ_A | Δ_V | gain | ρ (ep5) |
|--:|--:|--:|--:|--:|--:|
| 0.05 | 0.563 | 0.056 | 0.234 | 0.056 | 0.14 |
| 1.00 | 0.671 | 0.203 | 0.197 | 0.191 | 1.01 |
| 20.0 | 0.561 | 0.212 | 0.053 | 0.053 | 7.55 |

- **Strong, symmetric collapse** of the weak modality at both extremes; `acc_av`
  drops 0.67 → 0.56; multimodal gain roughly halves. Spearman(ratio, dominance)
  = **+0.97**. Automated verdict: **SUPPORTED**.
- `early ρ` → dominance correlation: **Spearman 0.98** (H1 signal is real).

**Interpretation (honest).**
Feature-magnitude disparity, with information held fixed, is a **sufficient cause
of *transient / early-training* modality collapse** (Run B), and it reliably
drives the activation-norm ratio ρ. But a trainable encoder given enough steps
**absorbs a constant scale** (a linear layer can undo it), so the collapse
**does not persist at convergence** in this clean setting (Run A). Therefore:
*static scale alone is not a sufficient cause of **persistent** collapse for this
architecture — it is an optimization/early-dynamics effect and a co-factor.*
This is the sharper, reviewer-resistant version of the thesis and directly
motivates H1 (early-ρ diagnostic), H3 (normalization as the mechanism that
enforces the compensation up-front), and H4 (scale × learning-rate boundary,
where compensation can be impeded).

**Failure modes / caveats.**
- The `sum`-fusion + absorb-capable linear encoder is exactly the case where a
  constant scale is compensable; results may differ for `concat`, frozen
  encoders, or cross-attention (ablations).
- `prototype_margin` and epoch budget strongly modulate the effect size; both
  are pre-registered in config, not tuned post-hoc.
- Geometric scaling halves the per-modality log-scale vs anchored scaling, so ρ
  ranges are milder than an anchored sweep would show.

**Next action.**
1. E1: per-epoch early-ρ → final-collapse predictor (AUROC), pre-register epoch 5.
2. E4 pilot: scale × LR grid — test whether LR coupling makes scale collapse
   persist (impeding encoder compensation).
3. E3: does per-modality normalization prevent the Run-B transient collapse
   without hurting Run-A? (normalization = enforced compensation.)

---

## 2026-09-19 — Debug sweeps (pre-milestone sanity)
- 8-epoch / n=800 smoke: at `alpha_A=10` (anchored) V collapsed (Δ_V 0.211→0.065)
  — consistent with the transient-collapse reading above (undertrained + tiny data).
- 15-epoch anchored margin=3: dominance shifted with scale (Δ_A rises), collapse
  mild — near-ceiling task leaves little headroom. Motivated the margin=1.5
  headroom regime used for the milestone.
