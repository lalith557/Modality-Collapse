# E5 FINAL REPORT — OGM-GE mechanistic comparison

_Preregistration: `PREREGISTRATION_E5.md` (frozen before run). New experiment; E1–E4-EXTREME unmodified._
_Labels: **[OBS]** observed · **[STAT]** statistical · **[MECH]** mechanistic · **[SPEC]** speculation._

---

## 1. Executive summary
Across a targeted 4-arm comparison {vanilla, z-score, OGM-GE, z-score+OGM-GE} over scale-only / LR-only / aligned / opposed conditions (5 seeds, 260 runs, 0 diverged):
- **[STAT] Per-modality z-score recovers the scale/magnitude-driven collapse** (aligned 100×: weak 0.002 → 0.113; aligned 20×: 0.087 → 0.139; z-score vs OGM effect d=−8.8, p=4×10⁻⁵), and it does so by **rebalancing the encoder gradients at the source** (aligned |g_A|/|g_V|: 41 → 0.52).
- **[STAT] OGM-GE has essentially no effect in any tested condition** (LR-only Δweak=−0.000, p=0.96; aligned Δweak=+0.001, p=0.29; positive control ±0.001), **despite verified-active modulation** (aligned k_A→0.57). Adding OGM-GE to z-score adds nothing (both ≈ z-score).
- **[MECH]** OGM attenuates the dominant modality's *update* but does not change the underlying gradient imbalance (grad-ratio unchanged 41→42.6) and cannot raise the suppressed/starved weak modality.
- **[OBS] Implementation verified faithful and active** (not E5-E).

**Classification: E5-D** — OGM-GE does not correct the collapse this project induces (modulation active but ineffective). **Real-data decision: HOLD** (the clean two-channel/two-method decomposition is not supported; the LR-starvation proxy differs from OGM's target).

## 2. Scientific question
Do representation normalization (scale channel) and optimization-level intervention (OGM-GE) correct *different, separable* components of modality imbalance? H5-A: z-score↓scale; H5-B: OGM-GE↓optimization; H5-C: combined complementary.

## 3. Preregistration
`PREREGISTRATION_E5.md`. Conditions, four arms, α=0.1, γ=0.5, 5 seeds, frozen collapse definition, primary metric = weak-modality contribution (not accuracy). A **difficulty positive control** (info asymmetry, equal LR/scale) was pre-committed to test OGM fidelity. Nothing changed after results.

## 4. OGM-GE implementation fidelity
`baselines/ogm_ge.py`. **Faithful:** per-modality softmax-score discrepancy ratio; dominant-modality coefficient `k = 1 − tanh(α·ratio)`; applied to uni-modal encoder gradients only. **Project-specific:** per-modality logit score for sum fusion uses the shared head weight (`s_m = h_m Wᵀ + b`). **Adapted:** GE = Gaussian noise restoring part of the modulation-reduced covariance (paper maintains covariance more precisely). **[OBS] Verified active:** in aligned 100×, k_A drops to ~0.57 (dominant attenuated), k_V=1.0 (`results/figures/e5_gradient_mechanism.png`). So the negative result is **not** an inactive implementation (rules out E5-E).

## 5. Experimental design
Conditions (both directions): C0 balanced (1,1); C1 scale-only (100,1); C2 LR-only (1,100); C3 aligned (20,20); C4 aligned (100,100); C5 opposed (100,1/100); + difficulty positive control (margin 2.5 vs 1.0, equal LR/scale). Arms M0–M3. Same generator/architecture (sum fusion, linear head)/optimizer/40 epochs/frozen collapse def. No per-method tuning.

## 6. QC
260/260 runs, 0 diverged; baseline weak (balanced vanilla) = **0.192** (identical to E4/E4-EXTREME); OGM modulation active (k<1). Config in `results/e5/e5_config.json`.

## 7. Scale-only results (Test A)
**[STAT]** Weak contribution: vanilla 0.184, z-score 0.202 (Δ=+0.018, d=1.1, p=0.06), OGM 0.182 (Δ=−0.002, p=0.23), both 0.193. **[OBS]** Scale-only 100× is already largely compensated (vanilla 0.184 ≈ baseline 0.192, consistent with E4-EXTREME); z-score gives a marginal top-up, OGM nothing.

## 8. LR-only results (Test B — critical)
**[STAT]** Weak: vanilla 0.086 (collapsed to ~½ baseline), **OGM 0.085 (Δ=−0.000, d=−0.02, p=0.96 — zero effect)**, z-score 0.102 (Δ=+0.017, d=1.3, p=0.04 — small), both 0.107. **[OBS]** OGM does **not** correct the LR-driven (optimization-starvation) collapse; z-score only marginally helps. This is the key negative control: the "optimization channel" is **not** fixed by OGM-GE here.

## 9. Aligned results (Test C)
**[STAT]** Aligned 100× weak: vanilla 0.002, **z-score 0.113**, **OGM 0.003**, both 0.112. z-score vs OGM: Δ=−0.110, d=−8.8, p=4×10⁻⁵. OGM→both: Δ=+0.110, d=12, p=1×10⁻⁵ (all benefit is the z-score component). Aligned 20×: vanilla 0.087, z-score 0.139, OGM 0.087, both 0.133. **[OBS]** z-score recovers aligned collapse substantially; OGM adds nothing, alone or combined.

## 10. Opposed results (Test D)
**[STAT]** Weak: vanilla 0.209 (healthy — favourable LR rescues the scale-weak modality), z-score 0.104 (**hurts**), OGM 0.202 (no change), both 0.094 (hurts). **[OBS]** z-score removes the scale advantage that was *helping* in the opposed regime — confirming it is a scale-channel intervention with a regime-dependent sign, not a generic regularizer.

## 11. Combined intervention (H5-C)
**[STAT]** both ≈ z-score in every condition (aligned100 both 0.112 vs z-score 0.113; opposed both 0.094 vs z-score 0.104). **[OBS] No complementary benefit** — OGM-GE contributes nothing on top of z-score. H5-C **not supported**.

## 12. Gradient analysis
**[STAT/MECH]** `results/figures/e5_gradient_mechanism.png`: aligned 100× pre-modulation grad-ratio |g_A|/|g_V| grows to ~41 identically for vanilla and OGM (OGM does **not** reduce the underlying gradient imbalance), while z-score holds it at **0.52**. OGM's coefficient k_A→0.57 modulates the *update* but the gradient imbalance and weak-modality starvation persist. **[MECH]** z-score fixes the imbalance at its source (representation magnitude); OGM acts downstream on the dominant update and cannot lift the suppressed/starved modality.

## 13. Compensation analysis
**[MECH]** Persistent collapse ↔ gradient imbalance ↔ weak-modality starvation, connecting E5 to E4/E4-EXTREME: the collapse signature is a large, growing |g_A|/|g_V| that vanilla never resolves; z-score resolves it (→0.5), OGM does not (→42).

## 14. Statistical analysis
5 seeds, paired by seed; mean/SD/95% CI + Cohen's d + paired-t p (§7–9). Key effects CI-separated: z-score recovers aligned (d≈−8.8 vs OGM); OGM null everywhere (|d|≤0.6, p≥0.29 except the trivially-signed scale-only). Effect sizes reported alongside p-values; the OGM nulls are consistent across conditions and the positive control, so multiplicity does not change the conclusion.

## 15. Architecture context
E4 concat-fusion robustness already showed the aligned interaction is architecture-general (weaker in concat). E5 was not repeated for concat (per plan); the mechanism (magnitude suppression → gradient imbalance) is fusion-general.

## 16. Negative results (reported, not hidden)
- **OGM-GE null across all conditions**, including its closest native setting here (difficulty control: Δweak ±0.001). Implementation verified active.
- z-score does **not** fix the pure LR-only channel (only +0.017) and **hurts** the opposed and balanced regimes (accuracy −0.036 balanced, −0.12 opposed).
- The pre-registered decomposition "z-score↔scale, OGM-GE↔optimization" is **only half-supported**: z-score↔scale holds; **no tested method cleanly fixes the optimization channel**.

## 17. Limitations
Our "optimization channel" is induced by an **imposed per-modality LR ratio** (starvation), which differs from the natural **greedy-learning dynamics under shared LR** that OGM targets — so E5 is a fair test of *whether OGM fixes our collapse*, but **not** a fair test of OGM in its intended setting. The difficulty positive control had low weak-modality headroom (easier modality also more useful), so it did not create an OGM-recoverable collapse. α=0.1, γ=0.5 fixed (not tuned); larger α was not explored (pre-registered). Single synthetic generator; sum fusion; 40 epochs.

## 18. Mechanistic interpretation
**[MECH]** Modality collapse in this project has two sources: (i) **representation-magnitude suppression** (the weak modality's activations are too small to be read at the fusion point) and (ii) **learning starvation** (the weak encoder cannot train). Per-modality normalization removes (i) by equalising activation magnitude, which **rebalances the gradients at the source** and lets the (normal-LR) head/encoder use the weak modality. OGM-GE addresses **neither**: attenuating the dominant modality's update does not enlarge the weak representation nor raise the starved encoder's effective step. **[SPEC]** In a collapse driven by genuine greedy dynamics under shared LR (OGM's setting), OGM might succeed; that regime was not induced here.

## 19. What E5 establishes
- **[STAT]** z-score corrects the scale/magnitude-driven collapse by rebalancing gradients (large, CI-separated effect), with a regime-dependent sign (helps under scale suppression, hurts when scale is favourable).
- **[STAT/OBS]** A faithful, active OGM-GE does **not** correct the scale- or LR-induced collapse in this synthetic setting, alone or combined with z-score.
- **[MECH]** The collapse signature and its fix are located in the encoder gradient-imbalance, upstream of where OGM acts.

## 20. What E5 does NOT establish
- It does **not** show OGM-GE is ineffective in general (its native greedy-dynamics regime was not induced; α not swept).
- It does **not** establish a clean two-channel/two-method decomposition (the optimization channel has no working method here).
- It does **not** validate the mechanism on real data, nor make any novelty/SOTA claim.

## 21. GO / HOLD / STOP decision → **HOLD** (mechanism needs refinement)
The scale/magnitude channel is clean and robust (z-score works, source-level gradient evidence) and is **real-data-testable**. But the **optimization channel is not resolved**: our LR-starvation proxy is not OGM's target, and no tested method corrects it — so the intended "normalization vs optimization method" decomposition is not supported as stated. Rushing CREMA-D on the *full* mechanism is not justified.
- **Minimum defensible claim now:** *Feature-scale (magnitude) disparity causes modality collapse via representation-magnitude suppression and encoder gradient imbalance; per-modality normalization corrects it at the source, specifically in the scale-driven regime, while it is neutral-to-harmful where scale is not the driver.* (The optimization/LR-starvation collapse is a separate phenomenon that normalization only partly addresses and that OGM-GE does not address in this setup.)

## 22. Recommended next experiment (before any real data)
Two options, pick one and pre-register:
1. **Reframe to the scale channel and validate that on CREMA-D** (conditional GO for a *narrowed* claim): audio/video have genuinely different raw feature scales — measure raw feature norms, ρ, whether per-modality normalization corrects scale-driven imbalance. Do **not** carry the OGM/optimization-channel claim into real data yet.
2. **Refine the optimization-collapse induction** (HOLD path): induce collapse via natural greedy dynamics under **shared LR** (e.g., matched-usefulness modalities with different learning difficulty), where OGM is applicable, and re-run E5. Only then compare normalization vs OGM as complementary channels.

**STOP here.** No CREMA-D / INCLUDE / OGM-sweep / paper auto-run.

---
_Machine-readable: `results/e5/e5_primary.csv` (per-run), `e5_aggregated.csv`, `e5_analysis.json`, `e5_config.json`. Figures: `results/figures/e5_{mechanism_decomposition,intervention_comparison,aligned100_trajectories,gradient_mechanism}.png`._
