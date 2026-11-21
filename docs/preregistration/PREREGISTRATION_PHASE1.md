# Phase 1 — Pre-registered predictions (written BEFORE running)

Date: 2026-11-09, before any Phase-1 run. Repo @ c76c5e6.
These are falsifiable; I will report them against the runs and mark each
SUPPORTED / WEAKENED / REFUTED. I commit to not rationalizing a miss.

## Decisive test: does synthetic Adam forward scaling degrade encoder A?
Observed (existing logs): under Adam, α=100, forward arm, masked acc_A drops
0.501 → 0.447 (ΔD = −0.070), while ρ_enc compensates to ≈0.09.

**Primary prediction (P-main): the acc_A drop is mostly a head / off-distribution
artifact, NOT genuine encoder degradation.** Concretely I predict the frozen-
encoder linear probe on encoder A will drop **less** than masked acc_A does:

- masked acc_A drop (baseline→forward): ≈ 0.05 (replicates the log).
- probe acc_A drop (baseline→forward): **< 0.02** (near-flat), because the probe
  standardizes features (α-invariant) and reads the encoder's class structure,
  which forward scaling + Adam compensation should largely preserve in direction.

Decision rule (fixed before the run):
- If probe acc_A also drops materially (≳ the masked drop, say ≥ 0.04): the
  encoder genuinely learned less → **real effect**. P-main REFUTED.
- If probe acc_A is ~flat while masked acc_A drops: **off-distribution / head
  artifact** → P-main SUPPORTED, and the paper's "read by magnitude" stance is
  vindicated for a concrete reason (not hand-waved).

I am genuinely uncertain here (~60/40 toward artifact). Strong Adam compensation
(ρ_enc→0.09, an ~11× norm shrink of h_A) could plausibly worsen conditioning
enough to hurt the encoder; the probe adjudicates.

## Secondary predictions
- **P2 (trajectories):** Adam forward ρ_enc dives well below 1 (→≈0.09) =
  strong compensation; SGD forward ρ_enc stays ≈1 (little compensation). acc_A
  and acc_B diverge more under SGD (|shift|≈0.24) than Adam (|shift|≈0.07).
- **P3 (Adam update ratio):** synthetic Adam gradient-only parameter-update
  ratio ρ_Δθ ≈ 1 (predict 1.0–2.0), matching CREMA-D's 1.61 and confirming
  Table I's Adam column is not a dataset artifact.
- **P4 (mean-ablation vs zeroing):** mean-ablation will REDUCE the apparent
  dominance shift vs zero-masking (less off-distribution), but keep the same
  sign on CREMA-like unbalanced structure; on the matched synthetic it stays
  near zero.
- **P5 (Shapley):** 2-modality Shapley split will track mean-ablation more than
  zero-masked dominance; under matched synthetic + Adam forward, Shapley of A
  changes little (consistent with artifact interpretation if P-main holds).
- **P6 (10 seeds):** wider n does not overturn signs; synthetic Adam forward
  shift stays small-negative (≈ −0.05 to −0.08), SGD stays ≈ +0.24.

## What would change the paper
- P-main SUPPORTED → add probe/Shapley columns, state the acc_A drop is a
  head artifact, keep the magnitude-based reading, strengthen H3.
- P-main REFUTED → the paper must say forward scaling under Adam *degrades the
  scaled encoder* (a new, non-trivial claim); rewrite H1/H3 accordingly.

---

## OUTCOME (2026-11-09, after the 10-seed run)

**P-main: REFUTED.** probe acc_A shift (baseline→forward, Adam, α=100) =
**−0.034 ± 0.005** (predicted < 0.02). Mean-ablation dominance shift −0.056 and
Shapley_A shift −0.043 move the *same* way, all CIs exclude 0. The acc_A drop is
a **real reduction in the scaled modality's encoder quality and contribution**,
NOT a head/zeroing artifact. I was wrong; the sign is meaningful.

- P2: SUPPORTED. Adam forward ρ_enc→0.09 (strong compensation); SGD forward
  ρ_enc≈0.97, ρ_fus stays ≈100 (no compensation). See figures/phase1_trajectories.png.
- P3: SUPPORTED. Synthetic Adam gradient-only ρ_Δθ = 1.001 ± 0.010 (vs SGD 95.8).
- P4: SUPPORTED (direction), WEAKENED (magnitude). Mean-ablation ≈ zero-mask
  (Adam forward −0.056 vs −0.063), so zeroing was not badly off-distribution.
- P5: mixed — Shapley tracks the real effect (drops under Adam forward).
- P6: SUPPORTED. 10 seeds keep all signs; Adam forward −0.060, SGD forward +0.235.
