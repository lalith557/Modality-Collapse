# EXPERIMENT_LOG.md

Running log. Newest first. Each entry: hypothesis / config / result / interpretation / failure modes / next action.

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
