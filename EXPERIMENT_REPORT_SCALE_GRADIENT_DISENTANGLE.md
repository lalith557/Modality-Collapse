# Forward-scale vs gradient-scale disentangling — final report

_Preregistration: `PREREGISTRATION_SCALE_GRADIENT_DISENTANGLE.md` (frozen before results). Start commit `4465ebc`._
_All prior experiments (E1–E5, CREMA-D R1–R3/H6, UTD-MHAD) are unchanged. Synthetic equal-information setup, Adam._
_Labels: **[OBS]** observed · **[STAT]** statistical · **[MECH]** mechanistic · **[SPEC]** speculation/extrapolation._

## Verdict: **CASE A — forward-dominated** (under Adam), with an essential optimizer caveat
The learned modality-contribution shift produced by the ordinary training-time scale `h_A'=α·h_A` is
**entirely reproduced by forward-magnitude scaling alone** and is **not** produced by encoder-gradient
scaling: forward-only ≡ ordinary, gradient-only ≡ baseline — even though gradient-only imposes a **99×
encoder-gradient imbalance**, which **Adam normalizes into an inert update.** 60 runs, **0 diverged**.

## 1. Exact computational definition of the four arms
`h_A` = enc_A(x_A) (normalization='none'). Modality A only is scaled at the fusion boundary in-graph:
| arm | forward value → fusion | gradient d/d h_A | implementation |
|---|---|---|---|
| baseline | `h_A` | ×1 | identity |
| forward-only | `α·h_A` | **×1** | `ForwardScale`: fwd `α·h`, bwd passes grad ×1 |
| gradient-only | `h_A` (unchanged) | **×α** | `GradScale`: fwd `h`, bwd grad ×α |
| ordinary | `α·h_A` | ×α | plain `α*h` |
Evaluation uses each arm's own forward transform (baseline/gradient present `h_A`; forward/ordinary present `α·h_A`).

## 2. Autograd sanity gate — PASSED (`disentangle_sanity.json`)
Scalar forward/backward multipliers exactly as designed: baseline (1,1), forward (α,1), gradient (1,α),
ordinary (α,α). Numeric checks: **gradient-only forward activations bit-identical to baseline** (True);
enc_A gradient-norm ratio vs baseline at α=10: **forward ×1.54** (α absent from the gradient), **gradient
×10.00** (exactly α), **ordinary ×15.40** (both compound: 1.54×10). The intervention is valid.

## 3. Configuration (identical to E4/E4-EXTREME/E5)
Equal-information generator (8 classes, dim 32+32, margin 1.5, σ=1, n_train 4000/val 1000/test 2000,
prototype_seed 12345), **data α_a=α_v=1** (scale imposed at the representation, not the data). Tiny MLP
encoders (hidden 64, rep 32), sum fusion, linear head. **Adam, base_lr 1e-3, batch 256, 40 epochs, wd 0,
lr_a=lr_v (no LR asymmetry), 5 seeds [0–4].** α ∈ {1,10,100}. GPU RTX 4070. 4×3×5 = 60 runs, 0 diverged.

## 4. Per-seed results
`results/disentangle/disentangle_primary.csv` (per run) and `_agg.csv`. Seed SD of dominance at α=100:
baseline 0.010, forward 0.028, gradient 0.010, ordinary 0.028 — low/stable (not CASE E).
**α=1 control:** all four arms give identical dominance (+0.003), confirming the arms coincide when α=1.

## 5–6. Effect sizes & 95% CIs (dominance = Acc_A − Acc_V; test; 5 seeds) [STAT]
| α | arm | dominance | shift vs baseline | Cohen d | paired p | gain | ρ_enc | ρ_fusion |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 100 | baseline | +0.003 | — | — | — | +0.195 | 0.99 | 0.99 |
| 100 | **forward** | −0.066 | **−0.070 ± 0.016** | **−3.76** | **0.0011** | +0.157 | 0.09 | 9.03 |
| 100 | **gradient** | +0.003 | **−0.0005 ± 0.0008** | −0.53 | **0.30** | +0.195 | 0.99 | 0.99 |
| 100 | ordinary | −0.066 | −0.069 ± 0.016 | −3.73 | 0.0011 | +0.157 | 0.09 | 9.03 |
| 10 | forward | −0.035 | −0.038 | (large) | | +0.172 | 0.20 | 2.02 |
| 10 | gradient | +0.003 | ~0 | | | +0.195 | 0.99 | 0.99 |
| 10 | ordinary | −0.035 | −0.038 | | | +0.171 | 0.20 | 2.02 |

**Direct pathway comparisons (α=100):** ordinary vs forward diff **+0.0002** (d=0.73, p=0.18, CI ±0.0002)
→ **indistinguishable**; ordinary vs gradient diff **−0.069** (d=−3.76, p=0.0011) → **large, separated**;
gradient vs baseline diff −0.0005 (d=−0.53, p=0.30) → **null**.

## 7. Forward-only result [OBS]
Forward scaling reproduces the *entire* ordinary effect: dominance shift −0.070 (α=100), −0.038 (α=10),
gain 0.195→0.157, and it drives the encoder to **compensate** (ρ_enc 0.99→0.09), leaving a residual net
ρ_fusion=9.03. The effect is dose-dependent (α=10 vs 100) and CI-separated from baseline.

## 8. Gradient-only result [OBS]
Gradient scaling has **no effect** on learned contribution: dominance +0.003 (= baseline), gain 0.195 (=
baseline), ρ unchanged — **despite imposing a 99× encoder-gradient ratio** (verified in the trajectory).
**[MECH]** Adam is invariant to constant per-parameter gradient rescaling (first/second moments both
scale by α, α²; the normalized update m̂/√v̂ is unchanged), so a 99× gradient imbalance produces an inert
update. The gradient pathway, *as a constant scaling under Adam*, is neutralized.

## 9. Ordinary-scale result [OBS]
Ordinary = forward exactly (dominance −0.066, gain 0.157, ρ_enc 0.09, ρ_fusion 9.03). The extra encoder-
gradient scaling ordinary carries over forward-only (grad ratio ~25–99 vs forward's ~0.3) is Adam-
neutralized and changes nothing in the outcome.

## 10. Direct pathway comparison (the dissociation)
- **forward pathway:** forward imbalance present (ρ_fusion↑, ρ_enc↓), gradient imbalance absent (grad
  ratio ~0.3–1), **outcome present** (dominance −0.07).
- **gradient pathway:** forward imbalance absent (ρ~1), gradient imbalance present (**99×**), **outcome
  absent** (dominance = baseline).
- **ordinary:** both imbalances present, **outcome = forward's**.
⇒ The outcome tracks the **forward** manipulation, not the gradient manipulation.

## 11. Training trajectories (α=100, `disentangle_trajectories.png`) [OBS]
- ρ_fusion: forward/ordinary start ~87 and relax to ~9 (encoder compensation); baseline/gradient flat ~1.
- ρ_enc: forward/ordinary crash 0.88→0.09 (encoder shrinks A); baseline/gradient flat ~1.
- grad_ratio: gradient ~99 (flat), ordinary ~100→25, **forward ~0.3** (forward scaling actually *lowers*
  enc_A's gradient share), baseline ~1.
- val_dominance: forward≡ordinary show an early **transient** rise (+0.12 by epoch ~3) then settle
  **negative** (~−0.055 by epoch 40); gradient≡baseline stay flat ~+0.005 throughout.
**[MECH]** A forward imbalance appears immediately in the forward arms and is progressively compensated;
the gradient imbalance in the gradient arm never translates into a contribution change. (Temporal
tracing only; no causal claim from correlation.)

## 12. Mechanism classification: **forward-dominated (CASE A)** — with the Adam caveat
The training-time scale effect on learned contribution is **forward-feature-magnitude-mediated** in this
Adam, equal-information setup. **[MECH, essential caveat]** This is **optimizer-specific.** The gradient
pathway is inert *because Adam normalizes constant gradient rescaling*, not because gradients are
irrelevant to modality competition. Under **SGD**, gradient-only would equal an α× effective learning-
rate boost and would **not** be inert — it would behave like the E4 learning-rate axis (which E4/E4-
EXTREME show drives persistent collapse). So the correct statement is: *under Adam, the training-time
scale intervention's causal effect on learned contribution runs through the forward magnitude; its
gradient side-effect is optimizer-neutralized.*

## 13. Effect on existing claims
| claim | before | after this experiment |
|---|---|---|
| Training-time scale is a **causal modulator** of learned contribution | supported (intervention level) | **SUPPORTED** — and now localized to the forward pathway |
| The operative variable is **feature magnitude** (not gradient/optimization) | *unresolved / confounded* (audit) | **SUPPORTED under Adam** (forward-dominated; gradient side-effect inert) — with the SGD caveat |
| Effect is entangled forward↔gradient | assumed | **REFUTED under Adam** — cleanly separable; forward carries it |
| Encoder compensation is real | supported | **SUPPORTED** (ρ_enc 0.99→0.09 in forward-scaled arms) |
| Scale disparity is a *sufficient cause of collapse* | NOT supported (E4-EXTREME) | still **NOT supported** (equal-info effect is small, −0.07 dominance, and compensated) |
| Gradient/optimization channel drives modality imbalance | supported for the **LR** knob (E4) | **NOT via constant gradient scaling under Adam**; the E4 result is an **LR**-magnitude effect, which Adam responds to (LR) but not to gradient rescaling — consistent, now sharpened |

## 14. Is "causal modulator" still justified? **Yes.** [STAT]
The intervention causally and reproducibly shifts learned contribution (forward/ordinary vs baseline
d=−3.76, p=0.0011, dose-dependent, 0 diverged, stable across seeds). "Causal modulator" stands and is now
mechanistically localized to the forward magnitude pathway (under Adam).

## 15. Is "feature magnitude is the operative mechanism" justified? **Qualified yes.**
In this Adam, equal-information setting: **yes** — the forward feature magnitude is the operative variable;
the coupled gradient effect is Adam-neutralized (the 99× gradient imbalance yields no contribution change).
**Caveat:** (i) optimizer-specific (SGD would revive the gradient/LR pathway); (ii) established for the
equal-information synthetic setup — extrapolation to H6/UTD is **[SPEC]** plausible (same Adam) but not
directly tested here; (iii) the magnitude effect is small and heavily compensated (consistent with
E4-EXTREME "scale is compensable").

## 16. Does the original title need to be abandoned? **Yes** (unchanged from the audit).
This experiment does not rescue "sufficient cause / silences / collapse / minimal cure." It sharpens the
*mechanism* (forward-magnitude, Adam-specific) but the effect remains small and compensable in equal-info;
"sufficient cause of modality collapse" stays refuted by E4-EXTREME. The conservative titles from the
audit still apply; this result supports framing the mechanism as *forward-feature-magnitude modulation
under adaptive optimization*.

## 17. Single strongest defensible conclusion
**[MECH]** *Under Adam, the causal effect of a training-time modality-feature-scale factor on learned
modality contribution is mediated by the forward feature magnitude at fusion, not by the induced
encoder-gradient scaling: forward-only scaling reproduces the full effect (d=−3.76) while gradient-only
scaling — despite a 99× encoder-gradient imbalance — is indistinguishable from baseline, because Adam
normalizes constant gradient rescaling. The effect is real but small and strongly compensated by the
encoders (ρ_enc 0.99→0.09). This is optimizer-specific: under SGD the gradient pathway would reduce to a
learning-rate effect and would not be inert.*

---
_Reproducibility: `results/disentangle/disentangle_{primary,trajectories,agg}.csv`, `_analysis.json`,
`_sanity.json`, `_config.json`; figures `disentangle_{arms_contribution,arms_distribution,trajectories,
compensation}.png`. Not committed pending review. Verdict chosen from evidence, not before analysis._
