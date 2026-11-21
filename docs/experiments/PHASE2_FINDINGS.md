# Phase 2 findings (detailed tables)

Companion to `../CLAIMS_LEDGER.md`. 10 seeds, paired 95% CIs. Synthetic unless noted.

## Per-experiment: which modality got the multiplier, and what was "weak"

| Experiment | Gets the multiplier (α or LR) | Dependent / "weak" measure | Direction of harm |
|---|---|---|---|
| synthetic disentangle (P0/1/2) | modality **A** (forward/grad/ordinary ×α) | dominance = acc_A − acc_B | Adam: A loses (scaled modality) |
| phase2 **lrmatch** | **A** (enc-A Adam LR ×α) | A probe / dominance | A loses (higher LR) |
| phase2 **LR-sweep** | **A** (enc-A LR ×r) | A probe vs r | — (reference curve) |
| CREMA-D disentangle | **A = audio** (×α) | dominance = acc_audio − acc_video | audio dominance ↑ (unbalanced) |
| **E4 / S_loss** | **both**: α_A=√s↑, α_V=1/√s↓; lr_A=base√l↑, lr_V=base/√l↓ | "weak" = min(δ_A, δ_V) | the **higher-LR** modality has the **lower** contribution |

**E4 consistency check:** E4's higher-LR modality loses contribution — the SAME
direction as phase2 (the scaled/higher-effective-LR modality A loses under Adam).
No conflict. "Feature-scale compensable (0.04–0.08)" vs "LR imbalance persistent
(→0.88 at 400×)" = small vs large magnitude of one monotone high-LR-degrades
mechanism. (e4_grid.csv, scale_r=1: lr_r=20 → δ_A 0.20→0.14.)

## #1 — probe_A/B and Shapley_A/B shifts vs baseline (α=100, paired CI)

| opt/arm | Δprobe_A | Δprobe_B | ΔShapley_A | ΔShapley_B | Δ(probe_A−probe_B) |
|---|---|---|---|---|---|
| adam/forward | −0.034±0.005 | +0.010±0.004 | −0.043±0.003 | +0.013±0.004 | −0.044±0.006 |
| adam/gradient | +0.000±0.001 | +0.000±0.001 | −0.000 | −0.000 | −0.000±0.001 |
| adam/ordinary | −0.034±0.005 | +0.010±0.004 | −0.043±0.003 | +0.013±0.004 | −0.044±0.006 |
| adam/lrmatch | −0.094±0.007 | +0.016±0.004 | −0.107±0.004 | +0.010±0.007 | −0.110±0.008 |
| sgd/forward | −0.038±0.007 | −0.043±0.007 | +0.066±0.010 | **−0.197±0.012** | +0.004±0.011 (ns) |
| sgd/gradient | +0.014±0.007 | −0.022±0.004 | +0.118±0.010 | −0.132±0.016 | +0.036±0.007 |
| sgd/ordinary | +0.039±0.008 | −0.042±0.006 | +0.180±0.009 | −0.195±0.012 | +0.081±0.010 |

**Reading:** (i) The **SGD dominance gain is B-starvation** (Shapley_B collapses;
probe-dominance ≈ 0). (ii) **SGD forward ≠ gradient in probe_A** (−0.038 vs
+0.015, non-overlapping) — do NOT claim forward ~ gradient under SGD. (iii) Adam
forward degrades A's encoder (probe_A −0.034) while B slightly improves.

## #6 — forward_k (k in EPOCHS of 40) + test-time scale removal

| variant | dom | probe_A | probe_B |
|---|---|---|---|
| baseline | +0.003 | 0.518 | 0.519 |
| forward_k k=1,5,10,20 ep | ≈0 | ≈0.531 | ≈0.529 |
| full forward, **scale removed at test** | −0.188 | **0.484** | 0.528 |

probe_A=0.484 with scale removed = identical to the normal forward arm → the
encoder degradation is real and test-scale-independent; only masked dominance
depends on deployment calibration.

## #9 — compensation c(α) under SGD ≈ 0 → H1 Adam-specific

SGD c: 0.057 / 0.036 / 0.013 / 0.008 / 0.032 (α 3/10/30/100/300). SGD does not
compensate (ρ_fus ≈ α). H1 ("a constant scale is compensated") is **Adam-specific**.

## Final-batch results (conditions #2, #3, #5, #7, #8)

- **#2 per-modality heads (logit-sum) vs shared head.** SGD forward −0.040±0.009
  vs gradient +0.012±0.009 — forward≠gradient PERSISTS with separate heads
  (shared was −0.038 / +0.015). Adam unchanged (forward −0.037, gradient 0.000).
  → shared-head explanation REJECTED; the pathway difference is intrinsic.
- **#3 multiplicativity, unsaturated (SGD).** ordinary(α) vs forward(α²) vs
  gradient(α²) (all m=α²): α=1.5→(.080/.065/.104), 2→(.123/.103/.168),
  3→(.172/.160/.209), 5→(.202/.217/.228). ordinary≈forward(α²); gradient(α²)
  slightly higher. m=α² approximately holds (saturated α=100 is uninformative).
- **#5 P3 extended ε.** gradient-only Adam: ε=1e-8→1e-2→1e-1→1 gives update ratio
  1.00→2.84→9.21→49.2 and dominance 0.003→−0.003→+0.232→+0.389. The cancellation
  breaks and the effect fully returns as ε grows — ε→0 is load-bearing.
- **#7 early stopping.** best-val epoch: baseline 13.1, forward 30.4 (of 40).
  probe_A shift forward−baseline: −0.034 (final), **−0.044±0.004 (best-val)**.
  The A degradation SURVIVES early stopping (not overtraining).
- **#8 redraw prototypes per seed.** Adam forward dom −0.056±0.010 / probe
  −0.036±0.008; gradient null; SGD forward dom +0.238 / probe −0.032. ≈ fixed-
  prototype results → robust to data variation (final tables valid).
