# Proposition 1 — derivation (Phase 2 working note)

Working derivation for the effective function-space learning-rate multipliers.
Not a paper edit (deferred to Phase 5). muP statements are marked **VERIFY**.

## Setup

Sum fusion, logits `z = W(h̃_A + h_B) ∈ R^C`; `W ∈ R^{C×d}` the head, held
fixed over one step and shared across modalities. `h̃_A` is A's rep at the
fusion input (post-operator), `h_B` is B's rep. Loss `L = CE(z,y)`,
`g := ∂L/∂z`. Encoder-A params `θ_A`, Jacobian `J_A := ∂h_A/∂θ_A`.

Each arm applies a **forward gain** `s_f` to the fusion input and a **gradient
gain** `s_g` to the encoder gradient:

| arm | operator | s_f | s_g |
|---|---|---|---|
| baseline | identity | 1 | 1 |
| forward | FWD (fwd ×α, bwd ×1) | α | 1 |
| gradient | GRAD (fwd ×1, bwd ×α) | 1 | α |
| ordinary | ×α both | α | α |

## Derivation

Encoder-A gradient (chain rule; `∂L/∂h̃_A = W^⊤ g`):
```
g_{θ_A} = ∂L/∂θ_A = s_g · J_A^⊤ W^⊤ g.
```
Parameter update:
```
SGD :  Δθ_A = −η s_g J_A^⊤ W^⊤ g.
Adam:  Δθ_A ≈ −η · normalize(g_{θ_A}) = −η · normalize(J_A^⊤ W^⊤ g)   (ε→0),
```
because per-coordinate normalization cancels the constant factor `s_g`.

A's logit contribution is `W h̃_A = s_f W h_A`, so to first order (local
linearity `h_A(θ+Δθ) ≈ h_A + J_A Δθ`):
```
Δz_A ≈ s_f · W J_A · Δθ_A.
```
Substituting the two updates gives the **effective function-space multiplier**
`m` (factor on A's per-step logit-contribution change vs baseline):
```
m_SGD  = s_f · s_g
m_Adam = s_f            (s_g cancels under Adam's normalization)
```

| arm | m_SGD | m_Adam |
|---|---|---|
| baseline | 1 | 1 |
| forward | α | α |
| gradient | α | 1 |
| ordinary | α² | α |

**Assumptions (state in the paper as a proposition):** (A1) sum fusion, shared
head fixed over the step; (A2) local linearity of `h_A` in `θ_A`; (A3) Adam in
the ε→0 steady-second-moment regime where normalization exactly cancels a
constant `s_g`; (A4) B held fixed when attributing `Δz_A`. First-order sketch,
not a proof.

## Predictions (each falsifiable; Phase 2 tests them)

- **P1 (Adam):** gradient-only `m=1` (null); ordinary = forward (`m=α`). *[Phase 1: SUPPORTED.]*
- **P2 (SGD):** ordinary `m=α²` vs isolated `α` → multiplicative, sub-additive under saturation. *[matches 0.322<0.244+0.200.]*
- **P-collapse:** dominance shift vs predicted `m` collapses onto one curve across arms/optimizers. *[α-sweep, this phase.]*
- **P-lr (the key mechanism test):** under Adam, `m_forward = s_f = α`. The same
  `m` is obtained with **no scaling** but encoder-A LR `η_A = α η` (Adam step
  ∝ η). So the **LR-matched control should reproduce forward-only.** If it does →
  the forward effect is an effective-LR effect (H-lr), not forward-magnitude /
  compensation per se. *[phase2 `lrmatch` arm.]*
- **P3 (large ε):** once `ε ≳ RMS(g_{θ_A})`, Adam's cancellation fails and the
  gradient-only effect returns. *[ε-sweep, this phase.]*
- **P4 (SGD cancellation):** gradient-only with `η_A = η/α` gives
  `m_SGD = s_f·s_g·(1/α) = (1)(α)(1/α) = 1` → reproduces baseline. *[this phase.]*

## What is NEW vs muP / Tensor Programs V (Yang et al. 2022, arXiv:2203.03466)

Checked against the paper text (2026-11-09). VERIFIED / NOT-FOUND tags reflect
what was read directly; items marked (summary) were not checked against the
full text.

**VERIFIED about muP — do NOT claim these as new:**
- muP is defined as a **width→∞ limit**: "a μP network has a well-defined
  infinite-width limit"; "each layer is updated on the same order during
  training regardless of width" (Intro, §2). Goal = stable Θ(1) feature learning
  + HP transfer across width (§2, Fig 3).
- muP **covers Adam as well as SGD**, with separate per-layer LR scalings:
  "While [57] only covered SGD, here we derive μP for Adam as well (Table 3)";
  Appendix J (summary) rescales LR by 1/θ² (SGD) vs 1/θ (Adam). So the
  per-layer-LR-governs-feature-learning idea, **including the Adam case**, is
  muP's, not ours. We cite muP for it and do not claim it.

**Distinctions that SURVIVE (our genuine contribution):**
1. **FWD/GRAD operator decoupling.** muP sets a parameter multiplier and a
   per-layer LR *together* as one parametrization; independently scaling the
   forward activation (`s_f`) vs the encoder gradient (`s_g`) is **NOT FOUND**
   in the muP text. Our FWD/GRAD split is the instrument that isolates the two,
   and it is what lets us *measure* `m_Adam=s_f` (gradient-only null) vs
   `m_SGD=s_f s_g` empirically rather than by width-limit derivation.
   **Prior art (cite, do not over-claim):** the GRAD operator (identity forward,
   gradient ×α) is the Gradient Reversal Layer (Ganin & Lempitsky 2015)
   generalized from negation to scaling; the FWD operator (forward ×α, identity
   gradient) is a less-standard counterpart. The autograd trick is NOT new.
   Claim only — **"to our knowledge"** — the *paired* use of FWD and GRAD to
   decouple forward- vs gradient-magnitude effects on **modality contribution**.
2. **Deliberate finite-width, per-modality imbalance.** muP's purpose is to make
   all layers learn at the *same* Θ(1) rate; deliberately scaling one
   branch/modality to create imbalance is **NOT FOUND**. We study exactly the
   Θ(α)/Θ(α²) imbalance regime muP is designed to avoid, at fixed width.
3. **Application to modality contribution.** Connecting the effective-LR
   multiplier to measured modality dominance / Shapley / probe is new.

**Honest framing for the paper:** muP already establishes that per-layer
effective learning rate (SGD and Adam) governs feature-learning rate. Our
contribution is the *operator decomposition* that isolates forward vs gradient
scaling, its use as a controlled per-modality imbalance probe, and the empirical
finding that raising a modality's effective LR *degrades* it (Phase 1). Frame the
theory as "muP-consistent," citing muP, not as a new learning-rate theory.
