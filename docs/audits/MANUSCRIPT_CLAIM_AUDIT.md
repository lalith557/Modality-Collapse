# MANUSCRIPT CLAIM AUDIT (sentence-level)

_Audit of PAPER_DRAFT.md against machine-readable results. Classification: **A** directly supported ·
**B** supported with scope qualification · **C** interpretation/inference · **D** unsupported · **E**
contradicted. Required revisions folded into PAPER_DRAFT_FINAL.md._

| # | Claim (paraphrased) | Section | Evidence / experiment / dataset | Scope | Class | Required revision |
|---|---|---|---|---|---|---|
| 1 | Static feature scaling is largely compensated by trainable encoders | §5.1 | E2 (compensated by ep40); E4-EXTREME (S≤0.08 @400×) | synthetic | **A** | keep |
| 2 | Persistent collapse is associated with the optimization/LR channel, not feature scale | §5.3, §9.2 | E4-EXTREME (LR/aligned S→1.0; scale-only S≤0.08) | synthetic | **B** | add "in the tested synthetic regime" |
| 3 | Training-time scale causally modulates learned contribution | §5.4,§6.4 | H6 Spearman 0.922; CREMA-D dis. d=8.5 | Adam, tested archs | **B** | "causal modulator (intervention-level)"; not "cause of collapse" |
| 4 | Under Adam the effect is forward-magnitude dominated; gradient-only inert | §5.4,§6.5 | synth d(grad)=−0.53 p0.30; CREMA-D d(grad)=−0.08 p0.87 @282× | Adam | **B** | "under the tested Adam configuration" |
| 5 | Gradient-only inert because parameter-update ratio ≈1 (Adam normalizes constant grad rescaling) | §5.4,§9.3 | update ratio ≈1 (synth), 1.61≈base (CREMA-D); Adam≈SignSGD theory | Adam, constant scaling | **B/C** | reword per Phase 4; cite theory; "largely normalized at the update level" |
| 6 | Under SGD both pathways contribute; gradient→update≈α | §5.5 | fwd d3.93, grad d3.90, ordinary d7.75; update≈96 | SGD lr1e-3 mom0.9 | **B** | add "tested SGD configuration; 40-ep horizon" |
| 7 | Parameter-update ratio (not gradient-norm) is the mediator | §5.5,§9.3 | grad-ratio 100–282× vs update ~1(Adam)/~α(SGD) | tested optimizers | **C** | "the relevant optimization quantity is the parameter-update scale" (interpretation) |
| 8 | CREMA-D: both modalities informative | §6.2 | R1 Δ_audio 0.32, Δ_video 0.25 | CREMA-D | **A** | keep; note video-alone masking degenerate |
| 9 | CREMA-D forward-only reproduces the ordinary/H6 shift; gradient-only does not | §6.5 | fwd d7.39; grad d−0.08; ordinary≡fwd (d0.59,p0.26) | Adam, matched-CNN | **A** (effect) / **B** (scope) | "five matched seeds on the fixed actor-disjoint split" |
| 10 | Amplifying the more-informative modality entrenches it and reduces gain | §6.4,§7 | H6 audio×10 gain→0.013; UTD inertial×10 gain→0.053 | 2 real datasets | **B** | "utility-gated"; UTD asymmetric |
| 11 | UTD is a utility-gated boundary case, not symmetric replication | §7 | depth≈chance 0.036; per-subject consistency | UTD | **A** | keep; emphasize 4 fixed subjects |
| 12 | Gradient-inertness robust across sum/concat/gated; gated attenuates forward effect | §8 | grad inert all 3; gated fwd d−0.70 p0.19 | Adam, synthetic, 3 fusions | **A** (grad) / **B** (gated fwd) | do NOT say "forward significant across all fusions" |
| 13 | z-score removes the imposed-scale pathway; not a cure; can hurt accuracy | §9.5 | H6 0.328→0.110; R2 acc −0.026 p0.014 | tested settings | **A** | keep; "scale-channel control" |
| 14 | OGM-GE null should not be generalized | §9.6 | E5 null in induced LR-starvation regime | our regime | **A** | keep scoped |
| 15 | Feature scale = causal modulator, not sufficient cause of persistent collapse | §11 | synthesis of above | tested settings | **B** | keep; primary framing |
| 16 | Information is invariant under scaling (reparam objection) | §5.1 | E2 linear probe | synthetic | **A** | expand into a dedicated reparameterization discussion (Phase 3) |
| 17 | "Adam's near-invariance … makes the gradient pathway inert" | §9.3 | interpretation + theory | Adam | **C** | soften to "consistent with"; cite; mark as candidate mechanism not proven |
| 18 | DGL differs (routing vs magnitude) | §2 | lit audit | — | **A** | keep; sharpen per Phase 5 |
| 19 | Effect persists (implicit) | §6/§8 | 30–40 ep; E4-EXTREME 200 ep | tested horizons | **B** | add explicit horizon caveat |
| 20 | Whether Adam-inertness is moment-norm vs shared-head absorption | (not claimed) | — | — | **D→disclose** | add to Limitations as open question (do not claim) |

**Net:** no **E** (contradicted) and no **D** (unsupported) claims remain in the draft; the majority are
**A** or scope-qualified **B**; three are explicit **C** (interpretation) and are reworded to signal that.
The hardening in PAPER_DRAFT_FINAL.md (a) scopes every optimizer/architecture claim, (b) adds the
reparameterization discussion, (c) reframes the Adam-invariance statement as cited theory + our empirical
observation, (d) states the gated-fusion forward effect as non-significant, (e) adds the moment-norm vs
head-absorption open question to Limitations.
