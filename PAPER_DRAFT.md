# Forward, Not Gradient (Usually): A Controlled Causal Decomposition of Training-Time Feature-Scale Effects on Modality Contribution

_Draft. All numbers trace to `results/**` (see EXPERIMENT_FREEZE_AUDIT.md, RESULTS_INDEX.md,
CLAIM_EVIDENCE_MATRIX.md). Statistical unit = seed on a fixed split unless stated. This is an
analysis paper (mechanism), not a new method / SOTA claim._

## Abstract
Modality imbalance in multimodal learning is usually addressed by gradient-modulation methods that
rebalance training. We instead ask a narrow causal question: if we directly manipulate a modality's
**feature scale during training**, does the learned modality contribution change, and through which
pathway? We introduce a controlled autograd decomposition that separates the **forward representation
magnitude** a modality contributes to fusion from the **backward gradient magnitude** delivered to its
encoder, and we measure the **parameter-update ratio** in addition to the gradient-norm ratio. In a
controlled synthetic setting we find that static feature scaling is largely compensated by trainable
encoders, and that persistent modality collapse is associated with the optimization/learning-rate axis
rather than with feature scale per se. When scale is imposed during training, it does modulate learned
contribution — and under Adam this modulation is carried almost entirely by the **forward** pathway:
isolated gradient scaling is statistically inert despite a 100–282× gradient-norm ratio, because its
parameter-update ratio remains ≈1 (consistent with Adam's known near-invariance to constant per-parameter
gradient rescaling). Under SGD the same isolated gradient scaling instead acts as an effective per-modality
learning rate (parameter-update ratio ≈ α) and both pathways contribute. On real audio-video data
(CREMA-D), the forward-dominated decomposition replicates under Adam (forward-only effect Cohen's d≈7.4;
gradient-only indistinguishable from baseline at a 282× gradient ratio), and it is robust across sum,
concat, and gated fusion (with gated fusion attenuating the forward effect). Across CREMA-D (audio-video)
and UTD-MHAD (depth-inertial) the effect is **utility-gated**: amplifying the more-informative modality
entrenches its contribution and reduces multimodal gain, while amplifying a weak modality is largely
compensated. Per-modality normalization removes the imposed-scale pathway but does not improve — and can
reduce — accuracy. We conclude that training-time feature scale is a **causal modulator of modality
contribution operating primarily through the forward pathway under adaptive optimization**, not a
universal sufficient cause of persistent modality collapse, and we separate a **scale channel** from an
**optimization channel** with the parameter-update ratio as the mediator.

## 1. Introduction
Multimodal networks often under-use one modality. Prior work frames this as an optimization problem
(gradient modulation), a representation-quality problem, or a decision-layer bias. We take a deliberately
narrow, intervention-first view: we *manipulate* a modality's feature scale during training and ask,
causally, whether and how learned contribution changes. Our contributions are (i) a validated autograd
decomposition isolating the forward-magnitude pathway from the encoder-gradient pathway of a feature-scale
intervention; (ii) evidence that the decomposition is optimizer-dependent, with the parameter-update ratio
as the mediator; (iii) real-data confirmation on CREMA-D and fusion-mechanism robustness; and (iv) a
scale-channel vs optimization-channel account with an explicit utility-gating boundary. We explicitly do
**not** claim feature scale is a sufficient cause of persistent collapse; our own synthetic experiments
falsify that broad version.

## 2. Related work
**Gradient-modulation fixes.** OGM-GE [Peng 2022] slows the dominant modality's updates via a softmax
discrepancy coefficient; AGM, PMR, CGGM, MMCosine and GMML follow related re-weighting ideas. DGL
[Wei, ICCV 2025] disentangles *which loss's gradient* trains the encoder vs the fusion module (gradient
routing). **Decision-layer / distributional views.** "Revisit Modality Imbalance at the Decision Layer"
[2510.14411] attributes imbalance to data-distribution rather than optimization. **Classification-ability
disproportion / multi-objective** methods [2502.20120; 2511.06686] rebalance by classification disparity.
**Optimizer theory.** Adam's coordinate-wise normalization makes it behave like a soft SignSGD and
approximately invariant to constant per-parameter gradient rescaling [Kunstner 2024; Balles 2018;
gradient-heterogeneity 2025]. **How we differ.** Prior multimodal work proposes *fixes* and modulates
*backward* gradients; the optimizer literature does not connect to multimodal contribution. We instead run
a *controlled forward-magnitude-vs-backward-magnitude decomposition of a feature-scale intervention*, show
its *optimizer dependence* via the *parameter-update ratio*, confirm it on *real data*, and test *fusion
robustness*. We cite (not claim) the Adam-invariance fact; our novelty is the multimodal decomposition and
its boundary conditions.

## 3. Research questions and hypotheses
RQ: How does a training-time modality-specific feature-scale intervention alter learned modality
contribution — through the forward-magnitude pathway, the encoder-gradient pathway, or both — and how does
this depend on optimizer, fusion mechanism, and modality utility? Hypotheses H1–H5 as in
CLAIM_EVIDENCE_MATRIX (C1–C9); the broad "sufficient cause / cure / silences" hypothesis is treated as
falsified and reported as such.

## 4. Method
**4.1 Setup.** Two-stream model: per-modality encoders → fusion (sum/concat/gated) → linear head.
Synthetic: equal-information generator (8 classes, 32+32 dims, margin 1.5, σ=1), tiny MLP encoders.
CREMA-D: matched CNN encoders on cached log-mel/frames, actor-disjoint split (64/9/18 actors). UTD-MHAD:
depth-CNN + inertial-1D-CNN, subject-disjoint split (train {1,3,5}/val {7}/test {2,4,6,8}).
**4.2 Training-time scale.** A modality's fusion-boundary representation h is multiplied by α in-graph
during every step; encoders remain trainable. **4.3–4.5 Interventions.** ForwardScale: y=α·h, ∂y/∂h=1
(forward magnitude only). GradScale: y=h, ∂y/∂h=α (encoder gradient only; forward bit-identical to
baseline). Ordinary: y=α·h (both). Baseline: y=h. Each is validated by a scalar autograd sanity gate
(forward/backward multipliers) and a numeric gate (gradient-only forward bit-identical to baseline).
**4.6 Contribution metrics.** Masking-based unimodal accuracies; dominance = Acc_A − Acc_B; multimodal
gain = Acc_AB − max(Acc_A, Acc_B). **4.7 Gradient/update metrics.** Per-encoder gradient-norm ratio
‖g_A‖/‖g_B‖ and **parameter-update ratio** ‖Δθ_A‖/‖Δθ_B‖. **4.8 Compensation.** ρ_enc = ‖h_A‖/‖h_B‖
(pre-scale); ρ_fusion (post-scale); compensation = 1 − ln(ρ_fusion)/ln(α). **4.9 Statistics.** 5 seeds
(init + minibatch order) on fixed splits; paired Cohen's d and paired t within a comparison; 95% CIs;
seeds are **not** independent populations (UTD reports per-subject consistency). Effect sizes lead;
p-values report within-split reproducibility.

## 5. Controlled synthetic study
**5.1 Static scale.** Scale imposed on the data/representation is compensated by trainable encoders by
convergence; early ρ tracks the imposed scale but is a *transient* diagnostic (E1: final-dominance
prediction unsupported); a linear probe confirms information invariance (E2).
**5.2 Scale × LR.** In a signed scale×LR grid, aligned scale+LR imbalance produces the largest imbalance;
opposed cancels; z-score removes the scale component but not LR sensitivity (E4).
**5.3 Persistent-collapse regime.** At extreme magnitudes (≤400×) scale-only remains compensated
(severity ≤0.08), while LR-only and aligned conditions drive persistent collapse (severity → 1.0,
surviving 200 epochs). **Persistent collapse belongs to the optimization/LR channel, not to feature scale**
(E4-EXTREME). z-score removes the scale channel; no tested method (incl. OGM-GE, E5) corrects the LR
channel in this induced regime.
**5.4 Adam disentanglement.** α=100, 5 seeds. Forward-only reproduces the ordinary contribution shift
(both ≈ −0.07 dominance; forward vs baseline d=−3.76, ordinary vs forward p=0.18); **gradient-only is
inert** (d=−0.53, p=0.30) despite a ~100× gradient-norm ratio, because its **parameter-update ratio ≈1**.
**Forward-dominated (CASE A).**
**5.5 SGD disentanglement.** Same design, SGD (lr 1e-3, momentum 0.9). **Both pathways contribute**:
forward vs baseline d=3.93, gradient vs baseline d=3.90, ordinary d=7.75, ordinary ≠ forward (d=2.86);
gradient-only **parameter-update ratio ≈ α (≈96)**. The optimizer changes the decomposition; the mediator
is the parameter-update scale, not the gradient norm (**CASE C, mixed**).

## 6. Real-data study: CREMA-D
**6.1–6.3.** Audio-video emotion; actor-disjoint split; natural feature-scale ratio ρ≈1.22, both
modalities informative (Δ_audio 0.32, Δ_video 0.25); z-score reduces ρ and rebalances encoder gradients
(2.74→0.14) but slightly reduces accuracy (−0.026, p=0.014). **6.4 Training-time scale.** Amplifying audio
(the more-informative modality) during training entrenches its contribution and collapses multimodal gain
(baseline dom 0.075 → audio×10 0.328; gain 0.248 → 0.013; Spearman(imposed, dominance)=0.922); amplifying
video is largely compensated (dom 0.009; gain 0.295). z-score suppresses the pathway (audio×10 dominance
0.328 → 0.110). **6.5 Forward-vs-gradient decomposition (central result).** Under the same Adam, α{10,100}:
forward-only reproduces the entire ordinary (H6) effect (α=100: dominance 0.333 vs ordinary 0.340; gain
both ≈0.006; forward vs baseline **d=7.39, p=8×10⁻⁵**; ordinary vs forward d=0.59, p=0.26), while
**gradient-only is indistinguishable from baseline** (dominance 0.073 vs 0.075; **d=−0.08, p=0.87**) despite
a **282× gradient-norm ratio** — its **parameter-update ratio is 1.61 ≈ baseline 1.65**. The real-data
mechanism is forward-magnitude-mediated under Adam (**CASE A**).

## 7. Cross-domain study: UTD-MHAD (boundary condition)
Depth-inertial action recognition; subject-disjoint split. Depth-only ≈ chance (0.036), inertial-only
0.227, multimodal 0.632, gain 0.405 — a strong **utility asymmetry**. Amplifying inertial (informative)
entrenches it and collapses gain (baseline dominance −0.191 → inertial×10 −0.615; gain 0.405 → 0.053;
Spearman 0.757); amplifying depth is largely compensated; z-score suppresses the pathway. This is a
**utility-gated / asymmetric boundary case**, not a symmetric replication: with depth ≈ chance the masking
dominance is dominated by the inertial-only term, so we read the effect off dominance/gain and ρ, and we
report per-subject consistency (all four test subjects) rather than population significance.

## 8. Fusion-mechanism robustness
α=100, Adam, synthetic. **Gradient-only is inert under all three fusion mechanisms** (gradient-norm ratio
~90–100×, parameter-update ratio ≈1). Sum and concat show clean forward-dominance (forward vs baseline
d=−3.76 and d=−5.16; ordinary ≡ forward). **Gated fusion attenuates** the forward effect (forward vs
baseline d=−0.70, p=0.19; not individually significant) and shows a mild interaction (ordinary slightly
exceeds forward), consistent with the learned gate partially absorbing the imposed scale. We call this
**fusion-mechanism robustness**, not architecture generality.

## 9. Discussion
**9.1 What feature scale does.** Applied during training, it is a causal modulator of learned modality
contribution, gated by modality utility, operating (under Adam) through the forward pathway.
**9.2 What it does not do.** It is compensable when static, and it is not a sufficient cause of persistent
collapse — which belongs to the optimization/LR channel in our synthetic study.
**9.3 Optimizer dependence.** Adam's near-invariance to constant per-parameter gradient rescaling makes the
gradient pathway inert; SGD converts gradient scaling into an effective per-modality LR. The
parameter-update ratio, not the gradient-norm ratio, mediates.
**9.4 Utility dependence.** Amplifying the useful modality entrenches it; amplifying a weak one is
compensated. Direction and magnitude are utility-dependent.
**9.5 Normalization.** Per-modality z-score removes the imposed-scale pathway (a scale-channel control) but
does not improve — and can reduce — accuracy; it is not a cure.
**9.6 Relation to prior work.** Complementary to gradient-modulation fixes (we analyze, not fix) and to
DGL (loss-routing vs forward/backward-magnitude); consistent with, and citing, the Adam-invariance theory;
our OGM-GE result is a scoped negative in an induced LR-starvation regime, not a general claim.

## 10. Limitations
Adam-only decomposition on real data (SGD real-data untested); synthetic equal-information for the
mechanism isolation; tiny matched encoders and sum/concat/gated fusion (no attention/transformer); UTD
depth is near chance (utility-gated boundary, not symmetric); persistence beyond the tested horizons
(≤40 ep for disentanglement; ≤200 ep for E4-EXTREME) untested; statistical unit is the seed on fixed
splits (18 CREMA-D actors, 4 UTD subjects), so seed p-values report reproducibility, not population
significance; we did not isolate whether Adam-inertness is specifically its moment normalization versus
the shared-head absorbing the constant.

## 11. Conclusion
Training-time modality feature scale is a causal modulator of learned modality contribution, operating
primarily through the forward pathway under adaptive optimization and through both pathways under SGD, with
the parameter-update ratio as the mediator, gated by modality utility, confirmed on real audio-video data
and robust across fusion mechanisms — but it is not a universal sufficient cause of persistent modality
collapse, which is better attributed to the optimization/learning-rate channel in the tested settings.
