# Forward, Not Gradient (Under Adam): A Controlled Causal Decomposition of Training-Time Feature-Scale Effects on Modality Contribution

_Final hardened draft. Every number traces to `results/**` (EXPERIMENT_FREEZE_AUDIT.md, RESULTS_INDEX.md).
Statistical unit = seed on a fixed split unless stated; seeds are not independent populations. This is a
mechanism/analysis paper, not a new method and not a state-of-the-art claim._

## Abstract
**Problem.** Modality-specific feature scaling imposed during training can change how much a multimodal
model relies on each modality, but it is unclear *through which pathway* — the forward representation
magnitude a modality contributes to fusion, or the backward gradient magnitude delivered to its encoder —
and whether this depends on the optimizer. **Method.** We introduce a controlled autograd intervention
that, holding the learning objective fixed, separates a forward-only scaling (forward ×α, backward ×1)
from a gradient-only scaling (forward unchanged, backward ×α) and an ordinary scaling (both), each
validated by scalar and numeric sanity gates; we additionally measure the **parameter-update ratio**
alongside the gradient-norm ratio. **Synthetic.** In a controlled equal-information setting, static
feature scale is largely compensated by trainable encoders, and persistent modality collapse is associated
with the optimization/learning-rate axis rather than feature scale. When scale is imposed during training,
under the tested **Adam** configuration the contribution shift is reproduced by forward-only scaling
(Cohen's d≈−3.8), while gradient-only scaling is statistically inert (d≈−0.5, p≈0.30) **despite a ~100×
gradient-norm ratio**, because its parameter-update ratio remains ≈1 — consistent with (and citing)
Adam's known near-invariance to constant per-parameter gradient rescaling. Under the tested **SGD**
configuration the same gradient-only scaling instead acts as an effective per-modality learning rate
(parameter-update ratio ≈α) and **both** pathways contribute. **Real data.** On CREMA-D audio-video under
the same Adam and matched CNN architecture, forward-only scaling reproduces the ordinary effect (forward
vs baseline d≈+7.4, five matched seeds on the fixed actor-disjoint split) while gradient-only is
indistinguishable from baseline (d≈−0.08, p≈0.87) at a ~282× gradient-norm ratio (update ratio ≈1.6 ≈
baseline). **Robustness.** The gradient-inertness result holds across sum, concat, and learned gated
fusion; the forward effect is strong under sum/concat but **attenuated (not individually significant)**
under gated fusion. **Boundary.** On UTD-MHAD (depth-inertial), where depth is near chance, the effect is
utility-gated and asymmetric — a boundary case, not a symmetric replication. **Conclusion.** Training-time
feature scale is a **causal modulator of learned modality contribution**, operating primarily through the
forward pathway under adaptive optimization and through both pathways under SGD, with the parameter-update
scale as the relevant optimization quantity — **not** a universal sufficient cause of persistent modality
collapse, which is better attributed to the optimization/learning-rate channel in the tested settings.
Per-modality normalization removes the imposed-scale pathway but is a control, not a cure.

## 1. Introduction
Multimodal networks frequently learn unequal modality contributions, under-using one stream even when it
carries useful information. The literature explains this through optimization imbalance and gradient
modulation, through decision-layer bias, and through modality "ability" differences, and it proposes
methods that rebalance training.

A simpler, more basic question is left open: *what happens if we directly manipulate one modality's
feature scale during training?* Feature magnitude is trivial to change, and an early intuition — that
larger-magnitude features dominate and "silence" the other modality — is appealing. We show that this
broad intuition is wrong as stated: **static feature scaling alone is generally compensated** by trainable
encoders, so it does not by itself explain persistent modality collapse; in our synthetic study,
persistent collapse is associated with the optimization/learning-rate axis, not with feature scale.

The scientifically productive question is therefore not *whether scale matters* but *through which causal
pathway a training-time scale intervention acts*. Multiplying a representation by α during training changes
both the forward magnitude entering fusion and the backward gradient reaching the encoder; these are
confounded in the ordinary intervention. We disentangle them with a controlled autograd construction:
forward-only (forward ×α, backward ×1), gradient-only (forward unchanged, backward ×α), and ordinary
(both), holding the objective fixed.

Our findings are: **(Adam)** the contribution shift is forward-magnitude dominated and isolated gradient
scaling is inert despite very large gradient-norm ratios, because Adam largely normalizes constant
per-parameter gradient rescaling at the update level; **(SGD)** gradient scaling instead becomes an
effective per-modality learning rate and both pathways contribute — so the decomposition is
optimizer-dependent and the relevant quantity is the parameter-update scale, not the gradient norm;
**(real data)** CREMA-D confirms the forward-mediated pathway under Adam; **(fusion)** gradient-inertness
is robust across sum/concat/gated fusion while the forward effect is attenuated under gating; **(utility)**
across CREMA-D and UTD-MHAD the effect is utility-gated and asymmetric.

**Contributions.** (1) A validated, objective-preserving decomposition of a training-time feature-scale
intervention into forward-magnitude and backward-gradient pathways, with the parameter-update ratio as an
additional readout. (2) Evidence that the decomposition is optimizer-dependent. (3) Real-data confirmation
on CREMA-D and fusion-mechanism robustness. (4) A scale-channel vs optimization-channel account with an
explicit utility-gating boundary. We do not propose a new balancing method and do not claim feature scale
is a sufficient cause of collapse.

## 2. Related work
**Gradient-modulation methods.** OGM-GE modulates the dominant modality's updates via a softmax-discrepancy
coefficient; AGM, PMR, CGGM, MMCosine, GMML and related methods rebalance backward gradients. These are
*fixes*; we run a controlled *analysis* and do not propose a competing method. **DGL** (Wei et al., ICCV
2025) is the closest in name: it disentangles *which loss's gradient trains which module* (routing the
unimodal-loss gradient to the encoder and removing the multimodal-loss gradient there). **Our intervention
axis is different:** we hold the objective and routing fixed and instead scale the *forward representation
magnitude* and, separately, the *backward gradient magnitude*, comparing forward-only, gradient-only and
ordinary. We do not claim to be "first to disentangle gradients"; we claim a specific, different
intervention. **Decision-layer / distributional views** (Revisit Modality Imbalance at the Decision Layer)
attribute imbalance to data distribution; this is complementary to our training-time intervention.
**Optimizer theory.** Adam's coordinate-wise normalization makes it approximately invariant to constant
per-parameter gradient rescaling and akin to a soft SignSGD (Kunstner et al. 2024; Balles & Hennig 2018).
This is a *known optimizer property*; we cite it. Our contribution is not this property but its
consequence for a *multimodal feature-scale* intervention and the contrasting SGD behavior.

## 3. Research questions and hypotheses
**RQ.** How does a training-time modality-specific feature-scale intervention alter learned modality
contribution — forward pathway, gradient pathway, or both — and how does this depend on optimizer, fusion
mechanism, and modality utility? Hypotheses and their evidence are enumerated in CLAIM_EVIDENCE_MATRIX.md.
The broad "sufficient cause / cure / silences" hypothesis is treated as falsified by our own synthetic
results and reported as a negative result.

## 4. Method
**4.1 Setup.** Two-stream model: per-modality encoders → fusion → linear head. Synthetic: equal-information
generator (8 classes, 32+32 dims, prototype margin 1.5, σ=1), tiny MLP encoders. CREMA-D: matched CNN
encoders on cached log-mel/frames, actor-disjoint split (train 64 / val 9 / test 18 actors). UTD-MHAD:
depth-CNN + inertial-1D-CNN, subject-disjoint split (train {1,3,5} / val {7} / test {2,4,6,8}).
**4.2 Training-time scale.** A modality's fusion-boundary representation h is multiplied by α in-graph
every step; encoders remain trainable. **4.3 Forward-only.** y=α·h with ∂y/∂h=1 (a straight-through-style
autograd function). **4.4 Gradient-only.** y=h (forward bit-identical to baseline) with ∂y/∂h=α.
**4.5 Ordinary.** y=α·h with ∂y/∂h=α. **Baseline.** y=h. Each is validated by a scalar sanity gate
(forward/backward multipliers 1/1, α/1, 1/α, α/α) and a numeric gate (gradient-only forward activations
bit-identical to baseline). **4.6 Contribution metrics.** Masking-based unimodal accuracies; dominance =
Acc_A − Acc_B; multimodal gain = Acc_AB − max(Acc_A, Acc_B). We note masking-based unimodal accuracy is
degenerate (→ majority class) for CREMA-D video and UTD depth; we therefore read effects primarily off
dominance, gain, and the activation-norm ratio. **4.7 Gradient/update metrics.** Per-encoder gradient-norm
ratio and **parameter-update ratio** ‖Δθ_A‖/‖Δθ_B‖. **4.8 Compensation.** ρ_enc (pre-scale), ρ_fusion
(post-scale), compensation = 1 − ln(ρ_fusion)/ln(α). **4.9 Statistics.** 5 seeds (initialization +
minibatch order) on fixed splits; paired Cohen's d and paired t within a comparison; 95% CIs; seeds are
not independent populations, so p-values report within-split reproducibility and effect sizes lead. For
UTD-MHAD we additionally report per-subject consistency across the four fixed test subjects.

## 5. Controlled synthetic study
**5.1 Static scale and information invariance.** A modality's imposed scale is compensated by trainable
encoders by convergence; early activation-norm ratio tracks the imposed scale but is only a transient
diagnostic (its pre-registered use as a *predictor* of final dominance was unsupported). A linear probe
confirms that scaling is **information-preserving**: accuracy attainable from the scaled representation is
unchanged. This matters for interpreting the training-time results (§9.1).
**5.2 Scale × learning rate.** In a signed scale×LR grid, aligned scale+LR imbalance produces the largest
imbalance, opposed conditions cancel, and z-score removes the scale component but not LR sensitivity.
**5.3 Persistent-collapse regime.** At magnitudes up to 400×, scale-only conditions remain compensated
(collapse severity ≤0.08), while learning-rate-only and aligned conditions drive persistent collapse
(severity → 1.0, surviving 200 epochs). **In this synthetic setting, persistent collapse is associated
with the optimization/learning-rate channel, not with feature scale.** z-score removes the scale channel;
no tested method, including OGM-GE, corrected the induced LR-starvation collapse (§9.6).
**5.4 Adam disentanglement.** α=100, 5 seeds. Forward-only reproduces the ordinary contribution shift
(forward vs baseline d=−3.76; ordinary vs forward diff +0.0002, p=0.18), while **gradient-only is inert**
(d=−0.53, p=0.30) despite a ~100× gradient-norm ratio, because its **parameter-update ratio ≈1**. Under
the tested Adam configuration the effect is **forward-dominated**.
**5.5 SGD disentanglement (essential contrast).** Same design, SGD (lr 1e-3, momentum 0.9, no schedule,
no clipping), 40 epochs. **Both pathways contribute**: forward vs baseline d=3.93, gradient vs baseline
d=3.90, ordinary d=7.75, and ordinary differs from forward (d=2.86, p=0.003). The gradient-only
**parameter-update ratio ≈ α (≈96)**. This shows the gradient-norm ratio alone is not the correct
mediator; the **parameter-update scale** is the relevant optimization quantity, and whether isolated
gradient scaling matters is optimizer-dependent. Scope: this SGD configuration only, lr 1e-3, momentum
0.9, 40-epoch horizon; broader SGD hyperparameters remain untested.

## 6. Real-data study: CREMA-D (central)
**6.1–6.3 Dataset, protocol, natural imbalance.** Audio-video emotion; frozen actor-disjoint split;
natural feature-scale ratio ρ≈1.22; **both modalities carry useful information** (Δ_audio 0.32,
Δ_video 0.25 on top of the other). Per-modality z-score reduces ρ and rebalances encoder gradients
(2.74→0.14) but slightly reduces accuracy (−0.026, p=0.014).
**6.4 Training-time scale.** Amplifying audio (the more-informative modality) during training entrenches
its contribution and collapses multimodal gain (baseline dominance 0.075 → audio×10 0.328; gain 0.248 →
0.013; Spearman(imposed, dominance)=0.922, five seeds); amplifying video is largely compensated
(dominance 0.009; gain 0.295). Per-modality z-score suppresses the pathway (audio×10 dominance 0.328 →
0.110).
**6.5 Forward-vs-gradient decomposition.** Under the same Adam, α{10,100}, **five matched seeds on the
fixed actor-disjoint split**. Forward-only reproduces the entire ordinary effect (α=100: dominance 0.333
vs ordinary 0.340; multimodal gain both ≈0.006; forward vs baseline Δ=+0.259, **Cohen's d=+7.39, p=8×10⁻⁵**;
ordinary vs forward d=0.59, p=0.26). **Gradient-only is statistically indistinguishable from baseline**
(dominance 0.073 vs 0.075; Δ=−0.001, **d=−0.08, p=0.87**) despite imposing a **~282× gradient-norm ratio**;
its **parameter-update ratio is ≈1.61, ≈ baseline (1.65)**. Under the tested Adam configuration and matched
architecture, the real-data training-time scale effect is consistent with a **forward-magnitude-mediated
pathway**; this decomposes the H6 result (the ordinary arm reproduces H6's audio-entrenchment) into the
forward pathway. We do not present these as population-level estimates; they are within-split
reproducibility across five seeds.

## 7. Cross-domain study: UTD-MHAD (utility-gated boundary case)
Depth-inertial action recognition; subject-disjoint split. **Depth-only accuracy is near chance (0.036);
inertial-only is 0.227; multimodal 0.632; gain 0.405** — a strong utility asymmetry. Amplifying inertial
(informative) entrenches its contribution and collapses gain (baseline dominance −0.191 → inertial×10
−0.615; gain 0.405 → 0.053; Spearman 0.757); amplifying depth is largely compensated; z-score suppresses
the pathway. **This is a utility-gated / asymmetric boundary case, not a symmetric universal replication.**
Because depth is near chance, the masking dominance is dominated by the inertial-only term; we read the
effect off dominance/gain/ρ and report **per-subject consistency across all four fixed test subjects**
rather than population-level significance from seed variation.

## 8. Fusion-mechanism robustness
α=100, Adam, synthetic; sum, concat, and learned gated fusion (per-modality self-gates). **Gradient-only
remains inert under all three** (gradient-norm ratio ~90–100×, parameter-update ratio ≈1). The forward
effect is **strong under sum and concat** (forward vs baseline d=−3.76 and d=−5.16; ordinary ≡ forward)
but **attenuated and not individually significant under gated fusion** (forward vs baseline d=−0.70,
p=0.19), with a mild interaction (ordinary slightly exceeds forward), consistent with the learned gate
partially absorbing the imposed scale. We therefore claim **gradient-inertness robustness across fusion
mechanisms** and a **fusion-sensitive forward effect** — this is fusion-mechanism robustness, not
architecture generality.

## 9. Discussion
We organize the account around two channels and two moderators.
**9.1 The scale channel (forward magnitude).** Scaling is information-preserving (§5.1), yet training-time
scaling changes learned contribution. The reconciliation (Phase-3 reparameterization discussion): a
constant multiplicative rescale is invertible at inference and can be absorbed by a trained linear head —
which is exactly why *static* scaling is largely compensated. During training, however, the scaled
representation is what enters fusion and shapes the loss and the updates at every step; the encoder
adapts (ρ_enc moves opposite to the imposed scale), but the net learned contribution shifts. The forward
change is therefore not "mere reparameterization" in the training dynamics, even though it preserves the
information content. We do not claim to have mathematically excluded all reparameterization effects.
**9.2 The optimization channel (parameter-update imbalance).** Static scale is compensable; aligned
scale+LR imbalance produces persistent collapse; the persistent regime belongs to the optimization/LR
channel in our synthetic study.
**9.3 Optimizer moderation.** The gradient pathway of the intervention is realized only through the
*parameter update*. Under the tested constant per-modality gradient scaling, **Adam largely normalized the
induced gradient-magnitude change at the parameter-update level** (update ratio ≈1 despite ~100–282×
gradient ratios), which is consistent with — and we cite — Adam's known near-invariance to constant
per-parameter gradient rescaling; under SGD the same scaling produced an ≈α× update and an active pathway.
We flag as an open question (Limitations) whether the Adam inertness is specifically its moment
normalization or also the shared head absorbing the constant.
**9.4 Utility moderation.** Amplifying the more-informative modality entrenches its contribution and
reduces multimodal gain; amplifying a weak modality is largely compensated. Direction and magnitude are
utility-dependent (CREMA-D positive/entrenching, synthetic equal-information negative/over-compensating).
**9.5 Normalization.** Per-modality z-score removes the imposed feature-scale pathway in the tested
settings; it does not necessarily improve accuracy and can reduce it, and it alters optimization as well
as scale (it rebalances encoder gradients at the source). It is used here as a **mechanism control**, not
a universal remedy.
**9.6 Prior work.** Complementary to gradient-modulation fixes and to DGL (loss-routing); our OGM-GE result
is a scoped negative in an induced LR-starvation regime that differs from OGM-GE's target greedy-learning
dynamics and must not be generalized.

## 10. Limitations
(1) Mechanism decomposition on real data uses one optimizer (Adam). (2) CREMA-D uses one matched CNN
architecture. (3) Fusion-mechanism robustness is not architecture generality (no attention/transformer).
(4) SGD real-data disentanglement is untested. (5) Long-horizon persistence of the exact SGD
disentanglement is untested (E4-EXTREME shows LR-driven collapse persists to 200 ep). (6) UTD-MHAD depth
is near chance. (7) UTD test population is four fixed subjects; seed variation is not population variation.
(8) We did not fully isolate whether Adam's gradient inertness is moment normalization or shared-head
absorption. (9) Attention/transformer architectures untested. (10) Broader optimizer/LR ranges untested.
(11) Normalization changes optimization as well as scale, so its effect is not a pure scale intervention.
These bound the claims but are not fatal to them.

## 11. Conclusion
Training-time modality feature scale is a causal modulator of learned modality contribution, operating
primarily through the forward pathway under adaptive optimization and through both pathways under SGD, with
the parameter-update scale as the relevant optimization quantity, gated by modality utility, confirmed on
real audio-video data under Adam, and robust in its gradient-inertness across fusion mechanisms. It is not
a universal sufficient cause of persistent modality collapse, which in the tested settings is better
attributed to the optimization/learning-rate channel. Per-modality normalization removes the imposed-scale
pathway as a control, not a cure.
