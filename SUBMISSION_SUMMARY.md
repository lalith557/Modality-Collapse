# SUBMISSION SUMMARY

**Title.** Forward, Not Gradient (Under Adam): A Controlled Causal Decomposition of Training-Time
Feature-Scale Effects on Modality Contribution.

**Research question.** How does a training-time modality-specific feature-scale intervention alter learned
modality contribution — through the forward-representation-magnitude pathway, the encoder-gradient pathway,
or both — and how does this depend on optimizer, fusion mechanism, and modality utility?

**Contributions.**
1. A validated, objective-preserving autograd decomposition of a training-time feature-scale intervention
   into a forward-magnitude pathway and a backward-gradient pathway, with the parameter-update ratio as an
   additional readout.
2. Evidence that the decomposition is optimizer-dependent: forward-dominated under Adam (gradient-only
   inert at ~100–282× gradient ratios because its update ratio ≈1), mixed under SGD (gradient scaling → ≈α
   update, an effective per-modality learning rate).
3. Real-data confirmation on CREMA-D (forward-only reproduces the effect, d≈7.4; gradient-only null) and
   fusion-mechanism robustness (gradient-inertness across sum/concat/gated; forward effect attenuated under
   gated fusion).
4. A scale-channel vs optimization-channel account with an explicit utility-gating boundary (CREMA-D vs
   UTD-MHAD), and normalization characterized as a scale-channel control rather than a cure.

**Main findings.** Static scale is compensable; training-time scale is a causal modulator of contribution;
under Adam the effect is forward-mediated and isolated gradient scaling is inert; under SGD both pathways
act; CREMA-D confirms the forward pathway under Adam; gradient-inertness is fusion-robust; the effect is
utility-gated; persistent collapse is optimization/LR-driven in the synthetic regime.

**Datasets.** Synthetic equal-information generator; CREMA-D (audio-video, actor-disjoint 64/9/18);
UTD-MHAD (depth-inertial, subject-disjoint {1,3,5}/{7}/{2,4,6,8}).

**Experimental scope.** 5 seeds throughout; 0 divergent runs. Adam (all real-data mechanism) + SGD
(synthetic); sum/concat/gated fusion; matched lightweight encoders; α up to 100 (synthetic to 400×).

**Main limitations.** Real-data decomposition Adam-only; one CNN architecture on CREMA-D; fusion-robustness
≠ architecture generality; UTD depth near chance (4 fixed test subjects); SGD real-data and long-horizon
disentanglement untested; Adam moment-norm vs shared-head absorption not disambiguated; no
attention/transformer.

**Novelty positioning.** An analysis/mechanism paper, not a balancing method. Distinct from gradient-
modulation fixes (OGM-GE/AGM/PMR/GMML) and from DGL (which routes *which loss* trains the encoder; we scale
*forward vs backward magnitude* with the objective fixed). The Adam-scale-invariance property is cited
(Kunstner 2024; SignSGD), not claimed as our contribution.

**Not claimed.** feature scale as sufficient cause of collapse; normalization as cure; universal
architecture/dataset generality; UTD symmetric replication; general OGM-GE ineffectiveness; SOTA/best.
