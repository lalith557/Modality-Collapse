# CLAIM–EVIDENCE MATRIX

_Every claim intended for the manuscript, mapped to its evidence, scope, and limitation. Claims not in
this matrix must not appear in the paper._ Statistical unit = seed (fixed split) unless noted.

| # | Claim | Evidence | Experiment / dataset | Condition | Statistical support | Scope | Limitation | Paper location |
|---|---|---|---|---|---|---|---|---|
| C1 | Training-time modality feature-scale intervention causally modulates learned modality contribution | dose-response + arm interventions | H6 (CREMA-D), UTD, synthetic | α{5,10}/{10,100} | H6 Spearman(imposed,dom)=0.922; CREMA-D dis. ordinary d=8.5 | tested archs/optimizers | not "sufficient cause of collapse" | §6.4, §5.4 |
| C2 | Under Adam, the effect is carried by the FORWARD-magnitude pathway; isolated gradient scaling is inert | 4-arm autograd decomposition | synthetic Adam + CREMA-D | α=100 | synth: fwd d=−3.76, grad d=−0.53 (p=0.30); CREMA-D: fwd d=+7.39, grad d=−0.08 (p=0.87) despite 282× grad | Adam, these archs | Adam-specific; not universal adaptive | §5.4, §6.5 |
| C3 | The mediator is the PARAMETER-UPDATE ratio, not the gradient-norm ratio | update vs grad ratio per arm | synthetic Adam/SGD + CREMA-D | α=100 | grad-only: grad-ratio 100–282× but update-ratio ≈1 (Adam) / ≈α (SGD) | tested optimizers | update-norm measured per-encoder | §5.4–5.5, §6.5 |
| C4 | Under SGD, gradient scaling becomes an effective per-modality LR and BOTH pathways contribute | 4-arm under SGD | synthetic SGD | α=100 | fwd d=3.93, grad d=3.90, ordinary d=7.75; ordinary≠fwd (d=2.86) | SGD (lr1e-3,mom0.9) | synthetic only; 40-ep horizon | §5.5 |
| C5 | Static/frozen scale is compensable; persistent collapse is optimization/LR-driven, not scale-driven | severity dose-response | E2, E4-EXTREME, R3 | scale-only vs LR/aligned | E4-EXTREME: scale-only S≤0.08 @400×; LR/aligned S→1.0; R3 dominance flat | synthetic (+R3 frozen real) | synthetic persistence; R3 frozen-rep | §5.1–5.3, §9.2 |
| C6 | The effect is utility-gated: amplifying the more-informative modality entrenches it and reduces multimodal gain; amplifying a weak modality is largely compensated | directional asymmetry | H6, UTD, CREMA-D dis. | audio/inertial vs video/depth | H6 audio×10 gain 0.248→0.013 vs video×10 0.248→0.295; UTD inertial×10 gain 0.405→0.053 | 2 real datasets | UTD depth≈chance (not symmetric) | §6.4, §7 |
| C7 | Per-modality z-score removes the imposed-scale pathway (a scale-channel intervention) without improving accuracy | norm control | E3, R2, E5, H6, UTD | none vs z-score | H6 audio×10 dom 0.328→0.110; R2 acc −0.026 (p=0.014) | tested settings | not a "cure"; can hurt acc | §9.5 |
| C8 | The forward-dominated / gradient-inert (Adam) decomposition is robust across sum, concat, and gated fusion; gated fusion attenuates the forward effect | fusion sweep | fusion robustness | α=100 | grad inert all 3 (upd_r~1); sum fwd d−3.76, concat d−5.16; gated d−0.70 (p=0.19) | 3 fusion ops, Adam, synthetic | not attention/transformer | §8 |
| C9 | The training-time forward-magnitude mechanism observed synthetically holds on real audio-video data (CREMA-D) | real-data decomposition | CREMA-D disentangle | α{10,100} | CASE A; fwd d=7.39, grad null; ordinary≡fwd (d=0.59, p=0.26) | Adam, matched-CNN sum | Adam only on real data | §6.5 |
| C10 | Early activation-norm ratio ρ is a transient diagnostic, not a reliable predictor of final dominance | E1 | synthetic | early ρ | AUROC≈0.91@ep5; final-dominance prediction UNSUPPORTED | synthetic | confounded with noise magnitude | §5.1 (brief) |
| C11 | OGM-GE did not correct the imposed LR-starvation collapse in our regime (NOT a general OGM claim) | E5 | synthetic | 4 arms | OGM null (p≥0.29); active (k→0.57) | our induced regime only | ≠ OGM's greedy-dynamics target | §9.6 (scoped) |

## Claims explicitly EXCLUDED (do not appear anywhere)
"feature magnitude is a sufficient cause of modality collapse"; "feature scale silences a modality";
"normalization cures modality collapse"; "gradient imbalance is irrelevant"; "Adam is universally
invariant to gradient scaling"; "the mechanism is universal"; "generalizes to all multimodal
architectures"; "UTD-MHAD proves universal replication"; any "first to" without literature backing;
any "state-of-the-art / best method" (this is an analysis paper, not a method paper).
