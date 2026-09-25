# FINAL CLAIM–EVIDENCE MATRIX

_Every major claim → evidence, with its scope and final (post-hardening) wording. Numbers verified
against `results/**` (see `EXPERIMENT_FREEZE_AUDIT.md`)._

| Claim | Evidence / Experiment | Metric | Strength | Scope | Reviewer objection (pre-empted) | Final wording |
|---|---|---|---|---|---|---|
| Constant scale is compensated | Synthetic E2/E4-EXTREME | dominance, ρ_enc, S_loss | Strong | evaluated synthetic regime | "does nothing" overclaim | "substantially compensated, leaving only a small residual contribution change" |
| Training-time scaling changes contribution | Synthetic + CREMA-D H6 | dominance, Spearman 0.922 | Strong | intervention-level | universal-cause overclaim | "produces substantial changes in contribution" |
| Adam: forward carries the effect | Synthetic + CREMA-D disentangle | Δd, d_z, p | Strong | tested Adam config | "Adam invariant to gradients" | "under the tested Adam configuration, forward scaling reproduces the effect" |
| Adam: gradient-only little change | Synthetic d=−0.53 p=0.30; CREMA d=−0.08 p=0.87 | Δd, r_g, r_Δθ | Strong (null) | tested Adam config | "no effect" (proof of null) | "produced no detectable shift in this five-seed experiment"; r_Δθ≈1 |
| SGD: gradient transmitted to updates | SGD disentangle | r_Δθ=95.9 @α=100 | Strong | tested SGD config | "update ratio = α" | "≈96× update ratio … behaving like a substantial effective per-modality learning-rate difference" |
| Both SGD pathways contribute | SGD disentangle | Δd (fwd/grad/ord) | Strong | tested SGD config | "additive"/"formal interaction" | "the two effects do not combine additively here" |
| Update ratio, not gradient norm, tracks effect | Adam vs SGD contrast | r_g vs r_Δθ | Moderate (interpretation) | tested optimizers | "formal mediation" | "optimization-level readout … not a formal mediation analysis" |
| Adam attenuation mechanism | Update-ratio observation | r_Δθ | Weak (open) | — | over-attribution to moment norm | "does not identify whether … moment normalization, shared-head interactions, or both" |
| CREMA-D real-data support | CREMA-D disentangle, 40 runs | Δd=+0.259, d=7.39 | Strong | Adam, one CNN, sum fusion | generalization | "consistent with a forward-magnitude-carried effect on real data under Adam" |
| Fusion dependence | Fusion robustness, 60 runs | Cohen's d per fusion | Moderate | 3 evaluated mechanisms | "significant interaction test" | "the observed forward effect varied across the evaluated fusion mechanisms" |
| Modality-utility boundary | UTD-MHAD | dominance, gain | Moderate | one utility-asymmetric dataset | universal utility law | "in this utility-asymmetric setting the response … differed sharply" |
| Normalization | H6 z-score; E3 | dominance, ρ_fus, acc | Moderate | evaluated settings | "cure" | "largely suppresses … substantially reduces … not a general cure" |
| Persistent loss = optimization channel | E4-EXTREME | S_loss | Moderate | synthetic scale×LR sweep | "collapse" universal | "persistent contribution loss … associated more with the optimization/LR channel" |
| OGM-GE null | E5 | recovery frac | Weak (scoped) | LR-starvation regime | "OGM-GE ineffective" | "little effect only in the evaluated learning-rate-starvation configuration" |

**Not claimed anywhere** (verified by global language audit): feature scale a sufficient cause of
collapse; universal collapse; gradient imbalance universally irrelevant; Adam universally invariant;
normalization a universal cure; universal fusion moderation; utility law from one dataset; a complete
causal DAG; masked accuracy as complete causal attribution; universal generalization; "first"/SOTA.
