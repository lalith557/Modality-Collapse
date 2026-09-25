# FINAL CORRECTION AUDIT (main(6) pass)

_Issues 1–25 were already resolved in the prior hardening round (see `FINAL_HARDENING_AUDIT.md` Round 3)
and were re-verified here; the Additional Fixes A–I are new to this pass. No experimental number was
changed; where the prompt's approximate value disagreed with the source artifact, the artifact value was
kept and the discrepancy flagged (see NUMBER-INTEGRITY note)._

| # | Location | Original problem | Fix applied | Verified? |
|---|---|---|---|---|
| 1 | §III-A | "estimate the effect of scale" | "quantify the response to the imposed scale intervention…" | RESOLVED |
| 2 | H1/§IV-B/concl. | constant scale "does nothing" | "substantially compensated … only a small residual contribution change" | RESOLVED |
| 3 | §IV-C | "collapse severity" | `S_loss` (study-specific), "not … a universal definition of modality collapse" | RESOLVED |
| 4 | Fig. 1 node | "normalization removes scale channel" | "normalization suppresses scale difference" | RESOLVED |
| 5 | §VI-C | "undoes the induced shift" | "substantially reduces the induced contribution shift" | RESOLVED |
| 6 | §VI-A/H5 | fusion "depends on" (implies interaction test) | "varied across the evaluated fusion mechanisms" | RESOLVED |
| 7 | §VI-B | UTD "depended strongly on utility" | "response … differed sharply between … inertial … and … depth" | RESOLVED |
| 8 | §V-B | compensation unmeasured | tied to metric: "ρ_enc shifts oppositely, consistent with partial encoder compensation" | RESOLVED |
| 9 | §IV-D/Disc. | Adam mechanism over-attributed | "consistent with the attenuation expected from Adam's normalized update under approximately constant gradient rescaling"; limitation retained | RESOLVED |
| 10 | §IV-E | "effective per-modality learning rate" | "behaving like a substantial effective per-modality learning-rate difference in this setting" | RESOLVED |
| 11 | §IV-E | "empirically sub-additive" | "the two effects do not combine additively here" | RESOLVED |
| 12 | §IV / STAT_AUDIT | statistical framing | within-split reproducibility; effect sizes/consistency emphasized | RESOLVED |
| 13 | global | "no effect" | uses "no detectable shift"/"little measurable shift" | RESOLVED |
| 14 | §V-A / Table III | CREMA arch. vague | exact 3-block CNN spec (verified `models/cremad_encoders.py`) | RESOLVED |
| 15 | Table III / REPRO | UTD arch. vague | depth 2D-CNN + inertial 1D-CNN spec (verified `models/utd_encoders.py`) | RESOLVED |
| 16 | §IV-A | "equal modality utility" | "matched prototype geometry and noise variance (matched generative signal-to-noise)" | RESOLVED |
| 17 | §IV-A | "margin 1.5" | "orthonormal prototypes of norm (separation scale) 1.5" (verified `data/synthetic.py`) | RESOLVED |
| 18 | Abstract | overloaded | boundary conditions condensed to one sentence | RESOLVED |
| 19 | Conclusion | false dichotomy | "jointly through forward representation scale and optimizer-dependent parameter updates…" | RESOLVED |
| 20 | H5 | too broad | "varies across … and differs qualitatively between the matched synthetic and utility-asymmetric real-data setting tested" | RESOLVED |
| 21 | §VI-A | "approximately null" | added range "\|ΔD\|≤0.001" | RESOLVED |
| 22 | §III-C | masking OOD | "masking is not necessarily part of the training distribution … a diagnostic … not a calibrated estimate of causal contribution" | RESOLVED |
| 23 | Abstract/Concl. | "real-data settings" | narrowed to "utility-asymmetric UTD-MHAD setting" | RESOLVED |
| 24 | §V-A | "speaker memorization" | "reducing the possibility that the reported effect is driven by actor overlap between splits" | RESOLVED |
| 25 | Related Work | novelty vs rebalancing | "unlike such rebalancing methods, our intervention is designed as a diagnostic…" | RESOLVED |

## Additional fixes A–I (this pass)
| ID | Location | Fix | Verified? |
|---|---|---|---|
| A | Table I | **No duplicate row found** — the table already has exactly forward/gradient/ordinary for each of Adam-synthetic, Adam-CREMA-D, SGD-synthetic. Nothing to delete. | RESOLVED (no-op; verified) |
| B | §IV-D, §V-C, Abstract, Disc. | forward-vs-ordinary "reproduces" → "closely matches"; cross-dataset "reproduces the Adam pattern" → "matches the Adam pattern" | RESOLVED |
| C | §VI-B | "near chance (0.036)" → "near the 27-class chance level (0.037), at 0.036" | RESOLVED |
| D | §VI-B | "amplifying inertial entrenched it" → "amplifying the informative inertial modality further increased its relative contribution" | RESOLVED |
| E | §VI-C | "does not address the optimization channel…" → "does not directly modify the optimization imbalance identified in the evaluated scale×LR experiments" | RESOLVED |
| F | Disc. | "gradient-based rebalancing is optimizer-conditional" → "the effect of gradient-based rebalancing may depend on optimizer dynamics" | RESOLVED |
| G | Limitations | "three datasets" → "one controlled synthetic generator and two real datasets on fixed splits" | RESOLVED |
| H | Limitations | "Adam-only on one CNN architecture" → "the real-data forward-vs-gradient decomposition being evaluated only under Adam and the specified CNN architecture" | RESOLVED |
| I | §V-B | "collapses gain (0.248→0.013)" → "reduces multimodal gain from 0.248 to 0.013" | RESOLVED |

## NUMBER-INTEGRITY note (flagged, not changed)
The prompt lists synthetic-Adam α=100 forward/ordinary shifts as **≈−0.066**. The manuscript uses
**−0.070 / −0.069**, which match the source artifact `results/disentangle/disentangle_analysis.json`
(forward `shift_vs_baseline=−0.0695`, ordinary `−0.0693`). Per the "do not change verified numbers /
flag discrepancies" rule, the artifact-verified values were **kept**; the prompt's −0.066 appears to be
a looser approximation and was not adopted. All other listed values (CREMA-D, UTD, SGD) match the
manuscript and artifacts.

## Global language audit result
Residual scan for `reproduces/reproduced/collapses gain/entrenched/speaker memorization/effective
per-modality learning rate/removes the/utility-gated/equal utility/bit-identical/sub-additive`: **NONE**.
Remaining "causal" occurrences are the disavowal in the Fig. 1 caption ("not an identified causal graph")
and the bibliography key `pearl2009causality`; "first" is the sequencing word "We first ask".
