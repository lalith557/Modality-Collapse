# FINAL HARDENING AUDIT

_Five independent hostile reviews of the near-final manuscript, then a triaged fix list.
Source of truth = experiment code + `results/**`. No number/test/detail invented. Items are
marked FIX NOW (do this pass), ALREADY RESOLVED (prior passes), or NEEDS-EXPERIMENT (out of scope)._

## Reviewer A — Multimodal ML
- **Strongest criticism:** "This is not a balancing method; what is the contribution?" — VALID as a
  framing risk. Evidence: the decomposition + optimizer/fusion/utility dependence. → ALREADY RESOLVED
  (Intro states it is a controlled mechanistic decomposition, not an algorithm). Reinforce Fig. 1 as
  *conceptual*. FIX NOW (caption wording).
- "UTD is just an easy/degenerate case." VALID → ALREADY RESOLVED (framed as utility-asymmetry boundary,
  depth near chance stated).

## Reviewer B — Optimization (Adam/SGD)
- **Strongest criticism:** "You claim Adam is invariant to gradient scaling." Would be fatal IF present.
  → ALREADY RESOLVED: paper says "under the tested Adam configuration, coordinate-wise normalization
  substantially attenuated…"; no "sign descent"; limitation stated.
- "SGD update ratio ≈ α overclaims exactness." Partly VALID. → FIX NOW: state "≈96× update ratio,
  demonstrating substantial transmission" rather than "=α" (keep `≈α` only as a parenthetical intuition).

## Reviewer C — Statistics / reproducibility
- **Strongest criticism 1:** "n=5, many pairwise tests, no multiplicity control." VALID. →
  FIX NOW: add a one-sentence multiple-comparisons statement (pre-specified pairwise contrasts, no
  correction, emphasize effect sizes + cross-seed consistency). Do NOT fabricate a correction.
- **Strongest criticism 2:** "p=0.87 is read as proof of no effect." Partly VALID. → FIX NOW: soften
  "inert" to "no detectable shift in this five-seed experiment" at the key occurrences.
- **Strongest criticism 3:** "Severity S is undefined." VALID. → FIX NOW: define
  `S = clip(1 − c_weak/c_weak^base, 0, 1)` (verified `e4_extreme.py:197`).
- Test/effect-size spec, generator equation, hyperparameters, CREMA preprocessing → ALREADY RESOLVED
  (two-sided paired `ttest_rel`, `d_z`; verified identical across all four scripts).

## Reviewer D — MLSP area chair
- **Strongest criticism:** "Title says 'Causal Study' — the work is interventional, not a causal-graph
  identification." VALID. → FIX NOW: retitle to "Forward or Gradient? Disentangling Training-Time
  Feature-Scale Effects on Modality Contribution".
- "6-page / 45-ref / format compliance." → ALREADY RESOLVED; re-verify after edits.

## Reviewer E — Hostile (looking to reject)
- **Strongest criticism (most dangerous):** "Logical contradiction: 'encoder-frozen scaling is
  compensated by trainable encoders.'" VALID and embarrassing. → FIX NOW: the synthetic E2/E4 scaling
  is a **constant scale held fixed throughout training with encoders trainable** (verified
  `e2_causal.py` uses `train_model`); rewrite H1 and the subsection to say exactly that; remove
  "frozen/test-time" wording for that experiment.
- "'nonlinear interaction' claimed without an interaction model." VALID. → FIX NOW: say "did not
  combine additively (empirically sub-additive)".
- "Reproducibility statement implies a public release." Partly VALID. → FIX NOW: "code and
  configurations are prepared for release" (not claimed public).
- "'mediator' overclaim." → ALREADY RESOLVED (optimization-level readout; "not a formal mediation").

## Triage summary
**CRITICAL (fix now):** static-scaling contradiction (H1 + subsection).
**MAJOR (fix now):** title de-causalization; severity S definition; multiple-comparisons statement;
"no detectable shift" wording; SGD "≈α" exactness; sub-additive wording.
**MINOR (fix now):** Fig. 1 "conceptual" caption; reproducibility "prepared for release"; add explicit
"what the paper does not show" list item (gradient imbalance not universally irrelevant).
**ALREADY RESOLVED:** mediator, Adam/sign-descent, fusion/UTD/OGM-GE/normalization scoping, statistical
spec, generator + hyperparameter reproducibility, autograd implementation audit, 6pp/45-ref/order.
**NEEDS NEW EXPERIMENT (out of scope, disclosed as limitations, NOT run):** SGD-on-real-data;
attention/transformer fusion; additional optimizers/schedules; >40-epoch SGD horizon; disambiguating
Adam's attenuation (moment-norm vs shared-head absorption). None is required to defend the scoped claims.

## Principle applied
Where a criticism is valid and fixable from existing evidence, the claim is scoped or the wording
corrected. Where it would require new evidence, it is disclosed as a limitation, not patched with an
invented result.

## Round 2 (final precision pass) — additional fixes applied
Verified against code/PDF; no numbers changed.
- **Section title** "Problem Formulation and Causal Interventions" → **"…Controlled Interventions"**;
  added the disclaimer "we use intervention terminology to distinguish exogenously assigned
  perturbations from observational correlations; we do not claim identification of a complete causal
  graph."
- **"bit-identical" removed** → "unchanged by construction ($\textsc{Grad}_\alpha$ is the identity in
  the forward pass)" — accurate, since `GradScale.forward` returns `x.clone()` (no bitwise check was
  run, so "bit-identical" was unjustified).
- **"Two channels are thus separable" → "The experiments thus support two experimentally separable
  pathways within the evaluated intervention."**
- **Adam limitation sharpened**: "the update-ratio result establishes attenuation at the
  optimizer-update level but does not identify whether it arises from Adam's moment normalization,
  shared-head interactions, or both" (and the duplicate limitations-list item removed).
- **Conclusion + abstract "≈α" → "≈96× at α=100"** (matches the observed value; no theoretical
  overstatement).
- **Normalization wording** "removes the forward-scale channel" → "largely suppresses the imposed
  forward-scale difference" (three occurrences); "persistent collapse" → "persistent contribution
  loss" in the normalization analysis.
- **UTD** reframed as "a boundary-condition experiment … not an independent replication of the central
  pathway decomposition."
- **Contributions consolidated 4→3** (methodology / mechanistic finding / validation+boundaries),
  explicitly "methodological rather than another rebalancing algorithm."
- Held **exactly 6 pages** by consolidating contributions, removing one duplicate limitation and one
  duplicate table reference, tightening the abstract/stats wording, and a small span-figure width
  reduction — no formatting hacks.

## Round 3 — 25 reviewer vulnerabilities (all resolved from existing evidence)
| # | Issue | Resolution (verified) |
|---|---|---|
| 1 | "estimate the effect of scale" | → "quantify the response to the imposed scale intervention rather than an observational association with naturally occurring feature magnitude" |
| 2 | constant scale "does nothing" | → "substantially compensated, leaving only a small residual contribution change" (H1, §IV-B, conclusion) |
| 3 | "collapse severity S" universal | renamed `S_loss` (study-specific), "not proposed as a universal definition of modality collapse" |
| 4 | figure "normalization removes scale channel" | TikZ node → "normalization suppresses scale difference" |
| 5 | "undoes the induced shift" | → "substantially reduces the induced contribution shift" |
| 6 | fusion "depended on" (implies interaction test) | → "the observed forward effect varied across the evaluated fusion mechanisms"; non-significance of gated reported separately (no interaction test claimed) |
| 7 | UTD "depended strongly on utility" | → "the response to scaling differed sharply between the informative inertial and near-chance depth modality" |
| 8 | encoder compensation unmeasured | tied to metric: "the pre-scale ratio ρ_enc shifts oppositely, consistent with partial encoder compensation" |
| 9 | Adam mechanism over-attributed | "consistent with Adam's coordinate-wise normalization…"; limitation: does not identify moment-norm vs shared-head vs both |
| 10 | "effective per-modality learning rate" | → "behaving like a substantial effective per-modality learning-rate difference in this setting" |
| 11 | "empirically sub-additive" (formal) | → "the two effects do not combine additively here" |
| 12 | statistical framing | retained: within-split reproducibility, effect sizes emphasized (see STATISTICAL_AUDIT.md) |
| 13 | "no effect" language | audited; uses "no detectable shift"/"little measurable shift" only |
| 14 | CREMA-D architecture | added exact 3-block CNN spec (Table III + REPRODUCIBILITY_AUDIT.md); verified `models/cremad_encoders.py` |
| 15 | UTD-MHAD architecture | added depth 2D-CNN + inertial 1D-CNN spec (Table III + audit); verified `models/utd_encoders.py` |
| 16 | "equal modality utility" | → "matched prototype geometry and noise variance (matched generative signal-to-noise)"; table "2 matched" |
| 17 | "margin 1.5" | → "orthonormal prototypes of norm (separation scale) 1.5" (verified `data/synthetic.py`) |
| 18 | abstract overload | condensed boundary conditions to one sentence |
| 19 | conclusion dichotomy | → "jointly through forward representation scale and optimizer-dependent parameter updates" (both pathways preserved) |
| 20 | H5 too broad | → "varies across the evaluated fusion mechanisms and differs between the matched synthetic and utility-asymmetric real-data setting tested" |
| 21 | "approximately null" context | added range "|ΔD|≤0.001" for gradient-only across fusions |
| 22 | masked-accuracy OOD | added: "masking is not necessarily part of the training distribution … a diagnostic of retained predictive utility, not a calibrated causal-contribution estimate" |
| 23 | "real-data settings" generalization | narrowed to "in the utility-asymmetric UTD-MHAD setting" (abstract + conclusion) |
| 24 | "speaker memorization" | → "reducing the possibility that the reported effect is driven by actor overlap between splits" |
| 25 | novelty vs rebalancing | added: "unlike such rebalancing methods, our intervention is designed as a diagnostic … which pathway carries the effect" |

**Resolved from existing evidence: 25 / 25.** No new experiment was required or run. Architecture
details (14/15) were recovered from the model code, not invented. Global forbidden-language audit:
remaining hits are disavowals ("not a cure", "not … universal", "not an identified causal graph"),
the sequencing word "first", or false positives (`im`**prove**, the bib key `pearl2009`**causal**`ity`).
