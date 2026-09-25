# STATISTICAL AUDIT

_All items verified against the experiment code (`experiments/*disentangle*.py`,
`fusion_robustness.py`) and result JSONs. No statistic was recomputed or invented._

| Item | Value / procedure | Source |
|---|---|---|
| Unit of analysis | random seed (initialization + minibatch order) | configs; all scripts |
| n | 5 seeds `[0..4]` | `configs/boundary.yaml`, `cremad.yaml` |
| Split | fixed per dataset (synthetic 4k/1k/2k; CREMA actor 64/9/18; UTD subj 1,3,5/7/2,4,6,8) | configs |
| Test | two-sided **paired** t-test `scipy.stats.ttest_rel(b,a)` | `disentangle_scale_grad.py:294`, `sgd_disentangle.py:102`, `cremad_disentangle.py:184`, `fusion_robustness.py:160` |
| Pairing | arms share the same 5 seeds → paired comparison is valid | all scripts (`sort_values("seed")`) |
| Effect size | Cohen's `d_z = mean(b−a) / std(b−a, ddof=1)` | `:296` etc. (identical helper) |
| CI | `1.96 · sd / sqrt(n)` (descriptive) | `:295` |
| Multiplicity | **none applied** (pre-specified pairwise contrasts) | analysis code; disclosed in text |
| Interpretation | within-split reproducibility, not population inference; effect sizes + cross-seed consistency emphasized | manuscript §IV, Discussion |
| UTD subject-level | 4 fixed test subjects; per-subject consistency reported; seed-level p-values **not** treated as population significance | manuscript §VI-B; `utd_train_scale_persubject.csv` |

**Consistency check.** Every p-value and Cohen's d printed in the manuscript (Tables I–II, §IV–VI)
is produced by the single `_paired` routine above; the manuscript states this globally ("Throughout,
significance is a two-sided paired t-test … the identical routine for every comparison"). The
n=5/fixed-split caveat and the multiplicity statement appear in §IV. No confidence intervals or
corrections beyond those in the code are claimed.

**Honest limitation retained in the paper.** With n=5 the paired denominator is small, so large |d|
reflects cross-seed consistency rather than a large population effect; this is stated explicitly.
