# MANUSCRIPT REVISION AUDIT

_Pre-revision audit of `main.tex` against the experiment artifacts (`results/**`,
`EXPERIMENT_FREEZE_AUDIT.md`, experiment scripts). Source of truth = the code and
result JSONs. No numbers were invented; where a detail was not in the artifacts it
is flagged `MISSING`. Ground-truth facts recovered from code are listed in §0._

## 0. Ground truth recovered from experiment code (previously implicit in the manuscript)
- **Statistical test:** `scipy.stats.ttest_rel` — a **two-sided paired** t-test across
  the 5 seeds (arms share the seed set) (`experiments/disentangle_scale_grad.py:290`,
  same helper in `sgd_disentangle.py`, `cremad_disentangle.py`, `fusion_robustness.py`).
- **Effect size:** Cohen's `d = mean(paired diff) / SD(paired diff)` with `ddof=1`
  (paired `d_z`), `CI95 = 1.96·SD/sqrt(n)` (`:295-296`).
- **Unit of observation:** seed; **n = 5** (`configs/boundary.yaml seed_policy.seeds=[0..4]`).
- **Synthetic generator** (`data/synthetic.py`): `x_m = α_m·(μ_y^m + σ_m·ε)`, `ε~N(0,I)`;
  orthonormal class prototypes from fixed `prototype_seed=12345`; a linear head can absorb a
  constant `α`, so information is scale-invariant. Config (`configs/boundary.yaml`):
  `C=8`, `d_a=d_v=32`, margin `1.5`, `σ_a=σ_v=1`, `n_train/val/test=4000/1000/2000`.
- **Synthetic model:** hidden `64`, rep `32`, ReLU, linear head, sum fusion (concat/gated for
  robustness).
- **Optimizers** (`experiments/train.py:20-39`): `torch.optim.Adam(lr=1e-3, weight_decay=0)`
  → PyTorch defaults `betas=(0.9,0.999)`, `eps=1e-8`; `torch.optim.SGD(lr=1e-3, momentum=0.9,
  weight_decay=0)`. **No LR scheduler, no gradient clipping** (training loop has neither).
  Synthetic: bs `256`, `40` epochs.
- **CREMA-D** (`configs/cremad.yaml`, `data/crema_d.py`, `experiments/cremad_disentangle.py`):
  6 emotion classes; audio = 16 kHz mono, 3.0 s, log-mel (64 mels, n_fft 1024, hop 256)
  → `[1,64,188]`; video = 8 uniformly-sampled 64×64 grayscale frames → `[8,1,64,64]`; rep 64,
  sum fusion, linear head; Adam (defaults) lr 1e-3, wd 0, bs 64, **30** epochs; audio scaled;
  α∈{10,100}; 5 seeds. Split: actor-disjoint 64/9/18 actors = **5234/738/1470** clips.
- **Gated fusion** (`experiments/fusion_robustness.py:54-66`): per-modality self-gates
  `g_m = σ(Linear_m(h_m))`, gated reps summed (rep-dim), linear head.
- **All headline numbers verified** against the result JSONs (see §2); every figure value is
  read from `results/**` by `scripts/make_figs.py` (no hand-entered points).

## 1. Major scientific weaknesses (in the pre-revision draft)
1. **"mediator" over-claim.** Abstract/metrics/discussion/conclusion call the parameter-update
   ratio "the mediator," implying a formal mediation analysis that was not run. → Reword to an
   *optimization-level readout/diagnostic*.
2. **Adam described as "sign descent."** Related Work: "Adam approximates sign descent and is
   largely invariant to constant per-parameter gradient rescaling." Too strong / imprecise.
   → "Adam's coordinate-wise normalization can substantially attenuate the effect of a constant
   gradient rescale on the update."
3. **"fusion-robust" (gradient inertness).** Reads as architecture generality. → "approximately
   null across the three evaluated fusion mechanisms."
4. **"the phenomenon is utility-gated"** stated as a universal mechanism. → "in the evaluated
   UTD-MHAD setting the effect depended on relative modality utility."
5. **Statistical spec absent.** The exact test/effect-size/unit were not stated; large `|d|` from
   a paired design could be misread as population effect size. → State test, `d_z`, n=5, and that
   effect sizes index reproducibility, not population magnitude.

## 2. Numerical results — verified (no changes needed to values)
All values in the draft trace to result JSONs at α=100:
- Adam synth: fwd Δ−0.070 d−3.76 p1.1e-3; grad Δ−0.0005 d−0.53 p0.30; ord Δ−0.069 d−3.73
  (`disentangle_analysis.json`). ✓
- Adam CREMA-D: fwd Δ+0.259 d7.39 p8e-5; grad Δ−0.001 d−0.08 p0.87; `r_g`=282.7, `r_Δθ`=1.61;
  ord Δ+0.266 d8.51; ord-vs-fwd d0.59 p0.26 (`cremad_disentangle_analysis.json`). ✓
- SGD synth: fwd +0.244 d3.93; grad +0.200 d3.90; ord +0.322 d7.75; `r_Δθ`(grad)=95.9,
  `r_g`=318.5; ord `r_Δθ`=18.3; ord-vs-fwd d2.86 p3e-3 (`sgd_disentangle_analysis.json`). ✓
- CREMA-D H6 dose: dom 0.075→0.299→0.328 (audio), →0.056→0.009 (video); gains 0.248→0.045→0.013,
  0.263→0.295; Spearman 0.922; z: 0.328→0.110, ρ 3.24→0.97 (`cremad_train_scale_analysis.json`). ✓
- Fusion: sum d−3.76 p1e-3; concat d−5.16 p3e-4; gated d−0.70 p0.19 (n.s.); grad null all three
  (`fusion_robustness_analysis.json`). ✓
- UTD: depth 0.036 / inertial 0.227 / mm 0.632 / gain 0.405; base dom −0.191; inertial×5/×10
  dom −0.501/−0.615 gain 0.136/0.053; depth×5/×10 dom −0.193/−0.198 gain 0.398/0.400; Spearman
  0.757 (`utd_train_scale_analysis.json`). ✓
**Conclusion: no numerical inconsistency between manuscript and artifacts.**

## 3. Unsupported claims (present → to remove/rescope)
- "mediator" as formal mediation (see 1.1). "fusion-robust" (1.3). "utility-gated phenomenon" (1.4).
- "sign descent" characterization of Adam (1.2).

## 4. Overstated claims (present → to soften)
- "causal modulator" — retain only in the controlled-intervention sense; prefer "controlled
  intervention that changes contribution." Title keeps "Controlled Causal Study" (intervention-level).
- Figure panel label "mediator" → "gradient vs update ratio."

## 5. Missing methodological details (to ADD from §0 ground truth; none fabricated)
- Generator equation + prototype construction; synthetic dims/sizes/margin/σ.
- Adam betas/eps, wd; SGD momentum; bs/epochs per dataset; no scheduler/clipping.
- CREMA-D audio/video preprocessing; class count; split clip counts (5234/738/1470).
- Gated-fusion definition. Contribution metric = masking-based (operational).

## 6. Statistical issues (to fix)
- State the exact test (two-sided paired t-test), `d_z` definition, n=5 = seeds on a fixed split.
- Add the caveat that the paired design's small denominator inflates `|d|`; read as reproducibility.
- Keep UTD seed-level p-values out of population claims (already scoped; keep).

## 7. Reproducibility issues (to fix)
- Add optimizer hyperparameters + preprocessing to the protocol table; note code/config availability.

## 8. Figure/table issues
- Fig. 3 panel (b) title "mediator" → neutral wording; regenerate that PDF from JSON (value-identical).
- All other figures/tables consistent with artifacts; no invented points. No change to data.

## 9. Citation/reference issues
- 45 references, all cited, first-appearance order [1]→[45] verified in the previous build; unchanged
  set. No fabricated entries. (Re-verified after revision.)

## 10. Required revisions
Items 1.1–1.5, §5 additions, §6 statistical spec, §8 figure label, contribution/abstract rewording
(Parts 4/5/8/13/15/16/19/24/25/26). Keep exactly 6 pages, 45 refs.

## 11. Optional revisions
- OBSERVATION/INTERPRETATION/LIMITATION structure in Discussion (adopted).
- H1–H5 explicit hypotheses (adopted, scoped).

## 12. Experiments genuinely necessary
**None.** Every scoped claim is supported by existing artifacts; each gap is disclosed as a
limitation, not a validity hole. No new experiment is required to defend the revised claims.

## 13. Experiments NOT necessary (explicitly)
- SGD-on-real-data, attention/transformer fusion, additional optimizers, >40-epoch SGD horizon,
  additional datasets — all are *scope extensions*, disclosed as open, not needed for the claims made.

## 14. Deeper implementation + statistical verification (second pass)
_Added after re-reading the code end-to-end, to answer the autograd (Part I) and per-p-value
(Parts U/V/W) asks with verified facts, not description._

**Autograd implementation (verified in `disentangle_scale_grad.py` and reused in
`cremad_disentangle.py`, `fusion_robustness.py`):**
- The scale is applied by `arm_apply(h_A, α, arm)` **at the fusion boundary** — on
  `h_A = norm_A(enc_A(x_A))`, i.e. *after* the encoder and its normalization and *before* `fuse(...)`
  (`arm_forward`/`arm_logits`). It affects **modality A only**; modality B and the head input are
  untouched except through fusion.
- `ForwardScale.forward = α·x`, `backward` returns `g` (×1): forward magnitude scaled, encoder-A
  gradient **not** scaled. `GradScale.forward = x.clone()` (values identical), `backward = α·g`:
  forward **bit-identical to baseline**, only encoder-A's gradient scaled. This is checked by the
  sanity gate (`gradient_only_forward_identical_to_baseline`). Hence in gradient-only the loss and
  **classifier-head gradients equal baseline**; only encoder A's gradient carries α.
- α enters the optimizer as a **constant per-parameter rescale of encoder A's gradient**, before
  `opt.step()`. `weight_decay=0`, **no gradient clipping, no scheduler, no gradient accumulation**
  (single backward per batch) — verified in the training loops and `configs/*.yaml`.
- These verified facts are now stated compactly in the manuscript (§ Forward and Gradient Pathways).

**Statistical routine (verified identical across all four scripts):** `disentangle_scale_grad.py:290`,
`sgd_disentangle.py:98`, `cremad_disentangle.py:180`, `fusion_robustness.py:156` all use the same
`_paired` helper: `scipy.stats.ttest_rel(b,a)` (**two-sided paired** t-test), `cohen_d =
mean(b−a)/std(b−a, ddof=1)` (**paired d_z**), `ci95 = 1.96·sd/√n`, `n=5` seeds. Therefore **every**
reported p-value and Cohen's d in the paper — including CREMA-D ordinary-vs-forward (d=0.59, p=0.26,
a paired seed comparison whose non-significance is used only to say "ordinary ≈ forward") — comes
from this one routine. The manuscript now states this globally ("Throughout … the identical routine
is used for every reported comparison").
