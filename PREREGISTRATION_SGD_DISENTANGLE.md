# Pre-registration — SGD forward-scale vs gradient-scale disentangling

Written **before** running. Authorized single experiment: repeat the validated Adam disentangling
experiment under **SGD**, to test whether the forward-dominated pathway is optimizer-dependent and
whether isolated gradient scaling becomes an effective learning-rate effect under SGD (connecting to
E4/E4-EXTREME). **All prior experiments (E1–E5, CREMA-D R1–R3/H6, UTD-MHAD, Adam disentangle) are
immutable and untouched. Adam runs are NOT rerun.** Start commit `4465ebc`. Date: 2026-09-20.

## Scientific question / hypothesis (not assumed true)
**H_SGD:** under SGD, isolated encoder-gradient scaling (gradient-only) will have a measurable effect on
learned modality contribution, unlike the Adam gradient-only null — because under SGD a constant gradient
multiply ≈ a modality-specific effective-LR change. A null / mixed / interaction / instability result is
equally acceptable and will be reported as-is. Not tuned toward any outcome.

## Only intended change vs the Adam experiment: Adam → SGD
Everything else identical: same generator (`configs/boundary.yaml` data: 8 classes, dim 32+32, margin
1.5, σ=1, n_train 4000/val 1000/test 2000, prototype_seed 12345, **data α_a=α_v=1**), same tiny MLP
encoders (hidden 64, rep 32), sum fusion, linear head, normalization='none'; batch 256; 40 epochs; **seeds
[0,1,2,3,4]** (paired to the Adam run by init/data order); α ∈ {1,10,100} on modality **A**; same metrics;
the **same validated `ForwardScale`/`GradScale`/`arm_apply`/`arm_forward`** reused from
`experiments/disentangle_scale_grad.py` (NOT redesigned).

## Optimizer configuration (fixed; NOT tuned)
- **Adam (reference, already run):** lr 1e-3, betas (0.9, 0.999), eps 1e-8, weight_decay 0, no scheduler,
  no gradient clipping.
- **SGD (this run):** the project's existing SGD default — `torch.optim.SGD(lr=1e-3, momentum=0.9,
  weight_decay=0)`, no scheduler, no gradient clipping. lr kept **identical** to Adam's base_lr (closest
  documented "optimization scale"); per-encoder LR equal (no asymmetry). **No hyperparameter sweep, no
  result-driven adjustment.** If this fixed config diverges, STOP and report (do not tune to a result). If
  a documented numerical instability from the exact scale requires the *minimum* stability fix, it is
  recorded explicitly and not optimized for effect size.

## Four arms (validated implementation reused)
| arm | forward → fusion | gradient into h_A |
|---|---|---|
| baseline | `h_A` | ×1 |
| forward-only | `α·h_A` (ForwardScale) | ×1 |
| gradient-only | `h_A` (GradScale, forward bit-identical to baseline) | ×α |
| ordinary | `α·h_A` (plain multiply) | ×α (natural) |

## Autograd sanity gate (rerun; MUST pass before training)
Scalar forward/backward multipliers: baseline (1,1), forward (α,1), gradient (1,α), ordinary (α,α);
gradient-only forward bit-identical to baseline; forward-only enc gradient free of the α multiplier;
ordinary carries both. (Autograd behaviour is optimizer-independent, but the gate is rerun for the record.)
**Fail ⇒ STOP.**

## SGD-specific measurement (critical): gradient imbalance vs PARAMETER-UPDATE imbalance
Per epoch, per arm, record: encoder gradient norms ‖g_A‖, ‖g_V‖ and ratio; **and parameter-update norms
‖Δθ_A‖, ‖Δθ_V‖ (θ after − before opt.step, summed over batches) and ratio ‖Δθ_A‖/‖Δθ_V‖.** Under Adam a
gradient ×α need not change the update; under SGD it should scale the update ~×α. This distinction is the
core of the cross-optimizer comparison. (The Adam run did not record update ratios; Adam's update ratio is
inferred ≈1 from its null gradient-only outcome and flagged as inference — Adam is NOT rerun.)

## Primary metrics
val/test accuracy; Δ_A, Δ_V; dominance D=Acc_A−Acc_V; multimodal gain; ρ_fusion; ρ_encoder; encoder
gradient ratio; **parameter-update ratio**; compensation `1−ln(ρ_fusion)/ln(α)`; per-seed values.

## Primary comparisons (α=100, paired by seed)
forward-only vs baseline; gradient-only vs baseline; ordinary vs baseline; ordinary vs forward-only;
ordinary vs gradient-only. **Key question:** does gradient-only now shift contribution under SGD?

## Persistence analysis (α=100 trajectories)
dominance(t), gain(t), gradient_ratio(t), update_ratio(t), ρ_fusion(t), ρ_encoder(t). Determine whether
any gradient-only effect is **transient (recovers)** or **persists to convergence.** Do not label a
transient effect "collapse"; distinguish contribution shift / accuracy change / gradient imbalance /
update imbalance / persistent collapse.

## Statistics
Same framework: mean ± SD, 95% CI, paired Cohen's d + paired t (same seeds → paired comparisons),
per-seed points plotted. Not p-values alone. Seeds are init/order variance, not population samples.

## Decision framework (choose from evidence; do not force)
- **CASE A:** SGD gradient-only produces a substantial shift → gradient scaling is optimizer-dependent
  (effective-LR pathway under SGD).
- **CASE B:** SGD gradient-only ≈ baseline → insufficient under this SGD config (no general optimizer-
  invariance claim).
- **CASE C:** gradient-only shifts but ordinary ≠ forward-only → mixed (forward + update dynamics).
- **CASE D:** gradient-only causes persistent collapse → connects to E4/E4-EXTREME (only if metrics match).
- **CASE E:** forward-only shifts but gradient-only does not under both Adam and SGD → forward robust,
  isolated gradient scaling insufficient in both configs.

## Stop conditions
autograd sanity fails; forward-only alters the forward rep incorrectly; gradient-only changes forward
activations; ordinary misbehaves; SGD config diverges consistently; SGD needs architecture changes; the
only stable run requires result-driven tuning. Report failures; do not tune to a result.

## Outputs (new dir `results/sgd_disentangle/`, no overwrite)
`PREREGISTRATION_SGD_DISENTANGLE.md` (this), `EXPERIMENT_REPORT_SGD_DISENTANGLE.md`,
`results/sgd_disentangle/sgd_disentangle_{primary,trajectories,agg}.csv`, `_analysis.json`, `_sanity.json`,
`_config.json`, `_adam_vs_sgd.csv`; figures `sgd_{dominance,contribution,grad_ratio,update_ratio,
adam_vs_sgd,dose_response,trajectories}.png`. Do not commit pending review. Then STOP.
