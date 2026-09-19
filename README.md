# modality-scale-collapse

**When Scale Silences a Modality:** investigating whether *feature-magnitude
disparity* between modalities is itself a cause of modality collapse, whether an
early activation-norm ratio (ρ) predicts collapse, whether per-modality
normalization cures it in the scale-driven regime, and where optimization
(learning-rate) imbalance takes over.

This is a **research** repository, not a leaderboard project. It is built so the
hypotheses **can fail**, and failures are reported (see `EXPERIMENT_LOG.md`).

## Scientific questions
- **H1 (diagnostic):** does early-training ρ = E‖h_A‖ / E‖h_V‖ (at the fusion
  boundary) predict which modality collapses?
- **H2 (scale sufficiency):** with *information held fixed*, does scale disparity
  alone cause modality dominance/collapse?
- **H3 (minimal cure) + boundary:** does per-modality normalization cure
  scale-driven collapse, and where does optimization-driven collapse take over?

## Current status (Milestone 1 — synthetic E2 prototype: DONE)
Generator → tiny encoders → fusion → classifier → scale injection → ρ →
masking → multi-seed scale sweep → plots. Everything runs on **CUDA**. 12/12
tests pass. See `PROJECT_STATUS.md`.

### Headline finding (honest, nuanced)
With information verified constant (alpha-invariant linear probe), scale disparity
causes **strong, symmetric collapse early in training** (epoch 5:
Spearman(ratio, dominance) = 0.97, multimodal gain halves at the extremes) — but a
trainable encoder **renormalizes the scale by convergence** (epoch 40: collapse
essentially gone; early ρ spans 0.14→7.55 while final ρ only 0.60→1.68).
**Conclusion:** static feature-scale disparity is a sufficient cause of *transient*
collapse and reliably drives ρ, but is **not sufficient for *persistent* collapse**
in this architecture — it is an early-dynamics / optimization effect. This
motivates the diagnostic (H1), normalization (H3), and scale×LR boundary (H4).

## Repository layout
```
configs/        YAML configs (one per experiment); nothing is hard-coded in scripts
data/           synthetic.py (causal env); crema_d.py / include.py (later)
models/         encoders.py, fusion.py, heads.py, multimodal_model.py
interventions/  scale_injection.py, per_modality_norm.py, diagnostic.py (ρ)
analysis/       metrics.py (masking, Δ, gain, macro-F1), probes.py (GPU linear probe),
                statistics.py (mean/CI, correlation), plots.py
experiments/    train.py (GPU loop), e2_causal.py (H2 sweep + QC + artifacts)
msc/            seed.py, device.py (GPU-first), runlog.py (provenance)
tests/          pytest suite (generator invariants, masking, ρ, GPU, smoke train)
results/        raw/ processed/ figures/ tables/   (auto-generated)
```

## Quickstart
```bash
# GPU required (torch CUDA build). From the repo root:
python -m pytest tests -q
python experiments/e2_causal.py --config configs/synthetic_causal.yaml --tag primary
```
Outputs land in `results/` (see `REPRODUCE.md`).

## Design principles
- Every run records seed, full config, git hash, hardware, params, metrics, runtime.
- Automated QC fails loudly on: broken information-invariance, scale/label
  mismatches, NaN/Inf, missing seeds/config.
- Verdicts are computed from evidence and reported as
  SUPPORTED / PARTIAL / INCONCLUSIVE — the report interprets, the code does not decide.
