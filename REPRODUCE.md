# REPRODUCE.md

All commands are run from the repo root (`modality-scale-collapse/`) on a machine
with a CUDA-capable GPU and a PyTorch CUDA build (`torch.cuda.is_available()`).

## 0. Environment check + tests
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pytest tests -q
```
Expected: torch reports `+cuXXX` and `True`; `12 passed`.

## E2 — causal scale experiment (H2)  [IMPLEMENTED]
```bash
# Primary: converged, 5 seeds, geometric scaling, headroom regime
python experiments/e2_causal.py --config configs/synthetic_causal.yaml --tag primary

# Undertrained contrast (shows transient collapse):
python experiments/e2_causal.py --config configs/synthetic_causal.yaml --tag headroom_ep5 --epochs 5

# Ceiling regime (less headroom):
python experiments/e2_causal.py --config configs/synthetic_causal.yaml --tag ceiling --margin 3.0
```
Produces, under `results/`:
- `raw/e2_causal/<tag>/run_r{ratio}_s{seed}.json` — per-run provenance + ρ/contribution history
- `raw/e2_causal/<tag>/run_record.json` — experiment record (QC, verdict)
- `processed/e2_causal_<tag>_{runs,agg,probes}.csv`
- `tables/e2_causal_<tag>_summary.md`
- `figures/<tag>/e2_scale_vs_{contribution,rho,gain,masked_acc}.png`

Sanity: probe accuracy ~constant across ratios (QC passes); at `--epochs 5` the
verdict is SUPPORTED (Spearman(ratio, dominance) ≈ 0.97); at 40 epochs it is
INCONCLUSIVE (encoder compensates). Both are expected — see `EXPERIMENT_LOG.md`.

## Planned (not yet implemented)
- **E1 — diagnostic (H1):** `python experiments/e1_diagnostic.py --config configs/diagnostic.yaml`
  → early-ρ → collapse AUROC/correlation, `figures/e1_*`.
- **E3 — normalization cure (H3):** `python experiments/e3_cure.py --config configs/normalization.yaml`
  → none/LayerNorm/z-score/BatchNorm across collapse regimes.
- **E4 — scale × LR boundary:** `python experiments/e4_boundary.py --config configs/boundary.yaml`
  → phase diagram `figures/e4_phase.png`.
- **E5 — baselines (OGM-GE, …):** `python experiments/e5_baselines.py --config configs/baselines.yaml`
- **E6 — ablation:** `python experiments/e6_ablation.py --config configs/*.yaml`
- **CREMA-D:** requires `git lfs pull` in `../CREMA-D-master` first (media are LFS stubs).
- **INCLUDE-50:** official splits in `../INCLUDE-master/train_test_paths/`; videos via HF (deferred).
