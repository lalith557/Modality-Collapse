# Reproducing the results

Every number in the papers and on the website comes from a CSV or JSON file under
`results/`. This guide lists the command that produced each one. All commands run from
the repository root and need a CUDA GPU; the reference machine is an RTX 4070 Laptop
(8 GB), Python 3.13 and PyTorch 2.11 (CUDA 12.8).

The files under `results/` are the frozen outputs that the papers, the website and the
tests are checked against. Re-running a script overwrites its own output files, so run
on a separate branch or copy if you want to compare against the originals.

## 1. Environment

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # pick the index for your CUDA driver
pip install -r requirements.txt            # experiments
pip install -r requirements-studio.txt     # Prediction Studio, CREMA-D/UTD-MHAD decoding, tests
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect a +cuXXX build and True
```

Seeds: the synthetic studies use seeds 0 to 9 and redraw the dataset per seed; the real-data
studies use seeds 0 to 4. A seed fixes initialisation, data order and, for synthetic data, the
data draw (`msc/seed.py`, deterministic cuDNN).

## 2. Data

The synthetic data is generated on the fly by `data/synthetic.py` and needs no download.

**CREMA-D.** Clone the dataset next to the repository, with its Git LFS media:

```bash
git lfs install
git clone https://github.com/CheyneyComputerScience/CREMA-D.git ../CREMA-D-full
```

`data/cremad_manifest.py` looks for `../CREMA-D-full`, then `../CREMA-D` and
`../CREMA-D-master`, and takes the first whose media are real files. Set `CREMAD_ROOT` to
point anywhere else. A clone made without Git LFS contains 130-byte pointer files instead of
audio and video; run `git lfs pull` inside it.

```bash
python data/cremad_manifest.py          # actor-disjoint split (seed 42): 5234 / 738 / 1470 clips
python experiments/cremad_extract.py    # caches log-mel audio and 8-frame video in data/cremad/cache/
```

**UTD-MHAD.** Download the Depth and Inertial archives from the
[UTD-MHAD page](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html) and extract them to
`../UTD-MHAD/Depth` and `../UTD-MHAD/Inertial` (the path is `data.raw_dir` in
`configs/utd_mhad.yaml`).

```bash
python experiments/utd_audit.py         # manifest + integrity report (docs/audits/DATA_AUDIT_UTD_MHAD.md)
python experiments/utd_extract.py       # caches encoder inputs in data/utd_mhad/cache/
```

## 3. The intervention operators

`experiments/disentangle_scale_grad.py` defines the instrument used everywhere:

- `ForwardScale`: forward multiplies by α, backward passes the gradient unchanged.
- `GradScale`: forward is the identity, backward multiplies the encoder gradient by α
  (a scaled generalisation of the gradient-reversal layer).
- `sanity_check()` verifies the four (forward, backward) multipliers numerically and that the
  gradient-only arm is bit-identical to the baseline in the forward pass. The decomposition
  scripts (`disentangle_scale_grad.py`, `sgd_disentangle.py`, `cremad_disentangle.py`,
  `fusion_robustness.py`) run it before training.

## 4. Commands and outputs

### Main study (Phases 1 to 4)

| What it produces | Command | Output |
|---|---|---|
| Decoupled metrics on synthetic data, Adam and SGD, α = 10 and 100 (CA paper Table II, Fig. 3) | `python experiments/phase1_decoupled.py --seeds 10` | `results/phase1_decoupled/phase1_primary_seeds10.csv` |
| α sweep, four arms, learning-rate-matched and forward-for-k arms (main paper Fig. 2, Table I synthetic row) | `python experiments/phase2_mechanism.py --seeds 10 --alphas 1,3,10,30,100,300` | `results/phase2_mechanism/phase2_primary_seeds10.csv` |
| Adam ε sweep and η/α cancellation | `python experiments/phase2_eps_cancel.py --seeds 10` | `results/phase2_mechanism/phase2_eps_cancel.csv` |
| Forward scaling for the first k epochs only | `python experiments/phase2_forwardk.py` | `results/phase2_mechanism/phase2_forwardk.csv` |
| Compensation curves c(α) | `python experiments/phase2_ccurve.py` | `results/phase2_mechanism/phase2_ccurve.csv` |
| Learning-rate sweep and tuned-baseline control | `python experiments/phase2_lrsweep.py`, `python experiments/phase2_tunedbase.py` | `results/phase2_mechanism/phase2_lrsweep.csv`, `…/phase2_tunedbase.csv` |
| Early-stopping and per-modality-head controls | `python experiments/phase2_earlystop.py`, `python experiments/phase2_heads.py` | `results/phase2_mechanism/phase2_earlystop.csv`, `…/phase2_heads.csv` |
| CREMA-D decoupled metrics (main paper Table I and Fig. 3a; CA paper Table IV and Fig. 6) | `python experiments/phase2_cremad_probe.py --seeds 5` | `results/phase2_mechanism/phase2_cremad_probe.csv` |
| CREMA-D with tuned SGD and the AdamW gradient arm | `python experiments/phase3_cremad_scoped.py` | `results/phase3_cremad/phase3_cremad_scoped.csv` |
| Positive control (audio noise) and SGD at α = 10 (main paper Fig. 3b) | `python experiments/phase3_controls.py` | `results/phase3_cremad/phase3_controls.csv` |
| OGM-GE and modality-specific learning-rate baselines | `python experiments/phase4_baselines.py` | `results/phase4_baselines/phase4_cremad.csv` |
| SGD stabilised by gradient clipping | `python experiments/phase4_sgd_stable.py` | `results/phase4_baselines/phase4_sgd_stable.csv` |
| UTD-MHAD replication | `python experiments/phase4_utd_decoupled.py` | `results/phase4_baselines/phase4_utd.csv` |

Each run also writes a `*_config.json` with the git commit, seeds and wall time.

### Forward-versus-gradient decomposition (5 seeds)

| What it produces | Command | Output |
|---|---|---|
| Synthetic, Adam | `python experiments/disentangle_scale_grad.py` | `results/disentangle/` |
| Synthetic, SGD | `python experiments/sgd_disentangle.py` | `results/sgd_disentangle/` |
| CREMA-D, Adam | `python experiments/cremad_disentangle.py` | `results/cremad_disentangle/` |
| Fusion robustness (sum, concatenation, gated) | `python experiments/fusion_robustness.py` | `results/fusion_robustness/` |

### Pre-registered programme E1 to E5 and H6

The earlier experiments (`e1_diagnostic.py`, `e2_causal.py`, `e3_cure.py`,
`e4_boundary.py`, `e4_extreme.py`, `e5_ogm.py`, `cremad_train_scale.py`,
`utd_train_scale.py`) each correspond to a plan in `docs/preregistration/` and a report in
`docs/experiments/`; their outputs are in `results/raw/`, `results/processed/`,
`results/tables/`, `results/e4/`, `results/e5/`, `results/cremad/` and `results/utd_mhad/`.

## 5. Figures, papers and website data

| Artifact | Command |
|---|---|
| Main paper figures (`fig_synth.pdf`, `fig_real.pdf`) | `python research_paper/scripts/make_paper_figs.py` |
| Earlier paper figures (`fig_disentangle.pdf`, `fig_fusion.pdf`, `fig_utd.pdf`) | `python research_paper/scripts/make_figs.py` |
| Main paper PDF | in `research_paper/`: `pdflatex main && pdflatex main` (the bibliography is embedded) |
| CA paper PDF | in `research_paper/`: `pdflatex ca_paper && pdflatex ca_paper` |
| Website evidence, architecture, thumbnails and training curves | `python scripts/export_site_evidence.py`, `python scripts/export_architecture.py`, `python scripts/export_architecture_media.py`, `python scripts/export_lab_trajectories.py` |
| Prediction Studio checkpoints | see [PREDICTION_STUDIO.md](PREDICTION_STUDIO.md) |

`export_site_evidence.py`, `export_architecture.py` and `export_lab_trajectories.py` also
have a `--check` mode, which `tests/test_site_artifacts.py` runs to confirm the generated
website data still matches `results/` and the checkpoints.
