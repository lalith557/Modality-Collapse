# PROJECT_STATUS.md — modality-scale-collapse

_Last updated: 2026-09-19_

## Environment

| Item | Value |
|---|---|
| OS | Windows 11 Home (10.0.26200) |
| Working dir | `C:\Users\Lalith Gona\HTML\Deep Learning Project 2` |
| Project root | `.\modality-scale-collapse` |
| Python | 3.13.12 (`C:\Program Files\Python313\python.exe`) |
| Package manager | pip 26.2.1 (no conda) |
| Git | 2.54.0 — **repo NOT yet initialized** in project root |
| Git LFS | 3.7.1 installed |

## GPU / CUDA / PyTorch  ✅ (experiments run on GPU)

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 **Laptop** GPU |
| VRAM | 8 GB (8188 MiB) — design target |
| Compute capability | sm_89 (8,9) |
| NVIDIA driver | 591.59 (CUDA driver 13.1) |
| PyTorch | **2.11.0+cu128**, `torch.cuda.is_available() == True` |
| torch CUDA runtime | 12.8 |
| Disk free (C:) | ~141 GB |

## Installed Python packages (verified)

`numpy 2.4.3`, `pandas 2.3.3`, `scikit-learn 1.8.0`, `scipy 1.17.1`,
`opencv 4.13.0`, `mediapipe 0.10.35`, `matplotlib 3.10.8`, `pyyaml 6.0.3`,
`tqdm 4.67.3`, `pytest 9.0.3`.

### Missing / mismatched (needed for later phases, NOT for the synthetic milestone)
- **`librosa` — MISSING.** Needed for CREMA-D log-mel features.
- **`torchvision` — MISSING.** Needed if we use any torchvision transforms/models.
- **`torchaudio 2.10.0+cpu` — CPU-ONLY BUILD, version-mismatched** with `torch 2.11.0+cu128`.
  Must be reinstalled as a matching CUDA build before the CREMA-D audio phase.

## Datasets (paths + availability)

### Synthetic — ✅ implemented, no download
`data/synthetic.py`. This is the controlled causal environment; it is the only
data used for the current milestone.

### CREMA-D — ⚠️ code/labels present, **media are Git-LFS pointer stubs**
- Path: `..\CREMA-D-master`
- `AudioWAV/` (7442 files), `AudioMP3/`, `VideoFlash/` — every media file is a
  **130-byte LFS pointer**, not real audio/video (`.gitattributes` routes
  `*.wav/*.mp3/*.flv` through LFS). CSV metadata + R/analysis scripts are real.
- **Blocker for the CREMA-D phase:** run `git lfs pull` inside `CREMA-D-master`
  (or otherwise fetch the ~5 GB media) before any feature extraction.

### INCLUDE-50 — ✅ official splits present, videos absent
- Path: `..\INCLUDE-master` (AI4Bharat pipeline code, 0.2 MB)
- `train_test_paths/include50_{train,val,test}.txt` — **official INCLUDE-50 splits present.**
- `label_maps/label_map_include50.json` present.
- Keypoint pipeline: `generate_keypoints.py` (MediaPipe). Videos themselves are
  NOT local; `dataset_script.txt` loads `ai4bharat/INCLUDE` from HuggingFace
  (~56.8 GB full) — **do not download until INCLUDE-50 experiments are justified.**

## Current milestone status

**Milestone 1 (synthetic E2 prototype): COMPLETE.** See `EXPERIMENT_LOG.md`.
Generator → tiny encoders → fusion → classifier → scale injection → rho →
masking → multi-seed scale sweep → plots, all on GPU; 12/12 tests pass.

## Immediate next step

1. `git init` + initial commit (scaffold + synthetic result).
2. E1 diagnostic experiment: formalize the early-rho → collapse predictor
   (the epoch-5 signal is already Spearman ≈ 0.98 vs final dominance).
3. E4 pilot: scale × learning-rate grid — the epoch-5-vs-epoch-40 finding
   (encoder compensates static scale) points to optimization as the key axis.
