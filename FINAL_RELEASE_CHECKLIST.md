# FINAL RELEASE CHECKLIST

_Pre-commit verification for the disentanglement program + paper package. Frozen experiments (E1–E5,
CREMA-D R1–R3/H6, UTD) NOT modified._

## Pre-commit verification
- [x] `git status` reviewed; only the disentanglement program + paper artifacts are new/uncommitted.
- [x] No raw media staged (no `.wav/.flv/.mp4/.mat`; verified `find results/... -type f`).
- [x] No feature caches staged (`data/cremad/cache/`, `data/utd_mhad/cache/`, `*.npy` gitignored;
      `git check-ignore` confirms).
- [x] No path-laden manifests staged (`results/*/utd_manifest.csv`, `results/cremad/cremad_manifest.csv`
      gitignored / not added).
- [x] No credentials / tokens (no secrets in code or configs; only local absolute paths in gitignored
      manifests).
- [x] No temporary files (scratchpad is outside the repo; tool-result PDFs are outside the repo).
- [x] New result dirs are small (160K–900K each; CSV/JSON/PNG only).
- [x] Scripts are deterministic (fixed seeds [0–4]; `set_seed(..., deterministic=True)`; cudnn
      deterministic).
- [x] Each experiment script runs an autograd/data sanity gate and refuses on failure.
- [x] Result paths in RESULTS_INDEX.md / FIGURE_REPRODUCTION.md match on-disk files.
- [x] Figures regenerate from saved results (verified for the normalization figure; scripts read CSV/JSON).
- [x] All headline numbers trace to `results/**` (EXPERIMENT_FREEZE_AUDIT.md; re-verified 2026-09-21).

## Commit structure (Phase 19)
- **COMMIT 1 — experiments+results:** `experiments/{disentangle_scale_grad,sgd_disentangle,
  cremad_disentangle,fusion_robustness}.py`; `PREREGISTRATION_{SCALE_GRADIENT_DISENTANGLE,SGD_DISENTANGLE,
  CREMAD_DISENTANGLE,FUSION_ROBUSTNESS}.md`; `EXPERIMENT_REPORT_*` for those; `results/{disentangle,
  sgd_disentangle,cremad_disentangle,fusion_robustness}/`.
- **COMMIT 2 — paper artifacts:** `PAPER_DRAFT.md`, `PAPER_DRAFT_FINAL.md`, `CLAIM_EVIDENCE_MATRIX.md`,
  `MANUSCRIPT_CLAIM_AUDIT.md`, `FINAL_CLAIM_LOCK.md`, `EXPERIMENT_FREEZE_AUDIT.md`, `RESULTS_INDEX.md`,
  `SUBMISSION_SUMMARY.md`.
- **COMMIT 3 — reproducibility/figures:** `FIGURE_REPRODUCTION.md`, `FINAL_RELEASE_CHECKLIST.md`.

## Post-commit
- [ ] `git log --oneline -6` shows the three new commits atop `4465ebc`.
- [ ] `git status` clean.
- [ ] Frozen commits (`b7a5295`, `c780d60`, `e35b2bd`, …) unchanged.
- [ ] No push (no remote / not requested).
