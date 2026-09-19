# CREMA-D REPORT — real-world validation (BLOCKED: media absent)

_Preregistration: `PREREGISTRATION_CREMAD.md` (frozen before results). E1–E5 unmodified._
_Labels: **[OBS]** observed · **[STAT]** statistical · **[MECH]** mechanistic · **[SPEC]** speculation._

> **Status: the causal experiments (R1–R3) were NOT run** because the CREMA-D audio
> and video media are **not present** on disk (git-LFS pointer stubs). Everything that
> is media-independent — audit, actor-disjoint split, manifest, and the frozen
> protocol — is complete. Classification: **CREMA-E (INVALID — data availability)**.

## 1. Executive summary
The local CREMA-D copy contains **only 130-byte git-LFS pointer stubs** for all 7442
audio and 7442 video files; the folder has **no `.git`**, so the media cannot be
fetched locally (`git lfs pull` needs a repo + remote). No audio or video can be
encoded, so R1 (observational), R2 (normalization) and R3 (controlled scale
intervention) cannot run. I did **not** fabricate results. The dataset **structure**
is fully audited and healthy; a frozen **actor-disjoint** split and manifest are
produced and committed, and the full experimental protocol is pre-registered, so
R1–R3 can run immediately once real media are provided.

## 2. Dataset and actor split  [OBS]
7442 clips · 91 actors · 6 emotions (ANG/DIS/FEA/HAP/SAD = 1271 each, NEU = 1087) ·
12 sentences · demographics 43 F / 48 M. **Actor-disjoint** split (stratified by sex,
seed 42, 70/10/20): **train 64 actors / 5234 clips · val 9 / 738 · test 18 / 1470**;
**0 actors span splits (asserted)**. Emotion is balanced across splits (see manifest).
`data/cremad/cremad_manifest.csv`, `results/cremad/cremad_manifest.csv`.

## 3. Preregistration  [OBS]
`PREREGISTRATION_CREMAD.md` (frozen): hypotheses, preprocessing (audio log-mel 16 kHz
/64 mel; video 8 frames/64×64), comparable-capacity CNN encoders, sum/concat fusion,
per-modality z-score, R1/R2/R3, metrics, leakage controls, classification tree.

## 4. Preprocessing  [BLOCKED]
Specified and frozen (Phase 4 / preregistration) but **not executed** — no media to
process. Media size per stub (declared): ~71 KB audio each (≈0.5 GB total), video
several GB. On disk: ~0.9 MB of pointer text per modality.

## 5. Architecture  [OBS, spec only]
Matched-capacity small CNN encoders (audio 2-D CNN over log-mel; video per-frame
2-D CNN + temporal mean-pool), rep_dim 64, sum fusion + linear head, per-modality
z-score before fusion (reusing `models/` + `interventions/`). Not instantiated
(depends on real feature shapes).

## 6. QC  [OBS]
- Git tree clean; E1–E5 commits intact (frozen).
- Manifest QC: 7442 clips parsed from real filenames; 91 actors; **actor-disjoint
  split verified (0 overlap)**; emotion×split balanced.
- **Media QC: audio_present = 0/7442, video_present = 0/7442** (all `lfs_stub`).
  This is the blocker.

## 7. Natural feature-scale analysis  [BLOCKED — needs media]
## 8. Unimodal baselines  [BLOCKED]
## 9. Multimodal baseline  [BLOCKED]
## 10. Normalization intervention (R2)  [BLOCKED]
## 11. Controlled scale intervention (R3)  [BLOCKED]
## 12. Normalization × scale intervention  [BLOCKED]
## 13. Gradient analysis  [BLOCKED]
## 14. Training dynamics  [BLOCKED]
## 15. Statistical analysis  [BLOCKED]
Sections 7–15 require encoded audio/video features and cannot be produced without the
media. No numbers are invented for them.

## 16. Confound analysis  [OBS, pre-run]
Capacity control (matched encoder budget), actor-disjoint split, train-only norm
stats, and R3-on-frozen-reps are pre-registered specifically to control the
confounds in Phase 15; they can only be evaluated once R1–R3 run.

## 17. Negative results  [OBS]
The only current negative is logistical, not scientific: **the media are absent**,
so no transfer test could be performed. This is reported rather than worked around.

## 18. Limitations  [OBS]
No real-data evidence yet. The synthetic mechanism (E1–E5) remains untested outside
the synthetic environment. Obtaining CREMA-D media via GitHub LFS may hit the
dataset's LFS bandwidth quota; an alternate authoritative local copy may be needed.

## 19. Comparison with synthetic findings  [n/a]
Deferred — requires R1–R3.

## 20. What CREMA-D establishes so far  [OBS]
Only that the dataset structure is intact and a rigorous, leakage-controlled,
actor-disjoint experimental protocol is ready. **No mechanistic claim** is or can be
made from the current state.

## 21. What CREMA-D does NOT establish  [OBS]
Nothing about real-world transfer of the feature-scale mechanism — the causal test
did not run.

## 22. Classification: **CREMA-E — INVALID (data availability)**
Leakage/preprocessing are controlled by design, but the **data-quality precondition
fails** (media absent). Per the decision tree, methodology/data must be fixed before
continuing.

## 23. GO / NO-GO for INCLUDE-50: **NO-GO / STOP**
Do **not** proceed to INCLUDE-50 (which has the same media-availability risk and is
even larger). Resolve CREMA-D media first. Recommended resolution (needs user
authorization for a multi-GB download):
1. `git clone https://github.com/CheyneyComputerScience/CREMA-D.git` then
   `git lfs install && git lfs pull` (may fail on LFS quota), **or**
2. provide a local copy of the real AudioWAV/VideoFlash (or a HuggingFace mirror), **or**
3. point the manifest at an existing media directory.
Once `cremad_manifest.py` reports `media_present > 0`, R1–R3 run under the frozen
protocol with no further design choices.

---
_Machine-readable produced now: `data/cremad/cremad_manifest.csv`,
`results/cremad/cremad_manifest.csv` (clip_id, actor_id, sentence, emotion, label,
level, split, audio_path, video_path, audio_status, video_status,
preprocessing_status). R1–R3 CSVs (`cremad_primary`, `cremad_scale_intervention`,
`cremad_aggregated`) and `cremad_config.json` will be produced when media are available._
