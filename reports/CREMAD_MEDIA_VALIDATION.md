# CREMA-D — media acquisition & validation report

_Task: unblock CREMA-D media, validate integrity, run ONLY a smoke test, then STOP._
_E1–E5 frozen and unmodified. No R1/R2/R3 run. Timestamp: 2026-09-20._

## DATA STATUS — **usable**
- **Source:** official repository `https://github.com/CheyneyComputerScience/CREMA-D`
  (HEAD `1658cd342dff90010aa843eaeebd53610a08b1dc`).
- **Acquisition method:** `git clone --depth 1 --config GIT_LFS_SKIP_SMUDGE=1`
  (pointers only), then `git lfs install --local` + `git lfs pull --include="AudioWAV/*"`
  and `--include="VideoFlash/*"`. **Git LFS was accessible (not over quota).**
  Location (outside the git repo, not committed): `../CREMA-D-full/`.
- **Provenance check:** LFS oid of `AudioWAV/1001_DFA_ANG_XX.wav`
  (`c6f94b2592487c7556bd86df7df6e8f4420d1b9c11414aeec9d8e5e640b51495`) matches the
  pointer stub in the pre-existing `CREMA-D-master` — same dataset.
- **Integrity audit** (`results/cremad/cremad_validation.json`):
  - Structure: **7442 clips · 91 actors · 6 emotions · 12 sentences** → IDENTITY_OK.
  - Media categories: **audio present 7442/7442, video present 7442/7442**; 0 stub, 0
    zero-byte, 0 missing, 0 duplicate clip IDs; manifest↔disk consistent (0 either way).
  - Stratified sample (72 clips, emotion×split): audio RIFF signature 72/72, video FLV
    signature 72/72, audio decoded 72/72, video decoded 72/72, **0 decode failures**.
  - Audio: sample rate 16000 Hz (all), duration 1.4–3.64 s (mean 2.44 s).
  - Video: 42–109 frames (mean 73). → **MEDIA_OK.**

## SPLIT STATUS — **preserved (frozen)**
Manifest regenerated to point at the real media; the deterministic actor-disjoint
split is **byte-identical** to the frozen one (seed 42, sex-stratified):
- train **64 actors / 5234 clips**, val **9 / 738**, test **18 / 1470**.
- **Actor overlap between splits: 0** (asserted).

## PIPELINE STATUS — **implemented & validated on real media**
- **Audio loading:** WAV (scipy) → 16 kHz mono, 3.0 s → log-mel (torchaudio, 64 mel,
  n_fft 1024, hop 256) → `[1, 64, 188]`. ✓
- **Video loading:** .flv (OpenCV/FFmpeg) → 8 uniform frames, center-crop 64×64 gray
  → `[8, 1, 64, 64]`. ✓
- **Fusion / normalization / scale intervention:** `CremadMultimodalModel` mirrors the
  synthetic `MultimodalModel` interface, reusing the frozen ρ diagnostic, masking
  contribution metrics, per-modality z-score, and fusion-boundary `ScaleInjector`
  unchanged. ✓
- **Capacity control:** audio encoder 16160 params = video encoder 16160 params (matched). ✓
- **Feature-norm & gradient logging:** ✓ (ρ, per-encoder grad norms).

## SMOKE TEST — **PASSED**
36 real clips (24 train / 12 val, actor-disjoint), 5 train steps + eval:
- actor leakage in subset: **0**; loss 1.850 → 1.783 (decreasing); checkpoint
  save/reload: OK; **GPU peak 244.6 MB**; runtime 4.1 s; no NaN/errors.
- **Scale-injection (R3 mechanism) verified:** α_audio=5 on frozen reps → ρ 2.28 → 11.38
  (**exactly ×5**), confirming feature-magnitude can be manipulated independently of
  information on real representations.
- **[OBS, descriptive only — not a result]** even at init/early, a natural
  audio/video feature-scale disparity is present (ρ≈2.28: audio norm 2.07 vs video
  0.91; encoder grad-ratio ≈5.5). This is a smoke observation, **not** an R1 finding;
  no scientific claim is made here.

## SCIENTIFIC STATUS
R1/R2/R3 are now **technically ready** to run under the frozen `PREREGISTRATION_CREMAD.md`
with no further design choices. **They have NOT been run** (per the stop condition).
No causal/observational scientific claim is made yet.

## Overall classification: **READY — media validated; smoke test passed**

## NEXT ACTION
"CREMA-D media and pipeline are validated. Frozen R1–R3 can now be executed." — pending
external review / go-ahead. Do not auto-run R1–R3, INCLUDE-50, or paper writing.

_Raw media are NOT committed (dataset licensing/size; `.gitignore` excludes media and
`CREMA-D-full/` is outside the repo). Committed: acquisition/validation scripts,
manifest, validation JSON, pipeline code, smoke, this report._
