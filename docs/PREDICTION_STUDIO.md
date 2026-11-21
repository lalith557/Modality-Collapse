# Prediction Studio

The Prediction Studio runs the study's real baseline checkpoints on files you upload:
audio and/or video for the CREMA-D emotion model, depth plus inertial recordings for the
UTD-MHAD action model. It applies exactly the preprocessing used in training, reports
per-modality outputs, and flags inputs that look unlike the training data.

It is a research prototype. The models are small and were trained from scratch on one
dataset each; held-out accuracy says nothing about accuracy on your recordings, and a
confident prediction can be wrong.

## Setup

All commands run from the repository root.

```bash
pip install -r requirements-studio.txt
cd website && npm install
```

### Start the backend

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health
```

Startup loads both checkpoint packages, verifies the SHA-256 of every file against
`metadata.json`, and warms up the models.

| Endpoint | Purpose |
|---|---|
| `GET /health`, `GET /models` | Status, device and loaded checkpoint IDs |
| `POST /predict/cremad/audio` | Audio only (audio encoder + its frozen-encoder probe) |
| `POST /predict/cremad/video` | Video only (video encoder + its probe) |
| `POST /predict/cremad/multimodal` | Audio + video through the joint head; `use_video_audio=true` takes the audio track from the video |
| `POST /predict/utd_mhad/multimodal` | Depth + inertial |
| `POST /preview/video` | Server-side frame preview for formats the browser cannot play (e.g. `.flv`) |

Interactive API docs are at http://127.0.0.1:8000/docs.

### Start the frontend

```bash
cd website && npm run dev
```

Open http://localhost:5173/#/studio. The API address defaults to `http://127.0.0.1:8000`
and can be changed with `VITE_STUDIO_API`. The backend accepts browser requests from
ports 5173 (dev server) and 4173 (`npm run preview`).

### Example request

```bash
curl -X POST http://127.0.0.1:8000/predict/cremad/multimodal \
  -F "audio=@../CREMA-D-full/AudioWAV/1001_DFA_SAD_XX.wav" \
  -F "video=@../CREMA-D-full/VideoFlash/1001_DFA_SAD_XX.flv"
```

### Rebuilding the checkpoints

The packages in `checkpoints/` are committed. To reproduce them (needs the datasets and
their caches, see [REPRODUCING.md](REPRODUCING.md)):

```bash
python scripts/build_studio_checkpoints.py --dataset all --force   # ~2 min CREMA-D, ~25 s UTD-MHAD on an RTX 4070
python scripts/calibrate_studio_quality.py                         # CREMA-D quality thresholds, ~10 min
```

The build trains the documented seed-0 baselines with the original experiment code and
checks parity: re-running the original evaluation on the new weights reproduces the frozen
seed-0 rows in `results/` to within 1e-16.

Each package holds `model.pth`, `config.yaml`, `metadata.json`, `label_mapping.json`,
`preprocessing.json`, `evaluation.json`, `probes.npz` and `ood_detector.npz`, plus
`normalization_stats.npz` (UTD-MHAD) or `quality_reference.json` (CREMA-D).

### Tests

```bash
python -m pytest tests/studio -q
cd website && npm test
```

The backend tests generate their external inputs in `samples/external/` on first run
(Windows text-to-speech and scikit-image photographs); `python scripts/make_external_samples.py`
creates them explicitly.

## Integrity rules

| Rule | How it is enforced |
|---|---|
| The checkpoints are the documented experiment | Trained by the original `phase2_cremad_probe.train_one` and `phase4_utd_decoupled.train` (seed 0, baseline arm, Adam lr 1e-3, weight decay 0, 30 / 60 epochs, fixed final epoch) and parity-checked against `results/`. |
| No training or selection on validation/test data | Models, unimodal probes and OOD reference statistics use the train split; OOD and quality thresholds use validation; test is only reported. |
| Baseline only | `ScaleInjector` is asserted disabled at load time. There is no inference-time α control. |
| Missing modalities are never simulated | Audio-only and video-only requests run only that encoder and its probe; nothing is zero-filled. |
| Only the project's own model files are loaded | Fixed package paths, SHA-256 check of every file, `torch.load(weights_only=True)`, `strict=True`, `.npz` loaded without pickle. |
| Uploads are not kept or logged | Each request uses a temporary directory that is deleted afterwards; file contents and names are not logged. |
| Confidence is not presented as reliability | The UI and API say so explicitly and report calibration error. |

## How a prediction is computed

### CREMA-D (6 emotions: anger, disgust, fear, happy, neutral, sad)

Audio, as in `data/crema_d.py::load_audio_logmel`:

1. Decode to PCM WAV (PyAV for non-WAV containers and for audio tracks inside videos), mix to mono, scale int16 by 1/32768.
2. Resample to 16 kHz with `torchaudio.functional.resample`.
3. Keep the first 3.0 s (zero-pad shorter clips): 48,000 samples.
4. 64-bin mel spectrogram (n_fft 1024, hop 256), `log(mel + 1e-6)`, time axis fixed at 188 frames.
5. Round-trip through float16, as the training cache did. Tensor `[1, 64, 188]`.

Video, as in `data/crema_d.py::load_video_frames`:

1. Decode with OpenCV and take 8 frames at `np.linspace(0, n-1, 8)` (floor).
2. Grayscale, centre square crop, resize to 64×64 with `INTER_AREA`, scale to [0, 1].
3. Float16 round-trip. Tensor `[8, 1, 64, 64]`.

Model: `CremadMultimodalModel` with 32,710 parameters. The audio CNN and the video CNN
(three 3×3 conv blocks each, applied per frame for video and averaged over time) produce
64-d representations that are summed and passed to a linear head. Unimodal predictions use
the paper's frozen-encoder linear probe (train-standardised features, Adam lr 0.05, weight
decay 1e-4, 400 full-batch steps), fitted on the train split and stored in `probes.npz`.

Split: actor-disjoint, stratified by sex, seed 42: 64 / 9 / 18 actors, 5234 / 738 / 1470 clips.

### UTD-MHAD (27 actions, reported as `a1` to `a27`)

The dataset folder and repository contain no action names, so actions are reported by ID.

- Depth (`.mat` variable `d_depth`, `(240, 320, T)`, millimetres, or an equivalent `.npy`):
  16 frames at `np.linspace(0, T-1, 16).round()`, resized to 32×32, per-clip min-max to [0, 1].
  Tensor `[16, 1, 32, 32]`.
- Inertial (`.mat` variable `d_iner` or a `T×6` CSV; accelerometer in g, gyroscope in deg/s):
  linear interpolation to 128 samples, per-channel z-score with the train statistics in
  `normalization_stats.npz`. Tensor `[6, 128]`.
- Model: `UTDMultimodalModel` with 35,467 parameters (depth CNN 16,160; inertial 1-D CNN
  17,552; head 1,755), sum fusion. Official cross-subject split: subjects 1, 3, 5 train (323),
  7 validation (108), 2, 4, 6, 8 test (430).

### Checks on every upload

- File validation by magic bytes, not by extension: wrong or unknown content returns 415,
  oversized files 413, corrupt files or missing streams 422, missing fields 400. A Git LFS
  pointer (the 130-byte stub left by a clone made without `git lfs pull`) returns 415 with
  code `lfs_pointer` and the size of the real file.
- Quality and domain checks: silence, no speech, too short, clipping and noise for audio;
  no face, dark, blurry, static, low resolution and too few frames for video; shape, units
  and NaN checks for UTD-MHAD.
- Distribution shift: class-conditional Mahalanobis distance on the encoder representations
  (joint and per modality), fitted on train with the threshold at the validation 95th percentile.
  It can only raise "potential distribution shift"; it never blocks a prediction on its own.

## Measured behaviour

### Held-out performance (for orientation only)

| Predictor | Validation | Test | Chance |
|---|---|---|---|
| CREMA-D joint head | 53.4% (ECE 0.050) | 50.8% (ECE 0.031) | 16.7% |
| CREMA-D audio probe / video probe | 56.1% / 18.2% | 51.8% / 19.3% | 16.7% |
| UTD-MHAD joint head | 50.9% (ECE 0.340) | 57.0% (ECE 0.265) | 3.7% |
| UTD-MHAD depth probe / inertial probe | 14.8% / 45.4% | 14.7% / 58.6% | 3.7% |

### False-alarm rates of the checks on in-distribution data

| Check | Validation (sets the threshold) | Test |
|---|---|---|
| CREMA-D joint shift detector | 5.0% | 9.5% |
| CREMA-D audio / video detector | 5.0% / 5.0% | 7.6% / 5.6% |
| UTD-MHAD joint / inertial detector | 5.0% / 5.0% | 2.8% / 2.3% |
| UTD-MHAD depth detector | 5.0% | 33.0% (diagnostic only, does not affect the status) |
| No speech | 2.2% | 2.2% |
| No face (Haar) | 0.0% | 0.07% |
| Blur | 1.1% | 13.5% |
| Clipping / noise | 1.1% / 1.1% | 1.0% / 0.9% |

### Behaviour on external and edge-case inputs

| Input | HTTP | Outcome |
|---|---|---|
| Text-to-speech as WAV / FLAC / Ogg / WebM | 200 | potential shift, anger 0.92 |
| Text-to-speech as M4A / MP3 | 200 | potential shift, anger 0.81 / 0.87 |
| 10 s of speech | 200 | truncated to the first 3.0 s (info note) |
| 0.3 s clip, 440 Hz tone, white noise, silence | 200 | unsupported, no prediction |
| Still face photograph as video | 200 | low quality (static frames) |
| Very dark video, video with no face | 200 | unsupported, no prediction |
| Face video with its own synthetic speech track | 200 | joint head anger 0.999 with joint shift score 9.9× the threshold |
| Video without an audio track, multimodal | 422 | `missing_audio_stream` |
| Text file as audio, WAV as video, WebM as depth | 415 | `unrecognized_format` / `wrong_content_type` |
| UTD-MHAD test trial a3 as `.npy` + `.csv` | 200 | compatible, predicts a15 (0.73), which is wrong |
| Same inertial data in m/s² | 200 | low quality (`inertial_units_accel`) |

The face-video case shows why confidence is never shown as reliability: a near-certain
softmax on an input far from the training data, flagged by the shift detector.

## Limitations

1. Accuracy on user recordings is unknown. Test accuracy covers unseen CREMA-D actors and
   UTD-MHAD subjects recorded under the datasets' own conditions.
2. The external test inputs are synthetic speech and still photographs; they exercise the
   whole system but say nothing about accuracy on natural recordings. No new Kinect + IMU
   recording was available for UTD-MHAD.
3. The video encoder is close to chance (video probe 19.3% against 16.7%).
4. UTD-MHAD confidence is poorly calibrated (validation ECE 0.34), so confident errors are common.
5. The shift detector's detection power is unknown: there is no labelled out-of-distribution
   data, so its threshold only fixes the in-distribution false-alarm rate.
6. The speech check cannot reliably separate speech from singing or rhythmic music, and the
   face check only detects frontal faces inside the centre crop. The blur check over-fires on
   new actors (13.5% on test) and only marks an input as low quality.
7. A prediction that exceeds its 120 s budget returns 504, but its worker thread finishes in
   the background; forward passes are serialised per model.
