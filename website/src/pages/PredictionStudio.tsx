import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Loader2, Play, RefreshCw, Square } from "lucide-react";
import { Callout, SectionHeader, Tag } from "../components/primitives";
import { API_BASE, StudioError, getHealth, predict } from "../studio/api";
import type { Dataset, HealthResponse, PredictionResponse } from "../studio/types";
import { planCremad, preValidate } from "../studio/validation";
import { UploadSlot } from "../studio/components/UploadSlot";
import { VideoPreview, WaveformPreview } from "../studio/components/WaveformPreview";
import { ResultPanel } from "../studio/components/ResultPanel";
import { ArchitectureTrace } from "../studio/components/ArchitectureTrace";
import { useStudio } from "../studio/StudioContext";

type Backend = { state: "checking" } | { state: "online"; health: HealthResponse } | { state: "offline"; error: StudioError };
type Run = { state: "idle" } | { state: "loading"; started: number } | { state: "error"; error: StudioError } | { state: "done"; response: PredictionResponse };

export function PredictionStudio() {
  const [backend, setBackend] = useState<Backend>({ state: "checking" });
  const [params] = useSearchParams();
  // ?dataset=utd_mhad preselects the model (e.g. when arriving from the Model Architecture page)
  const [dataset, setDataset] = useState<Dataset>(params.get("dataset") === "utd_mhad" ? "utd_mhad" : "cremad");
  const [audio, setAudio] = useState<File | null>(null);
  const [video, setVideo] = useState<File | null>(null);
  const [useVideoAudio, setUseVideoAudio] = useState(true);
  const [depth, setDepth] = useState<File | null>(null);
  const [inertial, setInertial] = useState<File | null>(null);
  const [run, setRun] = useState<Run>({ state: "idle" });
  const [elapsed, setElapsed] = useState(0);
  const abort = useRef<AbortController | null>(null);
  const { setLast } = useStudio();

  const checkBackend = useCallback(() => {
    setBackend({ state: "checking" });
    getHealth()
      .then((health) => setBackend({ state: "online", health }))
      .catch((error: StudioError) => setBackend({ state: "offline", error }));
  }, []);
  useEffect(checkBackend, [checkBackend]);

  useEffect(() => {
    if (run.state !== "loading") return;
    const t = setInterval(() => setElapsed((performance.now() - run.started) / 1000), 100);
    return () => clearInterval(t);
  }, [run]);

  const errs = {
    audio: preValidate(audio, "audio"),
    video: preValidate(video, "video"),
    depth: preValidate(depth, "depth"),
    inertial: preValidate(inertial, "inertial"),
  };
  const plan = planCremad(audio, video, useVideoAudio);
  const modelLoaded = backend.state === "online" && backend.health.models[dataset]?.loaded;

  const blocker = useMemo(() => {
    if (backend.state !== "online") return "The prediction backend is not reachable.";
    if (!modelLoaded) return `The ${dataset} checkpoint is not loaded on the server.`;
    if (dataset === "cremad") {
      if (plan.kind === "none") return "Add an audio file, a video file, or both.";
      if (errs.audio || errs.video) return "Fix the highlighted file problem first.";
    } else {
      if (!depth || !inertial) return "UTD-MHAD needs both a depth file and an inertial file.";
      if (errs.depth || errs.inertial) return "Fix the highlighted file problem first.";
    }
    return null;
  }, [backend, modelLoaded, dataset, plan.kind, errs.audio, errs.video, errs.depth, errs.inertial, depth, inertial]);

  const inputsLabel = () =>
    dataset === "cremad"
      ? [audio?.name, video?.name, plan.kind === "multimodal" && plan.audioSource === "video_track" ? "audio: video's own track" : undefined].filter(Boolean) as string[]
      : ([depth?.name, inertial?.name].filter(Boolean) as string[]);

  async function onRun() {
    if (blocker || run.state === "loading") return;
    const ctl = new AbortController();
    abort.current = ctl;
    setElapsed(0);
    setRun({ state: "loading", started: performance.now() });
    try {
      let response: PredictionResponse;
      if (dataset === "cremad") {
        if (plan.kind === "none") throw new StudioError(0, "no_input", "Add an audio file, a video file, or both.");
        const fields =
          plan.kind === "audio"
            ? { audio: audio! }
            : plan.kind === "video"
              ? { video: video! }
              : plan.audioSource === "file"
                ? { video: video!, audio: audio! }
                : { video: video!, use_video_audio: "true" };
        response = await predict(plan.path, fields, ctl.signal);
      } else {
        response = await predict("/predict/utd_mhad/multimodal", { depth: depth!, inertial: inertial! }, ctl.signal);
      }
      setRun({ state: "done", response });
      setLast({ response, inputs: inputsLabel() });
    } catch (e) {
      setRun({ state: "error", error: e as StudioError });
    } finally {
      abort.current = null;
    }
  }

  const planText =
    dataset === "utd_mhad"
      ? "Multimodal (joint head) on depth + inertial, plus each modality's own probe."
      : plan.kind === "multimodal"
        ? `Multimodal (joint head) on audio from ${plan.audioSource === "file" ? "your audio file" : "the video's own audio track"} + video, plus audio-only and video-only probes for comparison.`
        : plan.kind === "audio"
          ? "Audio-only: the audio encoder plus its frozen-encoder probe. Video is not used or simulated."
          : plan.kind === "video"
            ? "Video-only: the video encoder plus its frozen-encoder probe. Audio is not used or simulated."
            : "Nothing to run yet.";

  return (
    <div className="space-y-8">
      <SectionHeader
        eyebrow="Prediction Studio"
        title="Run the real models on your own recordings"
        subtitle="Upload new inputs; the server applies the exact training preprocessing, runs the saved baseline checkpoints and returns their actual outputs with quality and distribution-shift checks."
      />

      <Callout title="Research prototype — read before use" tone="warning">
        These are small models trained from scratch on one dataset each (CREMA-D acted emotion; UTD-MHAD actions).
        They are not an emotion- or action-recognition system. Held-out accuracy on the original test sets does not
        tell you how they behave on your recordings, a confident prediction can be wrong, and nothing here should be
        used to make decisions about people.
      </Callout>

      <div className="flex flex-wrap items-center gap-3 text-2xs" data-testid="backend-status">
        {backend.state === "checking" && (
          <span className="flex items-center gap-2 text-muted">
            <Loader2 size={14} className="animate-spin" /> Checking the backend at {API_BASE}…
          </span>
        )}
        {backend.state === "online" && (
          <>
            <Tag tone="positive">Backend online</Tag>
            <span className="font-mono text-muted">
              {API_BASE} · {backend.health.device} ·{" "}
              {Object.entries(backend.health.models)
                .map(([k, m]) => (m.loaded ? m.checkpoint_id : `${k}: not loaded`))
                .join(" · ")}
            </span>
          </>
        )}
        {backend.state === "offline" && (
          <>
            <Tag tone="critical">Backend offline</Tag>
            <span className="text-muted">{backend.error.message}</span>
            <code className="rounded bg-black/[0.04] px-2 py-0.5 font-mono">uvicorn backend.main:app --port 8000</code>
            <button type="button" className="btn-ghost !py-1" onClick={checkBackend}>
              <RefreshCw size={13} /> Retry
            </button>
          </>
        )}
      </div>

      <div className="grid gap-8 xl:grid-cols-[minmax(0,420px)_minmax(0,1fr)]">
        <div className="space-y-5">
          <div>
            <div className="mb-2 flex items-baseline justify-between gap-2">
              <div className="label">Dataset / model</div>
              <Link
                to={`/architecture?dataset=${dataset}`}
                className="text-2xs text-forward underline-offset-2 hover:underline"
                data-testid="inspect-architecture"
              >
                Inspect model architecture →
              </Link>
            </div>
            <div className="inline-flex flex-wrap gap-1.5" role="radiogroup" aria-label="Dataset">
              {(
                [
                  ["cremad", "CREMA-D · emotion (audio + video)"],
                  ["utd_mhad", "UTD-MHAD · action (depth + inertial)"],
                ] as const
              ).map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  role="radio"
                  aria-checked={dataset === id}
                  data-active={dataset === id}
                  className="seg border border-line"
                  onClick={() => {
                    setDataset(id);
                    setRun({ state: "idle" });
                  }}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {dataset === "cremad" ? (
            <>
              <UploadSlot role="audio" label="Audio" hint="Drop or choose a speech recording" file={audio} error={errs.audio} onChange={setAudio} />
              <UploadSlot role="video" label="Video" hint="Drop or choose a close-up video of a speaking face" file={video} error={errs.video} onChange={setVideo} />
              {video && !audio && (
                <label className="flex items-start gap-2 text-sm text-ink">
                  <input type="checkbox" className="mt-1" checked={useVideoAudio} onChange={(e) => setUseVideoAudio(e.target.checked)} />
                  <span>
                    Use the video's own audio track for multimodal inference
                    <span className="block text-2xs text-muted">Unchecked: video-only inference. The server rejects the request if the video has no audio stream.</span>
                  </span>
                </label>
              )}
            </>
          ) : (
            <>
              <UploadSlot role="depth" label="Depth" hint="Kinect depth stack: UTD .mat (d_depth, H×W×T, mm) or .npy (T×H×W)" file={depth} error={errs.depth} required onChange={setDepth} />
              <UploadSlot role="inertial" label="Inertial" hint="T×6 IMU signal: acc x/y/z (g), gyro x/y/z (deg/s) as .csv or UTD .mat" file={inertial} error={errs.inertial} required onChange={setInertial} />
              <p className="text-2xs leading-relaxed text-muted">
                An ordinary RGB video cannot stand in for either input: the model was trained on Kinect depth maps and a
                wrist-worn IMU signal. Classes are reported by UTD-MHAD action ID (a1–a27); action names are not stored
                in the project.
              </p>
            </>
          )}

          <div className="rounded-md border border-line bg-black/[0.015] p-3 text-2xs text-ink" data-testid="plan">
            <span className="label mr-2">Will run</span>
            {planText}
          </div>

          <div className="flex items-center gap-3">
            <button type="button" className="btn-primary disabled:cursor-not-allowed disabled:opacity-40" disabled={!!blocker || run.state === "loading"} onClick={onRun}>
              {run.state === "loading" ? <Loader2 size={15} className="animate-spin" /> : <Play size={15} />} Run prediction
            </button>
            {run.state === "loading" && (
              <button type="button" className="btn-ghost" onClick={() => abort.current?.abort()}>
                <Square size={13} /> Cancel
              </button>
            )}
          </div>
          {blocker && <p className="text-2xs text-muted" data-testid="blocker">{blocker}</p>}

          {dataset === "cremad" && (
            <div className="space-y-3">
              {audio && <WaveformPreview file={audio} title="Audio waveform (your file)" />}
              {!audio && video && useVideoAudio && <WaveformPreview file={video} title="Waveform of the video's audio track" />}
              {video && <VideoPreview file={video} />}
            </div>
          )}
        </div>

        <div className="min-w-0 space-y-6" aria-live="polite">
          {run.state === "idle" && (
            <div className="card flex min-h-[220px] flex-col items-center justify-center p-8 text-center" data-testid="empty-state">
              <div className="label mb-2">No prediction yet</div>
              <p className="max-w-md text-sm text-muted">
                Choose a model, add inputs and press Run. Results, the preprocessed tensors and a layer-by-layer trace
                of the real forward pass will appear here.
              </p>
            </div>
          )}
          {run.state === "loading" && (
            <div className="card flex min-h-[220px] flex-col items-center justify-center gap-3 p-8" data-testid="loading-state">
              <Loader2 size={22} className="animate-spin text-forward" />
              <div className="text-sm text-ink">Uploading and running on the server…</div>
              <div className="font-mono text-2xs text-muted">{elapsed.toFixed(1)} s elapsed</div>
            </div>
          )}
          {run.state === "error" && (
            <div role="alert" className="rounded-md border border-critical/40 bg-critical/[0.05] p-5" data-testid="error-state">
              <div className="flex flex-wrap items-center gap-2">
                <Tag tone="critical">{run.error.status ? `HTTP ${run.error.status}` : "Request failed"}</Tag>
                <span className="font-mono text-2xs text-muted">
                  {run.error.code}
                  {run.error.field ? ` · field ${run.error.field}` : ""}
                  {run.error.requestId ? ` · request ${run.error.requestId}` : ""}
                </span>
              </div>
              <p className="mt-2 text-sm text-ink">{run.error.message}</p>
            </div>
          )}
          {run.state === "done" && (
            <>
              <ResultPanel r={run.response} />
              <section className="card p-5">
                <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
                  <h3 className="label !text-ink">Your input through the model</h3>
                  <Link to="/methodology#trace" className="text-2xs text-forward underline-offset-2 hover:underline">
                    Open on the architecture page →
                  </Link>
                </div>
                <ArchitectureTrace response={run.response} inputs={inputsLabel()} />
              </section>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
