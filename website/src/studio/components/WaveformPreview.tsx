import { useEffect, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import { previewVideo, StudioError } from "../api";
import type { VideoPreviewResponse } from "../types";

/**
 * Draws the user's own file, decoded by the browser's Web Audio API. Purely a preview:
 * the model input is produced on the server. The shaded region is the first 3.0 s, the
 * only part the model sees (same crop as training).
 */
export function WaveformPreview({ file, title }: { file: File; title: string }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [info, setInfo] = useState<{ duration: number } | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setInfo(null);
    setErr(null);
    const Ctor: typeof AudioContext | undefined = (window as any).AudioContext ?? (window as any).webkitAudioContext;
    if (!Ctor) {
      setErr("This browser has no Web Audio support for previews.");
      return;
    }
    const ctx = new Ctor();
    file
      .arrayBuffer()
      .then((buf) => ctx.decodeAudioData(buf))
      .then((audio) => {
        if (cancelled || !canvas.current) return;
        const c = canvas.current;
        const W = c.clientWidth * devicePixelRatio;
        const H = c.clientHeight * devicePixelRatio;
        c.width = W;
        c.height = H;
        const g = c.getContext("2d")!;
        const css = getComputedStyle(document.documentElement);
        const ink = `rgb(${css.getPropertyValue("--ink-rgb").trim().split(" ").join(",")})`;
        const fwd = `rgb(${css.getPropertyValue("--forward-rgb").trim().split(" ").join(",")})`;
        g.clearRect(0, 0, W, H);
        const used = Math.min(1, 3.0 / audio.duration);
        g.fillStyle = fwd;
        g.globalAlpha = 0.1;
        g.fillRect(0, 0, W * used, H);
        g.globalAlpha = 1;
        const data = audio.getChannelData(0);
        const step = Math.max(1, Math.floor(data.length / W));
        g.fillStyle = ink;
        for (let x = 0; x < W; x++) {
          let lo = 1,
            hi = -1;
          for (let i = x * step; i < Math.min(data.length, (x + 1) * step); i++) {
            const v = data[i];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
          }
          if (hi < lo) continue;
          g.globalAlpha = x / W <= used ? 0.85 : 0.3;
          g.fillRect(x, ((1 - hi) / 2) * H, 1, Math.max(1, ((hi - lo) / 2) * H));
        }
        g.globalAlpha = 1;
        setInfo({ duration: audio.duration });
      })
      .catch(() => !cancelled && setErr("Your browser cannot preview this audio format; the server will still try to decode it."))
      .finally(() => ctx.close());
    return () => {
      cancelled = true;
    };
  }, [file]);

  return (
    <div className="card p-3">
      <div className="mb-1.5 flex items-baseline justify-between">
        <span className="label">{title}</span>
        {info && (
          <span className="font-mono text-2xs text-muted">
            {info.duration.toFixed(2)} s{info.duration > 3 ? " · shaded = first 3.0 s used" : " · zero-padded to 3.0 s"}
          </span>
        )}
      </div>
      {err ? (
        <p className="py-4 text-2xs text-muted">{err}</p>
      ) : (
        <canvas ref={canvas} className="h-20 w-full" aria-label={`Waveform of ${file.name}`} />
      )}
    </div>
  );
}

/** Containers no mainstream browser can play natively; go straight to the server preview. */
const BROWSER_UNPLAYABLE = new Set(["flv", "avi"]);

type Mode =
  | { kind: "native" }
  | { kind: "server-loading"; reason: string }
  | { kind: "server"; reason: string; data: VideoPreviewResponse }
  | { kind: "failed"; reason: string; error: StudioError };

/**
 * Plays the user's file in the browser when it can. When it cannot (CREMA-D's own .flv,
 * AVI, codecs the browser lacks), the server decodes it with the model's own loader and
 * returns the 8 frames the model would sample, so the preview never silently stays black.
 */
export function VideoPreview({ file }: { file: File }) {
  const [url, setUrl] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>({ kind: "native" });
  const abort = useRef<AbortController | null>(null);

  const fallback = (reason: string) => {
    abort.current?.abort();
    const ctl = new AbortController();
    abort.current = ctl;
    setMode({ kind: "server-loading", reason });
    previewVideo(file, ctl.signal)
      .then((data) => !ctl.signal.aborted && setMode({ kind: "server", reason, data }))
      .catch((e: StudioError) => !ctl.signal.aborted && setMode({ kind: "failed", reason, error: e }));
  };

  useEffect(() => {
    const u = URL.createObjectURL(file);
    setUrl(u);
    const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
    const probe = file.type ? document.createElement("video").canPlayType(file.type) : "maybe";
    if (BROWSER_UNPLAYABLE.has(ext) || probe === "") {
      fallback(`Your browser cannot play .${ext || "this"} files`);
    } else {
      setMode({ kind: "native" });
    }
    return () => {
      abort.current?.abort();
      URL.revokeObjectURL(u);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file]);

  return (
    <div className="card p-3" data-testid="video-preview">
      <div className="label mb-1.5">Video preview (your file)</div>
      {mode.kind === "native" && url && (
        <video
          src={url}
          controls
          muted
          playsInline
          data-testid="native-video"
          className="max-h-56 w-full rounded bg-black/80 object-contain"
          onError={() => fallback("Your browser could not decode this video")}
          onLoadedMetadata={(e) => {
            // e.g. HEVC without a decoder: the audio plays but there is no picture
            if (e.currentTarget.videoWidth === 0) fallback("Your browser has no decoder for this video codec");
          }}
        />
      )}
      {mode.kind === "server-loading" && (
        <p className="flex items-center gap-2 py-4 text-2xs text-muted" data-testid="preview-loading">
          <Loader2 size={13} className="animate-spin" /> {mode.reason}; decoding preview frames on the server…
        </p>
      )}
      {mode.kind === "server" && <ServerFrames reason={mode.reason} data={mode.data} />}
      {mode.kind === "failed" && (
        <p className="py-3 text-2xs text-muted" role="status" data-testid="preview-failed">
          {mode.reason}, and the server preview failed:{" "}
          <span className="font-mono">{mode.error.code}</span> — {mode.error.message}
        </p>
      )}
      <p className="mt-1.5 text-2xs text-muted">The model sees only the centre square of 8 evenly spaced frames, in grayscale at 64×64.</p>
    </div>
  );
}

function ServerFrames({ reason, data }: { reason: string; data: VideoPreviewResponse }) {
  const c = data.crop;
  return (
    <div data-testid="server-preview">
      <p className="mb-2 text-2xs text-muted">
        {reason}, so these are the 8 frames the server decoded: the same frames the model samples. The dashed
        square is the centre crop it uses.
      </p>
      <div className="grid grid-cols-4 gap-1.5">
        {data.frames.map((f, k) => (
          <figure key={k} className="m-0">
            <div className="relative overflow-hidden rounded bg-black/80">
              <img src={f.image} alt={`Frame ${f.index}`} className="block w-full" />
              {c && (
                <div
                  className="pointer-events-none absolute border border-dashed border-forward"
                  style={{ left: `${c.x * 100}%`, top: `${c.y * 100}%`, width: `${c.w * 100}%`, height: `${c.h * 100}%` }}
                />
              )}
            </div>
            <figcaption className="mt-0.5 font-mono text-2xs text-muted">
              #{f.index}
              {f.time_s != null ? ` · ${f.time_s.toFixed(2)} s` : ""}
            </figcaption>
          </figure>
        ))}
      </div>
      <p className="mt-1.5 font-mono text-2xs text-muted" data-testid="server-preview-meta">
        {data.width}×{data.height} · {data.fps ? `${data.fps.toFixed(2)} fps · ` : ""}
        {data.n_frames} frames · {data.duration_s.toFixed(2)} s ·{" "}
        {data.audio_stream
          ? `audio ${data.audio_stream.codec} ${data.audio_stream.sample_rate} Hz`
          : "no audio track"}
      </p>
    </div>
  );
}
