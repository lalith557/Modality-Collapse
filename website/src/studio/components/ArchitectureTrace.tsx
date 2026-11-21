import type { PredictionResponse, TraceLayer } from "../types";

const MOD = {
  cremad: { a: "audio", b: "video", aEnc: "AudioCNN", bEnc: "VideoCNN" },
  utd_mhad: { a: "depth", b: "inertial", aEnc: "DepthCNN", bEnc: "InertialCNN" },
} as const;

function Spark({ values, label }: { values: number[]; label: string }) {
  if (!values?.length) return null;
  const m = Math.max(...values.map(Math.abs), 1e-9);
  const w = 2;
  return (
    <figure>
      <svg width={values.length * w} height={36} className="block" role="img" aria-label={`${label}: ${values.length} values`}>
        <line x1={0} x2={values.length * w} y1={18} y2={18} className="stroke-line" />
        {values.map((v, i) => (
          <rect
            key={i}
            x={i * w}
            y={v >= 0 ? 18 - (v / m) * 17 : 18}
            width={w - 0.4}
            height={Math.max(0.5, (Math.abs(v) / m) * 17)}
            className={v >= 0 ? "fill-forward" : "fill-gradient"}
          />
        ))}
      </svg>
      <figcaption className="mt-1 font-mono text-2xs text-muted">
        {label} · {values.length}-d · max |x| {m.toFixed(2)}
      </figcaption>
    </figure>
  );
}

function Stage({ title, subtitle, layers }: { title: string; subtitle?: string; layers: TraceLayer[] }) {
  return (
    <div className="card p-4">
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <span className="text-sm font-semibold text-ink">{title}</span>
        {subtitle && <span className="font-mono text-2xs text-muted">{subtitle}</span>}
      </div>
      {layers.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[520px] text-left font-mono text-2xs">
            <thead className="text-muted">
              <tr>
                <th className="py-1 pr-3 font-normal">layer</th>
                <th className="py-1 pr-3 font-normal">type</th>
                <th className="py-1 pr-3 font-normal">output shape</th>
                <th className="py-1 pr-3 text-right font-normal">mean ± std</th>
                <th className="py-1 text-right font-normal">[min, max]</th>
              </tr>
            </thead>
            <tbody>
              {layers.map((l, i) => (
                <tr key={i} className="border-t border-line/70 text-ink">
                  <td className="py-1 pr-3">{l.name}</td>
                  <td className="py-1 pr-3 text-muted">{l.type}</td>
                  <td className="py-1 pr-3">[{l.output_shape.join(", ")}]</td>
                  <td className="py-1 pr-3 text-right">
                    {l.mean.toFixed(3)} ± {l.std.toFixed(3)}
                  </td>
                  <td className="py-1 text-right text-muted">
                    [{l.min.toFixed(2)}, {l.max.toFixed(2)}]
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export function ArchitectureTrace({ response, inputs }: { response: PredictionResponse; inputs?: string[] }) {
  const t = response.trace;
  const mod = MOD[response.dataset];
  if (!t.executed) {
    return (
      <p className="text-sm text-muted">
        No forward pass was executed for request {response.request_id}: the input was classified as unsupported, so the
        model was not run.
      </p>
    );
  }
  const A = t.layers.filter((l) => l.name.startsWith("enc_a"));
  const B = t.layers.filter((l) => l.name.startsWith("enc_v"));
  const fusion = t.layers.filter((l) => l.name.startsWith("fusion"));
  const head = t.layers.filter((l) => l.name === "head");
  const pre = response.preprocessed;
  const aShape = (pre.audio_tensor_shape ?? pre.depth_tensor_shape) as number[] | undefined;
  const bShape = (pre.video_tensor_shape ?? pre.inertial_tensor_shape) as number[] | undefined;
  const reps = t.representations;
  return (
    <div className="space-y-4" data-testid="architecture-trace">
      <p className="text-2xs leading-relaxed text-muted">
        Recorded by PyTorch forward hooks on the server during request <span className="measure">{response.request_id}</span>
        {inputs?.length ? ` (${inputs.join(", ")})` : ""} using checkpoint{" "}
        <span className="measure">{response.model.checkpoint_id}</span>. Shapes and statistics are the real activations
        of that pass; your browser did not run the model.
      </p>
      <div className="grid gap-3 sm:grid-cols-2">
        <Stage
          title="Original input → preprocessed tensor"
          subtitle={[aShape && `${mod.a} [${aShape.join(", ")}]`, bShape && `${mod.b} [${bShape.join(", ")}]`].filter(Boolean).join(" · ")}
          layers={[]}
        />
        <Stage title="Output" subtitle={head[0] ? `logits [${head[0].output_shape.join(", ")}] → softmax` : "probe logits → softmax"} layers={[]} />
      </div>
      {A.length > 0 && <Stage title={`Encoder A · ${mod.aEnc} (${mod.a})`} subtitle="models/*_encoders.py" layers={A} />}
      {B.length > 0 && <Stage title={`Encoder B · ${mod.bEnc} (${mod.b})`} subtitle="models/*_encoders.py" layers={B} />}
      {fusion.length > 0 && <Stage title="Fusion" subtitle="z = h_A + h_B (models/fusion.py)" layers={fusion} />}
      {head.length > 0 && <Stage title="Classification head" subtitle="Linear 64 → classes" layers={head} />}
      {response.mode !== "multimodal" && (
        <p className="text-2xs text-muted">
          Unimodal request: only the observed modality's encoder ran; its 64-d representation was classified by that
          modality's frozen-encoder probe. The fusion layer and joint head were not used.
        </p>
      )}
      <div className="card flex flex-wrap gap-6 p-4">
        {Object.entries(reps).map(([k, v]) => (
          <Spark key={k} label={k} values={v} />
        ))}
      </div>
      <ul className="list-disc space-y-1 pl-5 text-2xs text-muted">
        {t.notes.map((n, i) => (
          <li key={i}>{n}</li>
        ))}
      </ul>
    </div>
  );
}
