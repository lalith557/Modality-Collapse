import type { ReactNode } from "react";
import { Tag } from "../../components/primitives";
import { STATUS_LABEL } from "../validation";
import type { Prediction, PredictionResponse } from "../types";
import { ProbabilityBars } from "./ProbabilityBars";

const STATUS_TONE = {
  compatible: "positive",
  potential_distribution_shift: "warning",
  low_quality: "warning",
  unsupported: "critical",
} as const;

const SEV_TONE = { unsupported: "critical", low_quality: "warning", info: "neutral" } as const;

const pct = (x: number) => `${(100 * x).toFixed(1)}%`;

function Section({ title, children, note }: { title: string; children: ReactNode; note?: string }) {
  return (
    <section className="card p-5">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="label !text-ink">{title}</h3>
        {note && <span className="text-2xs text-muted">{note}</span>}
      </div>
      {children}
    </section>
  );
}

function PredictionCard({ p, title }: { p: Prediction; title: string }) {
  const r = p.reference_accuracy;
  return (
    <div className="rounded-md border border-line p-4">
      <div className="label mb-1">{title}</div>
      <div className="flex items-baseline gap-2">
        <span className="font-display text-2xl font-semibold text-ink">{p.predicted_code}</span>
        {p.predicted_name && <span className="text-sm text-muted">{p.predicted_name}</span>}
      </div>
      <div className="mt-1 font-mono text-2xs text-muted">
        top probability {p.max_probability.toFixed(3)} · normalized entropy {p.normalized_entropy.toFixed(2)}
      </div>
      <div className="mt-3">
        <ProbabilityBars prediction={p} compact />
      </div>
      <p className="mt-3 text-2xs leading-relaxed text-muted">{p.description}</p>
      <p className="mt-1 font-mono text-2xs text-muted">
        held-out accuracy of this predictor: val {pct(r.validation_accuracy)} · test (evaluation-only){" "}
        {pct(r.test_accuracy_evaluation_only)} · chance {pct(r.chance)}
      </p>
    </div>
  );
}

function KV({ data }: { data: Record<string, unknown> }) {
  const fmt = (v: unknown) =>
    typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(3)) : Array.isArray(v) ? `[${v.length} values]` : String(v);
  return (
    <dl className="grid grid-cols-1 gap-x-6 gap-y-1 sm:grid-cols-2">
      {Object.entries(data)
        .filter(([, v]) => !Array.isArray(v) || (v as unknown[]).length <= 8)
        .map(([k, v]) => (
          <div key={k} className="flex justify-between gap-3 border-b border-line/60 py-1">
            <dt className="text-2xs text-muted">{k}</dt>
            <dd className="measure text-2xs">{Array.isArray(v) && v.length <= 8 ? `[${(v as number[]).map((x) => (typeof x === "number" ? +x.toFixed(2) : x)).join(", ")}]` : fmt(v)}</dd>
          </div>
        ))}
    </dl>
  );
}

function InertialPlot({ rows, labels }: { rows: number[][]; labels: string[] }) {
  const W = 256,
    H = 36;
  return (
    <div className="grid gap-2 sm:grid-cols-2">
      {rows.map((r, c) => {
        const m = Math.max(...r.map(Math.abs), 1e-6);
        const d = r.map((v, i) => `${i ? "L" : "M"}${((i / (r.length - 1)) * W).toFixed(1)},${(H / 2 - (v / m) * (H / 2 - 2)).toFixed(1)}`).join(" ");
        return (
          <figure key={c}>
            <svg viewBox={`0 0 ${W} ${H}`} className="h-9 w-full">
              <path d={d} className="fill-none stroke-forward" strokeWidth={1.2} />
            </svg>
            <figcaption className="font-mono text-2xs text-muted">
              {labels[c]} · z-scored, max |z| {m.toFixed(2)}
            </figcaption>
          </figure>
        );
      })}
    </div>
  );
}

export function ResultPanel({ r }: { r: PredictionResponse }) {
  const tone = STATUS_TONE[r.status];
  const multimodal = r.mode === "multimodal";
  const pre = r.preprocessed;
  return (
    <div className="space-y-5" data-testid="result-panel">
      <div
        role="status"
        className={`rounded-md border p-5 ${
          tone === "positive" ? "border-positive/40 bg-positive/[0.05]" : tone === "warning" ? "border-warning/40 bg-warning/[0.06]" : "border-critical/40 bg-critical/[0.05]"
        }`}
      >
        <div className="flex flex-wrap items-center gap-2">
          <Tag tone={tone}>{STATUS_LABEL[r.status]}</Tag>
          <span className="font-mono text-2xs text-muted">
            mode {r.mode} · request {r.request_id} · {r.total_ms.toFixed(0)} ms server time
          </span>
        </div>
        <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-ink">
          {r.status_reasons.map((s, i) => (
            <li key={i}>{s}</li>
          ))}
        </ul>
        {r.status === "unsupported" && (
          <p className="mt-3 text-sm font-medium text-ink">No prediction was made for this input.</p>
        )}
      </div>

      {r.primary && (
        <Section title={multimodal ? "Multimodal prediction (joint head)" : `${r.mode.replace("_", "-")} prediction`}
          note="A high probability is the model's output, not evidence that it is correct.">
          <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
            <div>
              <div className="flex items-baseline gap-3">
                <span className="font-display text-4xl font-semibold text-ink" data-testid="predicted-code">{r.primary.predicted_code}</span>
                {r.primary.predicted_name && <span className="text-lg text-muted">{r.primary.predicted_name}</span>}
              </div>
              <dl className="mt-3 space-y-1 font-mono text-2xs text-muted">
                <div>top probability: <span className="text-ink">{r.primary.max_probability.toFixed(3)}</span></div>
                <div>entropy: <span className="text-ink">{r.primary.entropy_nats.toFixed(3)} nats</span> (normalized {r.primary.normalized_entropy.toFixed(2)})</div>
                <div>source: <span className="text-ink">{r.primary.source}</span></div>
              </dl>
              <p className="mt-3 text-2xs leading-relaxed text-muted">{r.primary.description}</p>
            </div>
            <ProbabilityBars prediction={r.primary} />
          </div>
        </Section>
      )}

      {Object.keys(r.unimodal).length > 0 && multimodal && (
        <Section title="Modality comparison" note="Each unimodal prediction uses only that modality.">
          <div className="grid gap-3 md:grid-cols-3">
            {r.primary && <PredictionCard p={r.primary} title="Multimodal · joint head" />}
            {Object.entries(r.unimodal).map(([m, p]) => (
              <PredictionCard key={m} p={p} title={`${m}-only · frozen-encoder probe`} />
            ))}
          </div>
        </Section>
      )}

      <Section title="Compatibility / distribution shift"
        note="Mahalanobis distance on encoder representations; fit on train, threshold from validation.">
        {r.shift.length === 0 ? (
          <p className="text-sm text-muted">Not computed: the model was not run for this input.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[560px] text-left font-mono text-2xs">
              <thead className="text-muted">
                <tr>
                  <th className="py-1 pr-3 font-normal">detector</th>
                  <th className="py-1 pr-3 text-right font-normal">score</th>
                  <th className="py-1 pr-3 text-right font-normal">threshold</th>
                  <th className="py-1 pr-3 text-right font-normal">× threshold</th>
                  <th className="py-1 pr-3 font-normal">role</th>
                  <th className="py-1 text-right font-normal">false-flag rate (val / test)</th>
                </tr>
              </thead>
              <tbody>
                {r.shift.map((s) => (
                  <tr key={s.detector} className="border-t border-line/70">
                    <td className="py-1 pr-3 text-ink">{s.detector}</td>
                    <td className="py-1 pr-3 text-right text-ink">{s.score.toFixed(1)}</td>
                    <td className="py-1 pr-3 text-right">{s.threshold.toFixed(1)}</td>
                    <td className={`py-1 pr-3 text-right ${s.exceeds ? "text-warning" : "text-ink"}`}>{s.ratio_to_threshold.toFixed(2)}</td>
                    <td className="py-1 pr-3 text-muted">{s.used_for_status ? "sets status" : "diagnostic"}</td>
                    <td className="py-1 text-right text-muted">
                      {pct(s.validation_flag_rate)} / {pct(s.test_flag_rate_evaluation_only)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>

      <Section title="Input quality checks">
        {r.issues.length === 0 ? (
          <p className="text-sm text-muted">No quality issues were detected by the implemented checks.</p>
        ) : (
          <ul className="space-y-2">
            {r.issues.map((i, k) => (
              <li key={k} className="flex flex-wrap items-start gap-2 text-sm">
                <Tag tone={SEV_TONE[i.severity]}>{i.severity.replace("_", " ")}</Tag>
                <span className="font-mono text-2xs text-muted">[{i.modality}] {i.code}</span>
                <span className="w-full text-ink sm:w-auto sm:flex-1">{i.message}</span>
                {i.measured_false_flag_rate_test != null && i.severity !== "info" && (
                  <span className="w-full font-mono text-2xs text-muted">
                    this check fires on {pct(i.measured_false_flag_rate_test)} of in-distribution test clips
                  </span>
                )}
              </li>
            ))}
          </ul>
        )}
      </Section>

      <Section title="Preprocessed representation (what the model received)" note="Returned by the server after training-identical preprocessing.">
        <div className="space-y-4">
          {pre.audio_logmel && (
            <figure>
              <img src={pre.audio_logmel} alt="log-mel spectrogram fed to the audio encoder" className="w-full max-w-xl rounded border border-line [image-rendering:pixelated]" />
              <figcaption className="mt-1 font-mono text-2xs text-muted">
                log-mel [{pre.audio_tensor_shape?.join(", ")}] · 64 mel bins × 188 frames (3.0 s) · values {pre.audio_logmel_range?.map((x) => x.toFixed(1)).join(" … ")}
              </figcaption>
            </figure>
          )}
          {(pre.video_frames || pre.depth_frames) && (
            <figure>
              <div className="flex flex-wrap gap-1.5">
                {(pre.video_frames ?? pre.depth_frames)!.map((src, i) => (
                  <img key={i} src={src} alt={`model frame ${i + 1}`} className="h-16 w-16 rounded border border-line [image-rendering:pixelated]" />
                ))}
              </div>
              <figcaption className="mt-1 font-mono text-2xs text-muted">
                {pre.video_frames ? `video [${pre.video_tensor_shape?.join(", ")}] · centre crop, grayscale` : `depth [${pre.depth_tensor_shape?.join(", ")}] · per-clip min-max`}
              </figcaption>
            </figure>
          )}
          {pre.inertial_zscored && pre.inertial_channels && (
            <InertialPlot rows={pre.inertial_zscored} labels={pre.inertial_channels} />
          )}
        </div>
      </Section>

      <Section title="Input report">
        <div className="space-y-3">
          {Object.entries(r.input_report).map(([m, d]) => (
            <div key={m}>
              <div className="label mb-1">{m}</div>
              <KV data={d} />
            </div>
          ))}
        </div>
      </Section>

      <Section title="Processing trace" note={`${r.total_ms.toFixed(0)} ms total on the server`}>
        <ol className="space-y-1.5">
          {r.steps.map((s, i) => (
            <li key={i} className="flex flex-wrap items-baseline gap-x-3 font-mono text-2xs">
              <span className="w-5 text-muted">{i + 1}</span>
              <span className="w-44 text-ink">{s.name}</span>
              <span className="w-16 text-right text-ink">{s.duration_ms.toFixed(1)} ms</span>
              <span className="flex-1 text-muted">{s.detail}</span>
            </li>
          ))}
        </ol>
      </Section>

      <Section title="Model & checkpoint">
        <KV
          data={{
            checkpoint: r.model.checkpoint_id,
            sha256: String(r.model.checkpoint_sha256).slice(0, 16) + "…",
            architecture: r.model.architecture,
            parameters: r.model.parameters,
            condition: r.model.condition,
            experiment: r.model.experiment_id,
            "seed / epochs": `${r.model.seed} / ${r.model.epochs}`,
            "optimizer / lr / wd": `${r.model.optimizer} / ${r.model.learning_rate} / ${r.model.weight_decay}`,
            selection: r.model.checkpoint_selection,
            "parity with frozen experiment": r.model.parity_with_frozen_artifact?.exact_match ? "exact" : "NOT exact",
            "val accuracy": r.model.validation_metrics?.accuracy,
            "test accuracy (evaluation-only)": r.model.test_metrics_evaluation_only?.accuracy,
            device: r.model.device,
          }}
        />
      </Section>

      <Section title="Limitations">
        <ul className="list-disc space-y-1.5 pl-5 text-sm text-ink">
          {r.limitations.map((l, i) => (
            <li key={i}>{l}</li>
          ))}
        </ul>
        <p className="mt-3 text-2xs text-muted">{r.disclaimer}</p>
      </Section>
    </div>
  );
}
