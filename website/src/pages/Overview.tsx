import { useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, GitFork, Sliders, Database, Lightbulb } from "lucide-react";
import { SectionHeader, MetricBlock, Card } from "../components/primitives";
import { LegendSwatch } from "../components/ScientificFigure";
import { ContributionScatter } from "../charts/ContributionScatter";
import { C } from "../charts/theme";
import { headline, cremad, findings } from "../data/research";

type ModFilter = "both" | "audio" | "video";
type ScaleFilter = "all" | "1" | "5" | "10";

const FINDING_ROUTES: Record<string, string> = {
  "01": "/disentanglement",
  "02": "/optimizer",
  "03": "/optimizer",
  "04": "/real-world",
};

const FINDING_ACCENTS: Record<string, { color: string; textClass: string }> = {
  "01": { color: C.forward, textClass: "text-forward" },
  "02": { color: C.forward, textClass: "text-forward" },
  "03": { color: C.gradient, textClass: "text-gradient" },
  "04": { color: C.neutral, textClass: "text-neutral" },
};

const SELECT_CLS =
  "h-7 rounded border border-line bg-canvas px-2 font-mono text-2xs text-ink " +
  "focus:outline-none focus:ring-1 focus:ring-ink/20 cursor-pointer";

function SectionMark({ n, label }: { n: string; label: string }) {
  return (
    <div className="mb-5 flex items-center gap-3">
      <span className="shrink-0 font-mono text-[0.56rem] uppercase tracking-[0.2em] text-muted/40">{n}</span>
      <span className="h-px w-5 shrink-0 bg-line" />
      <span className="label">{label}</span>
    </div>
  );
}

export function Overview() {
  const [modFilter, setModFilter] = useState<ModFilter>("both");
  const [scaleFilter, setScaleFilter] = useState<ScaleFilter>("all");

  const modalities: ("audio" | "video")[] =
    modFilter === "both" ? ["audio", "video"] : [modFilter];
  const scales: number[] =
    scaleFilter === "all" ? [1, 5, 10] : [Number(scaleFilter)];

  return (
    <div className="space-y-12">
      <SectionHeader
        eyebrow="Research overview"
        title="Research Overview"
        subtitle="What was studied, how broad the evidence is, what the main data shows, and where to go next."
      />

      {/* ── 01 Study Scope ── */}
      <section>
        <SectionMark n="01" label="Study Scope" />
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          {[
            { value: headline.conditions, label: "Experimental conditions" },
            { value: headline.datasets,   label: "Datasets",              sub: "Synthetic · CREMA-D · UTD-MHAD" },
            { value: headline.seeds,      label: "Seeds per condition" },
            { value: headline.fusionMechanisms, label: "Fusion mechanisms", sub: "Sum · Concat · Gated" },
          ].map((m) => (
            <Card key={m.label} className="card-hover p-5">
              <MetricBlock value={m.value} label={m.label} sub={m.sub} />
            </Card>
          ))}
        </div>
      </section>

      {/* ── 02 Evidence Landscape ── */}
      <section>
        <SectionMark n="02" label="Evidence Landscape" />
        <Card className="p-5">
          {/* Header + dropdowns */}
          <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
            <div>
              <div className="text-[0.85rem] font-semibold text-ink">
                Contribution Landscape · CREMA-D
              </div>
              <div className="mt-0.5 text-2xs text-muted">
                Imposed feature scale vs learned audio–video dominance (Table II)
              </div>
            </div>
            <div className="flex items-center gap-2">
              <select
                value={modFilter}
                onChange={(e) => setModFilter(e.target.value as ModFilter)}
                aria-label="Modality filter"
                className={SELECT_CLS}
              >
                <option value="both">Both modalities</option>
                <option value="audio">Audio · CREMA-D</option>
                <option value="video">Video · CREMA-D</option>
              </select>
              <select
                value={scaleFilter}
                onChange={(e) => setScaleFilter(e.target.value as ScaleFilter)}
                aria-label="Scale filter"
                className={SELECT_CLS}
              >
                <option value="all">All scales</option>
                <option value="1">1× (baseline)</option>
                <option value="5">5×</option>
                <option value="10">10×</option>
              </select>
            </div>
          </div>

          {/* Legend */}
          <div className="mb-3 flex flex-wrap gap-x-5 gap-y-1.5">
            {modalities.includes("audio") && (
              <LegendSwatch color={C.forward} label="Audio (CREMA-D)" />
            )}
            {modalities.includes("video") && (
              <LegendSwatch color={C.gradient} label="Video (CREMA-D)" />
            )}
            <LegendSwatch color={C.neutral} label="Baseline" />
          </div>

          <ContributionScatter modalities={modalities} scales={scales} />

          <p className="mt-2 font-mono text-[0.6rem] uppercase tracking-wider text-muted/70">
            Source: manuscript Table II · Spearman ρ(scale, dominance) ={" "}
            {cremad.spearman.toFixed(3)} · joint accuracy = {cremad.jointAccuracy.toFixed(3)}
          </p>
        </Card>
      </section>

      {/* ── 03 Key Findings ── */}
      <section>
        <SectionMark n="03" label="Key Findings" />
        <div className="grid gap-3 sm:grid-cols-2">
          {findings.map((f) => {
            const route = FINDING_ROUTES[f.n] ?? "/findings";
            const acc = FINDING_ACCENTS[f.n] ?? { color: C.neutral, textClass: "text-neutral" };
            return (
              <Link
                key={f.n}
                to={route}
                className="group relative overflow-hidden rounded-md border border-line bg-surface p-5 transition-all duration-150 hover:border-ink/20 hover:shadow-panel"
              >
                {/* Left accent bar */}
                <div
                  className="absolute left-0 top-0 h-full w-[3px]"
                  style={{ background: acc.color, opacity: 0.28 }}
                />

                <div className="pl-2">
                  {/* Finding number + arrow */}
                  <div className="mb-2 flex items-center justify-between">
                    <span
                      className={`font-mono text-[0.58rem] font-semibold uppercase tracking-[0.16em] ${acc.textClass}`}
                    >
                      Finding {f.n}
                    </span>
                    <ArrowRight
                      size={13}
                      className="text-muted/35 transition-colors group-hover:text-ink/50"
                    />
                  </div>

                  {/* Title */}
                  <div className="mb-3 text-[0.88rem] font-semibold leading-snug text-ink">
                    {f.title}
                  </div>

                  {/* OBSERVED */}
                  <div className="mb-2">
                    <span className="font-mono text-[0.5rem] uppercase tracking-[0.14em] text-muted/55">
                      Observed
                    </span>
                    <p className="mt-0.5 text-2xs leading-relaxed text-ink/70">
                      {f.observation}
                    </p>
                  </div>

                  {/* INTERPRETED */}
                  <div>
                    <span className="font-mono text-[0.5rem] uppercase tracking-[0.14em] text-muted/55">
                      Interpreted
                    </span>
                    <p className="mt-0.5 text-2xs leading-relaxed text-muted">
                      {f.interpretation}
                    </p>
                  </div>
                </div>
              </Link>
            );
          })}
        </div>
      </section>

      {/* ── 04 Baseline Context ── */}
      <section>
        <SectionMark n="04" label="Baseline Context" />
        <div className="rounded-md border border-line bg-surface/60 p-5">
          <div className="mb-4 text-[0.85rem] font-semibold text-ink">
            CREMA-D baseline is genuinely multimodal
          </div>
          <div className="grid grid-cols-3 gap-6">
            {[
              { value: cremad.jointAccuracy.toFixed(3), label: "Joint accuracy",   desc: "Both modalities combined" },
              { value: cremad.scale[0].gain.toFixed(3), label: "Multimodal gain 𝒢", desc: "Above best unimodal" },
              { value: cremad.scale[0].dom.toFixed(3),  label: "Dominance 𝒟",       desc: "Audio − Video at baseline" },
            ].map((m) => (
              <div key={m.label}>
                <div className="measure text-[1.65rem] font-semibold leading-none text-ink">
                  {m.value}
                </div>
                <div className="label mt-1.5">{m.label}</div>
                <div className="mt-1 text-2xs text-muted">{m.desc}</div>
              </div>
            ))}
          </div>
          <p className="mt-5 text-2xs leading-relaxed text-muted">
            A positive multimodal gain and near-equal dominance confirm both modalities contribute
            at baseline. Feature scaling then selectively disrupts this balance, making
            dominance shifts interpretable as a direct consequence of the intervention.
          </p>
        </div>
      </section>

      {/* ── 05 Explore the evidence ── */}
      <section>
        <SectionMark n="05" label="Explore the Evidence" />
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {[
            {
              to: "/disentanglement",
              icon: GitFork,
              label: "Disentanglement",
              desc: "Pathway isolation experiments",
            },
            {
              to: "/optimizer",
              icon: Sliders,
              label: "Optimizer Analysis",
              desc: "Adam vs SGD pathway effects",
            },
            {
              to: "/real-world",
              icon: Database,
              label: "Real-World Evidence",
              desc: "UTD-MHAD boundary case",
            },
            {
              to: "/findings",
              icon: Lightbulb,
              label: "All Findings",
              desc: "Scoped, evidenced conclusions",
            },
          ].map(({ to, icon: Icon, label, desc }) => (
            <Link
              key={to}
              to={to}
              className="group flex flex-col gap-2 rounded-md border border-line bg-surface p-4 transition-all duration-150 hover:border-ink/20 hover:shadow-panel"
            >
              <Icon size={15} className="text-muted/50 transition-colors group-hover:text-ink/55" />
              <div className="text-[0.82rem] font-semibold text-ink">{label}</div>
              <div className="text-2xs leading-snug text-muted">{desc}</div>
              <ArrowRight
                size={11}
                className="mt-auto self-end text-muted/30 transition-colors group-hover:text-ink/40"
              />
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
