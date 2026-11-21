import { useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, ChevronDown, GitFork, Database, FlaskConical, Network } from "lucide-react";
import { SectionHeader, MetricBlock, Card } from "../components/primitives";
import { LegendSwatch } from "../components/ScientificFigure";
import { ContributionScatter } from "../charts/ContributionScatter";
import { C } from "../charts/theme";
import { headline, cremad, findings, decoupled, decomposition, sgdStable, baselines } from "../data/research";
import { fixed, signed } from "../lib/format";
import { usePageContext } from "../state/page-context";
import { useSectionScroll } from "../lib/useSectionScroll";

type ModFilter = "both" | "audio" | "video";
type ScaleFilter = "all" | "1" | "5" | "10";

/** Where each finding's supporting evidence lives (sections of Disentanglement / Real-World Evidence). */
const FINDING_LINKS: Record<string, { label: string; to: string }[]> = {
  "01": [
    { label: "Forward vs gradient", to: "/disentanglement?section=forward-vs-gradient" },
    { label: "Masked vs decoupled · CREMA-D", to: "/real-world?section=masked-mislead" },
    { label: "UTD-MHAD", to: "/real-world?tab=utd&section=datasets" },
  ],
  "02": [
    { label: "Adam vs SGD", to: "/disentanglement?section=adam-vs-sgd" },
    { label: "Optimizer comparison", to: "/disentanglement?section=optimizer-comparison" },
  ],
  "03": [
    { label: "Optimizer comparison", to: "/disentanglement?section=optimizer-comparison" },
    { label: "Stable SGD · CREMA-D", to: "/real-world?section=sgd-stable" },
  ],
  "04": [{ label: "Published-method baseline", to: "/real-world?section=baselines" }],
};

const r2 = (x: number) => (x >= 10 ? x.toFixed(1) : x.toFixed(2));

/** Key numbers per finding, read from research.ts (no new values). */
const FINDING_METRICS: Record<string, { k: string; v: string }[]> = {
  "01": [
    { k: "CREMA-D masked Δ𝒟", v: signed(decoupled.cremadAdamW.masked) },
    { k: "CREMA-D probe", v: signed(decoupled.cremadAdamW.probe) },
    { k: "UTD-MHAD masked vs probe", v: `${signed(decoupled.utdAdamW.masked)} vs ${signed(decoupled.utdAdamW.probe)}` },
  ],
  "02": [
    { k: "Synthetic Adam r_g → r_Δθ", v: `${r2(decomposition.ratios.adam.rG)}× → ${r2(decomposition.ratios.adam.rTheta)}×` },
    { k: "CREMA-D Adam r_g → r_Δθ", v: `${r2(decomposition.ratiosCremad.rG)}× → ${r2(decomposition.ratiosCremad.rTheta)}×` },
    { k: "Gradient-only Δ𝒟 (synthetic)", v: fixed(decomposition.adamSynthetic.gradient.dDom, 4) },
  ],
  "03": [
    { k: "SGD r_g → r_Δθ", v: `${r2(decomposition.ratios.sgd.rG)}× → ${r2(decomposition.ratios.sgd.rTheta)}×` },
    { k: "Diverged seeds (no clip → clip 1)", v: `${sgdStable.divergedGradientArm.noClip} → ${sgdStable.divergedGradientArm.clip}` },
    { k: "Stable SGD forward: masked vs probe", v: `${signed(sgdStable.forwardArmMasked)} vs ${signed(sgdStable.forwardArmProbe)}` },
  ],
  "04": [
    { k: "OGM-GE masked Δ𝒟", v: signed(baselines.ogmGE.masked) },
    { k: "OGM-GE probe", v: signed(baselines.ogmGE.probe) },
    { k: "Mod-LR masked Δ𝒟", v: signed(baselines.modLR.masked) },
  ],
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

function FindingCard({ f }: { f: (typeof findings)[number] }) {
  const [open, setOpen] = useState(false);
  const acc = FINDING_ACCENTS[f.n] ?? { color: C.neutral, textClass: "text-neutral" };
  return (
    <article
      id={`finding-${f.n}`}
      className="relative scroll-mt-24 overflow-hidden rounded-md border border-line bg-surface p-5"
      data-testid={`finding-${f.n}`}
    >
      <div className="absolute left-0 top-0 h-full w-[3px]" style={{ background: acc.color, opacity: 0.28 }} />
      <div className="pl-2">
        <span className={`font-mono text-[0.58rem] font-semibold uppercase tracking-[0.16em] ${acc.textClass}`}>Finding {f.n}</span>
        <h3 className="mb-3 mt-2 text-[0.92rem] font-semibold leading-snug text-ink">{f.title}</h3>

        <div className="mb-3">
          <span className="font-mono text-[0.5rem] uppercase tracking-[0.14em] text-muted/55">Observed</span>
          <p className="mt-0.5 text-2xs leading-relaxed text-ink/75">{f.observation}</p>
        </div>

        <dl className="mb-3 grid gap-2 sm:grid-cols-3" data-testid={`finding-${f.n}-metrics`}>
          {FINDING_METRICS[f.n].map((m) => (
            <div key={m.k} className="rounded border border-line bg-canvas/40 px-2.5 py-1.5">
              <dt className="font-mono text-[0.55rem] uppercase tracking-wider text-muted">{m.k}</dt>
              <dd className="font-mono text-xs font-semibold text-ink">{m.v}</dd>
            </div>
          ))}
        </dl>

        <div className="mb-3">
          <span className="font-mono text-[0.5rem] uppercase tracking-[0.14em] text-muted/55">Interpreted</span>
          <p className="mt-0.5 text-2xs leading-relaxed text-muted">{f.interpretation}</p>
        </div>

        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          className="mb-2 inline-flex items-center gap-1 font-mono text-[0.6rem] uppercase tracking-wider text-muted hover:text-ink"
        >
          Evidence and scope <ChevronDown size={12} className={`transition-transform ${open ? "rotate-180" : ""}`} />
        </button>
        {open && (
          <div className="mb-3 space-y-2 rounded border border-line bg-canvas/40 p-3" data-testid={`finding-${f.n}-evidence`}>
            <div>
              <div className="font-mono text-[0.55rem] uppercase tracking-wider text-forward">Evidence</div>
              <p className="text-2xs leading-relaxed text-ink/80">{f.evidence}</p>
            </div>
            <div>
              <div className="font-mono text-[0.55rem] uppercase tracking-wider text-warning">Limitation</div>
              <p className="text-2xs leading-relaxed text-ink/80">{f.limitation}</p>
            </div>
          </div>
        )}

        <div className="flex flex-wrap gap-1.5">
          {FINDING_LINKS[f.n].map((l) => (
            <Link
              key={l.to}
              to={l.to}
              className="inline-flex items-center gap-1 rounded border border-line px-2 py-1 text-2xs text-ink transition-colors hover:border-ink/30"
            >
              {l.label} <ArrowRight size={11} className="text-muted" />
            </Link>
          ))}
        </div>
      </div>
    </article>
  );
}

export function Overview() {
  useSectionScroll();
  usePageContext({
    title: "Summary context",
    chips: [
      { k: "Scope", v: "all experiments" },
      { k: "Datasets", v: "Synthetic · CREMA-D · UTD-MHAD" },
      { k: "Optimizers", v: "Adam · SGD" },
    ],
    note: "Each figure and finding states its own dataset, optimizer, α and seeds",
  });

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
                Imposed feature scale vs masked-accuracy dominance, audio − video (Table II)
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

      {/* ── 03 Key Findings (formerly the Findings page) ── */}
      <section id="findings" className="scroll-mt-24">
        <SectionMark n="03" label="Key Findings" />
        <div className="grid gap-4 lg:grid-cols-2">
          {findings.map((f) => (
            <FindingCard key={f.n} f={f} />
          ))}
        </div>
        <div className="mt-4 rounded-md border border-line bg-black/[0.015] p-5" data-testid="not-claimed">
          <div className="label mb-2">What we do not claim</div>
          <p className="max-w-3xl text-2xs leading-relaxed text-ink/80">
            We do not claim that feature scale is a sufficient cause of persistent collapse, that gradient imbalance is universally
            irrelevant, that normalization cures imbalance, that Adam is universally invariant to gradient scaling, or that the phenomenon
            generalizes across architectures and optimizers. We identify experimentally separable pathways within the evaluated regimes.
          </p>
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
            at baseline. Scaling audio then moves masked-accuracy dominance strongly — but the
            decoupled metrics (frozen-encoder probe, Shapley, mean-ablation) barely move, so on
            CREMA-D that masked shift is a masking artifact. The same dissociation recurs on a
            second dataset (UTD-MHAD, where masked overstates the probe ~5×), under SGD stabilized
            by gradient clipping, and for a published rebalancing method (OGM-GE).
            See the findings above and Real-World Evidence.
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
              desc: "Pathways, Adam vs SGD, fusion mechanisms",
            },
            {
              to: "/real-world",
              icon: Database,
              label: "Real-World Evidence",
              desc: "CREMA-D, UTD-MHAD, baselines",
            },
            {
              to: "/lab",
              icon: FlaskConical,
              label: "Experiment Lab",
              desc: "Explore every measured condition",
            },
            {
              to: "/architecture",
              icon: Network,
              label: "Model Architecture",
              desc: "The real models, layer by layer",
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
