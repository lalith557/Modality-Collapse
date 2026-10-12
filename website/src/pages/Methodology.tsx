import { useNavigate } from "react-router-dom";
import { Github, Download, FileText, Settings2 } from "lucide-react";
import { SectionHeader, Card } from "../components/primitives";
import { Tabs } from "../components/Tabs";
import {
  protocol, metrics, paper,
  decomposition, cremad, utd, fusion, persistence,
} from "../data/research";

const pdfHref = `${import.meta.env.BASE_URL}${paper.pdfPath}`;

/* ── Download helpers ────────────────────────────────────────────── */
function triggerDownload(content: string, filename: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function downloadConfigs() {
  const yaml = `# Experiment Configurations
# ${paper.title}
# ${paper.subtitle}
# Authors: ${paper.authors.join(", ")}
# ${paper.affiliation}
# Generated from verified research artifacts — all values from manuscript Tables I–III.

datasets:

  synthetic:
    modalities: 2 matched
    encoder: 2-layer MLP
    rep_dim: 32
    split: "4k / 1k / 2k (train / val / test)"
    optimizer: [Adam, SGD]
    lr: 1.0e-3
    batch_size: 256
    epochs: 40
    fusion: [sum, concat, gated]
    seeds: 5

  cremad:
    modalities: audio / video
    classes: ${cremad.classes}
    split: "${cremad.split}"
    audio_spec: "${cremad.audioSpec}"
    video_spec: "${cremad.videoSpec}"
    encoder: 2D-CNN
    rep_dim: 64
    optimizer: Adam
    lr: 1.0e-3
    batch_size: 64
    epochs: 30
    fusion: sum
    seeds: 5

  utd_mhad:
    modalities: depth / inertial
    actions: ${utd.actions}
    samples: ${utd.samples}
    test_subjects: ${utd.testSubjects}
    encoder: "2D-CNN (depth) + 1D-CNN (inertial)"
    rep_dim: 64
    split: "subj 1,3,5 (train) / 7 (val) / 2,4,6,8 (test)"
    optimizer: Adam
    lr: 1.0e-3
    batch_size: 32
    epochs: 60
    fusion: sum
    seeds: 5

interventions:
  forward:  "h → α·h   (forward representation magnitude only)"
  gradient: "h → α·g   (gradient-norm rescale only; no forward change)"
  ordinary: "h → α·x   (standard scale: both pathways active)"
  scale_values: [1, 5, 10, 100]

optimizer_hyperparameters:
  adam:
    beta1: 0.9
    beta2: 0.999
    epsilon: 1.0e-8
    weight_decay: 0
    scheduler: none
    gradient_clipping: none
    # ${protocol.optimizerNotes}
  sgd:
    momentum: 0.9
    weight_decay: 0
    scheduler: none
    gradient_clipping: none

architecture:
  # ${protocol.architectureNote}
  cnn_2d:
    blocks: 3
    channels: [16, 32, 32]
    kernel_size: "3×3"
    pooling: "2×2 max-pool (×2)"
    after_pooling: global-average-pooling
    head: linear
  cnn_1d:
    blocks: 3
    channels: [16, 32, 64]
    kernel_size: 5
    pooling: "2× max-pool (×2)"
    after_pooling: global-average-pooling
    head: linear
  mlp:
    layers: 2
    head: linear

metrics:
${metrics.map((m) => `  ${m.symbol}:
    name: "${m.name}"
    definition: "${m.def}"`).join("\n")}

statistical_tests:
  type: "paired t-test (two-sided)"
  effect_size: "Cohen's d_z"
  seeds: 5
  note: "Within-split reproducibility; not population inference."
`;
  triggerDownload(yaml, "configs.yaml", "text/yaml");
}

function downloadResults() {
  const results = {
    _meta: {
      title: paper.title,
      subtitle: paper.subtitle,
      authors: paper.authors,
      affiliation: paper.affiliation,
      venue: paper.venue,
      note: "All values transcribed from the manuscript (Tables I–III) and result artifacts. Nothing fabricated.",
    },
    decomposition: {
      _note: "Alpha = 100, 5 seeds, paired t-test. dDom = signed dominance shift vs baseline; d = Cohen's d_z; p = two-sided p-value.",
      adam_synthetic: decomposition.adamSynthetic,
      adam_cremad: decomposition.adamCremad,
      sgd_synthetic: decomposition.sgdSynthetic,
      gradient_only_arm_ratios: {
        _note: "rG = gradient-norm ratio; rTheta = parameter-update ratio.",
        adam: decomposition.ratios.adam,
        sgd: decomposition.ratios.sgd,
      },
      ordinary_vs_forward: {
        _note: "Cohen's d_z for ordinary minus forward-only effect.",
        synthetic_sgd: decomposition.ordVsFwd.synthSgd,
        cremad: decomposition.ordVsFwd.cremad,
      },
    },
    persistence: {
      _note: "Fractional reduction of weak modality contribution below balanced baseline.",
      scale_only_range: persistence.scaleOnlyRange,
      scale_only_lo: persistence.scaleOnlyLo,
      scale_only_hi: persistence.scaleOnlyHi,
      lr_imbalance: persistence.lrImbalance,
      aligned: persistence.aligned,
      sweep_range: persistence.sweep,
    },
    cremad_scale_sweep: {
      _note: "Table II: Adam, 5 seeds. dom = dominance (audio − video); gain = multimodal gain; rhoFus = post-scale norm ratio.",
      joint_accuracy: cremad.jointAccuracy,
      spearman_correlation_dom_vs_rhoFus: cremad.spearman,
      conditions: cremad.scale,
      normalization_experiment: cremad.normalization,
    },
    utd_mhad: {
      _note: "Table III: Adam, 4 fixed test subjects. chance_level = 1/27.",
      chance_level: utd.chanceLevel,
      baseline: {
        depth_unimodal_acc: utd.base.depthAcc,
        inertial_unimodal_acc: utd.base.inertialAcc,
        multimodal_acc: utd.base.multimodalAcc,
        multimodal_gain: utd.base.gain,
        dominance: utd.base.dominance,
      },
      inertial_x10: {
        dominance: utd.inertialX10.dominance,
        multimodal_gain: utd.inertialX10.gain,
      },
      depth_x10: {
        dominance: utd.depthX10.dominance,
        multimodal_gain: utd.depthX10.gain,
      },
    },
    fusion_mechanisms: {
      _note: "Synthetic dataset, alpha = 100, Adam, 5 seeds. forward_d = Cohen's d_z under forward-only intervention. gradient_null = |ΔD| across all mechanisms.",
      gradient_only_null: fusion.gradientNull,
      mechanisms: fusion.mechanisms.map((m) => ({
        name: m.name,
        key: m.key,
        forward_cohens_dz: m.forwardD,
        p_value: m.p,
        significant: m.significant,
        diagram: m.diagram,
        note: m.note,
      })),
    },
  };
  triggerDownload(JSON.stringify(results, null, 2), "results.json", "application/json");
}

/* ── Page components ─────────────────────────────────────────────── */
function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card className="p-5">
      <div className="label mb-3">{title}</div>
      {children}
    </Card>
  );
}
function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-line py-1.5 last:border-0">
      <span className="text-2xs text-muted">{k}</span>
      <span className="measure text-2xs text-ink">{v}</span>
    </div>
  );
}

type ReproItem =
  | { icon: React.ElementType; label: string; note: string; kind: "link"; href: string }
  | { icon: React.ElementType; label: string; note: string; kind: "action"; onClick: () => void }
  | { icon: React.ElementType; label: string; note: string; kind: "disabled" };

const REPRO: ReproItem[] = [
  { icon: FileText,  label: "Read Full Paper (PDF)",     href: pdfHref,          kind: "link",     note: "main.pdf · 6 pages" },
  { icon: Github,    label: "View Code (GitHub)",        kind: "disabled",                         note: "repository URL to be added" },
  { icon: Settings2, label: "Download Configurations",  onClick: downloadConfigs, kind: "action",  note: "configs.yaml · all hyperparameters" },
  { icon: Download,  label: "Download Results",         onClick: downloadResults, kind: "action",  note: "results.json · Tables I–III" },
];

export function Methodology() {
  const nav = useNavigate();
  return (
    <div className="space-y-6">
      <SectionHeader eyebrow="Methodology" title="Methodology" subtitle="Experimental setup, configuration, and reproducibility." />

      <Tabs tabs={[{ id: "methodology", label: "Methodology" }, { id: "paper", label: "Paper" }]} active="methodology" onChange={(id) => id === "paper" && nav("/paper")} />

      <div className="grid gap-6 lg:grid-cols-[1.5fr_1fr]">
        {/* Experimental setup grid */}
        <div className="grid gap-4 sm:grid-cols-2">
          <Group title="Datasets">
            <div className="space-y-1">
              <Row k="Synthetic" v="controlled" />
              <Row k="CREMA-D" v="audio-video" />
              <Row k="UTD-MHAD" v="depth-inertial" />
            </div>
          </Group>
          <Group title="Interventions">
            <div className="space-y-1">
              <Row k="Forward scaling" v="h → αh" />
              <Row k="Gradient scaling" v="h → αg" />
              <Row k="Ordinary" v="h → αx" />
            </div>
          </Group>
          <Group title="Training Configuration">
            <div className="space-y-1">
              <Row k="Optimizers" v="Adam, SGD" />
              <Row k="Epochs" v="40 / 30 / 60" />
              <Row k="Seeds" v="5" />
              <Row k="LR / batch" v="1e−3 / 256·64·32" />
            </div>
          </Group>
          <Group title="Model Architecture">
            <div className="space-y-1">
              <Row k="Encoders" v="MLP / 2D-CNN / 1D-CNN" />
              <Row k="Fusion" v="Sum / Concat / Gated" />
              <Row k="Classifier" v="linear head" />
            </div>
          </Group>
          <Card className="p-5 sm:col-span-2">
            <div className="label mb-3">Evaluation Metrics</div>
            <div className="grid gap-x-6 gap-y-1 sm:grid-cols-2">
              {metrics.map((m) => (
                <div key={m.symbol} className="flex items-baseline gap-2 border-b border-line py-1.5">
                  <span className="measure w-8 shrink-0 font-semibold text-forward">{m.symbol}</span>
                  <span className="text-2xs text-ink/85">{m.name}</span>
                </div>
              ))}
            </div>
            <p className="mt-3 text-2xs text-muted">Statistical significance over 5 seeds (paired t-test, Cohen's d_z); within-split reproducibility, not population inference. {protocol.optimizerNotes}</p>
          </Card>
        </div>

        {/* Reproducibility */}
        <div>
          <div className="label mb-3">Reproducibility</div>
          <div className="space-y-2.5">
            {REPRO.map((r) => {
              const Icon = r.icon;
              if (r.kind === "link") {
                return (
                  <a key={r.label} href={r.href} target="_blank" rel="noreferrer"
                    className="card-hover flex items-center gap-3 p-4">
                    <Icon size={17} className="text-ink" />
                    <span className="flex-1">
                      <span className="block text-sm font-medium text-ink">{r.label}</span>
                      <span className="block font-mono text-[0.6rem] text-muted">{r.note}</span>
                    </span>
                  </a>
                );
              }
              if (r.kind === "action") {
                return (
                  <button key={r.label} onClick={r.onClick}
                    className="card-hover w-full flex items-center gap-3 p-4 text-left">
                    <Icon size={17} className="text-ink" />
                    <span className="flex-1">
                      <span className="block text-sm font-medium text-ink">{r.label}</span>
                      <span className="block font-mono text-[0.6rem] text-muted">{r.note}</span>
                    </span>
                  </button>
                );
              }
              return (
                <div key={r.label}
                  className="flex cursor-not-allowed items-center gap-3 rounded-md border border-dashed border-line bg-black/[0.01] p-4 opacity-70"
                  title="Not available in this explorer">
                  <Icon size={17} className="text-muted" />
                  <span className="flex-1">
                    <span className="block text-sm font-medium text-muted">{r.label}</span>
                    <span className="block font-mono text-[0.6rem] text-muted/70">{r.note}</span>
                  </span>
                </div>
              );
            })}
          </div>
          <p className="mt-3 text-2xs leading-relaxed text-muted">Every run records seed, full config, git hash, and metrics. Only the compiled paper is bundled with this explorer; code and raw results live in the research repository.</p>
        </div>
      </div>
    </div>
  );
}
