/**
 * Architecture data for the Model Architecture & Data Flow page.
 *
 * Every number (shapes, parameter counts, kernel/stride/padding, channels, layer
 * types, classes) comes from architecture.generated.json, which
 * scripts/export_architecture.py reads from the instantiated PyTorch modules after
 * a strict state-dict load of the baseline checkpoints. This file only adds
 * plain-language explanations, layout positions, and the conceptual-overlay facts
 * (whose provenance is stated next to each item). It introduces no numbers.
 */
import generated from "../data/architecture.generated.json";

export type ArchDataset = "cremad" | "utd_mhad";
export type Branch = "a" | "b" | "joint";
export type Segment = "pipeline" | "encoder" | "fusion" | "head" | "output";

export interface GenNode {
  id: string;
  branch: Branch;
  segment: Segment;
  label: string;
  module: string | null;
  op: string;
  summary: string;
  in_shape: number[] | null;
  out_shape: number[] | null;
  in_shapes?: Record<string, number[]>;
  shape_note?: string;
  per_frame: number | null;
  params: number;
  details: Record<string, unknown>;
  source: string;
}

export interface ArchNode extends GenNode {
  /** Compact label for the 3D view (full label shown when selected). */
  short: string;
  simple: string;
  technical: string;
  purpose: string;
  /** Layout grid position shared by the 3D and 2D views. */
  col: number;
  row: number;
}

export interface ArchGraph {
  dataset: ArchDataset;
  name: string;
  task: string;
  modalities: { a: string; b: string };
  encoders: { a: string; b: string };
  nClasses: number;
  classCodes: string[];
  classNamesAvailable: boolean;
  repDim: number;
  fusion: string;
  params: { encoder_a: number; encoder_b: number; fusion: number; head: number; total: number };
  training: Record<string, string | number>;
  checkpoint: { path: string; sha256: string; strict_state_dict_load: boolean };
  ctor: string;
  modelClass: string;
  preprocessingVersion: string;
  nodes: ArchNode[];
  edges: { from: string; to: string }[];
  forwardOrder: string[];
  /** Which branch the paper's real-data experiments scaled (see overlayFacts). */
  scaledBranch: "a" | "b";
}

/** Shape of one dataset entry in architecture.generated.json. */
interface GenDataset {
  display_name: string;
  task: string;
  model_class: string;
  constructor_call: string;
  modalities: { a: string; b: string };
  encoders: { a: string; b: string };
  rep_dim: number;
  fusion: string;
  n_classes: number;
  class_codes: string[];
  class_names: (string | null)[];
  params: { encoder_a: number; encoder_b: number; fusion: number; head: number; total: number };
  training: Record<string, string | number>;
  checkpoint: { path: string; sha256: string; strict_state_dict_load: boolean };
  preprocessing_version: string;
  nodes: unknown[];
  edges: { from: string; to: string }[];
  forward_order: string[];
}

export const SOURCE_SHA = generated.source_sha256 as Record<string, string>;

/* ── shape helpers ─────────────────────────────────────────────────── */

export function fmtShape(s: number[] | null | undefined): string {
  return s && s.length ? `[${s.join(", ")}]` : "—";
}

export function shapeText(n: Pick<GenNode, "per_frame" | "out_shape" | "shape_note">, _which: "in" | "out", shape: number[] | null): string {
  if (!shape) return n.shape_note ?? "variable (raw input)";
  return n.per_frame ? `${n.per_frame} frames × ${fmtShape(shape)}` : fmtShape(shape);
}

const fmt = (x: number) => x.toLocaleString("en-US");

/* ── explanations ──────────────────────────────────────────────────── */

const MOD_NOUN: Record<string, string> = {
  audio: "spectrogram",
  video: "frame",
  depth: "depth frame",
  inertial: "sensor signal",
};

function explain(ds: ArchDataset, n: GenNode, g: GenDataset): { simple: string; technical: string; purpose: string } {
  const mod = n.branch === "joint" ? "" : (g.modalities as Record<string, string>)[n.branch];
  const d = n.details as Record<string, any>;
  const perFrame = n.per_frame
    ? ` It runs on each of the ${n.per_frame} frames separately, with the same weights for every frame.`
    : "";
  switch (n.id.replace(/^[ab]_/, "")) {
    case "raw":
      return {
        simple: `The original ${mod} recording, before any processing. Nothing here is learned.`,
        technical: `${n.summary}. ${n.shape_note ?? ""}. Decoder: ${d.decoder ?? d.input}.`,
        purpose: "Raw input to the deterministic preprocessing pipeline.",
      };
    case "wave":
      return {
        simple:
          "Mixes the recording down to one channel, converts it to 16,000 samples per second, and keeps only the first 3 seconds (shorter clips are padded with silence).",
        technical: `Mono: ${d.mono}; int16 scaling: ${d.int16_scaling}; resample to ${d.sample_rate} Hz with ${d.resampler}; ${d.crop}. Output ${fmtShape(n.out_shape)} samples.`,
        purpose: "Fixes the sample rate and duration so every clip yields the same spectrogram size.",
      };
    case "logmel": {
      const m = d.mel as Record<string, unknown>;
      return {
        simple:
          "Turns the waveform into an image of how much energy each of 64 frequency bands carries over time (a log-mel spectrogram). The audio encoder treats it as a one-channel image.",
        technical: `MelSpectrogram(n_fft=${m.n_fft}, hop_length=${m.hop_length}, n_mels=${m.n_mels}, win_length=${m.win_length}, power=${m.power}, center=${m.center}) → ${d.log}; ${d.time_frames} time frames; no normalization; cached as ${d.cache_dtype}.`,
        purpose: "Produces the audio encoder's input tensor.",
      };
    }
    case "sample":
      if (ds === "cremad")
        return {
          simple: "Picks 8 evenly spaced frames across the whole video; all other frames are dropped.",
          technical: `Frame indices ${d.frame_index}; frames kept in colour at the original resolution for the next step.`,
          purpose: "Fixes the number of frames per clip regardless of video length.",
        };
      return {
        simple: "Picks 16 evenly spaced depth frames across the trial and shrinks each one to 32×32.",
        technical: `Frame indices ${d.frame_index}; resize to ${(d.resize as number[]).join("×")} with ${d.interpolation}.`,
        purpose: "Fixes the number and size of depth frames per trial.",
      };
    case "frames":
      if (ds === "cremad")
        return {
          simple: "Converts each frame to grayscale, crops the central square and shrinks it to 64×64 pixels with values between 0 and 1.",
          technical: `${d.color}; ${d.crop}; resize ${(d.resize as number[]).join("×")} (${d.interpolation}); ${d.scale}; no normalization; cached as ${d.cache_dtype}.`,
          purpose: "Produces the video encoder's input tensor.",
        };
      return {
        simple: "Rescales the clip's depth values so the nearest and farthest points in these 16 frames map to 0 and 1.",
        technical: `${d.normalization}; stored as ${d.dtype}. Output ${fmtShape(n.out_shape)}.`,
        purpose: "Produces the depth encoder's input tensor.",
      };
    case "resample":
      return {
        simple: "Stretches or compresses each of the 6 sensor channels in time to exactly 128 samples.",
        technical: `${d.resample}. Output ${fmtShape(n.out_shape)}.`,
        purpose: "Fixes the signal length regardless of how long the action took.",
      };
    case "zscore":
      return {
        simple:
          "Centres and rescales each sensor channel with the training set's average and spread, so accelerometer and gyroscope channels have comparable ranges.",
        technical: `${d.standardization}; stored as ${d.dtype}. Statistics come from the training split only.`,
        purpose: "Produces the inertial encoder's input tensor.",
      };
    case "block1":
    case "block2":
    case "block3": {
      const k = (d.kernel_size as number[]).join("×");
      const first = n.id.endsWith("1");
      return {
        simple:
          (first
            ? `Slides ${d.channels_out} small learned filters over the ${MOD_NOUN[mod] ?? mod} to detect local patterns, keeps only positive responses (ReLU), and halves the resolution (max-pooling).`
            : `Combines the previous block's ${d.channels_in} pattern maps into ${d.channels_out} larger-scale patterns, then again keeps positive responses and halves the resolution.`) +
          perFrame,
        technical: `${d.layer_types?.[0]}(${d.channels_in}→${d.channels_out}, kernel ${k}, stride ${(d.stride as number[]).join("×")}, padding ${(d.padding as number[]).join("×")}, bias=${d.bias}) → ${d.activation} → ${d.pooling}. ${shapeText(n, "in", n.in_shape)} → ${shapeText(n, "out", n.out_shape)}; ${fmt(n.params)} parameters. No normalization layer.`,
        purpose: "Learned feature extraction.",
      };
    }
    case "gap":
      return {
        simple: `Averages each of the ${n.out_shape?.[0]} feature maps over all positions, leaving one number per channel.` + perFrame,
        technical: `${d.pooling} followed by Flatten. ${shapeText(n, "in", n.in_shape)} → ${shapeText(n, "out", n.out_shape)}; no parameters.`,
        purpose: "Removes the spatial/temporal layout so the vector size does not depend on input size.",
      };
    case "tmean":
      return {
        simple: `Averages the per-frame feature vectors over the ${n.in_shape?.[0]} frames. The order of the frames is not modelled.`,
        technical: `Functional operation inside ${String(n.source).split(":")[0]} forward(): ${n.op}. ${fmtShape(n.in_shape)} → ${fmtShape(n.out_shape)}; no parameters and no nn.Module.`,
        purpose: "Collapses the frame axis into one clip-level vector.",
      };
    case "fc": {
      const h = n.branch === "a" ? "h_A" : "h_B";
      return {
        simple: `Projects the pooled features to the ${n.out_shape?.[0]}-dimensional representation ${h}, the vector this modality contributes to fusion.`,
        technical: `Linear(${d.features_in}→${d.features_out}, bias=${d.bias}); no activation afterwards. ${fmtShape(n.in_shape)} → ${fmtShape(n.out_shape)}; ${fmt(n.params)} parameters. Then norm_${n.branch === "a" ? "a" : "v"} = nn.Identity (normalization 'none').`,
        purpose: `Produces the modality representation ${h} (the fusion-boundary vector).`,
      };
    }
    case "fusion":
      return {
        simple:
          "Adds the two 64-number representations element by element. There is no learned weight here, so whichever representation has larger values has more influence on the sum.",
        technical: `${n.op}: ${fmtShape(n.in_shapes?.a)} + ${fmtShape(n.in_shapes?.b)} → ${fmtShape(n.out_shape)}; no parameters. Before fusion: ${d.pre_fusion}. Masked-unimodal evaluation multiplies one input by 0 here (mask_a / mask_v).`,
        purpose: "Combines the two modalities. This boundary is where the paper's scale intervention acts in the experiments.",
      };
    case "head":
      return {
        simple: `A single linear layer turns the fused vector into one score (logit) per class: ${n.out_shape?.[0]} scores.`,
        technical: `Linear(${d.features_in}→${d.features_out}, bias=${d.bias}); ${fmt(n.params)} parameters; no hidden layer (head='linear').`,
        purpose: "Classifier.",
      };
    case "output":
      return {
        simple: `Softmax turns the ${n.out_shape?.[0]} scores into probabilities that add up to 1; the highest is the predicted class. A high probability is not a guarantee of correctness.`,
        technical: `${d.activation}.`,
        purpose: "Prediction output.",
      };
  }
  return { simple: n.summary, technical: n.op, purpose: "" };
}

/* ── layout ────────────────────────────────────────────────────────── */

const COL: Record<string, number> = {
  raw: 0, wave: 1, sample: 1, resample: 1, logmel: 2, frames: 2, zscore: 2,
  block1: 3, block2: 4, block3: 5, gap: 6, tmean: 7, fc: 8,
};

function place(n: GenNode): { col: number; row: number } {
  if (n.id === "fusion") return { col: 9, row: 0 };
  if (n.id === "head") return { col: 10, row: 0 };
  if (n.id === "output") return { col: 11, row: 0 };
  const key = n.id.replace(/^[ab]_/, "");
  return { col: COL[key], row: n.branch === "a" ? 1 : -1 };
}

const SHORT: Record<string, (n: GenNode) => string> = {
  raw: () => "input",
  wave: () => "16 kHz · 3 s",
  logmel: () => "log-mel",
  sample: (n) => (n.id.startsWith("b") ? "8 frames" : "16 frames"),
  frames: (n) => (n.id.startsWith("b") ? "gray 64²" : "min–max"),
  resample: () => "128 pts",
  zscore: () => "z-score",
  block1: () => "conv 1",
  block2: () => "conv 2",
  block3: () => "conv 3",
  gap: () => "GAP",
  tmean: () => "mean t",
  fc: (n) => (n.branch === "a" ? "FC → h_A" : "FC → h_B"),
  fusion: () => "sum",
  head: () => "linear",
  output: () => "softmax",
};

/* ── graph ─────────────────────────────────────────────────────────── */

function build(ds: ArchDataset): ArchGraph {
  const g = (generated.datasets as unknown as Record<string, GenDataset>)[ds];
  const nodes: ArchNode[] = (g.nodes as unknown as GenNode[]).map((n) => ({
    ...n,
    short: SHORT[n.id.replace(/^[ab]_/, "")]?.(n) ?? n.label,
    ...explain(ds, n, g),
    ...place(n),
  }));
  return {
    dataset: ds,
    name: g.display_name,
    task: g.task,
    modalities: g.modalities as { a: string; b: string },
    encoders: g.encoders as { a: string; b: string },
    nClasses: g.n_classes,
    classCodes: g.class_codes,
    classNamesAvailable: (g.class_names as (string | null)[]).every((x) => !!x),
    repDim: g.rep_dim,
    fusion: g.fusion,
    params: g.params,
    training: g.training as Record<string, string | number>,
    checkpoint: g.checkpoint,
    ctor: g.constructor_call,
    modelClass: g.model_class,
    preprocessingVersion: g.preprocessing_version,
    nodes,
    edges: g.edges,
    forwardOrder: g.forward_order,
    // results/cremad_disentangle: modality A = audio is scaled; experiments/phase4_utd_decoupled.py: SCALED = "v" (inertial).
    scaledBranch: ds === "cremad" ? "a" : "b",
  };
}

export const GRAPHS: Record<ArchDataset, ArchGraph> = { cremad: build("cremad"), utd_mhad: build("utd_mhad") };

export function nodeById(g: ArchGraph, id: string | null | undefined): ArchNode | undefined {
  return id ? g.nodes.find((n) => n.id === id) : undefined;
}

/** All nodes upstream and downstream of `id` (its data-flow path through the graph). */
export function connectedPath(g: ArchGraph, id: string): { nodes: Set<string>; edges: Set<string> } {
  const nodes = new Set<string>([id]);
  const edges = new Set<string>();
  const walk = (start: string, dir: "up" | "down") => {
    const stack = [start];
    while (stack.length) {
      const cur = stack.pop()!;
      for (const e of g.edges) {
        const [a, b] = dir === "down" ? [e.from, e.to] : [e.to, e.from];
        if (a === cur) {
          edges.add(`${e.from}->${e.to}`);
          if (!nodes.has(b)) {
            nodes.add(b);
            stack.push(b);
          }
        }
      }
    }
  };
  walk(id, "down");
  walk(id, "up");
  return { nodes, edges };
}

export function nextStages(g: ArchGraph, id: string): ArchNode[] {
  return g.edges.filter((e) => e.from === id).map((e) => nodeById(g, e.to)!);
}

/* ── data-flow steps (forward call order) ──────────────────────────── */

export interface FlowStep {
  index: number;
  node: ArchNode;
  /** The edge that carries data INTO this stage (null for the first step of a branch). */
  inEdge: { from: string; to: string } | null;
  shape: string;
  next: ArchNode[];
}

export function flowSteps(g: ArchGraph): FlowStep[] {
  return g.forwardOrder.map((id, index) => {
    const node = nodeById(g, id)!;
    const ins = g.edges.filter((e) => e.to === id);
    return {
      index,
      node,
      inEdge: ins.length === 1 ? ins[0] : null,
      shape:
        node.id === "fusion"
          ? `${fmtShape(node.in_shapes?.a)} + ${fmtShape(node.in_shapes?.b)} → ${fmtShape(node.out_shape)}`
          : `${shapeText(node, "in", node.in_shape)} → ${shapeText(node, "out", node.out_shape)}`,
      next: nextStages(g, id),
    };
  });
}

/* ── comparison rows (all values read from the graphs) ─────────────── */

export interface CompareRow {
  field: string;
  cremad: string;
  utd_mhad: string;
}

function encSummary(g: ArchGraph, br: "a" | "b"): string {
  const blocks = g.nodes.filter((n) => n.branch === br && n.id.includes("block"));
  const d0 = blocks[0].details as Record<string, any>;
  const conv = d0.layer_types[0];
  const k = (d0.kernel_size as number[]).join("×");
  const ch = blocks.map((b) => (b.details as Record<string, any>).channels_out).join("/");
  const tm = g.nodes.some((n) => n.id === `${br}_tmean`);
  return `${g.encoders[br]}: 3 × [${conv} ${k} → ReLU → max-pool 2], channels ${ch}, global average pool${tm ? ", mean over frames" : ""}, Linear → ${g.repDim}`;
}

function inputShape(g: ArchGraph, br: "a" | "b"): string {
  const last = g.nodes.filter((n) => n.branch === br && n.segment === "pipeline").pop()!;
  return fmtShape(last.out_shape);
}

function pipe(g: ArchGraph, br: "a" | "b"): string {
  return g.nodes
    .filter((n) => n.branch === br && n.segment === "pipeline")
    .map((n) => n.label)
    .join(" → ");
}

export function compareRows(): CompareRow[] {
  const C = GRAPHS.cremad;
  const U = GRAPHS.utd_mhad;
  const both = (f: (g: ArchGraph) => string) => ({ cremad: f(C), utd_mhad: f(U) });
  return [
    { field: "Task", ...both((g) => g.task) },
    { field: "Modality A / B", ...both((g) => `${g.modalities.a} / ${g.modalities.b}`) },
    { field: "Preprocessing A", ...both((g) => pipe(g, "a")) },
    { field: "Preprocessing B", ...both((g) => pipe(g, "b")) },
    { field: "Model input A", ...both((g) => inputShape(g, "a")) },
    { field: "Model input B", ...both((g) => inputShape(g, "b")) },
    { field: "Encoder A", ...both((g) => encSummary(g, "a")) },
    { field: "Encoder B", ...both((g) => encSummary(g, "b")) },
    { field: "Parameters A / B", ...both((g) => `${fmt(g.params.encoder_a)} / ${fmt(g.params.encoder_b)}`) },
    { field: "Representation dim", ...both((g) => `${g.repDim} (h_A) + ${g.repDim} (h_B)`) },
    { field: "Normalization / scale", ...both(() => "none (nn.Identity) / ScaleInjector disabled (α = 1)") },
    { field: "Fusion", ...both((g) => `${g.fusion} (element-wise h_A + h_B), 0 parameters`) },
    {
      field: "Classifier",
      ...both((g) => {
        const h = nodeById(g, "head")!;
        return `Linear(${fmtShape(h.in_shape).slice(1, -1)}→${g.nClasses}), ${fmt(g.params.head)} parameters`;
      }),
    },
    { field: "Output", ...both((g) => `${g.nClasses} classes (${g.classCodes.slice(0, 3).join(", ")}, …)`) },
    { field: "Total parameters", ...both((g) => fmt(g.params.total)) },
    {
      field: "Training (baseline checkpoint)",
      ...both(
        (g) =>
          `${g.training.optimizer}, lr ${g.training.learning_rate}, weight decay ${g.training.weight_decay}, batch ${g.training.batch_size}, ${g.training.epochs} epochs, seed ${g.training.seed}`,
      ),
    },
  ];
}

/* ── conceptual overlay facts ──────────────────────────────────────── */

export const overlayFacts = {
  actual:
    "The nodes and connections are the implemented model, generated from the code and verified against the saved baseline checkpoint. In that checkpoint the scale operator is disabled (α = 1), so no intervention is present.",
  intervention: (g: ArchGraph) =>
    `In the paper's ${g.name} experiments, a scale operator with α = 100 acts on ${
      g.scaledBranch === "a" ? "h_A" : "h_B"
    } (${g.modalities[g.scaledBranch]}) just before fusion, implemented with custom autograd functions (experiments/disentangle_scale_grad.py): forward-only returns α·h but passes the gradient unchanged; gradient-only returns h but multiplies the gradient by α; ordinary returns α·h (both). The model layers are the same in every arm.`,
  conceptual:
    "The forward, gradient and update pathways are a conceptual summary of how the operator can affect training. They are not extra layers and not the model's computational graph; the optimizer is not part of the network.",
  scaledSource: {
    cremad: "results/cremad_disentangle (modality A = audio scaled)",
    utd_mhad: "experiments/phase4_utd_decoupled.py (SCALED = inertial)",
  } as Record<ArchDataset, string>,
};

/* ── figure components that are not layers of the checkpoint ─────────── */

export interface ConceptNode {
  id: string;
  label: string;
  /** e.g. "experiments only", "synthetic experiment", "analysis" */
  tag: string;
  simple: string;
  technical: string;
  facts: { k: string; v: string }[];
  source: string;
}

const fmtSigned = (x: number, dg = 3) => (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(dg);

export function conceptNodes(g: ArchGraph, evidence: {
  fusion: { key: string; name: string; formula: string; forwardD: number; p: number; forwardDDom: number; gradient: { dDom: number; d: number; p: number } }[];
  ratios: { adam: { rG: number; rTheta: number }; sgd: { rG: number; rTheta: number } };
  pValue: (p: number) => string;
}): Record<string, ConceptNode> {
  const scaledH = g.scaledBranch === "a" ? "h_A" : "h_B";
  const scaledE = g.scaledBranch === "a" ? "ℰ_A" : "ℰ_B";
  const fus = (key: string): ConceptNode => {
    const m = evidence.fusion.find((f) => f.key === key)!;
    return {
      id: `concept:fusion_${key}`,
      label: `${m.name} fusion`,
      tag: "synthetic experiment only",
      simple:
        key === "concat"
          ? "Places the two representations side by side instead of adding them, so the classifier sees each modality separately."
          : "Each modality's representation is multiplied by its own learned gate (values between 0 and 1) before the two are added.",
      technical: `${m.formula} (experiments/fusion_robustness.py; s(·) = the arm's scale operator on h_A${
        key === "gated" ? ", each gate computed from the unscaled representation" : ""
      }). Tested on matched synthetic data with Adam, α = 100, 5 seeds. The ${g.name} checkpoint uses sum fusion.`,
      facts: [
        { k: "Forward-only Δ𝒟", v: fmtSigned(m.forwardDDom) },
        { k: "Forward d_z · p", v: `${m.forwardD.toFixed(2)} · ${evidence.pValue(m.p)}` },
        { k: "Gradient-only Δ𝒟 · p", v: `${fmtSigned(m.gradient.dDom, 4)} · ${evidence.pValue(m.gradient.p)}` },
        ...(key === "gated" ? [{ k: "Gate values", v: "not logged in this experiment" }] : []),
      ],
      source: "results/fusion_robustness/fusion_robustness_analysis.json",
    };
  };
  return {
    "concept:intervention": {
      id: "concept:intervention",
      label: "Feature-scale intervention (α)",
      tag: "experiments only",
      simple: `In the experiments, the ${g.modalities[g.scaledBranch]} representation ${scaledH} is multiplied by α just before fusion and the other modality by 1. In this baseline checkpoint α = 1, so nothing is scaled.`,
      technical: overlayFacts.intervention(g),
      facts: [
        { k: "Scaled branch", v: `${g.modalities[g.scaledBranch]} (${scaledH})` },
        { k: "Other branch", v: "× 1 (unscaled)" },
        { k: "α in this checkpoint", v: "1 (ScaleInjector disabled)" },
        { k: "α in the decomposition", v: "100" },
        { k: "Where", v: "after the encoder (and its normalization), before fusion" },
      ],
      source: "interventions/scale_injection.py · experiments/disentangle_scale_grad.py",
    },
    "concept:fusion_concat": fus("concat"),
    "concept:fusion_gated": fus("gated"),
    "concept:variants": {
      id: "concept:variants",
      label: "Intervention variants",
      tag: "experiments only",
      simple: "Three ways to apply the same scale: change only what the next layer sees, only the learning signal sent back to the encoder, or both.",
      technical: `Custom autograd operators on ${scaledH}: ForwardScale returns α·h and passes the gradient unchanged; GradScale returns h and multiplies the gradient into ${scaledE} by α; ordinary returns α·h (both). All arms share one objective; a sanity gate checks each operator's forward and backward multipliers before training.`,
      facts: [
        { k: "Forward-only", v: `${scaledH}′ = α·${scaledH} · gradient unchanged` },
        { k: "Gradient-only", v: `${scaledH}′ = ${scaledH} · ∇${scaledE} × α` },
        { k: "Ordinary", v: `${scaledH}′ = α·${scaledH} · ∇${scaledE} × α` },
      ],
      source: "experiments/disentangle_scale_grad.py (ForwardScale, GradScale, arm_apply)",
    },
    "concept:optimizer": {
      id: "concept:optimizer",
      label: "Optimizer analysis",
      tag: "analysis · synthetic, α = 100, 5 seeds",
      simple: "The optimizer decides how much of a multiplied gradient reaches the weights. Adam divides it back out; SGD passes it through.",
      technical:
        "First-order analysis (paper, Proposition 1): with forward gain s_f and gradient gain s_g, the effective learning-rate multiplier is m = s_f·s_g under SGD and m = s_f under small-ε Adam, where the gradient gain cancels. Measured on the gradient-only arm (s_f = 1, s_g = α).",
      facts: [
        { k: "Adam", v: `m = s_f · r_g ${evidence.ratios.adam.rG.toFixed(1)}× → r_Δθ ${evidence.ratios.adam.rTheta.toFixed(2)}×` },
        { k: "SGD", v: `m = s_f·s_g · r_g ${evidence.ratios.sgd.rG.toFixed(1)}× → r_Δθ ${evidence.ratios.sgd.rTheta.toFixed(1)}×` },
      ],
      source: "results/fusion_robustness (Adam) · results/sgd_disentangle (SGD)",
    },
    "concept:utility": {
      id: "concept:utility",
      label: "Modality-utility measurements",
      tag: "analysis",
      simple: "Four ways the study measures how much each modality contributes. They can disagree, which is the paper's main caution.",
      technical:
        "Masked dominance 𝒟 = Acc_A − Acc_B with the other modality's fused input zeroed (off-distribution). With v(S) = test accuracy when modalities outside S are replaced by their pre-scale training-set mean: mean-ablation dominance = v({A}) − v({B}); Shapley φ_A = ½[v({A}) − v(∅)] + ½[v({A,B}) − v({B})]. The frozen-encoder probe is a multinomial-logistic classifier on each encoder's standardized output (L2 = 10⁻⁴), fit on train and scored on test.",
      facts: [
        { k: "Masked accuracy", v: "other modality zeroed at fusion" },
        { k: "Mean-ablation", v: "other modality set to its training mean" },
        { k: "Shapley", v: "two-modality value, mean-ablated empty set" },
        { k: "Frozen-encoder probe", v: "linear classifier per frozen encoder" },
      ],
      source: "analysis/ (contribution metrics, probes) · paper §III",
    },
  };
}

/* ── workflow stages (stage navigator, 3D camera focus, 2D captions) ────── */

export type StageKey = "inputs" | "preprocessing" | "encoding" | "intervention" | "fusion" | "classifier" | "output";

export const STAGE_ORDER: StageKey[] = ["inputs", "preprocessing", "encoding", "intervention", "fusion", "classifier", "output"];

/** Which workflow stage a node or figure component belongs to (null if none). */
export function stageOf(g: ArchGraph, id: string | null | undefined): StageKey | null {
  if (!id) return null;
  if (id === "concept:intervention" || id === "concept:variants") return "intervention";
  if (id.startsWith("concept:fusion") || id === "concept:utility") return "fusion";
  if (id === "concept:optimizer") return "encoding";
  const n = nodeById(g, id);
  if (!n) return null;
  switch (n.segment) {
    case "pipeline":
      return n.id.endsWith("_raw") ? "inputs" : "preprocessing";
    case "encoder":
      return "encoding";
    case "fusion":
      return "fusion";
    case "head":
      return "classifier";
    default:
      return "output";
  }
}

/** The node (or figure component) a stage opens on. */
export function stageEntry(g: ArchGraph, s: StageKey): string {
  switch (s) {
    case "inputs":
      return "a_raw";
    case "preprocessing":
      return g.nodes.find((n) => n.branch === "a" && n.segment === "pipeline" && !n.id.endsWith("_raw"))!.id;
    case "encoding":
      return "a_block1";
    case "intervention":
      return "concept:intervention";
    case "fusion":
      return "fusion";
    case "classifier":
      return "head";
    case "output":
      return "output";
  }
}

/** Stage caption; every value is read from the graph. */
export function stageInfo(g: ArchGraph, s: StageKey): { label: string; detail: string } {
  const head = nodeById(g, "head")!;
  switch (s) {
    case "inputs":
      return { label: "Inputs", detail: `${g.modalities.a} + ${g.modalities.b}` };
    case "preprocessing":
      return { label: "Preprocessing", detail: "deterministic, not learned" };
    case "encoding":
      return { label: "Encoding", detail: `${g.encoders.a} · ${g.encoders.b}` };
    case "intervention":
      return { label: "Intervention", detail: "α · experiments only" };
    case "fusion":
      return { label: "Fusion", detail: `${g.fusion} (this model)` };
    case "classifier":
      return { label: "Classifier", detail: `Linear ${head.in_shape?.[0]} → ${head.out_shape?.[0]}` };
    case "output":
      return { label: "Output", detail: `softmax · ${g.nClasses} classes` };
  }
}

/* ── sections (the platforms of the 3D view; one section rises when selected) ── */

export type SectionKey = "inputs" | "preprocessing" | "enc_a" | "enc_b" | "intervention" | "fusion" | "classifier" | "output";

/** The section a node, figure component or "section:<key>" id belongs to. */
export function sectionOf(g: ArchGraph, id: string | null | undefined): SectionKey | null {
  if (!id) return null;
  if (id.startsWith("section:")) return id.slice(8) as SectionKey;
  if (id === "concept:optimizer") return g.scaledBranch === "a" ? "enc_a" : "enc_b";
  const n = nodeById(g, id);
  if (n?.segment === "encoder") return n.branch === "a" ? "enc_a" : "enc_b";
  const s = stageOf(g, id);
  return s === "encoding" ? null : s;
}

export function sectionLabel(g: ArchGraph, s: SectionKey): string {
  switch (s) {
    case "enc_a":
      return `Encoder ℰ_A · ${g.modalities.a} (${g.encoders.a})`;
    case "enc_b":
      return `Encoder ℰ_B · ${g.modalities.b} (${g.encoders.b})`;
    case "intervention":
      return "Feature-scale intervention";
    default:
      return stageInfo(g, s).label;
  }
}
