/**
 * research.ts — single source of truth for every experimental value shown on
 * the site. All numbers are transcribed from the manuscript
 * ("Forward or Gradient? Disentangling Training-Time Feature-Scale Effects on
 * Modality Contribution", Tables I–III and the corresponding text) and its
 * result artifacts under `results/**`. Nothing here is invented or simulated.
 *
 * If a value changes in the paper, change it here once and it updates
 * everywhere in the UI.
 */

export const paper = {
  title: "Forward or Gradient?",
  subtitle:
    "Disentangling Training-Time Feature-Scale Effects on Modality Contribution",
  authors: [
    "Ankita Maji",
    "Komal Lamba",
    "Prabhutee Neb",
  ],
  affiliation: "Dept. of CSE, LPU, Phagwara, India",
  venue: "IEEE-style manuscript (MLSP format)",
  pdfPath: "main.pdf",
} as const;

export const headline = {
  conditions: "60+",
  datasets: 3,
  seeds: 5,
  fusionMechanisms: 3,
} as const;

/**
 * Forward-vs-gradient decomposition at alpha = 100, five seeds (Table I).
 * dDom = signed dominance shift vs baseline; d = Cohen's d_z; p = two-sided
 * paired t-test. rG = gradient-norm ratio, rTheta = parameter-update ratio
 * (gradient-only arm).
 */
export const decomposition = {
  adamSynthetic: {
    label: "Adam · synthetic",
    forward: { dDom: -0.07, d: -3.76, p: 1.1e-3 },
    gradient: { dDom: -0.0005, d: -0.53, p: 0.3 },
    ordinary: { dDom: -0.069, d: -3.73, p: 1.1e-3 },
  },
  adamCremad: {
    label: "Adam · CREMA-D",
    forward: { dDom: 0.259, d: 7.39, p: 8e-5 },
    gradient: { dDom: -0.001, d: -0.08, p: 0.87 },
    ordinary: { dDom: 0.266, d: 8.51, p: 4e-5 },
  },
  sgdSynthetic: {
    label: "SGD · synthetic",
    forward: { dDom: 0.244, d: 3.93, p: 9e-4 },
    gradient: { dDom: 0.2, d: 3.9, p: 1e-3 },
    ordinary: { dDom: 0.322, d: 7.75, p: 7e-5 },
  },
  // Gradient-only arm ratios (Table I footer).
  ratios: {
    adam: { rG: 282.7, rTheta: 1.61 },
    sgd: { rG: 318.5, rTheta: 95.9 },
  },
  ordVsFwd: {
    synthSgd: { d: 2.86, p: 3e-3 },
    cremad: { d: 0.59, p: 0.26 },
  },
} as const;

/**
 * Persistence statistic S_loss (study-specific), fractional reduction of the
 * weak modality's contribution below its balanced baseline. Text of §IV.
 */
export const persistence = {
  scaleOnlyRange: "0.04–0.08",
  scaleOnlyLo: 0.04,
  scaleOnlyHi: 0.08,
  lrImbalance: 0.88,
  aligned: 1.0,
  sweep: "20–400×",
} as const;

/**
 * CREMA-D training-time scale sweep (Table II, five seeds, Adam).
 * dom = dominance (audio − video); gain = multimodal gain; rhoFus = post-scale
 * norm ratio. Last row = per-modality z-score normalization on audio-10x.
 */
export const cremad = {
  classes: 6,
  split: "64 / 9 / 18 actors  (5234 / 738 / 1470 clips)",
  audioSpec: "16 kHz mono, 3 s, 64-bin log-mel",
  videoSpec: "8 frames, 64×64 grayscale",
  jointAccuracy: 0.493,
  spearman: 0.922,
  scale: [
    { condition: "baseline (1×)", scaled: "—", dom: 0.075, gain: 0.248, rhoFus: 1.22 },
    { condition: "audio ×5", scaled: "audio", dom: 0.299, gain: 0.045, rhoFus: 2.31 },
    { condition: "audio ×10", scaled: "audio", dom: 0.328, gain: 0.013, rhoFus: 3.24 },
    { condition: "video ×5", scaled: "video", dom: 0.056, gain: 0.263, rhoFus: 1.18 },
    { condition: "video ×10", scaled: "video", dom: 0.009, gain: 0.295, rhoFus: 1.02 },
  ],
  normalization: {
    condition: "audio ×10 + z",
    domBefore: 0.328,
    domAfter: 0.11,
    rhoFusBefore: 3.24,
    rhoFusAfter: 0.97,
  },
} as const;

/**
 * UTD-MHAD boundary experiment (Table III + text). Strongly asymmetric
 * modality utility. Accuracies are masked-unimodal.
 */
export const utd = {
  actions: 27,
  samples: 861,
  chanceLevel: 0.037,
  testSubjects: 4,
  base: {
    depthAcc: 0.036,
    inertialAcc: 0.227,
    multimodalAcc: 0.632,
    gain: 0.405,
    dominance: -0.191,
  },
  inertialX10: { dominance: -0.615, gain: 0.053 },
  depthX10: { dominance: -0.198, gain: 0.4 },
} as const;

/**
 * Fusion-mechanism robustness (Fig. 3, synthetic, alpha = 100).
 * Forward-only effect is Cohen's d; gradient-only stays approximately null
 * (|dDom| <= 0.001) across all three mechanisms.
 */
export const fusion = {
  gradientNull: "|Δ𝒟| ≤ 0.001",
  mechanisms: [
    {
      name: "Sum",
      key: "sum",
      forwardD: -3.76,
      p: 1e-3,
      significant: true,
      diagram: "hA + hB → prediction",
      note: "Strong forward effect.",
    },
    {
      name: "Concatenation",
      key: "concat",
      forwardD: -5.16,
      p: 3e-4,
      significant: true,
      diagram: "[hA ; hB] → prediction",
      note: "Strongest forward effect.",
    },
    {
      name: "Gated",
      key: "gated",
      forwardD: -0.7,
      p: 0.19,
      significant: false,
      diagram: "gate(hA, hB) → prediction",
      note: "Attenuated; not individually significant.",
    },
  ],
} as const;

/** Datasets & protocol (Table III / methodology). */
export const protocol = {
  columns: ["Synthetic", "CREMA-D", "UTD-MHAD"],
  rows: [
    { field: "Modalities", values: ["2 matched", "audio / video", "depth / inertial"] },
    { field: "Encoder", values: ["2-layer MLP", "2D-CNN", "2D + 1D-CNN"] },
    { field: "Rep. dim", values: ["32", "64", "64"] },
    { field: "Split (fixed)", values: ["4k / 1k / 2k", "actor 64/9/18", "subj. 1,3,5 / 7 / 2,4,6,8"] },
    { field: "Optimizer", values: ["Adam, SGD", "Adam", "Adam"] },
    { field: "LR / batch", values: ["1e−3 / 256", "1e−3 / 64", "1e−3 / 32"] },
    { field: "Epochs", values: ["40", "30", "60"] },
    { field: "Fusion", values: ["sum / concat / gated", "sum", "sum"] },
    { field: "Seeds", values: ["5", "5", "5"] },
  ],
  optimizerNotes:
    "Adam uses PyTorch defaults (β = (0.9, 0.999), ε = 1e−8); SGD uses momentum 0.9. No weight decay, scheduler, or gradient clipping.",
  architectureNote:
    "CNN encoders: three conv blocks (conv → ReLU → 2×2 max-pool; 2× pooling for the 1D encoder), then global-average pooling and a linear projection. 2D 3×3 with 16/32/32 channels; 1D kernel-5 with 16/32/64 channels.",
} as const;

/** Metric definitions (methodology). */
export const metrics = [
  { symbol: "𝒟", name: "Modality contribution / dominance", def: "Acc_A − Acc_B on masked unimodal accuracy (operational, not a causal attribution)." },
  { symbol: "𝒢", name: "Multimodal gain", def: "Acc_AB − max(Acc_A, Acc_B)." },
  { symbol: "r_g", name: "Gradient-norm ratio", def: "‖g_A‖ / ‖g_B‖ for modality-A vs modality-B encoder gradients." },
  { symbol: "r_Δθ", name: "Parameter-update ratio", def: "‖Δθ_A‖ / ‖Δθ_B‖ — an optimization-level readout, not a formal mediator." },
  { symbol: "ρ_f", name: "Post-scale norm ratio", def: "Feature-norm ratio at the fusion boundary, after the scale operator." },
  { symbol: "ρ_e", name: "Pre-scale (encoder) ratio", def: "Feature-norm ratio before the operator; shifts oppositely under compensation." },
] as const;

/** The four scoped findings (Findings page). Observation/Evidence/Interpretation/Limitation. */
export const findings = [
  {
    n: "01",
    title: "Forward pathway is dominant under the tested Adam settings",
    observation:
      "Under Adam, forward-only scaling reproduces almost the entire ordinary effect, while gradient-only scaling produces little measurable shift.",
    evidence:
      "Synthetic: forward Δ𝒟 = −0.070 (d = −3.76) vs gradient Δ𝒟 = −0.0005 (d = −0.53, p = 0.30). CREMA-D: forward Δ𝒟 = +0.259 (d = 7.39) vs gradient Δ𝒟 = −0.001.",
    interpretation:
      "The training-time intervention travels primarily through the forward representation-magnitude channel in these configurations.",
    limitation:
      "Shown for Adam with the specified CNN/MLP encoders; not a claim about all optimizers or architectures.",
  },
  {
    n: "02",
    title: "Gradient amplification is strongly attenuated under Adam",
    observation:
      "A large gradient-norm ratio does not translate into a proportional parameter-update ratio under Adam.",
    evidence:
      "Gradient-only arm: r_g ≈ 282.7 yet r_Δθ ≈ 1.61 (CREMA-D); a 100–282× gradient ratio leaves the update ratio ≈ 1.",
    interpretation:
      "Consistent with Adam's normalized update attenuating an approximately constant per-parameter gradient rescale.",
    limitation:
      "A readout at the update level, not an identified mechanism; does not establish invariance across Adam variants or schedules.",
  },
  {
    n: "03",
    title: "Under SGD, both pathways contribute",
    observation:
      "With SGD the gradient-only arm does shift dominance, and forward and gradient effects both act.",
    evidence:
      "SGD synthetic: gradient Δ𝒟 = +0.200 (d = 3.90) with r_Δθ ≈ 95.9; forward Δ𝒟 = +0.244; ordinary +0.322 < 0.244 + 0.200 (sub-additive).",
    interpretation:
      "Which pathway carries the intervention depends on optimizer dynamics; SGD passes the gradient rescale into the update.",
    limitation:
      "Evaluated on the synthetic generator; SGD-on-real-data was not tested.",
  },
  {
    n: "04",
    title: "Modality utility changes the observed behavior",
    observation:
      "In a strongly utility-asymmetric setting, the response to scaling differs sharply between modalities.",
    evidence:
      "UTD-MHAD: amplifying informative inertial pushes dominance −0.191 → −0.615 (gain → 0.053); amplifying near-chance depth barely moves contribution (gain ≈ 0.40).",
    interpretation:
      "The dominance metric is influenced by the large difference in unimodal predictive utility; scaling a near-chance modality cannot manufacture contribution.",
    limitation:
      "A boundary case (four fixed test subjects); reported as per-subject consistency, not population inference.",
  },
] as const;

export type Arm = { dDom: number; d: number; p: number };

/* ================================================================== *
 *  Interaction model — option lists + scenario lookups.
 *  Everything below only INDEXES the verified values above; it never
 *  introduces a new number. Combinations that were not run in the study
 *  return `null` so the UI can say so honestly instead of inventing data.
 * ================================================================== */

export type DatasetId = "synthetic" | "cremad" | "utd";
export type OptimizerId = "adam" | "sgd";
export type InterventionId = "forward" | "gradient" | "ordinary";
export type FusionId = "sum" | "concat" | "gated";
export type ScaleId = 1 | 5 | 10 | 100;

export const OPTIONS = {
  dataset: [
    { id: "synthetic", label: "Synthetic" },
    { id: "cremad", label: "CREMA-D" },
    { id: "utd", label: "UTD-MHAD" },
  ] as { id: DatasetId; label: string }[],
  optimizer: [
    { id: "adam", label: "Adam" },
    { id: "sgd", label: "SGD" },
  ] as { id: OptimizerId; label: string }[],
  intervention: [
    { id: "forward", label: "Forward-only" },
    { id: "gradient", label: "Gradient-only" },
    { id: "ordinary", label: "Ordinary (both)" },
  ] as { id: InterventionId; label: string }[],
  fusion: [
    { id: "sum", label: "Sum" },
    { id: "concat", label: "Concatenation" },
    { id: "gated", label: "Gated" },
  ] as { id: FusionId; label: string }[],
  scale: [1, 5, 10, 100] as ScaleId[],
  seed: [1, 2, 3, 4, 5],
} as const;

export const LABELS = {
  dataset: { synthetic: "Synthetic", cremad: "CREMA-D", utd: "UTD-MHAD" },
  optimizer: { adam: "Adam", sgd: "SGD" },
  intervention: { forward: "Forward-only", gradient: "Gradient-only", ordinary: "Ordinary" },
  fusion: { sum: "Sum", concat: "Concatenation", gated: "Gated" },
} as const;

/** The verified four-arm decomposition, keyed by dataset+optimizer (α = 100). */
export function getDecomposition(dataset: DatasetId, optimizer: OptimizerId) {
  if (dataset === "synthetic" && optimizer === "adam")
    return { data: decomposition.adamSynthetic, ratios: decomposition.ratios.adam, measured: true as const };
  if (dataset === "cremad" && optimizer === "adam")
    return { data: decomposition.adamCremad, ratios: decomposition.ratios.adam, measured: true as const };
  if (dataset === "synthetic" && optimizer === "sgd")
    return { data: decomposition.sgdSynthetic, ratios: decomposition.ratios.sgd, measured: true as const };
  return null; // cremad+sgd, and any utd decomposition were not run
}

/** The endpoint (ΔD, d, p) for one arm of a scenario, or null if not measured. */
export function armEndpoint(
  dataset: DatasetId,
  optimizer: OptimizerId,
  intervention: InterventionId,
): { endpoint: Arm | null; measured: boolean; label: string } {
  const dec = getDecomposition(dataset, optimizer);
  if (!dec) return { endpoint: null, measured: false, label: `${LABELS.dataset[dataset]} · ${LABELS.optimizer[optimizer]}` };
  return { endpoint: dec.data[intervention], measured: true, label: dec.data.label };
}

/** CREMA-D scale-sweep row for a modality (audio/video) at a given scale. */
export function cremadByScale(modality: "audio" | "video", scale: 1 | 5 | 10) {
  if (scale === 1) return cremad.scale[0];
  return cremad.scale.find((r) => r.condition === `${modality} ×${scale}`) ?? null;
}

/** UTD-MHAD conditions that were actually measured (base + ×10 amplification). */
export const utdConditions = [
  { id: "base", label: "Baseline", dominance: utd.base.dominance, gain: utd.base.gain, emphasis: "neutral" as const },
  { id: "inertialX10", label: "Inertial ×10", dominance: utd.inertialX10.dominance, gain: utd.inertialX10.gain, emphasis: "forward" as const },
  { id: "depthX10", label: "Depth ×10", dominance: utd.depthX10.dominance, gain: utd.depthX10.gain, emphasis: "gradient" as const },
];

/** Fusion mechanism record by id. */
export function fusionByKey(key: FusionId) {
  return fusion.mechanisms.find((m) => m.key === key)!;
}
