/**
 * research.ts — single source of truth for every experimental value shown on
 * the site. All numbers are transcribed from the manuscript
 * ("Feature Scale as a Hidden Per-Modality Learning Rate: When
 * Masked-Accuracy Dominance Misleads") and its result artifacts under
 * `results/**` / docs/CLAIMS_LEDGER.md. Nothing here is invented or simulated.
 *
 * If a value changes in the paper, change it here once and it updates
 * everywhere in the UI.
 */

export const paper = {
  // The manuscript served at website/public/ca_research_paper.pdf (copied from
  // "CA Resesrach Paper.pdf", 7 pages, 15 references). The site's measured values below
  // come from the frozen result artifacts, not from the PDF.
  title: "When Scale Silences a Modality",
  fullTitle:
    "When Scale Silences a Modality: Disentangling Representation Magnitude, Optimization Dynamics, and Evaluation Artifacts in Multimodal Collapse",
  subtitle: "Disentangling representation magnitude, optimization dynamics, and evaluation artifacts in multimodal collapse.",
  authors: ["Komal Lamba", "Ankita Maji", "Prabhutee Neb", "Premananda Sahu"],
  affiliation: "Lovely Professional University, Phagwara, Punjab, India",
  venue: "CA research paper",
  pdfPath: "ca_research_paper.pdf",
  pages: 7,
  references: 15,
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
  // Gradient-only arm ratios, SYNTHETIC, α = 100, 5 seeds (matched to the arm effects above).
  // Adam: results/fusion_robustness (sum fusion) — an exact repeat of the Adam decomposition
  // run (same seeds/config, identical dominance shifts), which itself did not log r_Δθ.
  // SGD: results/sgd_disentangle. Correction (2026-11-17): the site previously showed the
  // CREMA-D values (282.7 / 1.61) under a "synthetic" label; those are now `ratiosCremad`.
  ratios: {
    adam: { rG: 100.9, rTheta: 1.0 },
    sgd: { rG: 318.5, rTheta: 95.9 },
  },
  // CREMA-D, Adam, gradient-only arm, α = 100, 5 seeds (results/cremad_disentangle).
  // The CREMA-D BASELINE update ratio is itself 1.65, so 1.61 means "unchanged".
  ratiosCremad: { rG: 282.7, rTheta: 1.61, baselineRTheta: 1.65 },
  ordVsFwd: {
    synthSgd: { d: 2.86, p: 3e-3 },
    cremad: { d: 0.59, p: 0.26 },
  },
} as const;

/**
 * Decoupled-metric check (paper Table I). Forward scaling, α = 100, baseline →
 * forward shift, oriented so a POSITIVE value means the SCALED modality gains;
 * paired 95% CIs. Does a masked-accuracy dominance shift reflect a GENUINE
 * change in modality contribution, or does masking overstate it? All values from
 * logged runs (CLAIMS_LEDGER M3 / W2 / M-P4b / M-P4c and the Phase-3/4 controls);
 * nothing invented.
 *  - Synthetic (Adam, 10 seeds): every decoupled metric moves with masked → real.
 *  - CREMA-D (Adam, weight decay 0, 5 seeds): only masked moves; probe/Shapley/mean-ablation
 *    stay within noise → the masked shift is a masking artifact (probe saturated).
 *  - CREMA-D (stable SGD, clip = 1, 5 seeds): masked moves +0.13 while the probe
 *    moves −0.02 — masked and probe even disagree in SIGN.
 *  - UTD-MHAD (Adam, weight decay 0, 5 seeds, scaled = informative inertial): masked moves +0.53
 *    but the probe only +0.10 → masked OVERSTATES the real change ~5× (headroom).
 *    The gradient-only arm equals baseline on every metric (Adam cancellation).
 *  - Positive control (audio input noise, 0.5σ / 1.0σ): probe and Shapley DO
 *    fall (n = 3), evidence the diagnostics can detect this tested form of degradation.
 */
export const decoupled = {
  // Confidence intervals are paired Student-t 95% intervals over seeds
  // (t = 2.262 for n=10 synthetic, 2.776 for n=5 real), recomputed from the
  // seed-level CSVs per the reviewer report (Issue 1).
  syntheticAdam: { masked: -0.063, meanAbl: -0.056, shapley: -0.043, probe: -0.034 },
  syntheticCIs:  { masked: 0.011,  meanAbl: 0.005,  shapley: 0.003,  probe: 0.006 },
  cremadAdamW:   { masked: 0.259,  meanAbl: 0.012,  shapley: 0.014,  probe: 0.023 },
  cremadCIs:     { masked: 0.044,  meanAbl: 0.027,  shapley: 0.026,  probe: 0.017 },
  cremadNS:      { meanAbl: true,  shapley: true },
  // Phase 4 — CREMA-D under STABLE SGD (gradient-norm clip = 1, α = 100): forward arm.
  cremadSgdClip:    { masked: 0.132, meanAbl: -0.062, shapley: -0.059, probe: -0.021 },
  cremadSgdClipCIs: { masked: 0.079, meanAbl: 0.037,  shapley: 0.034,  probe: 0.020 },
  // Phase 4 — UTD-MHAD second dataset (Adam, weight decay 0, α = 100): forward arm, scaled = inertial.
  utdAdamW:      { masked: 0.525,  meanAbl: 0.107,  shapley: 0.113,  probe: 0.102 },
  utdCIs:        { masked: 0.048,  meanAbl: 0.039,  shapley: 0.044,  probe: 0.048 },
  utdProbeNS:    true,          // depth (unscaled) probe shift −0.011 is n.s.
  utdGradientNull: true,        // gradient-only arm ≡ baseline on EVERY metric
  positiveControl: {
    levels: ["0.5σ", "1.0σ"] as const,
    probe:   [-0.021, -0.064],
    shapley: [-0.029, -0.051],
  },
} as const;

/**
 * Phase-4 SGD stabilization (CREMA-D, α = 100, lr = 1e-2, momentum 0.9). The
 * earlier "SGD + scaling is unstable on real data" caveat was an optimization
 * artifact: standard gradient-norm clipping removes it. From
 * results/phase4_baselines/phase4_sgd_stable.csv (CLAIMS_LEDGER M-P4b / W3).
 */
export const sgdStable = {
  alpha: 100,
  lr: 0.01,
  clip: 1.0,
  divergedGradientArm: { noClip: "5 / 5", clip: "0 / 5" },
  // Under stable SGD (clip=1): gradient arm moves masked dominance but leaves the
  // decoupled metrics null; forward arm moves masked while the probe drops (sign flip).
  gradientArmMasked: 0.2, // decoupled metrics all null (≤0.002)
  gradientArmMaskedCI: 0.05, // paired Student-t, results/phase4_baselines/phase4_sgd_stable.csv (clip = 1)
  forwardArmMasked: 0.132,
  forwardArmProbe: -0.021,
} as const;

/**
 * Phase-4 payoff baselines on CREMA-D (Adam, weight decay 0, 5 seeds): method − vanilla shift,
 * no scaling. A published rebalancing method (OGM-GE, Peng et al. CVPR 2022) and
 * a modality-specific learning rate both move masked dominance while the decoupled
 * metrics barely move — so masked-accuracy dominance can mislead even for a method
 * in the literature. From results/phase4_baselines/phase4_cremad.csv
 * (CLAIMS_LEDGER M-P4a); nothing invented.
 */
export const baselines = {
  ogmGE: {
    label: "OGM-GE",
    cite: "Peng et al., CVPR 2022",
    kAudio: 0.273,
    kVideo: 0.999,
    masked: 0.24,  meanAbl: -0.023, shapley: -0.026, probe: -0.014,
    cis: { masked: 0.042, meanAbl: 0.024, shapley: 0.030, probe: 0.012 },
    // Under Student-t, mean-ablation and Shapley CIs now INCLUDE 0 (n.s.);
    // only the tiny probe (−0.014) is statistically resolved.
    decoupledNS: { meanAbl: true, shapley: true, probe: false },
  },
  modLR: {
    label: "Mod-LR (audio ×5)",
    masked: 0.117, meanAbl: 0.007, shapley: 0.012, probe: 0.005,
    cis: { masked: 0.110, meanAbl: 0.023, shapley: 0.027, probe: 0.022 },
    decoupledNS: { meanAbl: true, shapley: true, probe: true },
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
  // results/fusion_robustness/fusion_robustness_analysis.json — synthetic, Adam, α = 100, 5 seeds.
  // forward/gradient = arm − baseline (paired t-test, Cohen's d_z); ordinaryVsForward = ordinary − forward.
  // gradientArm.rG / rTheta = gradient-only arm gradient-norm and parameter-update ratios.
  // Gate activations were NOT logged, so no gate values exist to show.
  // s(·) is the arm's operator on h_A (experiments/fusion_robustness.py FusionModel.fuse_arm);
  // the gate is computed from the UNSCALED h_A and multiplies the scaled one.
  scope: "Synthetic · Adam · α = 100 · 5 seeds",
  mechanisms: [
    {
      name: "Sum",
      key: "sum",
      forwardD: -3.76,
      p: 1.1e-3,
      significant: true,
      forwardDDom: -0.07,
      gradient: { dDom: -0.0005, d: -0.53, p: 0.3 },
      ordinaryDDom: -0.069,
      ordinaryVsForward: { d: 0.73, p: 0.18 },
      gradientArm: { rG: 100.9, rTheta: 1.0 },
      formula: "z = s(h_A) + h_B",
      diagram: "hA + hB → prediction",
      note: "Strong forward effect.",
    },
    {
      name: "Concatenation",
      key: "concat",
      forwardD: -5.16,
      p: 3e-4,
      significant: true,
      forwardDDom: -0.057,
      gradient: { dDom: 0.0, d: 0.0, p: 1.0 },
      ordinaryDDom: -0.057,
      ordinaryVsForward: { d: 0.0, p: 1.0 },
      gradientArm: { rG: 101.6, rTheta: 1.02 },
      formula: "z = [s(h_A) ; h_B]",
      diagram: "[hA ; hB] → prediction",
      note: "Strongest forward effect.",
    },
    {
      name: "Gated",
      key: "gated",
      forwardD: -0.7,
      p: 0.19,
      significant: false,
      forwardDDom: -0.026,
      gradient: { dDom: -0.0003, d: -0.02, p: 0.96 },
      ordinaryDDom: -0.051,
      ordinaryVsForward: { d: -0.62, p: 0.24 },
      gradientArm: { rG: 92.8, rTheta: 1.02 },
      formula: "z = σ(W_A h_A + b_A) ⊙ s(h_A) + σ(W_B h_B + b_B) ⊙ h_B",
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

/** The four scoped findings (Research Overview § Findings). Observation/Evidence/Interpretation/Limitation.
 * ± values are paired Student-t 95% intervals (t = 2.262 for 10 seeds, 2.776 for 5), the same
 * intervals as `decoupled` above; corrected 2026-11-17 from normal-approximation (1.96) intervals. */
export const findings = [
  {
    n: "01",
    title: "The forward pathway carries the effect — real on synthetic data, but masked accuracy overstates it on real data",
    observation:
      "Under Adam, forward-only scaling reproduces almost the entire ordinary effect while gradient-only scaling barely moves masked dominance. On synthetic data the decoupled metrics confirm a genuine contribution change; on both real datasets masked dominance moves several times more than any decoupled metric.",
    evidence:
      "Synthetic (Adam, α=100, 10 seeds): forward moves every decoupled metric together — masked Δ𝒟 −0.063±0.011, frozen-encoder probe −0.034±0.006, mean-ablation −0.056±0.005, Shapley −0.043±0.003 (all CIs exclude 0). CREMA-D (Adam, weight decay 0, 5 seeds): masked Δ𝒟 +0.259±0.044 but mean-ablation +0.012±0.027 (n.s.), Shapley +0.014±0.026 (n.s.), probe only +0.023±0.017. UTD-MHAD (Adam, weight decay 0, 5 seeds): masked +0.525±0.048 but the scaled-modality probe only +0.102±0.048 — masked overstates ~5×.",
    interpretation:
      "On synthetic data the forward (feature-magnitude) channel produces a real, decoupled-metric-confirmed change. On real data the masked shift is far larger than the representational change — a masking artifact when the probe is saturated (CREMA-D) and a ~5× overstatement when it has headroom (UTD-MHAD). So ‘audio/inertial now dominates’ read from masked accuracy can mislead.",
    limitation:
      "Shown for Adam (weight decay 0) with the specified MLP/CNN encoders; the real-data result is scoped to our CREMA-D and UTD-MHAD splits and weak from-scratch encoders.",
  },
  {
    n: "02",
    title: "Gradient amplification is strongly attenuated under Adam — on synthetic and both real datasets",
    observation:
      "A large gradient-norm ratio does not translate into a proportional parameter-update ratio under Adam, and the gradient-only arm moves no metric.",
    evidence:
      "Gradient-only arm: a ≈100–282× gradient-norm ratio leaves the parameter-update ratio ≈ 1 (synthetic Adam r_Δθ = 1.00±0.01; CREMA-D 1.61), and no decoupled metric moves (synthetic probe shift +0.0001). On UTD-MHAD the gradient-only arm equals the baseline on EVERY metric (masked +0.003, mean-ablation +0.004, Shapley −0.002, probes within 0.01; all n.s.).",
    interpretation:
      "Consistent with Adam's normalized update cancelling an approximately constant per-parameter gradient rescale (m_Adam = s_f). The clean null replicates on a second real dataset.",
    limitation:
      "A readout at the update level, not an identified mechanism; does not establish invariance across Adam variants or schedules.",
  },
  {
    n: "03",
    title: "Under SGD the gradient scale reaches the weights — and clipping makes it stable without removing the mismatch",
    observation:
      "With SGD the gradient-only arm transmits the scale into the parameter update, unlike Adam. The scaled arms diverge on real data, but standard gradient-norm clipping stabilizes them — and the masked-vs-decoupled mismatch still holds.",
    evidence:
      "SGD synthetic gradient-only: parameter-update ratio r_Δθ ≈ 95.9 (vs ≈ 1.0 under Adam). CREMA-D, α=100: the gradient arm diverges 5/5 without clipping but 0/5 with gradient-norm clip = 1. Under this stable SGD the gradient arm moves masked Δ𝒟 +0.200±0.050 with mean-ablation, Shapley, and probe all null (≤0.002); the forward arm moves masked +0.132±0.079 while the probe moves −0.021±0.020 — opposite in sign.",
    interpretation:
      "The update-level mechanism — SGD passes the gradient rescale into the weights while Adam cancels it — is robust, and the instability was an optimization artifact fixed by clipping. Under stable SGD the masked shift is again not corroborated by the decoupled metrics, so the dissociation is not Adam-specific.",
    limitation:
      "Clipping disclosed; five seeds; the direction of the synthetic SGD contribution change remains regime-dependent (an undertrained-baseline artifact at lr 1e-3).",
  },
  {
    n: "04",
    title: "Masked-accuracy dominance can mislead even for a published rebalancing method",
    observation:
      "Applied to a method in the literature rather than our own operator, masked-accuracy dominance again moves while the representation barely changes.",
    evidence:
      "CREMA-D (zero-decay AdamW, 5 seeds), method − vanilla: OGM-GE (adapted to our shared sum-fusion setting, with two implementation modifications; attenuates audio, k_audio = 0.27) shifts masked Δ𝒟 +0.240±0.042 while its decoupled metrics are mostly unresolved under Student-t — mean-ablation −0.023±0.024 and Shapley −0.026±0.030 include zero; only the tiny probe −0.014±0.012 is resolved. A modality-specific learning rate (audio ×5) behaves the same (masked +0.117, decoupled n.s.).",
    interpretation:
      "An evaluator judging these rebalancing interventions by masked-accuracy dominance would read a large effect the representation-level metrics do not support — the caution our operator motivates, now for a published method. We claim nothing about OGM-GE's accuracy benefit, only about how its effect is measured.",
    limitation:
      "Our CREMA-D split and weak from-scratch encoders; measures the metric readout, not the method's downstream accuracy.",
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
    return { data: decomposition.adamCremad, ratios: decomposition.ratiosCremad, measured: true as const };
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
