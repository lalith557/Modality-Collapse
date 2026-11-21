# When Scale Silences a Modality

*Feature-Magnitude Disparity as a Sufficient Cause of Modality Collapse, and Per-Modality Normalization as its Minimal Cure*

**Repo handle:** `modality-scale-collapse` (rename freely)
**Target venue:** a solid peer-reviewed IEEE conference (IJCNN, an IEEE CV/ML/multimedia venue, or IEEE Access). *Not* framed as NeurIPS/ICML/ICLR main-track novelty — see §3.
**Compute envelope:** single laptop GPU, 8 GB VRAM. Every experiment is designed to run there.
**Status:** research blueprint. Novelty verified against a literature sweep (Sep 2026). Two neighbor papers must be read and differentiated on day one (§3).

---

## Table of contents
1. [TL;DR](#1-tldr)
2. [The seed: why this project exists](#2-the-seed)
3. [Background primer: modality collapse](#3-background-primer)
4. [Honest positioning & the gap](#4-positioning)
5. [Research questions & hypotheses](#5-questions)
6. [Explained simply](#6-simple)
7. [Technical formulation](#7-technical)
8. [Contributions](#8-contributions)
9. [Datasets — full details & download](#9-datasets)
10. [Method & architecture](#10-method)
11. [Baselines](#11-baselines)
12. [Metrics](#12-metrics)
13. [Experimental protocol (the matrix)](#13-protocol)
14. [Expected results](#14-expected)
15. [Compute plan](#15-compute)
16. [Environment & setup](#16-environment)
17. [Repository structure](#17-repo)
18. [Implementation roadmap](#18-roadmap)
19. [Reproducibility](#19-reproducibility)
20. [Paper outline](#20-paper)
21. [Risks & kill criteria](#21-risks)
22. [Why this project (fit)](#22-fit)
23. [References](#23-references)

---

<a name="1-tldr"></a>
## 1. TL;DR

In multimodal fusion, one modality frequently dominates and the model stops using the other ("modality collapse"). The dominant literature blames **optimization/gradient imbalance** and fixes it with gradient modulation or architectural surgery. This project pursues a different, simpler cause suggested by a real engineering incident: **raw feature-magnitude (scale) disparity between streams.** We aim to show — with controlled, causal experiments on small models — that scale disparity is a *sufficient* driver of collapse (induce it by scaling, cure it by normalizing), that a cheap **activation-norm ratio** predicts collapse before it locks in, and that **per-modality normalization** is a minimal, competitive fix in the scale-driven regime (and, honestly, not in the optimization-driven regime — a boundary that reconciles the two views).

The contribution is **mechanism + diagnostic + minimal fix + boundary**, not a leaderboard number.

<a name="2-the-seed"></a>
## 2. The seed: why this project exists

The project grew from a concrete, unexplained incident while building a multi-modal Indian Sign Language (ISL) recognizer. The model fused a **manual** stream (hand shape/motion) with a **non-manual** stream (facial expression / head pose — linguistically essential in sign languages). The manual stream steamrolled the non-manual one: the model effectively ignored the face channel. Adding **per-stream LayerNorm** rescued the face channel and restored its contribution. The fix worked but was never explained.

That unexplained, load-bearing hack *is* the research question. It is hard to scoop precisely because it came from hands-on work rather than from the literature.

<a name="3-background-primer"></a>
## 3. Background primer: modality collapse (so this doc is self-contained)

**What it is.** Train a network that fuses two or more input streams (e.g., audio + video) for a joint task. Ideally the multimodal model beats the best single-modality model. In practice it often *doesn't*: the network relies on one dominant modality and underuses (or ignores) the others. Symptoms: the multimodal accuracy barely exceeds — or is below — the strongest unimodal accuracy; masking the "weak" modality at test time barely changes predictions; the weak encoder is undertrained.

**Why it's believed to happen (existing accounts).**
- *Optimization / learning-rate imbalance.* Modalities converge at different rates; the faster one dominates the loss and suppresses the slower one. Fix: on-the-fly gradient modulation (OGM-GE, CVPR 2022).
- *Modality competition.* Joint training induces a winner-take-all dynamic (Huang et al., ICML 2022).
- *Feature entanglement.* Noisy features from one modality entangle with predictive features of the other through shared fusion-head neurons (Chaudhuri et al., ICML 2025).
- *Decision-layer magnitude bias.* Raw feature-magnitude differences bias the classifier weights under concatenation (2025; correlational — see §4).

**Common fixes today.** Gradient modulation, loss reweighting, unimodal auxiliary losses, knowledge distillation into weak encoders, asymmetric fusion architectures, rank/basis reallocation, modality dropout.

**What is under-explored — and what this project targets.** The role of *raw activation scale* as a **cause** (not a correlate), a **predictive diagnostic** for collapse, and **normalization** as the studied fix (rather than an appendix trick or a gating weight). See §4 for exactly what is and isn't already claimed.

<a name="4-positioning"></a>
## 4. Honest positioning & the gap

This wedge is **defensible, not pristine.** The following is already published and must be cited and differentiated:

| Prior work | What it holds | How we differ |
|---|---|---|
| Peng et al., **OGM-GE**, CVPR 2022 | Collapse = optimization imbalance; on-the-fly gradient modulation | We test a *different cause* (scale, not learning-rate) and a simpler fix; we map *when each* wins |
| Huang et al., **Modality Competition**, ICML 2022 | Theory: joint training induces competition | We give a controllable, causal *scale* account, not a competition theory |
| Chaudhuri et al., **A Closer Look at Multimodal Representation Collapse**, ICML 2025 (arXiv 2505.22483) | Collapse from noisy-feature entanglement via shared neurons | Different mechanism and fix (basis reallocation vs. normalization) |
| **Revisit Modality Imbalance at the Decision Layer**, 2025 (arXiv 2510.14411) | **Closest threat.** Feature-magnitude differences bias the *decision layer* under concatenation (correlational) | We claim *sufficiency* (inject→collapse, normalize→cure), *whole-network*, in *end-to-end recognition*, plus a predictive diagnostic |
| Retrieval + soft fusion, 2026 (arXiv 2606.15514) | Uses **modality-specific z-score normalization** because embeddings differ in scale (appendix trick) | We make normalization the *studied object* with a mechanism, in classification, not a distance-metric trick |
| Channel-Exchanging-Networks | BN scaling factor as modality-importance signal | Related signal, different use; we use scale as *cause* + *diagnostic*, not a gating weight |

**Required day-one task:** read arXiv **2510.14411** and **2606.15514** in full. If 2510.14411 already contains a causal scale-injection + normalization-cure experiment, the novelty narrows to the **diagnostic (H1)** + **boundary (H3)** — still publishable, but you must know this before investing. Adjust framing accordingly.

<a name="5-questions"></a>
## 5. Research questions & hypotheses

**Central question.** Is per-modality feature-scale (activation-magnitude) disparity a *sufficient cause* of modality collapse in end-to-end multimodal recognition?

- **H1 (Diagnostic).** The activation-norm ratio ρ (defined in §7), measured *early* in training, predicts which modality collapses (higher-norm stream dominates; lower-norm stream's contribution decays).
- **H2 (Sufficiency / causality).** Multiplying one stream's features by a scalar c — with its information content held fixed — monotonically shifts dominance toward that stream. Because only scale changed, this isolates magnitude from information as a *cause*.
- **H3 (Minimal cure + boundary).** Per-modality normalization at the fusion boundary equalizes contribution and prevents collapse **where scale drives it**, matching/beating gradient-modulation baselines at lower complexity — and **fails to help where learning-rate disparity is the true driver.** That boundary is the reconciling result.

Falsifiability is built in: each hypothesis has a concrete measurement and a concrete outcome that would refute it (§21).

<a name="6-simple"></a>
## 6. Explained simply

Two people shout in a room. One has a megaphone (large feature magnitudes), the other whispers (small magnitudes). The megaphone wins — not because it says anything more useful, but because it's louder. A fusion network sums/concatenates the two streams, so the louder one dominates the shared representation *and the gradients*: the network leans on it and never learns from the quiet one, which withers.

Per-stream normalization hands both people the same microphone: it equalizes loudness so **content, not volume**, decides who's heard. The research claim is that this "volume" story is a genuine, sufficient cause — provable by turning the volume knob (induce collapse), turning on normalization (cure it), and reading the volume meter early (predict collapse).

<a name="7-technical"></a>
## 7. Technical formulation

**Setup.** Two encoders produce modality features h_a, h_v ∈ R^d at the fusion boundary. Fusion φ (sum, concat, or cross-attention) yields z = φ(h_a, h_v), classified by g(z). Training minimizes cross-entropy on labels y.

**The diagnostic.** Activation-norm ratio over a batch B:

  ρ = ( E_B ||h_a||₂ ) / ( E_B ||h_v||₂ )

Track ρ per epoch. H1: ρ at an early epoch predicts final per-modality contribution.

**Why scale plausibly drives dominance (the hypothesis to test, not a theorem).**
- *Sum fusion:* z = h_a + h_v. If ||h_a|| ≫ ||h_v||, z ≈ h_a; the weak stream contributes negligibly to the representation and to ∂L/∂z routed back to its encoder.
- *Concat fusion:* z = [h_a ; h_v], classifier weight W = [W_a ; W_v]. The logit contribution of block m scales with ||h_m||·||W_m||. Early on, the larger-norm block produces larger logit swings, so its weights receive larger, more useful gradients and grow faster; the weak encoder is starved of learning signal → collapse.
- *Cure:* normalization rescales each block to comparable norm, equalizing both the representation contribution and the gradient flow.

**Interventions.**
- *Scale injection (tests H2):* replace h_m ← c · h_m for a chosen stream, sweeping c, with information fixed by construction (§9 synthetic).
- *Per-modality normalization (tests H3):* apply LayerNorm, z-score (running/batch stats), or BatchNorm to each h_m *before* fusion. Ablate operator and placement.

**Primary scientific object.** A **collapse phase diagram** over (scale gap c × per-modality information × learning-rate gap), with ρ as an order parameter and per-modality contribution as the outcome.

<a name="8-contributions"></a>
## 8. Contributions

- **Primary.** A controlled, *causal* demonstration that feature-magnitude disparity is a **sufficient driver** of modality collapse (induce via scaling, cure via normalization), plus a cheap **pre-convergence diagnostic** (ρ) that predicts collapse.
- **Secondary.** A **boundary characterization** separating scale-driven collapse (normalization suffices) from optimization-driven collapse (gradient modulation needed); a **synthetic controlled-collapse generator** decoupling scale from information; validation on a **real low-resource task** (ISL manual vs. non-manual) plus a standard AV benchmark.
- **Engineering.** A reproducible, 8 GB-native multimodal-fusion testbed with the interventions as drop-in modules.
- **Not claimed.** State-of-the-art accuracy; discovery of modality collapse. The value is mechanism + diagnostic + minimal fix + boundary.

<a name="9-datasets"></a>
## 9. Datasets — full details & download

Three tiers: (A) synthetic, for **causal** claims; (B) a standard audio-visual benchmark, for **comparability**; (C) the ISL data, the **real-world differentiator**. Optional (D) a second AV benchmark for generalization.

### A. Synthetic controlled-collapse generator (you build this — carries H2)

The only setting that cleanly separates **scale** from **information**, so it carries the causal claim. Spec:

- Draw label y ∈ {1..K} uniformly.
- For each modality m ∈ {a, v}, define K class prototypes μ^m_1..μ^m_K ∈ R^{d_m}. Feature: x_m = α_m · ( μ^m_y + σ_m · ε ),  ε ~ N(0, I).
  - **Information knob:** control per-modality predictability via the prototype separation / margin (distance between μ^m_k) and/or noise σ_m. Larger margin ⇒ more predictive stream. Measure it: fit a unimodal linear probe per stream and record its accuracy — this is the *operational* information level.
  - **Scale knob:** α_m is a scalar magnitude multiplier, **independent of information** (it multiplies signal and noise together, leaving the SNR/margin-to-noise ratio unchanged).
- To test H2: hold both streams' margins (information) fixed and equal, then sweep the scale ratio α_a/α_v. Any dominance shift is *caused by scale alone*.
- To test H3: at a fixed scale gap that induces collapse, toggle per-modality normalization and measure recovery.
- Encoders: tiny MLPs (2–3 layers). Trains in seconds–minutes on CPU/GPU.

*No download. Ship the generator (`data/synthetic.py`) with a seed and a config so results reproduce.*

### B. CREMA-D — standard audio-visual benchmark (carries comparability)

- **What:** Crowd-sourced Emotional Multimodal Actors Dataset. 7,442 clips, 91 actors (48 M / 43 F), 6 emotions (anger, disgust, fear, happy, neutral, sad), 12 sentences, 4 intensity levels. Audio + face video. The canonical benchmark for modality-imbalance work (used by OGM-GE).
- **Two streams:** audio (log-mel spectrogram) + visual (face-cropped frames). This is a genuine scale-mismatch setting — spectrogram magnitudes and CNN visual features differ in raw scale, exactly the regime under study.
- **Size / license:** ~470 MB audio-only; full AV via git-lfs repo. License: research-use; the HF audio mirror is ODbL.
- **Download:**
  - Full audio-visual (video + wav): GitHub — `https://github.com/CheyneyComputerScience/CREMA-D` (`VideoFlash/`, `VideoMP4/`, `AudioWAV/`; pull with `git lfs`).
  - Audio-only mirrors: HuggingFace `https://huggingface.co/datasets/myleslinder/crema-d` ; TensorFlow Datasets `tfds.audio.CremaD`.
- **Split:** speaker-independent (a speaker appears in exactly one split). A common split is ~6,698 train / 744 test; better, use a fixed speaker-independent train/val/test and report it. **Never** let a speaker straddle splits (leakage).

### C. INCLUDE — Indian Sign Language (the differentiator; manual vs. non-manual streams)

- **What:** Indian Lexicon Sign Language Dataset (AI4Bharat / IIT Madras). 4,287 videos, ~263–266 word signs, 15 categories, ~0.27 M frames, 7 signers (recorded in Chennai). **INCLUDE-50** is a 50-sign subset for fast iteration.
- **Why it fits perfectly:** the official tooling extracts **MediaPipe Hands** + **BlazePose** keypoints. This gives your two streams *for free*:
  - **Manual stream:** hand landmarks (MediaPipe Hands).
  - **Non-manual stream:** face-mesh landmarks (eyes/brows/mouth) + upper-body pose (BlazePose).
  This is exactly the manual/non-manual split your hypothesis needs, without a custom pipeline — and keypoint sequences are tiny, so encoders are small (LSTM/GRU/small Transformer) and 8 GB-trivial.
- **Size / license:** ~56.8 GB full video; keypoint sequences are far smaller. Start with **INCLUDE-50** or a ~10–15-sign subset (prior 8 GB / free-tier studies used ~13 classes, ~2 GB). License: CC BY-SA 4.0.
- **Download:**
  - Videos: Zenodo record `4010759` — `https://zenodo.org/record/4010759`.
  - Metadata + download script: HuggingFace `https://huggingface.co/datasets/ai4bharat/INCLUDE`.
  - Code + keypoint extraction + pretrained models: GitHub `https://github.com/AI4Bharat/INCLUDE` (`generate_keypoints.py` produces MediaPipe Hands + BlazePose keypoints).
- **Week-1 check (now largely answered):** confirm the keypoint extractor cleanly yields separable hand vs. face+pose tensors on your subset. If for any reason it doesn't, CREMA-D + synthetic still carry every core claim and ISL becomes a case study.

### D. Optional second AV benchmark (generalization)

- **AVE (Audio-Visual Event):** 4,143 ten-second clips, 28 event classes, audio + video, second-level annotations. Tian et al., ECCV 2018. Used in modality-imbalance work. Get the current official link from the ECCV-2018 AVE project page/repo (search "AVE Audio-Visual Event ECCV 2018"). *(URL intentionally not hard-coded here — confirm the live source rather than trust a stale link.)*
- **Kinetics-Sounds:** ~19k clips over 34 sound-producing Kinetics classes (Arandjelović & Zisserman, 2017; built on Kinetics, Kay et al. 2017). Also used by OGM-GE. Requires downloading the relevant Kinetics subset from its official source.

### Dataset summary

| Tier | Dataset | Modalities / streams | Size | Role | Source |
|---|---|---|---|---|---|
| A | Synthetic | 2 controllable streams | trivial | Causal (H2/H3) | you build (`data/synthetic.py`) |
| B | CREMA-D | audio (log-mel) + face video | ~0.5 GB audio / larger AV | Comparability, baselines | GitHub / HF / TFDS |
| C | INCLUDE(-50) | manual hand kpts + non-manual face+pose kpts | subset ~2 GB | Real-world differentiator | Zenodo / HF / GitHub |
| D | AVE / Kinetics-Sounds | audio + video | few GB | Optional generalization | official ECCV/Kinetics sources |

<a name="10-method"></a>
## 10. Method & architecture

Keep everything small — this is what makes it 8 GB-native.

- **Encoders (per modality).**
  - CREMA-D: small audio CNN on log-mel spectrograms; ResNet-18-scale (or lighter) on face-cropped frames.
  - INCLUDE: lightweight sequence encoders (LSTM / GRU / small Transformer) over keypoint sequences — one for hand landmarks, one for face+pose landmarks.
  - Synthetic: 2–3-layer MLPs.
- **Fusion boundary φ:** concatenation, summation, and cross-attention — **fusion type is a studied variable**, not a fixed choice.
- **Intervention modules (drop-in):**
  - `interventions/scale_injection.py` — multiply a stream's features by c (H2).
  - `interventions/per_modality_norm.py` — LayerNorm / z-score / BatchNorm per stream at the fusion boundary (H3); configurable operator & placement.
  - `interventions/diagnostic.py` — compute and log ρ (and decision-layer weight-norm ratio) each epoch (H1).
- **Head:** linear or shallow MLP.

Every module answers a specific claim. Cross-attention is included only because fusion type may modulate the effect (a question worth testing), not to look sophisticated.

<a name="11-baselines"></a>
## 11. Baselines

1. **Naive concat, no normalization** — the collapse baseline.
2. **OGM-GE** (Peng et al., CVPR 2022) — standard gradient-modulation fix.
3. **A loss-reweighting / balancing method** — one recent, reproducible baseline.
4. **Decision-layer fix** (arXiv 2510.14411) — closest-neighbor sanity check, if reproducible.
5. **Per-modality normalization (ours)** + ablations over operator/placement.

Each comparison isolates a question: (2)–(3) test "does the simple fix match complex ones where scale drives collapse?"; (4) checks the nearest neighbor; ablations isolate which part of normalization matters.

<a name="12-metrics"></a>
## 12. Metrics

- **Multimodal gain:** Acc_av − max(Acc_a, Acc_v). (Collapse ⇒ small or negative.)
- **Per-modality contribution (masking):** Δ_m = Acc_av − Acc_{av∖m}, i.e., accuracy drop when zeroing modality m at inference. A collapsed modality has Δ_m ≈ 0.
- **Collapse indicators:** activation-norm ratio ρ; decision-layer weight-norm ratio ||W_a||/||W_v||.
- **Diagnostic quality (H1):** correlation and/or AUROC between *early* ρ (e.g., epoch 5) and *final* collapse (which modality has Δ_m ≈ 0).
- **Causal effect (H2):** dominance (Δ_a − Δ_v, or a contribution share) as a function of injected scale c — report monotonicity and effect size.
- **Task metrics:** accuracy / macro-F1, **multi-seed mean ± std** (report the spread, not just the best run).

<a name="13-protocol"></a>
## 13. Experimental protocol (the matrix)

| Exp | Question | Setup | Key variable | Supports hypothesis if… | Refutes if… |
|---|---|---|---|---|---|
| E1 | Does ρ predict collapse? | Synthetic + CREMA-D + INCLUDE, natural training | early-epoch ρ | early ρ correlates with final collapsed modality (high AUROC) | ρ uninformative |
| E2 | Is scale a sufficient cause? | Synthetic, info fixed & equal | injected scale ratio c | dominance shifts monotonically with c | no shift, or shift explained by info |
| E3 | Does normalization cure it? | Synthetic + CREMA-D + INCLUDE, collapse regime | per-modality norm on/off & operator | collapse prevented; contribution equalized | normalization doesn't recover the weak stream |
| E4 | Boundary: scale vs. LR-driven | Synthetic, vary scale gap × LR gap | which fix wins where | normalization wins in scale-driven, loses in LR-driven | no clean boundary |
| E5 | Baseline comparison | CREMA-D (+AVE), collapse regime | ours vs. OGM-GE etc. | normalization matches/beats at lower complexity in scale regime | strictly dominated everywhere |
| E6 | Ablations | all | fusion type, norm type/placement, encoder size, injected noise | operative factor isolated | effect vanishes under controls |

For every experiment, pre-register the metric and the outcome that would refute the hypothesis. This is the discipline that turns "tables" into "answers."

<a name="14-expected"></a>
## 14. Expected results (honest)

- **E1/H1:** early ρ predicts which modality collapses — a positive, useful diagnostic.
- **E2/H2:** injected scale monotonically shifts dominance at fixed information — the causal claim.
- **E3/H3:** per-modality normalization prevents collapse and is competitive with OGM-GE **where scale drives collapse**; it does **not** help where learning-rate drives it.
- **Most likely headline finding:** the **boundary** — a one-line normalization matches elaborate gradient-modulation in a well-defined regime, and ρ tells you which regime you're in. On CREMA-D, expect competitive-but-not-necessarily-SOTA accuracy; that's fine, because the claim isn't the number.

<a name="15-compute"></a>
## 15. Compute plan

- Two small encoders + fusion head; small datasets.
- 8 GB VRAM ample; mixed precision optional; moderate batch sizes.
- Rough per-run time: synthetic seconds–minutes; INCLUDE-keypoints minutes–low hours; CREMA-D low hours. The full grid (fusion × norm × scale gap × LR gap × seeds) is a matter of days on a laptop — precisely why this direction fits your constraints.

<a name="16-environment"></a>
## 16. Environment & setup

**Core dependencies:** Python 3.10+, PyTorch, torchvision, torchaudio, `mediapipe` (ISL keypoints), `librosa` (audio features), `opencv-python` (frame/face extraction), `numpy`, `scikit-learn` (metrics/probes), `pandas`, `matplotlib`, `pyyaml`. Optional: `tensorflow-datasets` (CREMA-D audio), a lightweight tracker (e.g., `wandb` or CSV logging).

**Typical setup:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision torchaudio
pip install mediapipe librosa opencv-python scikit-learn pandas matplotlib pyyaml
# CREMA-D (audio-visual): git clone + git lfs pull from CheyneyComputerScience/CREMA-D
# INCLUDE: download videos from Zenodo (record 4010759), then run AI4Bharat/INCLUDE generate_keypoints.py
python experiments/run.py --config configs/synthetic_causal.yaml   # start here (E2)
```

Provide `requirements.txt` / `environment.yml`, and one config per experiment so a reader reproduces a result with a single command.

<a name="17-repo"></a>
## 17. Repository structure

```
modality-scale-collapse/
  configs/            # one YAML per experiment; seeds pinned
  data/
    synthetic.py      # controlled-collapse generator (scale ⟂ information)
    crema_d.py        # audio (log-mel) + face-video loaders
    include.py        # manual (hand kpts) vs non-manual (face+pose kpts) loaders
  models/             # encoders (audio CNN, visual CNN, keypoint seq), fusion, heads
  interventions/
    scale_injection.py
    per_modality_norm.py
    diagnostic.py     # activation-norm ratio ρ, decision-layer norm ratio
  baselines/          # ogm_ge.py, loss_reweight.py, decision_layer.py
  experiments/        # e1_diagnostic, e2_causal, e3_cure, e4_boundary, e5_baselines, e6_ablation
  analysis/           # metrics, masking contribution, phase-diagram plotting
  results/            # logs, checkpoints (named by config hash + seed)
  README.md
  requirements.txt / environment.yml
```

<a name="18-roadmap"></a>
## 18. Implementation roadmap

| Phase | Deliverable | Success criterion | Common failure |
|---|---|---|---|
| 1. Testbed + synthetic generator | Reproduce collapse; ρ probe live | collapse appears; ρ measurable | collapse doesn't reproduce → tune scale/info gap |
| 2. Causal + cure on synthetic (E2, E3) | injection & normalization curves | monotonic shift; normalization cures | effect confounded by info → fix generator |
| 3. Real data (E1, E5) | CREMA-D + INCLUDE results, baselines | ρ predicts on real data; norm competitive | INCLUDE not stream-separable → fall back to CREMA-D + synthetic |
| 4. Boundary (E4) | scale-vs-LR phase diagram | clear regime boundary | boundary fuzzy → report honestly as a gradient |
| 5. Ablations (E6) | fusion/norm/encoder/noise sweeps | operative factor isolated | too many knobs → pre-register the grid |
| 6. Error analysis + write-up | failure taxonomy + paper | — | — |

**Gate:** if H1 and H2 both hold on synthetic by end of Phase 2, the paper is real. If both fail, stop and reassess before touching real data.

<a name="19-reproducibility"></a>
## 19. Reproducibility

Fixed seeds (report multi-seed spread); config-driven runs (one YAML → one experiment); experiment tracking (CSV or a tracker); checkpoints named by config hash + seed; dataset versioning and split scripts committed; environment file pinned; normalize features **after** splitting (avoid leakage). Target: another researcher reproduces the phase diagram from your configs.

<a name="20-paper"></a>
## 20. Paper outline

1. Abstract · 2. Introduction (the lived hack → the question) · 3. Problem definition · 4. Related work (OGM-GE, competition, entanglement, decision-layer, z-score norm — differentiate explicitly) · 5. Research gap · 6. Method · 7. Conceptual motivation (the scale mechanism) · 8. Experimental setup · 9. Results (E1 diagnostic, E2 causal, E3 cure) · 10. Phase diagram / boundary · 11. Ablations · 12. Transfer to ISL · 13. Limitations · 14. Conclusion. Every claim traces Claim → Experiment → Result; drop any claim the experiments don't support.

<a name="21-risks"></a>
## 21. Risks & kill criteria (check by week 3)

- **Scooped:** if arXiv 2510.14411 already contains the causal injection + normalization cure → novelty collapses to diagnostic + boundary. *Mitigation: read it day one.*
- **H2 fails** (scale necessary but not sufficient) → pivot to a "necessary co-factor" characterization; weaker but still a paper.
- **H1 fails** (early ρ doesn't predict collapse) → a core selling point is gone; reconsider.
- **Normalization strictly dominated** by gradient modulation everywhere → the "minimal fix" claim dies; the boundary claim may survive.
- **INCLUDE not stream-separable** → not fatal; CREMA-D + synthetic carry the core; ISL becomes a case study.

If E1 and E2 both hold on synthetic by end of Phase 2, proceed. If both fail, stop and reassess.

<a name="22-fit"></a>
## 22. Why this project (fit)

- **Anchored in your real experience** — the seed is unscoopable because it's yours.
- **8 GB-native** — small encoders, small data, full ablation grid on a laptop.
- **IEEE-conference-appropriate novelty** — a clean mechanism + diagnostic + minimal fix, sharply positioned against the two closest neighbors.
- **Disjoint from your portfolio** — the contribution is a general fusion phenomenon, not the ISL application.
- **Plays to your demonstrated strength** — controlled, honest, mechanism-first experiments.

<a name="23-references"></a>
## 23. References

**Methods / phenomenon**
- Peng et al. "Balanced Multimodal Learning via On-the-fly Gradient Modulation" (OGM-GE), CVPR 2022.
- Huang et al. "Modality Competition: What Makes Joint Training of Multi-modal Networks Hard?", ICML 2022.
- Chaudhuri et al. "A Closer Look at Multimodal Representation Collapse", ICML 2025. arXiv:2505.22483.
- "Revisit Modality Imbalance at the Decision Layer", 2025. arXiv:2510.14411. *(closest neighbor — read first)*
- "…Soft Fusion… (modality-specific z-score normalization)", 2026. arXiv:2606.15514. *(read first)*
- Wang, Tran, Feiszli. "Channel-Exchanging-Networks / What Makes Multi-modal Learning Better than Single (Provably)", NeurIPS 2020.

**Datasets**
- Cao et al. "CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset", IEEE T-AC 2014. GitHub: CheyneyComputerScience/CREMA-D · HF: myleslinder/crema-d · TFDS: `crema_d`.
- Sridhar, Ganesan, Kumar, Khapra. "INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition", ACM MM 2020. Zenodo: record 4010759 · HF: ai4bharat/INCLUDE · Code: github.com/AI4Bharat/INCLUDE.
- Tian et al. "Audio-Visual Event Localization in Unconstrained Videos" (AVE), ECCV 2018.
- Arandjelović & Zisserman. "Look, Listen and Learn" (Kinetics-Sounds), ICCV 2017; Kay et al. "The Kinetics Human Action Video Dataset", 2017.

*Verification note:* dataset specs and links above were checked against primary sources (Sep 2026). AVE/Kinetics-Sounds links are intentionally not hard-coded — confirm the current official source before use.
