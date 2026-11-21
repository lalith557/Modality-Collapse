# RESULTS_LOG

Running log of what was run, its config/seed/commit, the outcome, and what it
means for each paper claim. One entry per run or analysis.

Provenance baseline: repo at commit `c76c5e6` (branch `master`), 3 uncommitted
files (paper edits from the prior task). All existing result CSV/JSON were
produced by earlier commits (Sep 19–21 runs); this log starts at the CVPR
revision.

---

## 2026-11-09 — Phase 0 (audit, NO training runs)

**What ran:** read-only analysis of existing logs only (no GPU training).
Scripts: a one-off diagnostic (`signflip_diag.py`, not kept in the repository) + inline pandas checks over
`results/{disentangle,sgd_disentangle,cremad_disentangle}/*.csv`.

**Verified against logs (Hard Rule 1 — every paper number traces to a run):**

| Paper Table I value | Logged value | Source file |
|---|---|---|
| Adam synth forward ΔD −0.070 | −0.0695 | results/disentangle/disentangle_primary.csv |
| Adam synth gradient ΔD −0.0005 | −0.0005 | " |
| Adam synth ordinary ΔD −0.069 | −0.0693 | " |
| Adam CREMA forward ΔD +0.259 | +0.2589 | results/cremad_disentangle/cremad_disentangle_primary.csv |
| Adam CREMA gradient ΔD −0.001 | −0.0011 | " |
| SGD synth forward ΔD +0.244 | +0.2443 | results/sgd_disentangle/sgd_disentangle_primary.csv |
| SGD synth gradient ΔD +0.200 | +0.2000 | " |
| SGD synth ordinary ΔD +0.322 | +0.3220 | " |
| gradient-only ρ_Δθ SGD 95.9 | 95.91 | results/sgd_disentangle/… |
| gradient-only ρ_Δθ Adam 1.61 | 1.61 (CREMA-D) | results/cremad_disentangle/… |
| gradient-only r_g SGD 318.5 | 318.47 | results/sgd_disentangle/… |
| gradient-only r_g Adam 282.7 | 282.69 (CREMA-D) | results/cremad_disentangle/… |

**Sign-flip diagnosis (from logs, no run):** NOT a bug. Baseline dominance is
+0.003 (synthetic, balanced by construction) vs +0.075 (CREMA-D). Forward arm:
synth acc_A 0.501→0.447, acc_B 0.498→0.514 (shift −0.070); CREMA acc_audio
0.245→0.502 (shift +0.259). Same operators, same masking, same scaled modality
(A). Sign is not robust at a ≈0 baseline; magnitude + forward≈ordinary hold in
both.

**Means for claims:** H1–H4 numbers SUPPORTED by logs. Sign flip = regime
effect, not error (paper framing already correct). Open: synthetic Adam
update-ratio not logged; decoupled metrics (probe on frozen encoder reps per
arm, mean-ablation, Shapley) not yet computed.

**Next:** Phase 1 (on "go").

---

## 2026-11-09 — Phase 1 (decoupled metrics + sign-flip decisive test)

**Run:** `experiments/phase1_decoupled.py --seeds 10 --alphas 10,100`,
commit c76c5e6, device RTX 4070 (8.6 GB), 160 runs, wall 376.6 s.
Outputs: `results/phase1_decoupled/phase1_primary_seeds10.csv`,
`…/phase1_trajectories_seeds10.csv`, `…/figures/phase1_trajectories.png`.
Pre-registered predictions in `preregistration/PREREGISTRATION_PHASE1.md` (written before the run).

**Decisive test (baseline→forward, Adam, α=100, 10 seeds):**
| metric | shift | 95% CI |
|---|---|---|
| masked acc_A | −0.0525 | ±0.0076 |
| **frozen-encoder probe acc_A** | **−0.0339** | **±0.0052** |
| mean-ablation dominance | −0.0559 | — |
| Shapley_A | −0.0432 | ±0.0028 |

All four move the same way; every CI excludes 0. **Verdict: the acc_A drop is a
REAL degradation of the scaled modality (encoder quality + contribution), not a
head/off-distribution artifact. Pre-registered prediction P-main REFUTED.**

**Mechanism (trajectories, α=100):** Adam compensates the forward scale
(ρ_fus 80→9, ρ_enc→0.09) and the compensation slightly degrades modality A;
SGD does NOT compensate (ρ_fus≈100) so A instead dominates (acc_B stuck ≈0.15).
The synthetic−CREMA sign difference reflects compensation strength + baseline
balance, NOT a convention bug.

**Fix (audit 4.1): synthetic Adam update ratio now logged.** gradient-only
ρ_Δθ = **1.001 ± 0.010** (Adam) vs **95.8 ± 4.0** (SGD), same dataset.

**Claims:** H3 (Adam forward≫gradient) SUPPORTED + strengthened (holds under
probe/Shapley/mean-ablation); gradient-only null under Adam SUPPORTED across all
metrics; metric-coupling objection (#3) addressed for synthetic (effect survives
decoupled metrics). WEAKENED/REFUTED: the paper's "read by magnitude, not sign"
sentence (added in the prior rewrite) — the sign is real; correction proposed.

**Next:** Phase 2 (on "go") — derive Prop 1 rigorously, α-sweep collapse, ε-sweep
(P3), η/α cancellation (P4), muP comparison (VERIFY).

---

## 2026-11-09 — Phase 2 (mechanism + theory), synthetic

Runs (commit c76c5e6, RTX 4070): `phase2_mechanism.py` (520 runs, 0 diverged,
1250 s), `phase2_forwardk.py`, `phase2_eps_cancel.py`, `phase2_ccurve.py`.
Prop 1 derived (`THEORY.md`). Paired 10-seed CIs.

- **lrmatch (H-lr):** reproduces forward DIRECTION at every α, overshoots
  (paired lrmatch−forward excludes 0 even at α=3, stable: −0.020±0.005; grows to
  −0.089±0.018 at α=300). → raising A's effective LR degrades A; forward's
  compensation buffers it. H-comp refuted.
- **forward_k multi-k:** k=1,5,10,20 all null → H-init refuted (not a short-k
  artifact). k=40 point invalid (eval-scale mismatch in script).
- **P3 ε-sweep:** gradient-only update ratio 1.00→1.02→2.84 as ε 1e-8→1e-4→1e-2;
  cancellation weakens at large ε (contribution effect still small). SUPPORTED.
- **P4 η/α:** SGD gradient-only with enc-A LR=η/α → dom +0.003 = baseline exactly
  (update ratio 0.99). Clean confirmation of m=s_f·s_g/α=1.
- **Adam forward≡ordinary / gradient-null:** ε→0 ALGORITHMIC IDENTITIES, not
  tests.
- **c(α):** synth Adam 0.76/0.69/0.62/0.52/0.44 (α 3–300), synth SGD ≈0, CREMA
  0.60/0.58/0.39. Matched-α: synth Adam > CREMA.
- **A quality (probe) sign flip:** SGD gradient/forward improve A at small α,
  degrade at large α; Adam forward always degrades, gradient null.
- Collapse: multiplier organizes effects within optimizer (Adam forward≡ordinary
  at m=α; SGD forward≈gradient at m=α, ordinary at m=α²); not one universal curve
  (opposite signs; lrmatch deviates).

**Phase 2 CLOSED (M1–M11).** LR-sweep, tuned-baseline, CREMA probe, multi-k,
per-modality heads (4 arms), early-stop, redraw, P3-ext, multiplicativity all
done. Two paper-weakening findings: **W1** SGD contribution direction is an
undertrained-baseline artifact (tuned lr=1e-2 → forward/gradient degrade A);
**W2** CREMA-D forward effect is a masked-accuracy artifact (decoupled metrics
ns). See `CLAIMS_LEDGER.md` (W1/W2) and `experiments/PHASE2_FINDINGS.md`. Figures:
phase2_collapse_quality.png, phase2_reference_crema.png.

## 2026-11-09 — Phase 3 feasibility (no runs) → deferred (compute and disk limits)
Disk 53 GB free (95% used); VRAM 8.6 GB. Local: CREMA-D-full (raw), UTD-MHAD.
Absent (need downloads): Kinetics-Sounds, AVE, VGGSound, CMU-MOSEI; OGM-GE/PMR/
AGM repos. Full Phase-3 dataset suite not feasible on this GPU/disk. W2 weakens
the real-data core claim, so the thesis was narrowed: Phase 3 is scoped to
CREMA-D only.

## 2026-11-09 — Phase 3 SCOPED (CREMA-D, tuned LR, AdamW+SGD, decoupled)
`phase3_cremad_scoped.py`, commit c76c5e6, 1731 s. Best SGD lr=3e-2.
- **AdamW (n=5): W2 CONFIRMED.** forward masked dom +0.259±0.031 but meanabl
  +0.012 (ns), Shapley +0.014 (ns), probe +0.023. gradient null. → real-data
  contribution shift NOT supported by decoupled metrics.
- **SGD (W3): unstable.** forward diverged 3/5, gradient 2/2 at tuned lr=3e-2;
  surviving 2 forward seeds show A degradation (probe −0.358, n=2 unreliable).
Phase 4 (payoff baselines) SKIPPED — OGM-GE/PMR/AGM repos+data absent at the time (run later as Phase 4 below).

## 2026-11-13 — Controls + Phase 5/6 COMPLETE (venue: IEEE ICRCICN 2027)
- Positive control: audio noise → probe −0.021/−0.064, Shapley −0.029/−0.051
  (metrics validated). SGD α=10 stable: masked +0.15 but decoupled ns (W2 holds).
- Phase 5: main.tex rewritten (IEEEtran, 6 pp, 44 refs, clean compile). New title/
  thesis; decoupled metrics; real-data scoped to CREMA-D/our-split/weak-encoders;
  H1 Adam-specific; Cohen's d removed; muP/GRL/MMPareto added, 4 padding cites
  removed; figs fig_synth.pdf/fig_real.pdf. Every number traces to CLAIMS_LEDGER.
- Reproduction guide: `REPRODUCING.md`.

---

## 2026-11-14 — Phase 4 (RUNNABLE): payoff baselines, SGD stabilization, 2nd dataset

Three runs, commit `c76c5e6`, RTX 4070 (8.6 GB), 5 seeds each. Scripts:
`experiments/phase4_baselines.py` (1997 s), `phase4_sgd_stable.py` (2476 s),
`phase4_utd_decoupled.py` (100 s). All CIs below recomputed from the CSVs with a
one-off analysis script (the scripts' own in-line SGD summary had a
`df.clip` bug — `clip` is a DataFrame *method*, so `df.clip==c` was always False;
fixed to `df["clip"]`; CSVs themselves are correct). Outputs in
`results/phase4_baselines/{phase4_cremad,phase4_sgd_stable,phase4_utd}.csv`.

**4A — CREMA-D payoff baselines (AdamW, NO scaling; method−vanilla, paired 95% CI, n=5).**
Does a published rebalancing / modality-LR method move the DECOUPLED metrics or
only masked accuracy?
| method | masked dom | mean-abl dom | Shapley_a | probe_a | probe_v |
|---|---|---|---|---|---|
| **OGM-GE** (k_a=0.273, k_v=0.999) | **+0.240±0.030** | −0.023±0.017 | −0.026±0.021 | −0.014±0.009 | +0.009±0.013 |
| modlr_video ×5 | +0.019±0.039 ns | −0.004 ns | −0.002 ns | −0.000 ns | −0.014±0.008 |
| modlr_audio ×5 | **+0.117±0.077** | +0.007 ns | +0.012 ns | +0.005 ns | +0.003 ns |
OGM-GE (faithful re-impl, α=0.5, GE γ=0.5) strongly attenuates the dominant audio
modality (k_audio=0.27) and shifts **masked dominance +0.24** while every decoupled
metric stays ≤0.026 in magnitude. modlr_audio shifts masked +0.12 with all decoupled
flat. → **An evaluator reading masked-accuracy dominance would see a large rebalancing
effect the representation-level metrics do not corroborate. Payoff baseline CONFIRMS
the thesis on a published method (OGM-GE, Peng et al. CVPR'22).**

**4B — CREMA-D SGD stabilization (α=100, lr=1e-2, momentum 0.9; arm−baseline, paired 95% CI, n=5).**
Divergence by (clip,arm): clip=0 → gradient **5/5 diverged**; clip=1.0 → **0/5
diverged** (all converge). Also clip=0 forward had 2/5 seeds collapse to a
degenerate near-chance solution (probe≈0.19). **Gradient-norm clipping (max 1.0)
STABILIZES SGD+scaling at α=100.** Under stable SGD (clip=1.0):
| arm | masked dom | mean-abl dom | Shapley_a | probe_a |
|---|---|---|---|---|
| forward | +0.132±0.056 | −0.062±0.026 | −0.059±0.024 | −0.021±0.014 |
| gradient | +0.200±0.035 | −0.001±0.015 ns | +0.002±0.013 ns | −0.001±0.014 ns |
→ (1) the W3 "SGD unstable at α=100" caveat is an OPTIMIZATION artifact, fixable
with standard clipping. (2) The masked-vs-decoupled MISMATCH HOLDS under stable
SGD: the gradient arm moves masked dominance **+0.20** with probe/Shapley/mean-abl
all flat — a clean masked-accuracy artifact under SGD. The forward arm moves
decoupled metrics modestly (probe −0.021) but masked (+0.132) still overstates ~6×.
Consistent with m_SGD=s_f·s_g (gradient scaling has a real training effect under SGD,
unlike Adam) whose representational footprint is small because the audio probe is
near its weak-encoder ceiling (~0.49).

**4C — UTD-MHAD (2nd real dataset: depth=near-chance, inertial=informative; scale
informative inertial by α=100; Adam; arm−baseline, paired 95% CI, n=5).**
baseline means: masked dom −0.191, probe_depth 0.131, probe_inertial 0.633.
| arm | masked dom | mean-abl dom | Shapley_scaled | probe_depth | probe_inertial |
|---|---|---|---|---|---|
| forward | **−0.525±0.034** | −0.107±0.028 | +0.113±0.031 | −0.011±0.033 ns | **+0.102±0.034** |
| gradient | +0.003±0.010 ns | +0.004±0.006 ns | −0.002±0.004 ns | −0.010±0.007 | −0.001±0.002 ns |
→ (1) **Gradient arm ≡ baseline on ALL metrics** — Adam cancels gradient-scaling
(m_Adam=s_f); clean replication on a 2nd real dataset. (2) **Headroom nuance (honest
refinement of W2):** on CREMA-D the scaled modality (audio) was saturated so forward
scaling moved masked but NOT the probe (pure masked artifact); on UTD the scaled
modality (inertial) has headroom (0.633→0.735), so forward scaling — which raises
the effective LR under Adam — produces a GENUINE +0.102 probe gain. BUT masked
dominance still OVERSTATES it ~5× (−0.525 masked vs +0.102 probe). So "masked can
mislead" generalizes; "forward is a pure masked artifact" was specific to the
saturated-probe case.

**Download-gated breadth NOT run (honest future work):** Kinetics-Sounds, AVE,
VGGSound (100s of GB); ImageNet/AudioSet-pretrained encoders; attention fusion at
scale — infeasible on this 8 GB / ~50 GB-free box. UTD-MHAD stands in as the genuine
second dataset. (Unchanged from Phase 3 feasibility note.)

**Means for claims:** see CLAIMS_LEDGER M-P4a/b/c and W3 status change. Core thesis
(scale = hidden per-modality LR; masked-accuracy dominance can mislead) CONFIRMED and
STRENGTHENED (published baseline + 2nd dataset + stable SGD). W3 downgraded from
"unstable/exploratory" to "resolved by clipping". W2 refined (headroom mechanism).

---

## 2026-11-15 — Reviewer-report revision (Student-t CIs + Issue 2 clip measurement)

**Issue 1 (CIs):** recomputed EVERY paper interval as a paired Student-$t$ 95% CI
(t_{0.975,n-1}; n=10 synth t=2.262, n=5 real t=2.776, n=3 pos-control t=4.303) from the
seed-level CSVs. Sources: synth phase1_primary_seeds10;
CREMA-D AdamW fwd phase2_cremad_probe; pos-control phase3_controls (n=**3**, not 5);
Phase-4 CSVs. **Significance changes under t (report honestly):** OGM-GE mean-ablation
(-0.023±0.024) and Shapley (-0.026±0.030) CIs now INCLUDE 0 (were excl under 1.96);
OGM probe -0.014±0.012 still excl (barely). UTD gradient probe_depth -0.010±0.010 now
incl 0 (cleaner gradient≡baseline). All headline effects (masked shifts, synth forward,
UTD fwd probe_inert +0.102±0.048, stable-SGD gradient masked +0.200±0.050) still exclude 0.
Full t-CI table in the paper (Tables I-II).

**Issue 2 (clip spec + measurement):** clipping in phase4_sgd_stable.py is
`clip_grad_norm_(model.parameters(), 1.0)` = GLOBAL over ALL params, applied AFTER the
GradScale operator's backward, BEFORE opt.step, every step; SGD lr=1e-2 mom=0.9 wd=0.
Measured (one-off instrumentation run, gradient arm, 164 batches): per-modality gradient
ratio r_g = **1011.3 pre-clip and 1011.3 post-clip** (max |Δ| 9.7e-5), 100% of steps
clipped. → global norm clipping is a single scalar on all params, so it PRESERVES the
per-modality ratio exactly; it caps step size (prevents divergence) without removing the
×α amplification. The +0.200 masked shift is therefore the amplified enc_A gradient
reaching the weights, not a clipping artifact.

**Issue 3 (OGM-GE):** baselines/ogm_ge.py = discrepancy-ratio coeff k=1-tanh(α·ratio) on
ENCODER grads only (α=0.5), per-modality softmax score via shared sum-fusion head, GE =
Gaussian-noise approx (γ=0.5); AdamW lr=1e-3 30ep 5 seeds, seed-matched to vanilla. Two
documented adaptations (sum-fusion score via shared head; GE Gaussian approx) → call it a
faithful reimplementation WITH documented adaptations, no accuracy claim.
(Superseded wording, round 3: "OGM-GE reimplementation adapted to our shared sum-fusion
setting, with two implementation modifications" — "faithful" dropped per reviewer.)

## 2026-11-15 — Reviewer round 3 (protocol, clipping, probe, wording)

Facts verified from source before writing (no new training except one measurement):
- **CREMA-D split** (data/cremad/cremad_manifest.csv): train 64 actors / 5234 clips, val 9 /
  738, test 18 / 1470. Val actor IDs: 1001, 1018, 1027, 1039, 1042, 1047, 1049, 1068, 1073.
- **UTD-MHAD split** (configs/utd_mhad.yaml): train {1,3,5}=323, val {7}=108, test
  {2,4,6,8}=430. Phase-4 decoupled runs loaded train/test only (val subject held out, unused).
- **Model selection:** reported real-data runs train a FIXED number of epochs (30/60), no early
  stopping or checkpoint selection; validation used only to tune the SGD base LR (phase 3).
- **Probe** (analysis/probes.py): multinomial logistic, Adam lr 0.05, 400 steps, FIXED L2
  weight_decay=1e-4, train-stat standardization — identical across modalities/conditions.
- **Baseline gradient ratio** (new measurement, CREMA-D baseline arm, SGD, first epoch):
  enc_audio/enc_video grad-norm ratio mean 8.59, median 7.92. Explains the clipped gradient
  arm's ~1011 = ×100 operator gain × ~10× baseline audio-vs-video asymmetry.
- Paper: 6 pp, 0 overfull, 36 refs ascending by year. To fit: Mod-LR moved from Table I to
  text (numbers unchanged), venue names abbreviated (Proc. ICML etc.), long author lists et al.

## 2026-11-15 — Reviewer round 4 (AdamW weight decay, protocol separation, title)

**Optimizer audit (code, not memory) — CORRECTS the paper's "AdamW" labels:**
- phase2_cremad_probe.py (CREMA-D forward/gradient, Table I row): `torch.optim.Adam`, wd = cfg 0.0.
- phase4_utd_decoupled.py (UTD-MHAD): `torch.optim.Adam`, wd 0.0. cremad_disentangle.py: Adam, wd 0.
- phase4_baselines.py (OGM-GE / Mod-LR) and phase3_cremad_scoped adamw arm: `AdamW(weight_decay=0.0)`
  = identical update to Adam.
- phase3_controls.py (POSITIVE CONTROL): `AdamW(params, lr=lr)` with NO weight_decay arg →
  PyTorch default **λ = 0.01** (verified: torch 2.11 AdamW default 0.01). Applied to all control
  arms incl. its noise-free reference; no scaling operator involved.
- Update ratio (cremad_disentangle.py:113, phase1_decoupled.py:104) = ‖θ_after − θ_before‖ from
  parameter snapshots → full update; with wd 0 it has no decay component.
→ Paper now labels real-data optimizer "Adam (weight decay 0)", states the positive-control
  exception, and conditions the cancellation claim on zero weight decay. No numbers change.
Also: real-vs-synthetic protocol separated explicitly; OGM-GE called an "adapted" baseline in
abstract/contributions/conclusion; title → "Feature Scale as a Hidden Per-Modality Learning Rate:
When Masked-Accuracy Dominance Misleads" (reviewer-preferred, optional). Paper 6 pp, 0 overfull.
