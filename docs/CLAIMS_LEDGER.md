# CLAIMS_LEDGER

Every claim in play, tagged **MEASURED** (logged run, with source), **HYPOTHESIZED**
(plausible, not yet established), or **REFUTED**. Updated each phase. Paper edits
are deferred to Phase 5; nothing here is in the paper yet unless noted.

---

## MEASURED

- **M1 (Phase 0).** Operators (ForwardScale ×α/×1, GradScale ×1/×α), sanity gate,
  metrics, seeds, splits in code = paper. Every Table I number reproduces from
  logs. Source: `RESULTS_LOG.md` (Phase 0 audit).
- **M2 (Phase 0).** The dominance sign flip is NOT a convention bug: CREMA-D
  imports the same `arm_apply`, masking, and dominance = acc_A−acc_B; scaled
  modality = A/audio in both. Source: cremad_disentangle.py:36.
- **M3 (Phase 1).** Synthetic Adam forward (α=100, 10 seeds) lowers the scaled
  modality on ALL metrics, every CI excludes 0: masked acc_A −0.053±0.008,
  frozen-encoder probe acc_A −0.034±0.005, mean-ablation dom −0.056, Shapley_A
  −0.043±0.003. → the scaled modality's encoder quality AND contribution really
  fall; not a masking artifact. Source: phase1_primary_seeds10.csv.
- **M4 (Phase 1).** Gradient-only is null under Adam across every metric
  (probe shift +0.0001±0.0007). forward ≈ ordinary on every metric.
- **M5 (Phase 1).** Synthetic same-dataset update ratio (gradient-only, α=100):
  Adam ρ_Δθ = 1.001±0.010, SGD = 95.8±4.0.
- **M6 (this turn).** At MATCHED α, synthetic Adam compensates MORE than CREMA-D
  (baseline-corrected c = 1−ln(ρ_fus(α)/ρ_fus(1))/ln α):
  α=10 → 0.69 (synth) vs 0.58 (CREMA); α=100 → 0.52 (synth) vs 0.39 (CREMA).
  NOTE: the code's logged `compensation` omits the ρ_fus(1) baseline term, so it
  undercounts c when baseline ρ_fus≠1 (CREMA audio=1.22). Paper's c=0.58–0.60
  (α=5,10) is the baseline-corrected value and matches the disentangle data.
- **M7 (Phase 1).** Mechanism (trajectories): Adam compensates the forward scale
  (ρ_fus 80→9, ρ_enc→0.09); SGD does not (ρ_fus≈100), so under SGD the scaled
  modality dominates and the other under-trains (acc_B stuck ≈0.15).
- **M-P4a (Phase 4, 2026-11-14) — PAYOFF BASELINE (OGM-GE).** CREMA-D, AdamW, no
  scaling, method−vanilla paired 95% CI (n=5). A faithful OGM-GE re-impl (Peng et
  al. CVPR'22; α=0.5, GE γ=0.5) attenuates the dominant audio gradient
  (mean k_audio=0.273, k_video=0.999) and shifts **masked dominance +0.240±0.030**
  while every decoupled metric is ≤0.026: mean-abl −0.023±0.017, Shapley_audio
  −0.026±0.021, probe_audio −0.014±0.009, probe_video +0.009 (ns). modlr_audio ×5:
  masked +0.117±0.077, all decoupled ns; modlr_video ×5: masked ns. → a published
  rebalancing/modality-LR method moves masked-accuracy dominance far more than any
  representation-level metric. **Masked-accuracy dominance can mislead about a real
  method, not only about our synthetic operator.** Source: phase4_cremad.csv.
  **UPDATE (2026-11-15, Student-t CIs):** under paired t_{0.975,4}=2.776, OGM-GE
  mean-ablation (-0.023±0.024) and Shapley (-0.026±0.030) CIs now INCLUDE 0 (n.s.);
  only the tiny probe (-0.014±0.012) is resolved. Reframe as "masked moves +0.240 while
  the decoupled metrics are mostly unresolved" (cleaner, not weaker). ALL paper CIs
  switched to paired Student-t per reviewer Issue 1 (see RESULTS_LOG 2026-11-15).
- **M-P4b (Phase 4) — SGD STABILIZED; mismatch holds.** CREMA-D, SGD α=100,
  lr=1e-2, mom=0.9. Gradient-norm clip (max 1.0): divergence 5/5→0/5 for the
  gradient arm (clip=0 also collapsed 2/5 forward seeds to near-chance). Under
  stable SGD (clip=1.0, arm−baseline paired 95% CI n=5): gradient masked dom
  **+0.200±0.035** with mean-abl −0.001, Shapley +0.002, probe −0.001 (all ns);
  forward masked dom +0.132±0.056 with probe −0.021±0.014 (tiny), mean-abl
  −0.062±0.026. → under STABLE SGD the masked shift is again not matched by the
  decoupled metrics (gradient arm = clean masked artifact; forward overstates ~6×).
  Consistent with m_SGD=s_f·s_g. Source: phase4_sgd_stable.csv.
- **M-P4c (Phase 4) — SECOND REAL DATASET (UTD-MHAD).** Depth (near-chance) /
  inertial (informative); scale informative inertial by α=100, Adam, arm−baseline
  paired 95% CI (n=5). **Gradient arm ≡ baseline on ALL metrics** (masked
  +0.003 ns, mean-abl +0.004 ns, Shapley −0.002 ns, probes ns) — clean Adam
  gradient-cancellation replicated on a 2nd real dataset. Forward: **masked dom
  −0.525±0.034** (inertial dominance balloons), inertial probe **+0.102±0.034**,
  mean-abl −0.107±0.028, Shapley +0.113±0.031, depth probe −0.011 (ns). → masked
  dominance OVERSTATES the representational change ~5×. Source: phase4_utd.csv.

## HYPOTHESIZED (to test in Phase 2+)

- **H-mech → RESOLVED (Phase 2, refined).** Raising A's effective LR degrades A
  (H-lr direction, SUPPORTED): lrmatch = the enc-A LR-sweep exactly (lrmatch
  α=100 probe 0.423 = LR-sweep r=100 0.423); P4 exact. **H-comp is a MODULATOR,
  not refuted:** forward ≈ an effective LR of ~α^(1−c) at small α (lands on the
  LR curve: α=10 r_eff 2.8 vs α^(1−c) 2.0), but forward's degradation SATURATES
  at large α (r_eff plateaus ~3–5 while α^(1−c)→24 at α=300) — compensation
  increasingly buffers it. So forward is an effective-LR effect whose gain is set
  AND capped by compensation. Do NOT say "H-comp refuted".
- **H-init → REFUTED (Phase 2).** forward_k null at k=1,5,10,20 (scaling for up to
  half of training then removing it → no effect). The effect needs sustained
  scaling, consistent with a sustained effective-LR mechanism.
- **P3/P4 (Prop 1 non-trivial tests) → SUPPORTED.** P4 exact (dom=baseline); P3
  mechanism (update ratio 1.00→2.84 as ε→1e-2). Adam forward≡ordinary and
  gradient-null are ε→0 identities, NOT independent evidence.
- **H-crema.** CREMA-D forward RAISES audio contribution partly because it
  compensates less (M6). Only half-tested: no CREMA probe yet (Phase 2 #5).
- **H-mup → now partly MEASURED (web-verified 2026-11-09).** muP (arXiv:2203.03466)
  is a width→∞ limit and DOES derive per-layer LR scalings for both SGD and Adam
  (verified verbatim) → we must NOT claim the per-layer-effective-LR idea (Adam
  included) as new; cite muP. What survives as novel (NOT FOUND in muP text): the
  FWD/GRAD operator decoupling of s_f vs s_g, deliberate finite-width per-modality
  imbalance, and the modality-contribution application. See `THEORY.md`.
- **H-collapse.** Prop 1 predicts dominance-shift vs predicted multiplier
  collapses onto one curve across arms/optimizers. Test: α-sweep (Phase 2).

## Phase 2 condition resolutions (2026-11-09, no new training)

- **#2 RECONCILED (with data).** The paper's S_loss/E4 experiment varies BOTH
  encoders' LR symmetrically (`lr_A=base·√l`, `lr_V=base/√l`) and tracks the weak
  modality. E4 `results/processed/e4_grid.csv`, scale_r=1 (pure LR imbalance):
  the modality with the HIGHER LR has the LOWER contribution — lr_r=20 (lr_A↑):
  δ_A 0.20→0.14; lr_r=0.05 (lr_V↑): δ_V 0.20→0.14. **Same direction as Phase 2**
  (raising a modality's effective LR lowers its contribution). So "forward
  scaling = effective-LR increase for A → A loses" does NOT contradict the paper;
  "feature scale compensable (0.04–0.08)" = small magnitude, "LR imbalance
  persistent (→0.88 at 400×)" = large magnitude, one monotone mechanism.
  **Caveat (HYPOTHESIZED):** this means the forward intervention's *contribution*
  effect is mediated by the effective LR, so the paper's "two separable channels"
  (feature-scale vs optimization) is not clean — forward scaling acts through the
  optimization channel. Confirm with the lrmatch 10-seed result before asserting.
- **M8 (#4 CONFIRMED).** CREMA-D trains CNN encoders END-TO-END from spectrogram/
  frame inputs, not frozen embeddings: `AudioCNN` log-mel `[B,1,64,T]`, `VideoCNN`
  frames `[B,N,1,64,64]` (cremad_encoders.py); cremad_disentangle trains all
  params. The phase-2 CREMA probe run will use this same end-to-end setup.
- **M9 (#3).** Old uncorrected c = `1−log(ρ_fus)/log α` is in 5 code sites
  (disentangle_scale_grad:221, cremad_disentangle:130, phase1_decoupled:121,
  fusion_robustness:134, sgd_disentangle:88). Only cremad_disentangle:130 is
  materially biased (CREMA baseline ρ_fus=1.22); the synthetic sites (baseline≈1)
  are fine. The PAPER uses the baseline-corrected c; its c=0.58–0.60 is correct.
  c(α) curve (baseline-corrected): synth {10:0.69, 100:0.52}; CREMA {5:0.60,
  10:0.58, 100:0.39}. Full synthetic curve pending a post-sweep ρ_fus pass.
- **#5 (novelty softened).** GRAD operator = the Gradient Reversal Layer
  (Ganin & Lempitsky 2015) generalized from negation to scaling (web-confirmed).
  FWD operator (scale forward, identity gradient) is less standard. Claim, "to
  our knowledge," only the PAIRED use to decouple forward- vs gradient-magnitude
  effects on modality contribution; cite GRL for the operator.

## Phase 2 framing corrections (user conditions, round 2)

- **Adam forward≡ordinary and gradient-null are ALGORITHMIC IDENTITIES, not
  tests.** Under Adam with ε→0, the normalization cancels the gradient gain
  `s_g`, so forward (s_g=1) and ordinary (s_g=α) are identical by construction,
  and gradient-only (s_f=1) equals baseline. These confirm *known* Adam
  scale-invariance (already cited: balles2018, bernstein2018), they are NOT
  evidence for our theory. **The real (falsifiable) tests are:** SGD ordinary
  (m=α²) vs isolated arms (m=α); P3 (ε-sweep — does the cancellation break?);
  P4 (η/α cancellation); and lrmatch / the LR-sweep reference curve.
- **lrmatch supports H-lr in DIRECTION ONLY (it overshoots).** Paired
  lrmatch−forward (dominance) excludes 0 at every α, including α=3 (stable
  lr=0.003): −0.020±0.005; grows to −0.089±0.018 at α=300 (lr=0.3, possible
  instability — train loss logged in the LR-sweep run to check). So: raising A's
  effective LR degrades A (direction confirmed), but forward ≠ a clean LR×α
  because forward-magnitude compensation buffers it. Do NOT claim exact
  equivalence.

## Phase 2 condition answers, stated explicitly (user #5)

- **(a) S_loss / which modality's LR.** The E4 experiment scales BOTH encoders'
  LR symmetrically: `lr_A = base·√l`, `lr_V = base/√l` (e4_boundary.py:89). It
  tracks the WEAK modality (min δ). From `results/processed/e4_grid.csv`
  (scale_r=1, pure LR imbalance): the HIGHER-LR modality has the LOWER
  contribution — lr_A↑20×: δ_A 0.20→0.14; lr_V↑: δ_V→0.14. **Same direction as
  Phase 2 forward-scaling.** NO conflict with the feature-scale result: "feature
  scale compensable (0.04–0.08)" = small magnitude, "LR imbalance persistent
  (→0.88 at 400×)" = large magnitude of the SAME monotone mechanism. Caveat: this
  blurs the paper's "two separable channels" (forward acts via the optimization
  channel) — Phase-5 wording fix.
- **(b) CREMA-D end-to-end.** CONFIRMED: `AudioCNN` (log-mel [B,1,64,T]),
  `VideoCNN` (frames [B,N,1,64,64]) are trainable CNNs; cremad_disentangle trains
  all params; the cache holds INPUTS not embeddings. (cremad_encoders.py,
  cremad_model.py.)
- **(c) Old c formula — every place.** Code only (5 sites):
  disentangle_scale_grad.py:221, cremad_disentangle.py:130, phase1_decoupled.py:121,
  fusion_robustness.py:134, sgd_disentangle.py:88. Only cremad_disentangle:130 is
  materially biased (CREMA baseline ρ_fus=1.22). The PAPER (main.tex:98) uses the
  baseline-corrected `c=1−ln(ρ_fus(α)/ρ_fus(1))/ln α`; its c=0.58–0.60 is correct.
  No old-formula usage in the paper.

## Phase 2 compute results — two findings that WEAKEN paper claims

- **W1 — SGD "improves A at small α" is an UNDERTRAINED-BASELINE artifact (#4).**
  SGD baseline at lr=1e-3 is badly undertrained (val 0.527, train loss 1.67;
  best SGD lr=1e-2, val 0.708). The enc-A LR sweep is an inverted-U peaking at
  r≈10; the lr=1e-3 baseline sits on the rising (undertrained) side, so scaling
  moves A UP toward the peak = apparent "improvement." **At the tuned SGD baseline
  (lr=1e-2), forward AND gradient DEGRADE A's probe (−0.049±0.005, −0.110±0.009)
  and gradient dominance goes NEGATIVE (−0.093).** So the direction of the SGD
  contribution effect is regime-dependent. The UPDATE-RATIO mechanism (SGD
  transmits the gradient scale, ρ_Δθ=95.9; Adam cancels, 1.0) is robust and
  unaffected; what is confounded is the SIGN of the contribution change. The
  paper's SGD disentangle numbers (lr=1e-3) are in the undertrained regime.
  Adam's forward degradation is ROBUST to LR tuning (−0.057±0.005 at tuned lr).
- **W2 — the CREMA-D forward effect is a masked-accuracy artifact. CONFIRMED
  (Phase 3 scoped, tuned LR, AdamW, 5 seeds).** baseline→forward: masked dominance
  **+0.259±0.031**, mean-ablation **+0.012±0.019 (ns)**, Shapley_audio
  **+0.014±0.019 (ns)**, probe_audio **+0.023±0.012**. Gradient null (all ~0). So
  under AdamW (=Adam at wd=0) the real-data contribution shift is NOT supported by
  any decoupled metric — it is off-distribution zeroing. The synthetic effect
  survives decoupled metrics; the CREMA-D one does not. **The paper's real-data
  evidence for a genuine contribution shift is not supported (d≈7.4 is on the
  coupled metric).**
  **REFINED (Phase 4, UTD-MHAD, M-P4c): "masked overstates", not "forward is a
  pure artifact everywhere".** On CREMA-D the scaled modality (audio) probe is at
  its weak-encoder ceiling, so forward scaling moves masked but not the probe (pure
  masked artifact). On UTD the scaled modality (inertial) has headroom, so forward
  scaling — which raises the effective LR under Adam (m_Adam=s_f) — yields a GENUINE
  inertial probe gain (+0.102), yet masked still overstates it ~5× (−0.525). The
  invariant claim across datasets is that masked-accuracy dominance responds far more
  strongly than the frozen-encoder probe; whether the probe moves at all depends on
  headroom. State W2 as "masked overstates/can mislead", mechanism = effective-LR +
  probe headroom.
- **W3 — SGD + scaling operators are UNSTABLE on real data → RESOLVED (Phase 4,
  2026-11-14).** Original (Phase 3): CREMA-D SGD at tuned lr=3e-2 diverged (forward
  3/5, gradient 2/2). **Phase 4 fix:** at lr=1e-2, α=100 with gradient-norm clipping
  (max 1.0) divergence drops to 0/5 on ALL arms (M-P4b). The instability was an
  optimization artifact, not a property of the scaling operators. Under the now-
  STABLE SGD the masked-vs-decoupled mismatch still holds (gradient masked +0.200,
  decoupled ns). So the paper CAN now make a confident stable-SGD real-data claim,
  with clipping disclosed. The earlier n=2 "A degradation" numbers are superseded
  and should NOT be cited.

## Phase 3 control close-out (venue: IEEE ICRCICN 2027)

- **M3 `ordinary` + per-modality heads — DONE** (reported, was already run):
  adam ordinary probe_A −0.037±0.007 / dom −0.050±0.012; sgd ordinary probe_A
  +0.037±0.006 / dom +0.255±0.025. forward≠gradient persists → shared-head
  explanation rejected (all 4 arms).
- **Probe-shift statement corrected:** CREMA-D AdamW forward probe_audio shift =
  **+0.023 ± 0.012** (slightly positive, CI excludes 0 but tiny) — audio encoder
  is NOT degraded on CREMA-D; the masked +0.259 is off-distribution.
- **Positive control (audio noise) — metrics VALIDATED.** AdamW baseline, audio
  input noise: probe_audio shift −0.021±0.007 (0.5σ), −0.064±0.001 (1.0σ);
  Shapley_audio −0.029±0.009, −0.051±0.005. Monotonic, CIs exclude 0 → the
  decoupled metrics DO detect genuine audio degradation, so the W2 null for
  forward-scaling (probe +0.023, Shapley +0.014 ns) is a real no-effect.
- **SGD α=10 on CREMA (stable) — W2 holds under SGD.** forward/gradient masked
  dominance +0.150/+0.156 (big) but mean-ablation +0.010/+0.002 (ns), Shapley
  +0.010/+0.002 (ns), probe +0.017/+0.020. → masked shift is not a genuine
  contribution shift, under stable SGD too. (W3 was SGD α=100 instability only.)
  ALL EXPERIMENTS COMPLETE.

## REFUTED

- **R1 (Phase 1).** Pre-registered P-main ("probe barely moves; the acc_A drop
  is a pure head/zeroing artifact"). REFUTED: probe drops 0.034, CI excludes 0,
  mean-ablation agrees. Source: `preregistration/PREREGISTRATION_PHASE1.md` (outcome section).
- **R2 (Phase 1).** The paper sentence (added in the prior rewrite) "read the
  Adam forward arm by magnitude, not sign; a near-zero baseline does not
  determine the sign." REFUTED: the sign is real and directional. Correction
  deferred to Phase 5.
