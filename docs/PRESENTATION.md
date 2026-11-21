# PRESENTATION

**When Scale Silences a Modality — Feature Scale, Optimizers and Misleading Metrics in Multimodal Learning**

Three equal parts, about **5 minutes each** (≈15 minutes total, then questions). Each part has four slides. Every slide has four points to show and a short note of what to say. All numbers are measured results from the project's logs.

---

## Part 1 — The problem and how we built the study

### Slide 1 · What this project is about

- Multimodal networks learn from two inputs at once, for example the **voice and face** of a speaker, or a **depth camera and a motion sensor**.
- They often **collapse**: they lean on one input and almost ignore the other.
- We asked whether a simple numeric cause, **one modality's features being larger in magnitude**, makes this happen.
- We also asked whether the **standard way of measuring** this collapse can be trusted.

> **Say:** "A two-input network should use both inputs. Ours often doesn't, and a popular explanation is that the bigger numbers win. We tested that explanation, and also the ruler people use to measure it."

### Slide 2 · Why modality collapse matters

- If a model ignores one input, it **wastes data** and **loses robustness**: when the used input is noisy or missing, accuracy drops.
- The usual fix is to **rebalance training**, for example by slowing the gradient of the dominant modality (methods such as OGM-GE, PMR).
- Collapse is usually measured with **masked accuracy**: switch one modality off (set it to zero) and see how much accuracy falls.
- If that measurement is wrong, researchers may "fix" a problem that is not there, or miss a real one.

> **Say:** "Everything here depends on measurement. If you measure dominance by switching an input off, you had better be sure the drop means the encoder is worse, not just that the classifier is surprised."

### Slide 3 · Our research questions

- **RQ1** — Does making one modality's features **larger** really damage what the network learns from the other?
- **RQ2** — Does the effect travel through the **forward signal** (what the next layer sees) or the **gradient** (how fast that encoder learns)?
- **RQ3** — Does the answer depend on the **optimizer**, adaptive **Adam** versus plain **SGD**?
- **RQ4** — Does **masked-accuracy dominance** agree with measurements that look at the representation itself?

> **Say:** "The key twist is RQ2. Making a feature bigger does two things at once: a bigger signal forward, and a bigger gradient backward. Nobody had pulled these two apart for modality imbalance."

### Slide 4 · Data and models

- **Synthetic data:** two 32-d inputs, 8 classes, *equal information by construction*, so any imbalance comes from training, not content.
- **CREMA-D:** 7,442 clips, 91 actors, 6 emotions, audio + video, actor-disjoint split (5,234 / 738 / 1,470 clips).
- **UTD-MHAD:** 861 sequences, 27 actions, depth + inertial sensor, subject-disjoint split (323 / 108 / 430).
- **Small CNN encoders** trained from scratch (32,710 and 35,467 parameters), sum fusion, linear classifier, all on one RTX 4070 with 5–10 seeds.

> **Say:** "We kept the models small on purpose. That let us run hundreds of controlled trainings with several random seeds and compare them statistically, instead of one big model we can't repeat. The synthetic data is our laboratory, where we know the truth; the two real datasets test whether the lesson survives outside the lab."

---

## Part 2 — Method and findings

### Slide 5 · The key idea: split forward from gradient

- We wrote custom **autograd operators** that act on modality A's feature vector h_A just before fusion.
- **Forward-only:** the next layer sees α·h_A, but the gradient is left unchanged. This isolates *signal size*.
- **Gradient-only:** the next layer sees h_A, but the encoder receives α × the gradient. This isolates *learning speed*.
- **Ordinary:** both at once, like a normal feature scaling. The training objective is identical in all arms; α = 100 in the main test.

> **Say:** "Think of it as two dials that are normally welded together. We cut the weld, so we can turn up only the volume, or only the learning signal, and see which one does the damage."

### Slide 6 · How we measure who contributes

- **Masked dominance** (the standard): accuracy with only A minus accuracy with only B, where "only" means the other modality is set to zero.
- **Frozen-encoder probe:** a simple linear classifier trained on one frozen encoder, which asks *how much the representation itself knows*.
- **Mean-ablation and Shapley value:** replace a modality by its **average** instead of zero, and share credit fairly between the two modalities.
- **Optimizer readout:** gradient ratio versus actual **parameter-update ratio**, which shows what reaches the weights.

> **Say:** "Zeroing an input the classifier always saw at a certain size pushes it into territory it never trained on. The probe and mean-ablation avoid that trap. A positive control, adding noise to audio, proved they do detect real damage."

### Slide 7 · Finding: the optimizer decides the pathway

- **Adam cancels gradient scaling:** a 100.9× gradient ratio became a **1.00×** update ratio (282.7× → 1.61× on CREMA-D, whose baseline is 1.65×). Gradient-only did nothing.
- **SGD passes it through:** a 318.5× gradient ratio became a **95.9×** update ratio, and gradient-only *did* change dominance (+0.200).
- **Theory** (first order): the effective learning-rate multiplier is **s_f · s_g for SGD** but only **s_f for Adam**. Feature scale acts as a **hidden per-modality learning rate**.
- **Confirmed by tests:**
  - raising Adam's ε breaks the cancellation (1.00 → 2.84);
  - dividing A's learning rate by α under SGD restores the baseline exactly (+0.003).

> **Say:** "Adam divides each update by the gradient's own typical size, so multiplying the gradient by 100 changes nothing. SGD has no such division, so the same trick is just a 100× learning rate for one encoder."

### Slide 8 · Finding: masked accuracy can mislead

- **Synthetic data:** forward scaling truly hurts the scaled modality. Every metric agrees (probe −0.034, Shapley −0.043).
- **CREMA-D:** masked dominance jumps **+0.259**, but mean-ablation (+0.012) and Shapley (+0.014) are not significant and the probe is **not** worse (+0.023). The "collapse" is mostly a measurement artifact.
- **UTD-MHAD:** masked dominance moves **0.525** but the representation improves only **0.102**, about a **5×** overstatement.
- **Same pattern elsewhere:** under stable SGD (masked +0.200, decoupled ≈ 0) and for **OGM-GE**, a published method (masked +0.240, decoupled mostly not resolved).

> **Say:** "This is the headline. The standard ruler reported a big collapse on real data while the encoders had barely changed. And it happened again on a second dataset, with a different optimizer, and for a published rebalancing method."

---

## Part 3 — What it means, what we built, what comes next

### Slide 9 · Why it happens, and honest negative results

- **Why masking misleads:** the classifier head adapts to the scaled input, so zeroing it moves the head *off its training distribution*. Accuracy falls because the head is confused, not because the encoder learned less.
- **Headroom matters:** CREMA-D's audio encoder was already at its ceiling, so nothing could improve; UTD-MHAD's inertial encoder had room, so a real but much smaller gain appeared.
- **Kept and reported:**
  - our own prediction "the synthetic drop is only an artifact" was **refuted**;
  - SGD's direction depended on an undertrained baseline;
  - gated fusion was not significant.
- **SGD diverged** on real data at α = 100 (5 of 5 seeds); gradient clipping fixed it completely (0 of 5) without hiding the effect.

> **Say:** "We pre-registered predictions and published the ones that failed. That is why we trust the parts that held up."

### Slide 10 · What we built around the research

- **Research website** with an Overview, an **Experiment Lab** (every measured setting, now with the real per-epoch training curves) and Real-World Evidence pages.
- **360° 3D Model Architecture:** the real networks layer by layer. Click a section to raise it and hover any part to see what it does. Moving data shows the forward pass.
- **Prediction Studio:** upload audio, video, depth or sensor files and run the *actual* checkpoints (test accuracy 0.508 on CREMA-D, 0.570 on UTD-MHAD), with out-of-distribution warnings.
- **Data integrity:** every number on the site is generated from the frozen results and checked by automated tests. No result is simulated.

> **Say:** "Anyone can check our claims: see the curves, inspect the model, even run it. The site also refuses to treat a high softmax confidence as proof of being right."

### Slide 11 · Limitations and future work

- **Scope:** two real datasets with fixed splits, small from-scratch encoders, five seeds on real data, sum fusion on real data.
- **Theory** is first-order and verified on synthetic data; weight decay, schedules and finite ε are not fully covered.
- **Not run (compute and disk):** Kinetics-Sounds, AVE, VGGSound, pretrained or transformer encoders, attention fusion.
- **Next steps:** larger benchmarks, pretrained encoders, more rebalancing methods (PMR, AGM, MLA), more seeds, logged gate values.

> **Say:** "Our real-data claim is deliberately narrow: for these models and splits, masked accuracy overstated the change. Testing large pretrained models is the most important next step."

### Slide 12 · Take-home messages

1. **Feature scale behaves like a hidden per-modality learning rate**, and the optimizer decides how it acts: Adam cancels gradient scaling, SGD does not.
2. **Zero-masking dominance can strongly overstate collapse or rebalancing.** Do not use it alone.
3. **Report decoupled metrics** (frozen probe, mean-ablation, Shapley) with a positive control whenever you claim one modality dominates.
4. **Everything is reproducible and inspectable** through the logs, the paper and the interactive website. **Questions?**

> **Say:** "If you remember one thing: before saying a modality was silenced, check the encoder itself, not just what happens when you switch it off."
