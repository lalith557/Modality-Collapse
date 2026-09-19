# E1 diagnostic (primary) — n=150 runs

**Verdict: UNSUPPORTED**

- overall Spearman(log rho5, D_final) = 0.063 [-0.092, 0.209], n=150
- AUROC(final A_stronger | log rho5) = 0.359
- **partial Spearman | log(scale) = 0.600 [0.486, 0.703]** (excl. 0: True)
- **balanced-scale subfamily Spearman = 0.653 [0.508, 0.766]** (n=90)
- incremental R² of log rho5 over log(scale): 0.029 → 0.454 (Δ=0.425)
- AUROC decay: epoch5=0.911 → final=0.399 (decays=True)

### Per-family Spearman(log rho5, D_final) and (snr_a-snr_v, D)
- equal: rho5→D = -0.825, trueAsym→D = nan (n=70)
- info_diff: rho5→D = 0.957, trueAsym→D = 0.969 (n=40)
- noise_diff: rho5→D = 0.250, trueAsym→D = 0.969 (n=40)
