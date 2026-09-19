# E3 normalization intervention (H3)

| norm | early\|D\|@extreme | final\|D\|@extreme | test_acc | test_f1 | resid\|logρ\| | conv_ep |
|------|---:|---:|---:|---:|---:|---:|
| none | 0.152 | 0.036 | 0.690 | 0.690 | 0.452 | 10.6 |
| layernorm | 0.020 | 0.013 | 0.648 | 0.648 | 0.011 | 3.5 |
| zscore | 0.025 | 0.019 | 0.668 | 0.667 | 0.014 | 3.8 |
| batchnorm | 0.019 | 0.018 | 0.645 | 0.644 | 0.009 | 3.5 |

### Effect vs none (early |D| reduction @ extreme scales, epoch 5)
- layernorm: +0.132 ± 0.024 (CI-separated: True, n=20)
- zscore: +0.127 ± 0.023 (CI-separated: True, n=20)
- batchnorm: +0.133 ± 0.024 (CI-separated: True, n=20)

### Final accuracy effect vs none
- layernorm: -0.047 ± 0.003
- zscore: -0.027 ± 0.003
- batchnorm: -0.049 ± 0.002

### H3 verdict per intervention
- **layernorm**: CURES early imbalance | prevents_early=True | improves_final_acc=False | reduces_compensation_need=True
- **zscore**: CURES early imbalance | prevents_early=True | improves_final_acc=False | reduces_compensation_need=True
- **batchnorm**: CURES early imbalance | prevents_early=True | improves_final_acc=False | reduces_compensation_need=True
