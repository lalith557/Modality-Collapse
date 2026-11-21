"""Phase 4 (runnable): make SGD on real data STABLE at alpha=100, then test the claim.

The paper reports SGD + scaling is unstable on CREMA-D at alpha=100 (forward diverges
3/5, gradient 2/2), so SGD real-data results were 'exploratory'. The standard fix for
a blown-up gradient is gradient-norm clipping. Here we add clip_grad_norm_ and re-run
baseline / forward / gradient at alpha=100, SGD(momentum=0.9), and ask two things:
  (1) does clipping stabilize the runs (0 divergence)?
  (2) under STABLE SGD at alpha=100, does the masked-vs-decoupled MISMATCH still hold
      (masked dominance moves, probe/Shapley/mean-ablation do not)?

Same model / split / decoupled metrics as phase3_cremad_scoped. 5 seeds.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, torch, torch.nn as nn
from experiments.cremad_run import load_features
from experiments.cremad_disentangle import arm_logits
from experiments.phase2_cremad_probe import decoupled, ALPHA
from models.cremad_model import CremadMultimodalModel
from msc.device import device_report, get_device
from msc.runlog import git_hash, load_config
from msc.seed import set_seed

ARMS = ["baseline", "forward", "gradient"]
LR = 1e-2            # tuned SGD baseline lr (from phase3 tuning)
CLIP = 1.0           # gradient-norm clip


def train(cfg, feats, device, seed, arm, clip):
    set_seed(seed, True); nc = cfg["data"]["n_classes"]
    a_tr, v_tr, y_tr = feats["train"]
    model = CremadMultimodalModel(nc, rep_dim=cfg["model"]["rep_dim"], fusion=cfg["model"]["fusion"],
                                  normalization="none", head=cfg["model"]["head"]).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    lf = nn.CrossEntropyLoss(); epochs = cfg["train"]["epochs"]; bs = cfg["train"]["batch_size"]; n = y_tr.numel()
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            loss = lf(arm_logits(model, a_tr[idx], v_tr[idx], ALPHA, arm), y_tr[idx])
            if not torch.isfinite(loss):
                return {"seed": seed, "arm": arm, "clip": clip, "diverged": True}
            loss.backward()
            if clip and clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
    m = decoupled(model, feats, nc, device, arm)
    return {"seed": seed, "arm": arm, "clip": clip, "diverged": False, **m}


def main():
    cfg = load_config(ROOT / "configs" / "cremad.yaml")
    device = get_device(cfg["device"]); print("Device:", device_report(device))
    feats = load_features(device)
    out = ROOT / "results" / "phase4_baselines"; out.mkdir(parents=True, exist_ok=True)
    seeds = cfg["seed_policy"]["seeds"]; rows = []; t0 = time.time()
    # run WITHOUT clip (reproduce instability) and WITH clip (the fix)
    for clip in [0.0, CLIP]:
        print(f"=== SGD alpha=100 lr={LR:g} clip={clip or 'none'} ===")
        for arm in ARMS:
            for seed in seeds:
                r = train(cfg, feats, device, seed, arm, clip); rows.append(r)
                print(f"  {arm:9s} seed={seed}: diverged={r['diverged']} "
                      f"dom={r.get('dominance',float('nan')):+.3f} "
                      f"probe_a={r.get('probe_audio',float('nan')):.3f} ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows); df.to_csv(out / "phase4_sgd_stable.csv", index=False)

    def pci(clip, arm, col):
        b = df[(df.clip == clip) & (df.arm == "baseline") & (~df.diverged)].sort_values("seed")[col].values
        x = df[(df.clip == clip) & (df.arm == arm) & (~df.diverged)].sort_values("seed")[col].values
        k = min(len(b), len(x))
        if k < 2:
            return None
        dd = x[:k] - b[:k]; sd = dd.std(ddof=1)
        return dd.mean(), 1.96 * sd / math.sqrt(k), k

    print("\n=== divergence counts (by clip, arm) ===")
    print(df[df.diverged].groupby(["clip", "arm"]).size().to_string() if df.diverged.any() else "  none diverged")
    print("\n=== STABLE SGD (clip=1.0) baseline->arm shift (paired 95% CI): mismatch hold? ===")
    for arm in ["forward", "gradient"]:
        print(f"-- {arm} --")
        for col in ["dominance", "meanabl_dominance", "shapley_audio", "probe_audio"]:
            r = pci(CLIP, arm, col)
            print(f"   {col:18s}: {'n<2 (diverged)' if r is None else f'{r[0]:+.4f} ± {r[1]:.4f} (n={r[2]})'}")
    json.dump({"git_commit": git_hash(), "lr": LR, "clip": CLIP, "alpha": ALPHA,
               "seeds": seeds, "wall_s": round(time.time() - t0, 1)},
              open(out / "phase4_sgd_stable_config.json", "w"), indent=2, default=str)
    print(f"\nWALL {time.time()-t0:.1f}s -> {out}/phase4_sgd_stable.csv")


if __name__ == "__main__":
    main()
