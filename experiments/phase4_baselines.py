"""Phase 4 (runnable): payoff baselines on CREMA-D, measured with the DECOUPLED metrics.

Question the paper needs: a rebalancing / modality-LR method changes masked-accuracy
contribution -- does that change show up on the DECOUPLED metrics (frozen probe,
Shapley, mean-ablation), or is it (like our forward scaling) a masked-accuracy effect?

Methods (all AdamW, lr=1e-3, wd=0, 30 epochs, 5 seeds, sum fusion, weak CNN encoders):
  * vanilla        -- plain joint training (reference)
  * ogm            -- OGM-GE gradient modulation (Peng et al. CVPR'22), faithful
                      re-implementation in baselines/ogm_ge.py (alpha=0.5, GE gamma=0.5)
  * modlr_video    -- modality-specific LR: video (weak) encoder LR x5
  * modlr_audio    -- modality-specific LR: audio (dominant) encoder LR x5

Every method is evaluated with experiments.phase2_cremad_probe.decoupled at arm=
"baseline" (no scaling), i.e. the SAME masked / probe / Shapley / mean-ablation
metrics the paper uses. Reported as method - vanilla paired shift over seeds.
Nothing is scaled here; this isolates the rebalancing method's own effect.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, torch, torch.nn as nn
from experiments.cremad_run import load_features
from experiments.phase2_cremad_probe import decoupled
from baselines.ogm_ge import per_modality_scores, ogm_coefficients, apply_modulation
from models.cremad_model import CremadMultimodalModel
from msc.device import device_report, get_device
from msc.runlog import git_hash, load_config
from msc.seed import set_seed

OGM_ALPHA, OGM_GAMMA = 0.5, 0.5     # modulation strength / GE noise (documented)
MODLR_K = 5.0                        # modality-specific LR multiplier
METHODS = ["vanilla", "ogm", "modlr_video", "modlr_audio"]


def build_opt(model, method, base_lr):
    if method == "modlr_video":
        groups = [{"params": model.enc_a.parameters(), "lr": base_lr},
                  {"params": model.enc_v.parameters(), "lr": base_lr * MODLR_K},
                  {"params": model.head.parameters(), "lr": base_lr}]
    elif method == "modlr_audio":
        groups = [{"params": model.enc_a.parameters(), "lr": base_lr * MODLR_K},
                  {"params": model.enc_v.parameters(), "lr": base_lr},
                  {"params": model.head.parameters(), "lr": base_lr}]
    else:
        groups = model.parameters()
    return torch.optim.AdamW(groups, lr=base_lr, weight_decay=0.0)


def train(cfg, feats, device, seed, method):
    set_seed(seed, True); nc = cfg["data"]["n_classes"]
    a_tr, v_tr, y_tr = feats["train"]
    model = CremadMultimodalModel(nc, rep_dim=cfg["model"]["rep_dim"], fusion=cfg["model"]["fusion"],
                                  normalization="none", head=cfg["model"]["head"]).to(device)
    opt = build_opt(model, method, cfg["train"]["lr"])
    lf = nn.CrossEntropyLoss(); epochs = cfg["train"]["epochs"]; bs = cfg["train"]["batch_size"]; n = y_tr.numel()
    ea = list(model.enc_a.parameters()); ev = list(model.enc_v.parameters())
    is_ogm = method == "ogm"; k_a_sum = k_b_sum = 0.0; nb = 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            logits, (h_a, h_v) = model(a_tr[idx], v_tr[idx], return_reps=True)
            loss = lf(logits, y_tr[idx])
            if not torch.isfinite(loss):
                return {"seed": seed, "method": method, "diverged": True}
            loss.backward()
            if is_ogm:
                s_a, s_b = per_modality_scores(model, h_a.detach(), h_v.detach())
                k_a, k_b, _ = ogm_coefficients(s_a, s_b, y_tr[idx], OGM_ALPHA)
                apply_modulation(ea, ev, k_a, k_b, OGM_GAMMA)
                k_a_sum += k_a; k_b_sum += k_b; nb += 1
            opt.step()
    m = decoupled(model, feats, nc, device, "baseline")
    out = {"seed": seed, "method": method, "diverged": False, **m}
    if is_ogm and nb:
        out["k_audio"] = k_a_sum / nb; out["k_video"] = k_b_sum / nb
    return out


def main():
    cfg = load_config(ROOT / "configs" / "cremad.yaml")
    device = get_device(cfg["device"]); print("Device:", device_report(device))
    feats = load_features(device)
    out = ROOT / "results" / "phase4_baselines"; out.mkdir(parents=True, exist_ok=True)
    seeds = cfg["seed_policy"]["seeds"]; rows = []; t0 = time.time()
    for method in METHODS:
        for seed in seeds:
            r = train(cfg, feats, device, seed, method); rows.append(r)
            print(f"  {method:12s} seed={seed}: dom={r.get('dominance',float('nan')):+.3f} "
                  f"probe_a={r.get('probe_audio',float('nan')):.3f} probe_v={r.get('probe_video',float('nan')):.3f} "
                  f"({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows); df.to_csv(out / "phase4_cremad.csv", index=False)

    def pci(method, col):
        b = df[(df.method == "vanilla") & (~df.diverged)].sort_values("seed")[col].values
        x = df[(df.method == method) & (~df.diverged)].sort_values("seed")[col].values
        k = min(len(b), len(x))
        if k < 2:
            return None
        dd = x[:k] - b[:k]; sd = dd.std(ddof=1)
        return dd.mean(), 1.96 * sd / math.sqrt(k), k

    print("\n=== Phase 4: method - vanilla shift (paired 95% CI). "
          "Does a rebalancing/LR method move the DECOUPLED metrics or only masked? ===")
    cols = ["dominance", "meanabl_dominance", "shapley_audio", "probe_audio", "probe_video"]
    for method in METHODS[1:]:
        print(f"-- {method} --")
        for col in cols:
            r = pci(method, col)
            print(f"   {col:18s}: {'n<2' if r is None else f'{r[0]:+.4f} ± {r[1]:.4f} (n={r[2]})'}")
    og = df[df.method == "ogm"]
    if "k_audio" in df.columns and len(og):
        print(f"   OGM modulation active? mean k_audio={og.k_audio.mean():.3f} k_video={og.k_video.mean():.3f} "
              f"(k<1 => that modality attenuated)")
    json.dump({"git_commit": git_hash(), "ogm_alpha": OGM_ALPHA, "ogm_gamma": OGM_GAMMA,
               "modlr_k": MODLR_K, "seeds": seeds, "wall_s": round(time.time() - t0, 1)},
              open(out / "phase4_cremad_config.json", "w"), indent=2, default=str)
    print(f"\nWALL {time.time()-t0:.1f}s -> {out}/phase4_cremad.csv")


if __name__ == "__main__":
    main()
