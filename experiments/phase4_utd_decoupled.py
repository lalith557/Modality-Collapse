"""Phase 4 (runnable): SECOND real dataset -- UTD-MHAD decoupled-metric protocol.

Cross-dataset test of the CREMA-D finding (masked-accuracy dominance shifts under
forward scaling while the decoupled metrics do not). Modality A = depth (near-chance),
B = inertial (informative). We forward-scale the INFORMATIVE modality (inertial) by
alpha=100 -- the parallel of scaling audio on CREMA-D -- and measure masked dominance,
frozen-encoder probes, mean-ablation dominance, and the two-modality Shapley value.

Adam, lr=1e-3, wd=0, 60 epochs, 5 seeds, sum fusion, weak from-scratch CNNs.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, torch, torch.nn as nn
from experiments.cremad_run import _chunks
from experiments.disentangle_scale_grad import arm_apply
from models.utd_model import UTDMultimodalModel
from models.fusion import fuse
from analysis.probes import linear_probe_accuracy
from msc.device import device_report, get_device
from msc.runlog import git_hash, load_config
from msc.seed import set_seed

ALPHA = 100.0
SCALED = "v"            # scale the INFORMATIVE modality (inertial = enc_v)
ARMS = ["baseline", "forward", "gradient"]


def load_utd(cache, device):
    d = {}
    for sp in ("train", "test"):
        a = torch.from_numpy(np.load(cache / f"{sp}_depth.npy")).float().to(device)
        v = torch.from_numpy(np.load(cache / f"{sp}_inertial.npy")).float().to(device)
        y = torch.from_numpy(np.load(cache / f"{sp}_y.npy")).long().to(device)
        d[sp] = (a, v, y)
    return d


def _reps(model, x_a, x_v):
    return model.norm_a(model.enc_a(x_a)), model.norm_v(model.enc_v(x_v))


def arm_logits(model, x_a, x_v, alpha, arm, mask_a=1.0, mask_v=1.0):
    h_a, h_v = _reps(model, x_a, x_v)
    if SCALED == "a":
        h_a = arm_apply(h_a, alpha, arm)
    else:
        h_v = arm_apply(h_v, alpha, arm)
    return model.head(fuse(model.fusion, mask_a * h_a, mask_v * h_v))


@torch.no_grad()
def extract(model, a, v):
    model.eval(); ha, hv = [], []
    for sl in _chunks(a.shape[0]):
        h_a, h_v = _reps(model, a[sl], v[sl]); ha.append(h_a); hv.append(h_v)
    return torch.cat(ha).detach(), torch.cat(hv).detach()


def decoupled(model, feats, nc, device, arm):
    a_tr, v_tr, y_tr = feats["train"]; a_te, v_te, y_te = feats["test"]
    ha_tr, hv_tr = extract(model, a_tr, v_tr); ha_te, hv_te = extract(model, a_te, v_te)
    probe_a = linear_probe_accuracy(ha_tr, y_tr, ha_te, y_te, nc, device)   # depth
    probe_v = linear_probe_accuracy(hv_tr, y_tr, hv_te, y_te, nc, device)   # inertial
    with torch.no_grad():
        sc = ALPHA if arm in ("forward", "ordinary") else 1.0
        if SCALED == "a":
            fa_te = sc * ha_te; fv_te = hv_te
            mfa = (sc * ha_tr).mean(0, keepdim=True).expand_as(fa_te); mfv = hv_tr.mean(0, keepdim=True).expand_as(fv_te)
        else:
            fa_te = ha_te; fv_te = sc * hv_te
            mfa = ha_tr.mean(0, keepdim=True).expand_as(fa_te); mfv = (sc * hv_tr).mean(0, keepdim=True).expand_as(fv_te)

        def acc(fa, fv):
            return float((model.head(fuse(model.fusion, fa, fv)).argmax(1) == y_te).float().mean())
        v_both = acc(fa_te, fv_te); v_A = acc(fa_te, mfv); v_B = acc(mfa, fv_te); v_empty = acc(mfa, mfv)
        z_A = acc(fa_te, torch.zeros_like(fv_te)); z_B = acc(torch.zeros_like(fa_te), fv_te)
    # shapley of the SCALED modality (inertial = B)
    shap_scaled = 0.5 * (v_B - v_empty) + 0.5 * (v_both - v_A) if SCALED == "v" else \
                  0.5 * (v_A - v_empty) + 0.5 * (v_both - v_B)
    return {"probe_depth": probe_a, "probe_inertial": probe_v,
            "acc_depth": z_A, "acc_inertial": z_B,
            "dominance": z_A - z_B,              # masked: depth - inertial
            "meanabl_dominance": v_A - v_B,
            "shapley_scaled": shap_scaled}


def train(cfg, feats, device, seed, arm):
    set_seed(seed, True); nc = cfg["data"]["n_classes"]
    a_tr, v_tr, y_tr = feats["train"]
    model = UTDMultimodalModel(nc, rep_dim=cfg["model"]["rep_dim"], fusion=cfg["model"]["fusion"],
                               normalization="none", head=cfg["model"]["head"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    lf = nn.CrossEntropyLoss(); epochs = cfg["train"]["epochs"]; bs = cfg["train"]["batch_size"]; n = y_tr.numel()
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            loss = lf(arm_logits(model, a_tr[idx], v_tr[idx], ALPHA, arm), y_tr[idx])
            if not torch.isfinite(loss):
                return {"seed": seed, "arm": arm, "diverged": True}
            loss.backward(); opt.step()
    m = decoupled(model, feats, nc, device, arm)
    return {"seed": seed, "arm": arm, "diverged": False, **m}


def main():
    cfg = load_config(ROOT / "configs" / "utd_mhad.yaml")
    device = get_device(cfg["device"]); print("Device:", device_report(device))
    feats = load_utd(ROOT / cfg["data"]["cache_dir"], device)
    print("shapes:", {k: tuple(feats[k][0].shape) for k in feats}, "scaled modality:", SCALED)
    out = ROOT / "results" / "phase4_baselines"; out.mkdir(parents=True, exist_ok=True)
    seeds = cfg["seed_policy"]["seeds"]; rows = []; t0 = time.time()
    for arm in ARMS:
        for seed in seeds:
            r = train(cfg, feats, device, seed, arm); rows.append(r)
            print(f"  {arm:9s} seed={seed}: dom={r.get('dominance',float('nan')):+.3f} "
                  f"probe_depth={r.get('probe_depth',float('nan')):.3f} "
                  f"probe_inert={r.get('probe_inertial',float('nan')):.3f} ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows); df.to_csv(out / "phase4_utd.csv", index=False)

    def pci(arm, col):
        b = df[(df.arm == "baseline") & (~df.diverged)].sort_values("seed")[col].values
        x = df[(df.arm == arm) & (~df.diverged)].sort_values("seed")[col].values
        k = min(len(b), len(x))
        if k < 2:
            return None
        dd = x[:k] - b[:k]; sd = dd.std(ddof=1)
        return dd.mean(), 1.96 * sd / math.sqrt(k), k

    print("\n=== UTD-MHAD baseline->arm shift (paired 95% CI). "
          "Second real dataset: masked moves but decoupled don't? ===")
    for arm in ["forward", "gradient"]:
        print(f"-- {arm} (scaled inertial, alpha=100) --")
        for col in ["dominance", "meanabl_dominance", "shapley_scaled", "probe_depth", "probe_inertial"]:
            r = pci(arm, col)
            print(f"   {col:18s}: {'n<2' if r is None else f'{r[0]:+.4f} ± {r[1]:.4f} (n={r[2]})'}")
    json.dump({"git_commit": git_hash(), "alpha": ALPHA, "scaled": SCALED,
               "seeds": seeds, "wall_s": round(time.time() - t0, 1)},
              open(out / "phase4_utd_config.json", "w"), indent=2, default=str)
    print(f"\nWALL {time.time()-t0:.1f}s -> {out}/phase4_utd.csv")


if __name__ == "__main__":
    main()
