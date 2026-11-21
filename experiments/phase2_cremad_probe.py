"""Phase 2 #5: CREMA-D real-data sign-flip test with the frozen-encoder probe.

Trains the SAME end-to-end CNN encoders as the paper (cremad_disentangle), adds
the decoupled metrics (probe on enc_a/enc_v, mean-ablation, Shapley). Arms:
baseline vs forward-only, Adam, alpha=100, 5 seeds. Tests whether forward scaling
RAISES audio contribution (as the paper's dominance says) while also degrading
or improving the audio encoder's probe — the real-data counterpart of Phase 1/2.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, torch, torch.nn as nn
from experiments.cremad_run import _chunks, load_features
from experiments.cremad_disentangle import _reps, arm_logits
from experiments.disentangle_scale_grad import arm_apply
from models.cremad_model import CremadMultimodalModel
from models.fusion import fuse
from msc.device import device_report, get_device
from msc.runlog import git_hash, load_config
from msc.seed import set_seed
from analysis.probes import linear_probe_accuracy

ALPHA = 100.0
ARMS = ["baseline", "forward"]


def extract(model, a, v):
    model.eval(); has = []; hvs = []
    with torch.no_grad():
        for sl in _chunks(a.shape[0]):
            ha, hv = _reps(model, a[sl], v[sl]); has.append(ha); hvs.append(hv)
    return torch.cat(has).detach(), torch.cat(hvs).detach()


def decoupled(model, feats, nc, device, arm):
    a_tr, v_tr, y_tr = feats["train"]; a_te, v_te, y_te = feats["test"]
    ha_tr, hv_tr = extract(model, a_tr, v_tr); ha_te, hv_te = extract(model, a_te, v_te)
    probe_a = linear_probe_accuracy(ha_tr, y_tr, ha_te, y_te, nc, device)
    probe_v = linear_probe_accuracy(hv_tr, y_tr, hv_te, y_te, nc, device)
    with torch.no_grad():
        fa_scale = ALPHA if arm in ("forward", "ordinary") else 1.0
        fa_te = fa_scale * ha_te; fv_te = hv_te
        mfa = (fa_scale * ha_tr).mean(0, keepdim=True).expand_as(fa_te)
        mfv = hv_tr.mean(0, keepdim=True).expand_as(fv_te)

        def acc(fa, fv):
            return float((model.head(fuse(model.fusion, fa, fv)).argmax(1) == y_te).float().mean())
        v_both = acc(fa_te, fv_te); v_A = acc(fa_te, mfv); v_B = acc(mfa, fv_te); v_empty = acc(mfa, mfv)
        z_A = acc(fa_te, torch.zeros_like(fv_te)); z_B = acc(torch.zeros_like(fa_te), fv_te)
    return {"probe_audio": probe_a, "probe_video": probe_v, "acc_audio": z_A, "acc_video": z_B,
            "dominance": z_A - z_B, "meanabl_dominance": v_A - v_B,
            "shapley_audio": 0.5 * (v_A - v_empty) + 0.5 * (v_both - v_B)}


def train_one(cfg, feats, device, seed, arm):
    set_seed(seed, True); nc = cfg["data"]["n_classes"]
    a_tr, v_tr, y_tr = feats["train"]
    model = CremadMultimodalModel(nc, rep_dim=cfg["model"]["rep_dim"], fusion=cfg["model"]["fusion"],
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
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); args = ap.parse_args()
    cfg = load_config(ROOT / "configs" / "cremad.yaml")
    device = get_device(cfg["device"]); print("Device:", device_report(device))
    feats = load_features(device)
    out = ROOT / "results" / "phase2_mechanism"; rows = []; t0 = time.time()
    for arm in ARMS:
        for seed in range(args.seeds):
            r = train_one(cfg, feats, device, seed, arm); rows.append(r)
            print(f"  {arm:9s} seed={seed}: dom={r.get('dominance',float('nan')):+.3f} "
                  f"probe_audio={r.get('probe_audio',float('nan')):.3f} ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows); df.to_csv(out / "phase2_cremad_probe.csv", index=False)

    def pci(arm, col):
        x = df[(df.arm == "baseline") & (~df.diverged)].sort_values("seed")[col].values
        y = df[(df.arm == arm) & (~df.diverged)].sort_values("seed")[col].values
        dd = y - x; sd = dd.std(ddof=1); return dd.mean(), 1.96 * sd / math.sqrt(len(dd))
    print("\n=== CREMA-D baseline->forward (Adam, alpha=100) ===")
    for col in ["dominance", "meanabl_dominance", "shapley_audio", "probe_audio", "probe_video", "acc_audio"]:
        m, c = pci("forward", col); print(f"  {col:18s}: shift={m:+.4f} ± {c:.4f}")
    json.dump({"git_commit": git_hash(), "seeds": args.seeds, "wall_s": round(time.time() - t0, 1)},
              open(out / "phase2_cremad_probe_config.json", "w"), indent=2, default=str)
    print(f"WALL {time.time()-t0:.1f}s -> {out}/phase2_cremad_probe.csv")


if __name__ == "__main__":
    main()
