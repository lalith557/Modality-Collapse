"""Phase 3 controls (close-out, then no more experiments):
 (A) POSITIVE CONTROL: add Gaussian noise to CREMA-D audio input (2 levels), no
     scaling, AdamW baseline. Verify the decoupled metrics (probe_audio, Shapley)
     DETECT genuine audio degradation -> the W2 null for forward-scaling is
     meaningful, not metric insensitivity.
 (B) SGD at alpha=10 on CREMA-D (baseline/forward/gradient) -> is the SGD effect
     measurable at lower alpha where it may not diverge?
3 seeds. Reuses the CREMA model + decoupled logic.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, torch, torch.nn as nn
from experiments.cremad_run import _chunks, load_features
from experiments.cremad_disentangle import arm_logits, _reps
from experiments.disentangle_scale_grad import arm_apply
from models.cremad_model import CremadMultimodalModel
from models.fusion import fuse
from msc.device import device_report, get_device
from msc.runlog import git_hash, load_config
from msc.seed import set_seed
from analysis.probes import linear_probe_accuracy


def noisy(feats, level):
    out = {}
    for k, (a, v, y) in feats.items():
        out[k] = (a + level * float(a.std()) * torch.randn_like(a), v, y)
    return out


def extract(model, a, v):
    model.eval(); ha = []; hv = []
    with torch.no_grad():
        for sl in _chunks(a.shape[0]):
            x, z = _reps(model, a[sl], v[sl]); ha.append(x); hv.append(z)
    return torch.cat(ha).detach(), torch.cat(hv).detach()


def decoupled(model, feats, nc, device, arm, alpha):
    a_tr, v_tr, y_tr = feats["train"]; a_te, v_te, y_te = feats["test"]
    ha_tr, hv_tr = extract(model, a_tr, v_tr); ha_te, hv_te = extract(model, a_te, v_te)
    probe_a = linear_probe_accuracy(ha_tr, y_tr, ha_te, y_te, nc, device)
    probe_v = linear_probe_accuracy(hv_tr, y_tr, hv_te, y_te, nc, device)
    with torch.no_grad():
        fs = alpha if arm in ("forward", "ordinary") else 1.0
        fa_te = fs * ha_te; mfa = (fs * ha_tr).mean(0, keepdim=True).expand_as(fa_te)
        mfv = hv_tr.mean(0, keepdim=True).expand_as(hv_te)

        def acc(fa, fv): return float((model.head(fuse(model.fusion, fa, fv)).argmax(1) == y_te).float().mean())
        v_both = acc(fa_te, hv_te); v_A = acc(fa_te, mfv); v_B = acc(mfa, hv_te); v_e = acc(mfa, mfv)
        z_A = acc(fa_te, torch.zeros_like(hv_te)); z_B = acc(torch.zeros_like(fa_te), hv_te)
    return {"probe_audio": probe_a, "probe_video": probe_v, "acc_audio": z_A, "acc_video": z_B,
            "dominance": z_A - z_B, "meanabl_dominance": v_A - v_B,
            "shapley_audio": 0.5 * (v_A - v_e) + 0.5 * (v_both - v_B)}


def train(cfg, feats, device, seed, arm, optimizer, lr, alpha):
    set_seed(seed, True); nc = cfg["data"]["n_classes"]
    a_tr, v_tr, y_tr = feats["train"]
    model = CremadMultimodalModel(nc, rep_dim=cfg["model"]["rep_dim"], fusion=cfg["model"]["fusion"],
                                  normalization="none", head=cfg["model"]["head"]).to(device)
    opt = (torch.optim.AdamW(model.parameters(), lr=lr) if optimizer == "adamw"
           else torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9))
    lf = nn.CrossEntropyLoss(); epochs = cfg["train"]["epochs"]; bs = cfg["train"]["batch_size"]; n = y_tr.numel()
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            loss = lf(arm_logits(model, a_tr[idx], v_tr[idx], alpha, arm), y_tr[idx])
            if not torch.isfinite(loss):
                return {"seed": seed, "arm": arm, "optimizer": optimizer, "diverged": True}
            loss.backward(); opt.step()
    return {"seed": seed, "arm": arm, "optimizer": optimizer, "diverged": False,
            **decoupled(model, feats, nc, device, arm, alpha)}


def main():
    cfg = load_config(ROOT / "configs" / "cremad.yaml")
    device = get_device(cfg["device"]); print("Device:", device_report(device))
    feats = load_features(device); out = ROOT / "results" / "phase3_cremad"; out.mkdir(parents=True, exist_ok=True)
    rows = []; t0 = time.time()

    print("=== (A) POSITIVE CONTROL: audio noise, AdamW baseline, 3 seeds ===")
    for level in [0.0, 0.5, 1.0]:
        nf = noisy(feats, level) if level > 0 else feats
        for seed in range(3):
            r = train(cfg, nf, device, seed, "baseline", "adamw", 1e-3, 100.0)
            rows.append({"control": "noise", "level": level, **r})
        g = [x for x in rows if x.get("control") == "noise" and x["level"] == level and not x.get("diverged")]
        print(f"  audio noise={level}: probe_audio={np.mean([x['probe_audio'] for x in g]):.3f} "
              f"shapley_audio={np.mean([x['shapley_audio'] for x in g]):.3f} "
              f"acc_audio={np.mean([x['acc_audio'] for x in g]):.3f} ({time.time()-t0:.0f}s)")

    print("=== (B) SGD at alpha=10, lr=1e-2, 3 seeds ===")
    for arm in ["baseline", "forward", "gradient"]:
        for seed in range(3):
            r = train(cfg, feats, device, seed, arm, "sgd", 1e-2, 10.0)
            rows.append({"control": "sgd_a10", "level": None, **r})
        g = [x for x in rows if x.get("control") == "sgd_a10" and x["arm"] == arm and not x.get("diverged")]
        nd = sum(1 for x in rows if x.get("control") == "sgd_a10" and x["arm"] == arm and x.get("diverged"))
        if g:
            print(f"  sgd a=10 {arm:9s}: dom={np.mean([x['dominance'] for x in g]):+.3f} "
                  f"probe_audio={np.mean([x['probe_audio'] for x in g]):.3f} div={nd} ({time.time()-t0:.0f}s)")
        else:
            print(f"  sgd a=10 {arm:9s}: ALL DIVERGED (div={nd})")

    pd.DataFrame(rows).to_csv(out / "phase3_controls.csv", index=False)
    json.dump({"git_commit": git_hash(), "wall_s": round(time.time() - t0, 1)},
              open(out / "phase3_controls_config.json", "w"), indent=2, default=str)
    print(f"\nWALL {time.time()-t0:.1f}s -> {out}/phase3_controls.csv")


if __name__ == "__main__":
    main()
