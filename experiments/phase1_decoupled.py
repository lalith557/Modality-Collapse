"""Phase 1 (CVPR revision): decoupled metrics + sign-flip decisive test.

Reuses the VALIDATED operators and training setup from
experiments.disentangle_scale_grad (same model, data, arms, masking) and only
ADDS measurement:
  * frozen-encoder linear probe on encoder A and B  (decisive sign-flip test)
  * mean-ablation accuracy (replace a modality by its train mean, not zero)
  * 2-modality Shapley value (mean-ablation as the empty coalition)
  * per-epoch train_loss, rho_enc, rho_fus, acc_A, acc_B
  * parameter-update ratio logged for BOTH Adam and SGD (fixes Table I footer)

Nothing in prior experiments is modified. 10 seeds. Synthetic only (cheap).

Usage:
  python experiments/phase1_decoupled.py --pilot      # 1 seed, time it
  python experiments/phase1_decoupled.py --seeds 10   # full run
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data.synthetic import generate_dataset
from experiments.train import _optimizer, _to_device
from experiments.disentangle_scale_grad import arm_apply, _reps, arm_forward, arm_contrib, arm_rho
from models.fusion import fuse
from models.multimodal_model import build_model
from msc.device import device_report, get_device
from msc.runlog import git_hash, load_config
from msc.seed import set_seed
from analysis.probes import linear_probe_accuracy

ARMS = ["baseline", "forward", "gradient", "ordinary"]
NAN = float("nan")


def decoupled_metrics(model, data, alpha, arm, nc, device):
    """Probe (encoder quality), mean-ablation, Shapley, and zero-mask (paper metric).

    NOT under a global no_grad: the linear probe trains its own classifier and
    needs grad. Encoder features are detached; ablation accuracies use no_grad.
    """
    a_tr, v_tr, y_tr = data["train"]; a_te, v_te, y_te = data["test"]
    model.eval()
    with torch.no_grad():
        ha_tr, hv_tr = _reps(model, a_tr, v_tr)      # pre-arm encoder outputs
        ha_te, hv_te = _reps(model, a_te, v_te)
    ha_tr, hv_tr, ha_te, hv_te = ha_tr.detach(), hv_tr.detach(), ha_te.detach(), hv_te.detach()
    probe_a = linear_probe_accuracy(ha_tr, y_tr, ha_te, y_te, nc, device)   # grad enabled here
    probe_b = linear_probe_accuracy(hv_tr, y_tr, hv_te, y_te, nc, device)

    with torch.no_grad():
        fa_te = arm_apply(ha_te, alpha, arm); fv_te = hv_te     # fusion inputs (A post-arm)
        fa_tr = arm_apply(ha_tr, alpha, arm)
        mfa = fa_tr.mean(0, keepdim=True).expand_as(fa_te)
        mfv = hv_tr.mean(0, keepdim=True).expand_as(fv_te)

        def acc(fa, fv):
            return float((model.head(fuse(model.fusion, fa, fv)).argmax(1) == y_te).float().mean())

        v_both = acc(fa_te, fv_te); v_A = acc(fa_te, mfv); v_B = acc(mfa, fv_te); v_empty = acc(mfa, mfv)
        z_A = acc(fa_te, torch.zeros_like(fv_te)); z_B = acc(torch.zeros_like(fa_te), fv_te)
    shap_a = 0.5 * (v_A - v_empty) + 0.5 * (v_both - v_B)
    shap_b = 0.5 * (v_B - v_empty) + 0.5 * (v_both - v_A)
    return {"probe_a": probe_a, "probe_b": probe_b,
            "meanabl_acc_a": v_A, "meanabl_acc_b": v_B, "meanabl_dominance": v_A - v_B,
            "zeromask_acc_a": z_A, "zeromask_acc_b": z_B, "zeromask_dominance": z_A - z_B,
            "shapley_a": shap_a, "shapley_b": shap_b, "v_both": v_both, "v_empty": v_empty}


def train_one(cfg, ds, device, seed, alpha, arm, optimizer):
    set_seed(seed, True)
    nc = ds.meta["n_classes"]
    data = _to_device(ds, device)
    a_tr, v_tr, y_tr = data["train"]; a_va, v_va, y_va = data["val"]; a_te, v_te, y_te = data["test"]
    cfg2 = json.loads(json.dumps(cfg)); cfg2["train"]["optimizer"] = optimizer
    model = build_model(cfg2, ds.meta["n_dim_a"], ds.meta["n_dim_v"], nc).to(device)
    opt = _optimizer(model, cfg2, device)
    lf = nn.CrossEntropyLoss()
    epochs = int(cfg2["train"]["epochs"]); bs = int(cfg2["train"]["batch_size"]); n = y_tr.numel()
    ea = list(model.enc_a.parameters()); ev = list(model.enc_v.parameters())
    hist = []; diverged = False
    for ep in range(1, epochs + 1):
        model.train(); perm = torch.randperm(n, device=device)
        ga = gb = ua = ub = 0.0; nb = 0; ep_loss = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            logits, *_ = arm_forward(model, a_tr[idx], v_tr[idx], alpha, arm)
            loss = lf(logits, y_tr[idx])
            if not torch.isfinite(loss):
                diverged = True; break
            loss.backward()
            ga += sum(float(p.grad.pow(2).sum()) for p in ea) ** 0.5
            gb += sum(float(p.grad.pow(2).sum()) for p in ev) ** 0.5
            pa = [p.detach().clone() for p in ea]; pv = [p.detach().clone() for p in ev]
            opt.step()
            ua += sum(float((p.detach() - q).pow(2).sum()) for p, q in zip(ea, pa)) ** 0.5
            ub += sum(float((p.detach() - q).pow(2).sum()) for p, q in zip(ev, pv)) ** 0.5
            ep_loss += float(loss.detach()) * len(idx); nb += 1
        if diverged:
            break
        re, rf = arm_rho(model, a_va, v_va, alpha, arm)
        c = arm_contrib(model, a_va, v_va, y_va, alpha, arm, nc)
        hist.append({"epoch": ep, "train_loss": ep_loss / n, "rho_enc": re, "rho_fus": rf,
                     "acc_a": c["acc_a"], "acc_b": c["acc_v"], "dominance": c["dominance"],
                     "grad_ratio": (ga / gb) if gb > 0 else NAN,
                     "update_ratio": (ua / ub) if ub > 0 else NAN})
    if diverged:
        return {"seed": seed, "alpha": alpha, "arm": arm, "optimizer": optimizer, "diverged": True}, hist
    te = arm_contrib(model, a_te, v_te, y_te, alpha, arm, nc)
    re, rf = arm_rho(model, a_te, v_te, alpha, arm)
    comp = (1.0 - math.log(rf) / math.log(alpha)) if (alpha != 1.0 and rf > 0) else NAN
    dec = decoupled_metrics(model, data, alpha, arm, nc, device)
    final = {"seed": seed, "alpha": alpha, "arm": arm, "optimizer": optimizer, "diverged": False,
             "acc_a": te["acc_a"], "acc_v": te["acc_v"], "dominance": te["dominance"],
             "multimodal_gain": te["multimodal_gain"], "rho_enc": re, "rho_fus": rf, "compensation": comp,
             "grad_ratio_final": hist[-1]["grad_ratio"], "update_ratio_final": hist[-1]["update_ratio"], **dec}
    return final, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="1 seed, alpha=100, baseline+forward, both opt; time it")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--alphas", type=str, default="10,100")
    args = ap.parse_args()

    device = get_device({"type": "cuda", "allow_cpu": False})
    print("Device:", device_report(device))
    cfg = load_config(ROOT / "configs" / "boundary.yaml")
    out = ROOT / "results" / "phase1_decoupled"; out.mkdir(parents=True, exist_ok=True)

    if args.pilot:
        seeds = [0]; alphas = [100.0]; arms = ["baseline", "forward"]; opts = ["adam", "sgd"]
    else:
        seeds = list(range(args.seeds)); alphas = [float(x) for x in args.alphas.split(",")]; arms = ARMS; opts = ["adam", "sgd"]

    finals, trajs = [], []; t0 = time.time()
    for opt_name in opts:
        for alpha in alphas:
            for arm in arms:
                for seed in seeds:
                    p = dict(cfg["data"]); p.update(data_seed=seed, alpha_a=1.0, alpha_v=1.0)
                    ds = generate_dataset(p)
                    final, hist = train_one(cfg, ds, device, seed, alpha, arm, opt_name)
                    finals.append(final)
                    for h in hist:
                        trajs.append({"optimizer": opt_name, "alpha": alpha, "arm": arm, "seed": seed, **h})
                g = [f for f in finals if f["optimizer"] == opt_name and f["alpha"] == alpha and f["arm"] == arm and not f.get("diverged")]
                dom = np.mean([f["dominance"] for f in g]) if g else NAN
                pa = np.mean([f.get("probe_a", NAN) for f in g]) if g else NAN
                print(f"  {opt_name} a={alpha:g} {arm:9s} dom={dom:+.3f} probe_a={pa:.3f} ({time.time()-t0:.0f}s)")

    tag = "pilot" if args.pilot else f"seeds{args.seeds}"
    fin = pd.DataFrame(finals); tj = pd.DataFrame(trajs)
    fin.to_csv(out / f"phase1_primary_{tag}.csv", index=False)
    tj.to_csv(out / f"phase1_trajectories_{tag}.csv", index=False)
    json.dump({"git_commit": git_hash(), "seeds": seeds, "alphas": alphas, "arms": arms, "optimizers": opts,
               "n_runs": len(finals), "wall_s": round(time.time() - t0, 1)},
              open(out / f"phase1_config_{tag}.json", "w"), indent=2, default=str)
    print(f"\nWALL: {time.time()-t0:.1f}s for {len(finals)} runs -> {out}/phase1_primary_{tag}.csv")

    # Decisive sign-flip read-out (alpha=100, Adam)
    if not fin.empty:
        print("\n=== DECISIVE: encoder-A probe vs masked acc_A (Adam, alpha=100) ===")
        for arm in ["baseline", "forward"]:
            g = fin[(fin.optimizer == "adam") & (fin.alpha == 100.0) & (fin.arm == arm) & (~fin.diverged)]
            if len(g):
                print(f"  {arm:9s}: masked acc_A={g.acc_a.mean():.4f}  probe_A={g.probe_a.mean():.4f}  "
                      f"meanabl_dom={g.meanabl_dominance.mean():+.4f}  shapley_A={g.shapley_a.mean():+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
