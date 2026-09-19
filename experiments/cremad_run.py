"""CREMA-D R1/R2/R3 execution (frozen; PREREGISTRATION_CREMAD.md + configs/cremad.yaml).

R1  = natural multimodal baseline (norm=none), 5 seeds. Observational.
R2  = none vs per-modality z-score (before fusion), 5 seeds. Intervention.
R3  = controlled scale intervention on the FIXED trained reps: alpha_audio=c,
      alpha_video=1/c for c in {0.1..10}, no retraining. Primary causal test.
Modality contribution = the frozen synthetic (masking) definition:
      Delta_audio = Acc_AV - Acc_video(masked), Delta_video = Acc_AV - Acc_audio(masked).
Trains on cached GPU features. Test metrics use ONLY the frozen test actors.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from analysis.metrics import macro_f1
from analysis.statistics import bootstrap_ci
from interventions.diagnostic import norm_stats
from models.cremad_model import CremadMultimodalModel, count_params
from msc.device import device_report, get_device
from msc.runlog import git_hash, load_config
from msc.seed import set_seed

CACHE = ROOT / "data" / "cremad" / "cache"
EVAL_BS = 512   # batched inference (CNN activations are large; chunk to avoid OOM)


def _chunks(n, bs=EVAL_BS):
    for i in range(0, n, bs):
        yield slice(i, i + bs)


@torch.no_grad()
def batched_reps(model, a, v):
    ha, hv = [], []
    for sl in _chunks(a.shape[0]):
        h_a, h_v = model.encode(a[sl], v[sl])
        ha.append(h_a); hv.append(h_v)
    return torch.cat(ha), torch.cat(hv)


@torch.no_grad()
def measure_rho(model, a, v):
    was = model.training; model.eval()
    ha, hv = batched_reps(model, a, v)
    s = norm_stats(ha, hv)
    if was:
        model.train()
    return s


@torch.no_grad()
def contribution_metrics(model, a, v, y, nc):
    """Batched replica of analysis.metrics.contribution_metrics (identical numbers)."""
    was = model.training; model.eval()
    accs = {}
    for name, (ma, mv) in {"both": (1.0, 1.0), "a_only": (1.0, 0.0), "v_only": (0.0, 1.0)}.items():
        preds = []
        for sl in _chunks(a.shape[0]):
            preds.append(model(a[sl], v[sl], mask_a=ma, mask_v=mv).argmax(1))
        preds = torch.cat(preds)
        accs[name] = (float((preds == y).float().mean()), macro_f1(preds, y, nc))
    if was:
        model.train()
    acc_av, acc_a, acc_v = accs["both"][0], accs["a_only"][0], accs["v_only"][0]
    return {"acc_av": acc_av, "acc_a": acc_a, "acc_v": acc_v,
            "f1_av": accs["both"][1], "f1_a": accs["a_only"][1], "f1_v": accs["v_only"][1],
            "delta_a": acc_av - acc_v, "delta_v": acc_av - acc_a,
            "multimodal_gain": acc_av - max(acc_a, acc_v),
            "dominance_a_minus_v": acc_a - acc_v}


def load_features(device):
    d = {}
    for sp in ("train", "val", "test"):
        a = torch.from_numpy(np.load(CACHE / f"{sp}_audio.npy")).float().to(device)
        v = torch.from_numpy(np.load(CACHE / f"{sp}_video.npy")).float().to(device)
        y = torch.from_numpy(np.load(CACHE / f"{sp}_y.npy")).long().to(device)
        d[sp] = (a, v, y)
    return d


@torch.no_grad()
def modality_norms(model, a, v):
    return measure_rho(model, a, v)


def train_one(cfg, feats, device, seed, norm, fusion="sum"):
    set_seed(seed, True)
    nc = cfg["data"]["n_classes"]
    a_tr, v_tr, y_tr = feats["train"]; a_va, v_va, y_va = feats["val"]; a_te, v_te, y_te = feats["test"]
    model = CremadMultimodalModel(nc, rep_dim=cfg["model"]["rep_dim"], fusion=fusion,
                                  normalization=norm, head=cfg["model"]["head"]).to(device)
    if norm == "zscore":
        with torch.no_grad():
            h_a, h_v = batched_reps(model, a_tr, v_tr)   # pre-fit norm is identity
            model.norm_a.fit(h_a); model.norm_v.fit(h_v)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                           weight_decay=cfg["train"]["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    epochs = cfg["train"]["epochs"]; bs = cfg["train"]["batch_size"]; n = y_tr.numel()
    enc_a_p = list(model.enc_a.parameters()); enc_v_p = list(model.enc_v.parameters())
    hist = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train(); perm = torch.randperm(n, device=device)
        ga = gv = 0.0; nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(a_tr[idx], v_tr[idx]), y_tr[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")
            loss.backward()
            ga += sum(float(p.grad.pow(2).sum()) for p in enc_a_p) ** 0.5
            gv += sum(float(p.grad.pow(2).sum()) for p in enc_v_p) ** 0.5
            opt.step(); nb += 1
        st = measure_rho(model, a_va, v_va)
        c = contribution_metrics(model, a_va, v_va, y_va, nc)
        hist.append({"epoch": ep, "rho": st["rho"], "norm_audio": st["mean_norm_a"],
                     "norm_video": st["mean_norm_v"], "val_acc_av": c["acc_av"],
                     "val_delta_audio": c["delta_a"], "val_delta_video": c["delta_v"],
                     "val_gain": c["multimodal_gain"],
                     "grad_audio": ga / nb, "grad_video": gv / nb,
                     "grad_ratio": (ga / gv) if gv > 0 else float("nan")})
    # final TEST metrics (frozen held-out actors)
    te = contribution_metrics(model, a_te, v_te, y_te, nc)
    ns = modality_norms(model, a_te, v_te)
    final = {"seed": seed, "norm": norm, "fusion": fusion,
             "acc_av": te["acc_av"], "acc_audio": te["acc_a"], "acc_video": te["acc_v"],
             "f1_av": te["f1_av"], "f1_audio": te["f1_a"], "f1_video": te["f1_v"],
             "delta_audio": te["delta_a"], "delta_video": te["delta_v"],
             "abs_imbalance": abs(te["delta_a"] - te["delta_v"]),
             "multimodal_gain": te["multimodal_gain"],
             "rho": ns["rho"], "log_rho": ns["log_rho"],
             "norm_audio": ns["mean_norm_a"], "norm_video": ns["mean_norm_v"],
             "grad_audio_final": hist[-1]["grad_audio"], "grad_video_final": hist[-1]["grad_video"],
             "grad_ratio_final": hist[-1]["grad_ratio"], "runtime_s": round(time.time() - t0, 1)}
    return model, final, hist


@torch.no_grad()
def r3_intervention(model, feats, cfg, device, seed, norm):
    """alpha_audio=c, alpha_video=1/c on FIXED trained reps; eval on TEST."""
    nc = cfg["data"]["n_classes"]; a_te, v_te, y_te = feats["test"]
    rows = []
    model.eval()
    for c in cfg["r3"]["scale_factors"]:
        model.scale.enabled = True
        model.scale.alpha_a.fill_(float(c)); model.scale.alpha_v.fill_(1.0 / float(c))
        st = modality_norms(model, a_te, v_te)
        contrib = contribution_metrics(model, a_te, v_te, y_te, nc)
        rows.append({"seed": seed, "norm": norm, "scale_c": float(c),
                     "favored": "audio" if c > 1 else ("video" if c < 1 else "none"),
                     "rho": st["rho"], "log_rho": st["log_rho"],
                     "norm_audio": st["mean_norm_a"], "norm_video": st["mean_norm_v"],
                     "acc_av": contrib["acc_av"], "acc_audio": contrib["acc_a"], "acc_video": contrib["acc_v"],
                     "delta_audio": contrib["delta_a"], "delta_video": contrib["delta_v"],
                     "abs_imbalance": abs(contrib["delta_a"] - contrib["delta_v"]),
                     "dominance_audio_minus_video": contrib["acc_a"] - contrib["acc_v"],
                     "multimodal_gain": contrib["multimodal_gain"], "f1_av": contrib["f1_av"]})
    model.scale.enabled = False
    return rows


def agg(df, by, cols):
    out = []
    for key, g in df.groupby(by):
        row = dict(zip(by if isinstance(by, list) else [by],
                       key if isinstance(key, tuple) else [key]))
        row["n"] = len(g)
        for c in cols:
            v = g[c].values
            row[f"{c}_mean"] = float(np.mean(v)); row[f"{c}_std"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            row[f"{c}_ci95"] = 1.96 * row[f"{c}_std"] / math.sqrt(len(v)) if len(v) > 1 else 0.0
        out.append(row)
    return pd.DataFrame(out)


def main():
    cfg = load_config(ROOT / "configs" / "cremad.yaml")
    device = get_device(cfg["device"])
    print("Device:", device_report(device))
    seeds = cfg["seed_policy"]["seeds"]
    fusion = cfg["model"]["fusion"]
    feats = load_features(device)
    print("features:", {k: tuple(feats[k][0].shape) for k in feats})

    out = ROOT / cfg["output"]["root"]; out.mkdir(parents=True, exist_ok=True)
    finals, trajs, r3rows = [], [], []
    models = {}
    for norm in ("none", "zscore"):
        for seed in seeds:
            model, final, hist = train_one(cfg, feats, device, seed, norm, fusion)
            finals.append(final)
            for h in hist:
                trajs.append({"norm": norm, "seed": seed, **h})
            models[(norm, seed)] = model
            r3rows += r3_intervention(model, feats, cfg, device, seed, norm)
            print(f"  {norm:6s} seed{seed}  acc_av={final['acc_av']:.3f} "
                  f"d_audio={final['delta_audio']:+.3f} d_video={final['delta_video']:+.3f} "
                  f"rho={final['rho']:.2f} gain={final['multimodal_gain']:+.3f}")

    fin_df = pd.DataFrame(finals); traj_df = pd.DataFrame(trajs); r3_df = pd.DataFrame(r3rows)
    fin_df.to_csv(out / "cremad_primary.csv", index=False)
    traj_df.to_csv(out / "cremad_trajectories.csv", index=False)
    r3_df.to_csv(out / "cremad_scale_intervention.csv", index=False)

    metrics = ["acc_av", "acc_audio", "acc_video", "delta_audio", "delta_video",
               "abs_imbalance", "multimodal_gain", "rho", "grad_ratio_final", "f1_av"]
    agg_df = agg(fin_df, ["norm"], metrics)
    agg_df.to_csv(out / "cremad_aggregated.csv", index=False)
    r3_agg = agg(r3_df, ["norm", "scale_c"],
                 ["rho", "delta_audio", "delta_video", "abs_imbalance",
                  "dominance_audio_minus_video", "multimodal_gain", "acc_av"])
    r3_agg.to_csv(out / "cremad_scale_intervention_agg.csv", index=False)

    json.dump({"config": cfg, "git_commit": git_hash(), "seeds": seeds,
               "device": device_report(device),
               "features": {k: list(feats[k][0].shape) for k in feats}},
              open(out / "cremad_config.json", "w"), indent=2, default=str)

    _analysis(fin_df, r3_df, r3_agg, out)
    _figures(traj_df, r3_agg, r3_df, fin_df, out)
    print("\nsaved to", out)


def _analysis(fin_df, r3_df, r3_agg, out):
    from scipy import stats as sps
    a = {}
    none = fin_df[fin_df.norm == "none"]; zs = fin_df[fin_df.norm == "zscore"]
    a["R1_baseline_none"] = {m: [round(float(none[m].mean()), 4), round(float(none[m].std(ddof=1)), 4)]
                             for m in ["acc_av", "acc_audio", "acc_video", "delta_audio", "delta_video",
                                       "abs_imbalance", "multimodal_gain", "rho", "grad_ratio_final"]}
    a["R1_natural_rho_ci"] = bootstrap_ci(none.rho.values, none.rho.values, "pearson") if False else None
    # R2 paired none vs zscore
    a["R2"] = {}
    for m in ["abs_imbalance", "rho", "multimodal_gain", "acc_av", "grad_ratio_final"]:
        n = none.sort_values("seed")[m].values; z = zs.sort_values("seed")[m].values
        diff = z - n; sd = np.std(diff, ddof=1)
        a["R2"][m] = {"none_mean": float(np.mean(n)), "zscore_mean": float(np.mean(z)),
                      "delta": float(np.mean(diff)), "ci95": float(1.96 * sd / math.sqrt(len(diff))),
                      "cohen_d": float(np.mean(diff) / sd) if sd > 0 else float("nan"),
                      "p": float(sps.ttest_rel(z, n).pvalue)}
    # R3 causal: slope of dominance vs log2(scale_c) (none), monotonicity
    dn = r3_df[r3_df.norm == "none"]
    x = np.log2(dn.scale_c.values); ycon = dn.dominance_audio_minus_video.values
    sl = np.polyfit(x, ycon, 1)[0]
    a["R3_none"] = {
        "slope_dominance_vs_log2c": float(sl),
        "spearman_c_vs_dominance": float(sps.spearmanr(dn.scale_c, dn.dominance_audio_minus_video).correlation),
        "spearman_c_vs_rho": float(sps.spearmanr(dn.scale_c, dn.rho).correlation),
        "dominance_at_c": {float(c): round(float(r3_agg[(r3_agg.norm == "none") & (r3_agg.scale_c == c)]["dominance_audio_minus_video_mean"].values[0]), 4)
                           for c in sorted(dn.scale_c.unique())},
    }
    # R3 under z-score: does the response flatten?
    dz = r3_df[r3_df.norm == "zscore"]
    a["R3_zscore"] = {"slope_dominance_vs_log2c": float(np.polyfit(np.log2(dz.scale_c.values), dz.dominance_audio_minus_video.values, 1)[0]),
                      "spearman_c_vs_dominance": float(sps.spearmanr(dz.scale_c, dz.dominance_audio_minus_video).correlation)}
    a["R3_flattening_ratio_zscore_over_none"] = (a["R3_zscore"]["slope_dominance_vs_log2c"] /
                                                 a["R3_none"]["slope_dominance_vs_log2c"]) if a["R3_none"]["slope_dominance_vs_log2c"] != 0 else float("nan")
    # verdict inputs
    strong = abs(a["R3_none"]["spearman_c_vs_dominance"]) > 0.6 and abs(a["R3_none"]["slope_dominance_vs_log2c"]) > 0.01
    a["r3_effect_present"] = bool(strong)
    json.dump(a, open(out / "cremad_analysis.json", "w"), indent=2, default=float)
    print("\n=== R1 baseline (none) ===", a["R1_baseline_none"])
    print("=== R2 (zscore - none) ===", {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in a["R2"].items()})
    print("=== R3 none: slope=%.4f spearman(c,dominance)=%.3f ===" % (
        a["R3_none"]["slope_dominance_vs_log2c"], a["R3_none"]["spearman_c_vs_dominance"]))
    print("    dominance by c:", a["R3_none"]["dominance_at_c"])
    print("=== R3 zscore slope=%.4f (flattening ratio z/none=%.2f) ===" % (
        a["R3_zscore"]["slope_dominance_vs_log2c"], a["R3_flattening_ratio_zscore_over_none"]))


def _figures(traj_df, r3_agg, r3_df, fin_df, out):
    fig_dir = out / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    # R1 trajectories (none): rho, dominance, grad_ratio, acc
    tn = traj_df[traj_df.norm == "none"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for seed, g in tn.groupby("seed"):
        g = g.sort_values("epoch")
        ax[0, 0].plot(g.epoch, g.rho, alpha=0.7, label=f"s{seed}")
        ax[0, 1].plot(g.epoch, g.val_delta_audio - g.val_delta_video, alpha=0.7)
        ax[1, 0].plot(g.epoch, g.grad_ratio, alpha=0.7)
        ax[1, 1].plot(g.epoch, g.val_acc_av, alpha=0.7)
    ax[0, 0].set_title(r"$\rho$(t) = $\|h_{audio}\|/\|h_{video}\|$"); ax[0, 0].axhline(1, color="k", lw=0.6); ax[0, 0].legend(fontsize=7)
    ax[0, 1].set_title(r"$\Delta_{audio}-\Delta_{video}$ (t)"); ax[0, 1].axhline(0, color="k", lw=0.6)
    ax[1, 0].set_title("encoder grad ratio audio/video (t)"); ax[1, 0].axhline(1, color="k", lw=0.6); ax[1, 0].set_yscale("log")
    ax[1, 1].set_title("val accuracy (t)")
    for a_ in ax.flat:
        a_.set_xlabel("epoch"); a_.grid(alpha=0.25)
    fig.suptitle("CREMA-D R1 training trajectories (vanilla, per seed)")
    fig.tight_layout(); fig.savefig(fig_dir / "cremad_R1_trajectories.png", dpi=150); plt.close(fig)

    # R3 scale-response (none vs zscore): c vs rho, dominance, gain
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for norm, col in [("none", "#d62728"), ("zscore", "#1f77b4")]:
        g = r3_agg[r3_agg.norm == norm].sort_values("scale_c")
        x = np.log2(g.scale_c.values)
        ax[0].errorbar(x, g.rho_mean, yerr=g.rho_ci95, marker="o", color=col, label=norm, capsize=2)
        ax[1].errorbar(x, g.dominance_audio_minus_video_mean, yerr=g.dominance_audio_minus_video_ci95, marker="o", color=col, label=norm, capsize=2)
        ax[2].errorbar(x, g.multimodal_gain_mean, yerr=g.multimodal_gain_ci95, marker="o", color=col, label=norm, capsize=2)
    ax[0].set_yscale("log"); ax[0].set_ylabel(r"$\rho$"); ax[0].set_title("scale c -> rho")
    ax[1].axhline(0, color="k", lw=0.6); ax[1].set_ylabel("Acc_audio - Acc_video"); ax[1].set_title("scale c -> modality dominance (CAUSAL)")
    ax[2].set_ylabel("multimodal gain"); ax[2].set_title("scale c -> multimodal gain")
    for a_ in ax:
        a_.set_xlabel(r"$\log_2 c$  (audio x c, video x 1/c)"); a_.grid(alpha=0.25); a_.legend(fontsize=8); a_.axvline(0, color="0.6", ls=":")
    fig.suptitle("CREMA-D R3 controlled scale intervention (test set, mean +/- 95% CI over 5 seeds)")
    fig.tight_layout(); fig.savefig(fig_dir / "cremad_R3_scale_response.png", dpi=150); plt.close(fig)

    # R2 comparison bars
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.2))
    for j, m in enumerate(["abs_imbalance", "multimodal_gain", "acc_av"]):
        for k, norm in enumerate(["none", "zscore"]):
            g = fin_df[fin_df.norm == norm]
            ax[j].bar(k, g[m].mean(), yerr=1.96 * g[m].std(ddof=1) / math.sqrt(len(g)), capsize=4,
                      color=["#d62728", "#1f77b4"][k], label=norm)
        ax[j].set_xticks([0, 1]); ax[j].set_xticklabels(["none", "z-score"]); ax[j].set_title(m); ax[j].grid(alpha=0.25, axis="y")
    fig.suptitle("CREMA-D R2: vanilla vs per-modality z-score (test)")
    fig.tight_layout(); fig.savefig(fig_dir / "cremad_R2_comparison.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
