"""CREMA-D training-time scale intervention (H6). See PREREGISTRATION_CREMAD_TRAINING_SCALE.md.

Scale applied IN-GRAPH during every training step (encoders trainable): tests whether
imposing modality feature-scale disparity during training causes persistent learned
modality imbalance, or whether trainable encoders compensate. R1-R3 / E1-E5 unchanged.

Conditions: none {(1,1),(5,1),(10,1),(1,5),(1,10)} + zscore {(1,1),(10,1),(1,10)}.
Reuses cached ENCODER INPUTS (log-mel/frames) -> encoders run & update each step.
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
from scipy import stats as sps

from experiments.cremad_run import (_chunks, batched_reps, contribution_metrics,
                                    load_features, measure_rho)
from interventions.diagnostic import norm_stats
from models.cremad_model import CremadMultimodalModel
from msc.device import device_report, get_device
from msc.runlog import git_hash, load_config
from msc.seed import set_seed

CONDS_NONE = [(1.0, 1.0), (5.0, 1.0), (10.0, 1.0), (1.0, 5.0), (1.0, 10.0)]
CONDS_ZS = [(1.0, 1.0), (10.0, 1.0), (1.0, 10.0)]


@torch.no_grad()
def rho_encoder(model, a, v):
    """Learned encoder-output ratio ||enc_audio||/||enc_video|| (pre-scale, pre-norm)."""
    ha, hv = [], []
    for sl in _chunks(a.shape[0]):
        h_a, h_v = model.raw_encode(a[sl], v[sl])
        ha.append(h_a); hv.append(h_v)
    return norm_stats(torch.cat(ha), torch.cat(hv))


def train_ts(cfg, feats, device, seed, norm, aa, av):
    set_seed(seed, True)
    nc = cfg["data"]["n_classes"]
    a_tr, v_tr, y_tr = feats["train"]; a_va, v_va, y_va = feats["val"]; a_te, v_te, y_te = feats["test"]
    model = CremadMultimodalModel(nc, rep_dim=cfg["model"]["rep_dim"], fusion=cfg["model"]["fusion"],
                                  normalization=norm, head=cfg["model"]["head"],
                                  scale_injection={"enabled": True, "alpha_a": aa, "alpha_v": av},
                                  scale_first=True).to(device)
    if norm == "zscore":                       # fit on SCALED enc outputs (scale_first) at init
        with torch.no_grad():
            h_a, h_v = batched_reps(model, a_tr, v_tr)   # = aa*enc_a, av*enc_v (pre-fit norm=identity)
            model.norm_a.fit(h_a); model.norm_v.fit(h_v)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    epochs = cfg["train"]["epochs"]; bs = cfg["train"]["batch_size"]; n = y_tr.numel()
    enc_a_p = list(model.enc_a.parameters()); enc_v_p = list(model.enc_v.parameters())
    hist = []; t0 = time.time(); diverged = False
    for ep in range(1, epochs + 1):
        model.train(); perm = torch.randperm(n, device=device); ga = gv = 0.0; nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(a_tr[idx], v_tr[idx]), y_tr[idx])   # scale is IN-GRAPH
            if not torch.isfinite(loss):
                diverged = True; break
            loss.backward()
            ga += sum(float(p.grad.pow(2).sum()) for p in enc_a_p) ** 0.5
            gv += sum(float(p.grad.pow(2).sum()) for p in enc_v_p) ** 0.5
            opt.step(); nb += 1
        if diverged:
            break
        rf = measure_rho(model, a_va, v_va)            # fusion-boundary rho (post-scale)
        re = rho_encoder(model, a_va, v_va)            # learned encoder-output ratio (pre-scale)
        c = contribution_metrics(model, a_va, v_va, y_va, nc)
        hist.append({"epoch": ep, "rho_fusion": rf["rho"], "rho_enc": re["rho"],
                     "val_delta_audio": c["delta_a"], "val_delta_video": c["delta_v"],
                     "val_dominance": c["dominance_a_minus_v"], "val_acc_av": c["acc_av"],
                     "grad_ratio": (ga / gv) if gv > 0 else float("nan")})
    if diverged:
        return {"seed": seed, "norm": norm, "alpha_audio": aa, "alpha_video": av,
                "diverged": True}, hist
    te = contribution_metrics(model, a_te, v_te, y_te, nc)
    rf = measure_rho(model, a_te, v_te); re = rho_encoder(model, a_te, v_te)
    imposed = aa / av
    final = {"seed": seed, "norm": norm, "alpha_audio": aa, "alpha_video": av,
             "imposed_ratio": imposed, "log2_imposed": math.log2(imposed), "diverged": False,
             "acc_av": te["acc_av"], "acc_audio": te["acc_a"], "acc_video": te["acc_v"],
             "delta_audio": te["delta_a"], "delta_video": te["delta_v"],
             "abs_imbalance": abs(te["delta_a"] - te["delta_v"]),
             "dominance_audio_minus_video": te["dominance_a_minus_v"],
             "multimodal_gain": te["multimodal_gain"], "f1_av": te["f1_av"],
             "rho_fusion": rf["rho"], "rho_enc": re["rho"],
             # compensation: fraction of imposed log-ratio counteracted by the encoders
             "compensation": 1.0 - (math.log(rf["rho"]) / math.log(imposed)) if imposed != 1 else float("nan"),
             "grad_ratio_final": hist[-1]["grad_ratio"] if hist else float("nan"),
             "runtime_s": round(time.time() - t0, 1)}
    return final, hist


def main():
    cfg = load_config(ROOT / "configs" / "cremad.yaml")
    device = get_device(cfg["device"]); print("Device:", device_report(device))
    seeds = cfg["seed_policy"]["seeds"]; feats = load_features(device)
    out = ROOT / cfg["output"]["root"]; out.mkdir(parents=True, exist_ok=True)

    finals, trajs = [], []
    runset = [("none", aa, av) for (aa, av) in CONDS_NONE] + [("zscore", aa, av) for (aa, av) in CONDS_ZS]
    for norm, aa, av in runset:
        for seed in seeds:
            final, hist = train_ts(cfg, feats, device, seed, norm, aa, av)
            finals.append(final)
            for h in hist:
                trajs.append({"norm": norm, "alpha_audio": aa, "alpha_video": av, "seed": seed, **h})
            tag = "DIVERGED" if final.get("diverged") else \
                f"dom={final['dominance_audio_minus_video']:+.3f} |imb|={final['abs_imbalance']:.3f} " \
                f"rho_fus={final['rho_fusion']:.2f} rho_enc={final['rho_enc']:.2f} comp={final['compensation']:.2f}"
            print(f"  {norm:6s} a{aa:g}/v{av:g} seed{seed}  {tag}")

    fin = pd.DataFrame(finals); tj = pd.DataFrame(trajs)
    fin.to_csv(out / "cremad_train_scale_primary.csv", index=False)
    tj.to_csv(out / "cremad_train_scale_trajectories.csv", index=False)
    json.dump({"config": cfg, "git_commit": git_hash(), "seeds": seeds,
               "conditions_none": CONDS_NONE, "conditions_zscore": CONDS_ZS},
              open(out / "cremad_train_scale_config.json", "w"), indent=2, default=str)

    _analyze_and_figures(fin, tj, out)


def _agg(df, cols):
    rows = []
    for (norm, aa, av), g in df.groupby(["norm", "alpha_audio", "alpha_video"]):
        r = {"norm": norm, "alpha_audio": aa, "alpha_video": av, "n": len(g),
             "n_diverged": int(g.diverged.sum()) if "diverged" in g else 0}
        gg = g[g.diverged == False] if "diverged" in g else g
        for c in cols:
            if c in gg and len(gg):
                v = gg[c].values
                r[f"{c}_mean"] = float(np.mean(v)); r[f"{c}_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                r[f"{c}_ci95"] = 1.96 * r[f"{c}_sd"] / math.sqrt(len(v)) if len(v) > 1 else 0.0
        rows.append(r)
    return pd.DataFrame(rows)


def _analyze_and_figures(fin, tj, out):
    ok = fin[fin.diverged == False]
    cols = ["dominance_audio_minus_video", "abs_imbalance", "delta_audio", "delta_video",
            "multimodal_gain", "acc_av", "rho_fusion", "rho_enc", "compensation", "grad_ratio_final"]
    ag = _agg(fin, cols); ag.to_csv(out / "cremad_train_scale_agg.csv", index=False)

    none = ok[ok.norm == "none"]
    base_dom = none[(none.alpha_audio == 1) & (none.alpha_video == 1)].dominance_audio_minus_video
    # primary: does training-time scale change learned dominance vs baseline?
    a = {"n_diverged": int(fin.diverged.sum()), "baseline_dominance_mean": float(base_dom.mean())}
    for (aa, av) in CONDS_NONE:
        g = none[(none.alpha_audio == aa) & (none.alpha_video == av)]
        key = f"a{aa:g}_v{av:g}"
        a[key] = {"dominance": float(g.dominance_audio_minus_video.mean()),
                  "dominance_sd": float(g.dominance_audio_minus_video.std(ddof=1)),
                  "abs_imbalance": float(g.abs_imbalance.mean()),
                  "rho_fusion": float(g.rho_fusion.mean()), "rho_enc": float(g.rho_enc.mean()),
                  "compensation": float(g.compensation.mean()) if (aa != av) else None,
                  "gain": float(g.multimodal_gain.mean()), "acc_av": float(g.acc_av.mean())}
    # slope dominance vs log2 imposed (none)
    x = none.log2_imposed.values; y = none.dominance_audio_minus_video.values
    a["slope_dominance_vs_log2imposed"] = float(np.polyfit(x, y, 1)[0])
    a["spearman_imposed_vs_dominance"] = float(sps.spearmanr(none.imposed_ratio, none.dominance_audio_minus_video).correlation)
    a["spearman_imposed_vs_rho_enc"] = float(sps.spearmanr(none.imposed_ratio, none.rho_enc).correlation)
    # compensation: audio10 & video10
    a10 = none[(none.alpha_audio == 10) & (none.alpha_video == 1)]
    v10 = none[(none.alpha_audio == 1) & (none.alpha_video == 10)]
    a["compensation_audio10"] = {"imposed_ratio": 10.0, "rho_enc_final": float(a10.rho_enc.mean()),
                                 "rho_fusion_final": float(a10.rho_fusion.mean()),
                                 "compensation_frac": float(a10.compensation.mean())}
    a["compensation_video10"] = {"imposed_ratio": 0.1, "rho_enc_final": float(v10.rho_enc.mean()),
                                 "rho_fusion_final": float(v10.rho_fusion.mean()),
                                 "compensation_frac": float(v10.compensation.mean())}
    # normalization: does zscore change the audio10/video10 outcome vs none?
    zs = ok[ok.norm == "zscore"]
    a["normalization"] = {}
    for (aa, av) in [(10.0, 1.0), (1.0, 10.0)]:
        n_ = none[(none.alpha_audio == aa) & (none.alpha_video == av)]
        z_ = zs[(zs.alpha_audio == aa) & (zs.alpha_video == av)]
        a["normalization"][f"a{aa:g}_v{av:g}"] = {
            "none_dominance": float(n_.dominance_audio_minus_video.mean()),
            "zscore_dominance": float(z_.dominance_audio_minus_video.mean()) if len(z_) else float("nan"),
            "none_rho_fusion": float(n_.rho_fusion.mean()),
            "zscore_rho_fusion": float(z_.rho_fusion.mean()) if len(z_) else float("nan")}
    # verdict
    dom_shift_10 = abs(a["a10_v1"]["dominance"] - a["baseline_dominance_mean"])
    comp = a["compensation_audio10"]["compensation_frac"]
    strong = abs(a["spearman_imposed_vs_dominance"]) > 0.6 and dom_shift_10 > 0.05 and comp < 0.5
    partial = dom_shift_10 > 0.02 or (comp < 0.8 and abs(a["slope_dominance_vs_log2imposed"]) > 0.005)
    a["verdict"] = ("SUPPORT" if strong else ("PARTIAL SUPPORT" if partial else "NO SUPPORT"))
    if a["n_diverged"] > 0.2 * len(fin):
        a["verdict"] = "TECHNICAL FAILURE (excess divergence)"
    json.dump(a, open(out / "cremad_train_scale_analysis.json", "w"), indent=2, default=float)

    print("\n=== H6 primary: learned dominance by condition (none) ===")
    for (aa, av) in CONDS_NONE:
        d = a[f"a{aa:g}_v{av:g}"]
        print(f"  a{aa:g}/v{av:g}: dominance={d['dominance']:+.3f} |imb|={d['abs_imbalance']:.3f} "
              f"rho_fus={d['rho_fusion']:.2f} rho_enc={d['rho_enc']:.2f} comp={d['compensation']} gain={d['gain']:+.3f}")
    print(f"slope(dominance vs log2 imposed)={a['slope_dominance_vs_log2imposed']:.4f} "
          f"spearman={a['spearman_imposed_vs_dominance']:.3f}")
    print(f"compensation audio10: rho_enc={a['compensation_audio10']['rho_enc_final']:.3f} "
          f"comp_frac={a['compensation_audio10']['compensation_frac']:.2f}")
    print("normalization:", a["normalization"])
    print("VERDICT:", a["verdict"])

    _figures(tj, ag, none, out)


def _figures(tj, ag, none, out):
    fig_dir = out / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    COND = [(1.0, 1.0, "baseline", "#000000"), (5.0, 1.0, "audio x5", "#ff7f0e"),
            (10.0, 1.0, "audio x10", "#d62728"), (1.0, 5.0, "video x5", "#2ca02c"),
            (1.0, 10.0, "video x10", "#1f77b4")]
    tn = tj[tj.norm == "none"]

    def mean_traj(aa, av, key):
        g = tn[(tn.alpha_audio == aa) & (tn.alpha_video == av)]
        if not len(g):
            return None, None
        m = g.groupby("epoch")[key].mean()
        return m.index.values, m.values

    # Fig 1 & 2: rho_fusion(t), rho_enc(t), grad_ratio(t)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for aa, av, lab, c in COND:
        for j, key in enumerate(["rho_fusion", "rho_enc", "grad_ratio"]):
            e, y = mean_traj(aa, av, key)
            if e is not None:
                ax[j].plot(e, y, color=c, label=lab, lw=1.7)
    ax[0].set_title(r"$\rho_{fusion}$(t) (post-scale, what fusion sees)"); ax[0].set_yscale("log")
    ax[1].set_title(r"$\rho_{enc}$(t) = learned $\|enc_A\|/\|enc_V\|$ (compensation)"); ax[1].set_yscale("log")
    ax[2].set_title("encoder grad ratio audio/video (t)"); ax[2].set_yscale("log")
    for a_ in ax:
        a_.axhline(1, color="0.6", ls=":"); a_.set_xlabel("epoch"); a_.grid(alpha=0.25); a_.legend(fontsize=7)
    fig.suptitle("CREMA-D H6 training-time scale: representation & gradient dynamics (none, mean of 5 seeds)")
    fig.tight_layout(); fig.savefig(fig_dir / "cremad_TS_dynamics.png", dpi=150); plt.close(fig)

    # Fig 3 & 4: learned dominance & gain vs imposed log2 ratio
    an = ag[ag.norm == "none"].copy(); an["log2_imposed"] = np.log2(an.alpha_audio / an.alpha_video)
    an = an.sort_values("log2_imposed")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].errorbar(an.log2_imposed, an.dominance_audio_minus_video_mean, yerr=an.dominance_audio_minus_video_ci95,
                   marker="o", capsize=3, color="#d62728")
    # per-seed points
    for _, r in none.iterrows():
        ax[0].scatter(r.log2_imposed, r.dominance_audio_minus_video, s=12, color="0.6", alpha=0.6, zorder=0)
    ax[0].axhline(0, color="k", lw=0.6); ax[0].set_ylabel("learned dominance Acc_audio-Acc_video")
    ax[0].set_title("H6: learned modality dominance vs imposed scale")
    ax[1].errorbar(an.log2_imposed, an.multimodal_gain_mean, yerr=an.multimodal_gain_ci95, marker="o", capsize=3, color="#ff7f0e")
    ax[1].set_ylabel("multimodal gain"); ax[1].set_title("H6: multimodal gain vs imposed scale")
    for a_ in ax:
        a_.set_xlabel(r"imposed $\log_2(\alpha_{audio}/\alpha_{video})$"); a_.grid(alpha=0.25); a_.axvline(0, color="0.6", ls=":")
    fig.suptitle("CREMA-D H6 learned contribution (test, 5 seeds)")
    fig.tight_layout(); fig.savefig(fig_dir / "cremad_TS_contribution.png", dpi=150); plt.close(fig)

    # Fig 5: compensation — imposed vs final learned rho_enc and rho_fusion
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    an2 = ag[ag.norm == "none"].copy(); an2["imp"] = an2.alpha_audio / an2.alpha_video
    an2 = an2.sort_values("imp")
    ax.plot(np.log2(an2.imp), np.log2(an2.imp), "k--", label="no compensation (rho_enc=imposed)")
    ax.plot(np.log2(an2.imp), np.log2(an2.rho_enc_mean), "o-", color="#2ca02c", label=r"learned $\log_2\rho_{enc}$")
    ax.plot(np.log2(an2.imp), np.log2(an2.rho_fusion_mean), "s-", color="#d62728", label=r"$\log_2\rho_{fusion}$ (post-scale)")
    ax.axhline(0, color="0.6", ls=":")
    ax.set_xlabel(r"imposed $\log_2(\alpha_{audio}/\alpha_{video})$"); ax.set_ylabel(r"$\log_2$ ratio")
    ax.set_title("H6 compensation: do encoders counteract the imposed scale?")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(fig_dir / "cremad_TS_compensation.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
