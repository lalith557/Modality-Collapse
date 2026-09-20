"""UTD-MHAD training-time scale intervention (Phase 6) — cross-dataset validation of CREMA-D H6.
See PREREGISTRATION_UTD_MHAD_SCALE.md. E1-E5 and CREMA-D R1-R3/H6 are unchanged.

Modality A = DEPTH, B = INERTIAL. Scale injected IN-GRAPH before fusion (encoders trainable).
Conditions: none {(1,1),(5,1),(10,1),(1,5),(1,10)} + zscore {(1,1),(10,1),(1,10)} (alpha_depth,alpha_inertial).
Reuses the frozen measurement helpers unchanged. Test = fixed subjects {2,4,6,8}.
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

from experiments.cremad_run import _chunks, batched_reps, contribution_metrics, measure_rho
from interventions.diagnostic import norm_stats
from models.utd_model import UTDMultimodalModel, count_params
from msc.device import device_report, get_device
from msc.runlog import git_hash, load_config
from msc.seed import set_seed

# (alpha_depth, alpha_inertial). A=depth, B=inertial.
CONDS_NONE = [(1.0, 1.0), (5.0, 1.0), (10.0, 1.0), (1.0, 5.0), (1.0, 10.0)]
CONDS_ZS = [(1.0, 1.0), (10.0, 1.0), (1.0, 10.0)]


@torch.no_grad()
def rho_encoder(model, a, v):
    ha, hv = [], []
    for sl in _chunks(a.shape[0]):
        h_a, h_v = model.raw_encode(a[sl], v[sl]); ha.append(h_a); hv.append(h_v)
    return norm_stats(torch.cat(ha), torch.cat(hv))


def load_features(cache, device):
    d = {}
    for sp in ("train", "val", "test"):
        a = torch.from_numpy(np.load(cache / f"{sp}_depth.npy")).float().to(device)
        v = torch.from_numpy(np.load(cache / f"{sp}_inertial.npy")).float().to(device)
        y = torch.from_numpy(np.load(cache / f"{sp}_y.npy")).long().to(device)
        s = torch.from_numpy(np.load(cache / f"{sp}_subject.npy")).long().to(device)
        d[sp] = (a, v, y, s)
    return d


def train_ts(cfg, feats, device, seed, norm, aa, av):
    set_seed(seed, True)
    nc = cfg["data"]["n_classes"]
    a_tr, v_tr, y_tr, _ = feats["train"]; a_va, v_va, y_va, _ = feats["val"]
    a_te, v_te, y_te, s_te = feats["test"]
    model = UTDMultimodalModel(nc, rep_dim=cfg["model"]["rep_dim"], fusion=cfg["model"]["fusion"],
                               normalization=norm, head=cfg["model"]["head"],
                               scale_injection={"enabled": True, "alpha_a": aa, "alpha_v": av},
                               scale_first=True).to(device)
    if norm == "zscore":
        with torch.no_grad():
            h_a, h_v = batched_reps(model, a_tr, v_tr)
            model.norm_a.fit(h_a); model.norm_v.fit(h_v)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    epochs = cfg["train"]["epochs"]; bs = cfg["train"]["batch_size"]; n = y_tr.numel()
    ea = list(model.enc_a.parameters()); ev = list(model.enc_v.parameters())
    hist = []; t0 = time.time(); diverged = False
    for ep in range(1, epochs + 1):
        model.train(); perm = torch.randperm(n, device=device); ga = gv = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(a_tr[idx], v_tr[idx]), y_tr[idx])     # scale IN-GRAPH
            if not torch.isfinite(loss):
                diverged = True; break
            loss.backward()
            ga += sum(float(p.grad.pow(2).sum()) for p in ea) ** 0.5
            gv += sum(float(p.grad.pow(2).sum()) for p in ev) ** 0.5
            opt.step()
        if diverged:
            break
        rf = measure_rho(model, a_va, v_va); re = rho_encoder(model, a_va, v_va)
        c = contribution_metrics(model, a_va, v_va, y_va, nc)
        hist.append({"epoch": ep, "rho_fusion": rf["rho"], "rho_enc": re["rho"],
                     "val_dominance": c["dominance_a_minus_v"], "val_acc_ab": c["acc_av"],
                     "grad_ratio": (ga / gv) if gv > 0 else float("nan")})
    if diverged:
        return {"seed": seed, "norm": norm, "alpha_depth": aa, "alpha_inertial": av, "diverged": True}, hist, []
    te = contribution_metrics(model, a_te, v_te, y_te, nc)
    rf = measure_rho(model, a_te, v_te); re = rho_encoder(model, a_te, v_te)
    imposed = aa / av
    final = {"seed": seed, "norm": norm, "alpha_depth": aa, "alpha_inertial": av,
             "imposed_ratio": imposed, "log2_imposed": math.log2(imposed), "diverged": False,
             "acc_ab": te["acc_av"], "acc_depth": te["acc_a"], "acc_inertial": te["acc_v"],
             "delta_depth": te["delta_a"], "delta_inertial": te["delta_v"],
             "abs_imbalance": abs(te["delta_a"] - te["delta_v"]),
             "dominance_depth_minus_inertial": te["dominance_a_minus_v"],
             "multimodal_gain": te["multimodal_gain"], "f1_ab": te["f1_av"],
             "rho_fusion": rf["rho"], "rho_enc": re["rho"],
             "compensation": 1.0 - (math.log(rf["rho"]) / math.log(imposed)) if imposed != 1 else float("nan"),
             "grad_ratio_final": hist[-1]["grad_ratio"] if hist else float("nan"),
             "runtime_s": round(time.time() - t0, 1)}
    # per-test-subject breakdown (transparency; test subjects fixed)
    persub = []
    for s in sorted(set(int(x) for x in s_te.tolist())):
        m = (s_te == s)
        cs = contribution_metrics(model, a_te[m], v_te[m], y_te[m], nc)
        persub.append({"seed": seed, "norm": norm, "alpha_depth": aa, "alpha_inertial": av, "subject": s,
                       "n": int(m.sum()), "acc_ab": cs["acc_av"], "acc_depth": cs["acc_a"],
                       "acc_inertial": cs["acc_v"], "dominance_depth_minus_inertial": cs["dominance_a_minus_v"],
                       "multimodal_gain": cs["multimodal_gain"]})
    return final, hist, persub


def _agg(df, cols):
    rows = []
    for (norm, aa, av), g in df.groupby(["norm", "alpha_depth", "alpha_inertial"]):
        r = {"norm": norm, "alpha_depth": aa, "alpha_inertial": av, "n": len(g),
             "n_diverged": int(g.diverged.sum()) if "diverged" in g else 0}
        gg = g[g.diverged == False] if "diverged" in g else g
        for c in cols:
            if c in gg and len(gg):
                v = gg[c].values
                r[f"{c}_mean"] = float(np.mean(v)); r[f"{c}_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                r[f"{c}_ci95"] = 1.96 * r[f"{c}_sd"] / math.sqrt(len(v)) if len(v) > 1 else 0.0
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    cfg = load_config(ROOT / "configs" / "utd_mhad.yaml")
    device = get_device(cfg["device"]); print("Device:", device_report(device))
    cache = ROOT / cfg["data"]["cache_dir"]
    seeds = cfg["seed_policy"]["seeds"]; feats = load_features(cache, device)
    print("features:", {k: tuple(feats[k][0].shape) for k in feats})
    m0 = UTDMultimodalModel(cfg["data"]["n_classes"], rep_dim=cfg["model"]["rep_dim"]).to(device)
    print(f"params depth-enc={count_params(m0.enc_a)} inertial-enc={count_params(m0.enc_v)} total={count_params(m0)}")
    out = ROOT / cfg["output"]["root"]; out.mkdir(parents=True, exist_ok=True)

    finals, trajs, persubs = [], [], []
    runset = [("none", aa, av) for (aa, av) in CONDS_NONE] + [("zscore", aa, av) for (aa, av) in CONDS_ZS]
    for norm, aa, av in runset:
        for seed in seeds:
            final, hist, ps = train_ts(cfg, feats, device, seed, norm, aa, av)
            finals.append(final)
            for h in hist:
                trajs.append({"norm": norm, "alpha_depth": aa, "alpha_inertial": av, "seed": seed, **h})
            persubs += ps
            tag = "DIVERGED" if final.get("diverged") else \
                (f"dom={final['dominance_depth_minus_inertial']:+.3f} gain={final['multimodal_gain']:+.3f} "
                 f"rho_fus={final['rho_fusion']:.2f} rho_enc={final['rho_enc']:.2f} comp={final['compensation']:.2f}")
            print(f"  {norm:6s} d{aa:g}/i{av:g} seed{seed}  {tag}")

    fin = pd.DataFrame(finals); tj = pd.DataFrame(trajs); ps = pd.DataFrame(persubs)
    fin.to_csv(out / "utd_train_scale_primary.csv", index=False)
    tj.to_csv(out / "utd_train_scale_trajectories.csv", index=False)
    ps.to_csv(out / "utd_train_scale_persubject.csv", index=False)
    json.dump({"config": cfg, "git_commit": git_hash(), "seeds": seeds,
               "conditions_none": CONDS_NONE, "conditions_zscore": CONDS_ZS,
               "params": {"depth_enc": count_params(m0.enc_a), "inertial_enc": count_params(m0.enc_v)}},
              open(out / "utd_train_scale_config.json", "w"), indent=2, default=str)
    _analyze(fin, tj, out)


def _analyze(fin, tj, out):
    ok = fin[fin.diverged == False]
    cols = ["dominance_depth_minus_inertial", "abs_imbalance", "delta_depth", "delta_inertial",
            "multimodal_gain", "acc_ab", "acc_depth", "acc_inertial", "rho_fusion", "rho_enc",
            "compensation", "grad_ratio_final"]
    ag = _agg(fin, cols); ag.to_csv(out / "utd_train_scale_agg.csv", index=False)
    none = ok[ok.norm == "none"]
    base = none[(none.alpha_depth == 1) & (none.alpha_inertial == 1)]
    a = {"n_diverged": int(fin.diverged.sum()),
         "baseline_depth_only_acc": float(base.acc_depth.mean()),
         "baseline_inertial_only_acc": float(base.acc_inertial.mean()),
         "baseline_multimodal_acc": float(base.acc_ab.mean()),
         "baseline_gain": float(base.multimodal_gain.mean()),
         "baseline_dominance": float(base.dominance_depth_minus_inertial.mean()),
         "more_informative_modality": ("depth" if base.acc_depth.mean() > base.acc_inertial.mean() else "inertial")}
    for (aa, av) in CONDS_NONE:
        g = none[(none.alpha_depth == aa) & (none.alpha_inertial == av)]
        a[f"d{aa:g}_i{av:g}"] = {
            "dominance": float(g.dominance_depth_minus_inertial.mean()),
            "dominance_sd": float(g.dominance_depth_minus_inertial.std(ddof=1)) if len(g) > 1 else 0.0,
            "abs_imbalance": float(g.abs_imbalance.mean()), "gain": float(g.multimodal_gain.mean()),
            "acc_ab": float(g.acc_ab.mean()), "rho_fusion": float(g.rho_fusion.mean()),
            "rho_enc": float(g.rho_enc.mean()),
            "compensation": float(g.compensation.mean()) if aa != av else None}
    x = none.log2_imposed.values; y = none.dominance_depth_minus_inertial.values
    a["slope_dominance_vs_log2imposed"] = float(np.polyfit(x, y, 1)[0])
    a["spearman_imposed_vs_dominance"] = float(sps.spearmanr(none.imposed_ratio, none.dominance_depth_minus_inertial).correlation)
    a["spearman_imposed_vs_rho_enc"] = float(sps.spearmanr(none.imposed_ratio, none.rho_enc).correlation)
    d10 = none[(none.alpha_depth == 10) & (none.alpha_inertial == 1)]
    i10 = none[(none.alpha_depth == 1) & (none.alpha_inertial == 10)]
    a["compensation_depth10"] = {"rho_enc": float(d10.rho_enc.mean()), "rho_fusion": float(d10.rho_fusion.mean()),
                                 "comp_frac": float(d10.compensation.mean())}
    a["compensation_inertial10"] = {"rho_enc": float(i10.rho_enc.mean()), "rho_fusion": float(i10.rho_fusion.mean()),
                                    "comp_frac": float(i10.compensation.mean())}
    zs = ok[ok.norm == "zscore"]; a["normalization"] = {}
    for (aa, av) in [(10.0, 1.0), (1.0, 10.0)]:
        n_ = none[(none.alpha_depth == aa) & (none.alpha_inertial == av)]
        z_ = zs[(zs.alpha_depth == aa) & (zs.alpha_inertial == av)]
        a["normalization"][f"d{aa:g}_i{av:g}"] = {
            "none_dominance": float(n_.dominance_depth_minus_inertial.mean()),
            "zscore_dominance": float(z_.dominance_depth_minus_inertial.mean()) if len(z_) else float("nan"),
            "none_rho_fusion": float(n_.rho_fusion.mean()),
            "zscore_rho_fusion": float(z_.rho_fusion.mean()) if len(z_) else float("nan")}
    # verdict (mirror H6 thresholds; direction-agnostic on |dominance shift|)
    base_dom = a["baseline_dominance"]
    dom_shift_d10 = abs(a["d10_i1"]["dominance"] - base_dom)
    dom_shift_i10 = abs(a["d1_i10"]["dominance"] - base_dom)
    max_shift = max(dom_shift_d10, dom_shift_i10)
    strong = abs(a["spearman_imposed_vs_dominance"]) > 0.6 and max_shift > 0.05
    partial = max_shift > 0.02 or abs(a["slope_dominance_vs_log2imposed"]) > 0.005
    zc = a["normalization"]["d10_i1"]
    zscore_suppresses = abs(zc["zscore_dominance"] - base_dom) < 0.5 * dom_shift_d10 + 1e-9
    if a["n_diverged"] > 0.2 * len(fin):
        a["verdict"] = "TECHNICAL FAILURE (excess divergence)"
    elif strong and zscore_suppresses:
        a["verdict"] = "REPLICATION SUPPORT"
    elif partial:
        a["verdict"] = "PARTIAL REPLICATION"
    else:
        a["verdict"] = "NON-REPLICATION"
    a["zscore_suppresses_depth10"] = bool(zscore_suppresses)
    json.dump(a, open(out / "utd_train_scale_analysis.json", "w"), indent=2, default=float)

    print("\n=== UTD-MHAD baseline utility ===")
    print(f"  depth-only={a['baseline_depth_only_acc']:.3f} inertial-only={a['baseline_inertial_only_acc']:.3f} "
          f"multimodal={a['baseline_multimodal_acc']:.3f} gain={a['baseline_gain']:+.3f} "
          f"-> more informative = {a['more_informative_modality']}")
    print("=== learned dominance by condition (none) ===")
    for (aa, av) in CONDS_NONE:
        d = a[f"d{aa:g}_i{av:g}"]
        print(f"  d{aa:g}/i{av:g}: dominance={d['dominance']:+.3f} gain={d['gain']:+.3f} "
              f"rho_fus={d['rho_fusion']:.2f} rho_enc={d['rho_enc']:.2f} comp={d['compensation']}")
    print(f"slope={a['slope_dominance_vs_log2imposed']:.4f} spearman={a['spearman_imposed_vs_dominance']:.3f}")
    print("normalization:", a["normalization"])
    print("VERDICT:", a["verdict"])
    _figures(tj, ag, none, a, out)


def _figures(tj, ag, none, a, out):
    fig_dir = out / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    COND = [(1.0, 1.0, "baseline", "#000000"), (5.0, 1.0, "depth x5", "#ff7f0e"),
            (10.0, 1.0, "depth x10", "#d62728"), (1.0, 5.0, "inertial x5", "#2ca02c"),
            (1.0, 10.0, "inertial x10", "#1f77b4")]
    tn = tj[tj.norm == "none"]

    def mtraj(aa, av, key):
        g = tn[(tn.alpha_depth == aa) & (tn.alpha_inertial == av)]
        if not len(g):
            return None, None
        m = g.groupby("epoch")[key].mean(); return m.index.values, m.values

    # dynamics
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for aa, av, lab, c in COND:
        for j, key in enumerate(["rho_fusion", "rho_enc", "grad_ratio"]):
            e, y = mtraj(aa, av, key)
            if e is not None:
                ax[j].plot(e, y, color=c, label=lab, lw=1.7)
    ax[0].set_title(r"$\rho_{fusion}$(t) post-scale"); ax[1].set_title(r"$\rho_{enc}$(t) learned (compensation)")
    ax[2].set_title("encoder grad ratio depth/inertial (t)")
    for a_ in ax:
        a_.set_yscale("log"); a_.axhline(1, color="0.6", ls=":"); a_.set_xlabel("epoch"); a_.grid(alpha=0.25); a_.legend(fontsize=7)
    fig.suptitle("UTD-MHAD training-time scale: representation & gradient dynamics (none, mean 5 seeds)")
    fig.tight_layout(); fig.savefig(fig_dir / "utd_TS_dynamics.png", dpi=150); plt.close(fig)

    # contribution + gain vs imposed
    an = ag[ag.norm == "none"].copy(); an["l2"] = np.log2(an.alpha_depth / an.alpha_inertial); an = an.sort_values("l2")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].errorbar(an.l2, an.dominance_depth_minus_inertial_mean, yerr=an.dominance_depth_minus_inertial_ci95, marker="o", capsize=3, color="#d62728")
    for _, r in none.iterrows():
        ax[0].scatter(r.log2_imposed, r.dominance_depth_minus_inertial, s=12, color="0.6", alpha=0.6, zorder=0)
    ax[0].axhline(0, color="k", lw=0.6); ax[0].set_ylabel("dominance Acc_depth-Acc_inertial")
    ax[0].set_title("learned modality dominance vs imposed scale")
    ax[1].errorbar(an.l2, an.multimodal_gain_mean, yerr=an.multimodal_gain_ci95, marker="o", capsize=3, color="#ff7f0e")
    ax[1].set_ylabel("multimodal gain"); ax[1].set_title("multimodal gain vs imposed scale")
    for a_ in ax:
        a_.set_xlabel(r"imposed $\log_2(\alpha_{depth}/\alpha_{inertial})$"); a_.grid(alpha=0.25); a_.axvline(0, color="0.6", ls=":")
    fig.suptitle("UTD-MHAD learned contribution (test, 5 seeds)")
    fig.tight_layout(); fig.savefig(fig_dir / "utd_TS_contribution.png", dpi=150); plt.close(fig)

    # compensation
    an2 = ag[ag.norm == "none"].copy(); an2["imp"] = an2.alpha_depth / an2.alpha_inertial; an2 = an2.sort_values("imp")
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.plot(np.log2(an2.imp), np.log2(an2.imp), "k--", label="no compensation")
    ax.plot(np.log2(an2.imp), np.log2(an2.rho_enc_mean), "o-", color="#2ca02c", label=r"learned $\log_2\rho_{enc}$")
    ax.plot(np.log2(an2.imp), np.log2(an2.rho_fusion_mean), "s-", color="#d62728", label=r"$\log_2\rho_{fusion}$ post-scale")
    ax.axhline(0, color="0.6", ls=":"); ax.set_xlabel(r"imposed $\log_2(\alpha_{depth}/\alpha_{inertial})$")
    ax.set_ylabel(r"$\log_2$ ratio"); ax.set_title("UTD-MHAD compensation: do encoders counteract imposed scale?")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(fig_dir / "utd_TS_compensation.png", dpi=150); plt.close(fig)

    # normalization
    fig, ax = plt.subplots(figsize=(7, 4.4))
    conds = ["d1_i1", "d10_i1", "d1_i10"]; labels = ["baseline", "depth x10", "inertial x10"]
    xn = np.arange(len(conds))
    none_dom = [a["normalization"].get(c, {}).get("none_dominance", a["baseline_dominance"] if c == "d1_i1" else np.nan) for c in conds]
    zs_dom = [a["baseline_dominance"] if c == "d1_i1" else a["normalization"].get(c, {}).get("zscore_dominance", np.nan) for c in conds]
    none_dom[0] = a["baseline_dominance"]
    ax.bar(xn - 0.2, none_dom, 0.4, label="none", color="#d62728")
    ax.bar(xn + 0.2, zs_dom, 0.4, label="z-score", color="#1f77b4")
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(xn); ax.set_xticklabels(labels)
    ax.set_ylabel("dominance Acc_depth-Acc_inertial"); ax.set_title("UTD-MHAD: z-score suppresses the scale pathway?")
    ax.legend(); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(fig_dir / "utd_TS_normalization.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
