#!/usr/bin/env python3
"""Generate IEEE-column publication figures for the MLSP paper.
All plotted values are read from results/**/*.json (no hand-typed data).
Outputs vector PDFs into research_paper/figures/.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# resolve paths relative to this file: research_paper/scripts/ -> project root
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))          # project root
RES = os.path.join(ROOT, "results")
OUT = os.path.join(os.path.dirname(_HERE), "figures")   # research_paper/figures
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.3,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

COL = 3.45   # single column width (in)
SPAN = 7.10  # double column width (in)

def load(p):
    with open(os.path.join(RES, p), "r", encoding="utf-8") as f:
        return json.load(f)

C_FWD = "#1f77b4"; C_GRAD = "#d62728"; C_ORD = "#2ca02c"; C_BASE = "#7f7f7f"
C_A = "#1f77b4"; C_B = "#ff7f0e"

# ---------------------------------------------------------------- Fig: phase
def fig_phase():
    d = load("e4/e4_extreme_analysis.json")["dose_response"]
    mags = [20, 100, 400]
    def sev(k): return [d[k]["severity_by_mag"][str(m)] for m in mags]
    fig, ax = plt.subplots(figsize=(COL, 2.35))
    ax.plot(mags, sev("scale_only_none"), "o-", color=C_FWD, label="feature scale only")
    ax.plot(mags, sev("lr_only_none"), "s-", color=C_GRAD, label="learning-rate only")
    ax.plot(mags, sev("aligned_none"), "^-", color="#111111", label="scale+LR aligned")
    ax.set_xscale("log"); ax.set_xticks(mags); ax.set_xticklabels([f"{m}$\\times$" for m in mags])
    ax.set_xlabel("imposed imbalance magnitude")
    ax.set_ylabel("collapse severity $S$")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.legend(loc="center left", frameon=False, handlelength=1.6)
    fig.savefig(os.path.join(OUT, "fig_phase.pdf")); plt.close(fig)
    print("phase:", {k: round(v,3) for k,v in zip(["scale20","scale400","lr400","al400"],
          [sev("scale_only_none")[0], sev("scale_only_none")[2], sev("lr_only_none")[2], sev("aligned_none")[2]])})

# ------------------------------------------------ Fig: disentanglement (span)
def fig_disentangle():
    syn = load("disentangle/disentangle_analysis.json")["by_alpha"]["100.0"]
    cre = load("cremad_disentangle/cremad_disentangle_analysis.json")["by_alpha"]["100.0"]
    sgd = load("sgd_disentangle/sgd_disentangle_analysis.json")["by_alpha"]["100.0"]
    fus = load("fusion_robustness/fusion_robustness_analysis.json")["by_fusion"]["sum"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(SPAN, 2.7))
    # panel a: signed dominance shift by arm across settings
    settings = ["Adam\n(synthetic)", "Adam\n(CREMA-D)", "SGD\n(synthetic)"]
    fwd = [syn["forward"]["shift_vs_baseline"], cre["forward"]["shift_vs_baseline"], sgd["forward"]["shift_vs_baseline"]]
    grd = [syn["gradient"]["shift_vs_baseline"], cre["gradient"]["shift_vs_baseline"], sgd["gradient"]["shift_vs_baseline"]]
    ordv = [syn["ordinary"]["shift_vs_baseline"], cre["ordinary"]["shift_vs_baseline"], sgd["ordinary"]["shift_vs_baseline"]]
    x = np.arange(len(settings)); w = 0.26
    ax1.axhline(0, color="k", lw=0.6)
    ax1.bar(x-w, fwd, w, color=C_FWD, label="forward-only")
    ax1.bar(x,   grd, w, color=C_GRAD, label="gradient-only")
    ax1.bar(x+w, ordv, w, color=C_ORD, label="ordinary ($\\alpha h$)")
    ax1.set_xticks(x); ax1.set_xticklabels(settings)
    ax1.set_ylabel("dominance shift $\\Delta$ vs baseline")
    ax1.set_title("(a) pathway decomposition ($\\alpha{=}100$)")
    ax1.grid(True, axis="y", alpha=0.3, lw=0.5)
    ax1.legend(frameon=False, handlelength=1.4, loc="lower left")
    for xi, g in zip(x, grd):
        ax1.annotate(f"{g:+.3f}", (xi, g), textcoords="offset points", xytext=(0, 3 if g>=0 else -8),
                     ha="center", fontsize=6, color=C_GRAD)

    # panel b: gradient-only arm -- gradient ratio vs update ratio
    labels = ["Adam\n(synthetic)", "Adam\n(CREMA-D)", "SGD\n(synthetic)"]
    grad_r = [fus["gradient"]["grad_ratio"], cre["gradient"]["grad_ratio"], sgd["gradient"]["grad_ratio"]]
    upd_r  = [fus["gradient"]["update_ratio"], cre["gradient"]["update_ratio"], sgd["gradient"]["update_ratio"]]
    x2 = np.arange(len(labels))
    ax2.bar(x2-0.2, grad_r, 0.4, color="#9467bd", label="gradient-norm ratio")
    ax2.bar(x2+0.2, upd_r, 0.4, color="#e377c2", label="param-update ratio")
    ax2.set_yscale("log"); ax2.set_xticks(x2); ax2.set_xticklabels(labels)
    ax2.set_ylabel("ratio A:B  (log scale)")
    ax2.set_title("(b) gradient-only arm: gradient vs update ratio")
    ax2.axhline(1.0, color="k", lw=0.6, ls=":")
    ax2.grid(True, axis="y", alpha=0.3, lw=0.5, which="both")
    ax2.legend(frameon=False, handlelength=1.4, loc="upper left")
    for xi, v in zip(x2, upd_r):
        ax2.annotate(f"{v:.2g}", (xi+0.2, v), textcoords="offset points", xytext=(0, 2), ha="center", fontsize=6)
    fig.savefig(os.path.join(OUT, "fig_disentangle.pdf")); plt.close(fig)
    print("disentangle grad shifts:", [round(g,4) for g in grd], "upd_r:", [round(u,2) for u in upd_r])

# ---------------------------------------------------------------- Fig: CREMA-D
def fig_cremad():
    a = load("cremad/cremad_train_scale_analysis.json")
    order = ["a1_v10", "a1_v5", "a1_v1", "a5_v1", "a10_v1"]
    labels = ["video\n$\\times$10", "video\n$\\times$5", "base", "audio\n$\\times$5", "audio\n$\\times$10"]
    dom = [a[k]["dominance"] for k in order]
    gain = [a[k]["gain"] for k in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    ax.axhline(0, color="k", lw=0.6)
    ax.bar(x, dom, 0.6, color=C_A, label="dominance (audio$-$video)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("dominance", color=C_A)
    ax.tick_params(axis="y", labelcolor=C_A)
    ax.set_ylim(-0.05, 0.38)
    ax2 = ax.twinx()
    ax2.plot(x, gain, "o--", color=C_B, label="multimodal gain")
    ax2.set_ylabel("multimodal gain", color=C_B)
    ax2.tick_params(axis="y", labelcolor=C_B)
    ax2.set_ylim(-0.02, 0.32)
    ax.set_title("CREMA-D training-time scaling")
    fig.savefig(os.path.join(OUT, "fig_cremad.pdf")); plt.close(fig)
    print("cremad dom:", [round(v,3) for v in dom], "spearman", a["spearman_imposed_vs_dominance"])

# ---------------------------------------------------------------- Fig: fusion
def fig_fusion():
    f = load("fusion_robustness/fusion_robustness_analysis.json")["by_fusion"]
    fusions = ["sum", "concat", "gated"]
    fwd_d = [f[k]["cmp"]["forward_vs_baseline"]["cohen_d"] for k in fusions]
    grd_d = [f[k]["cmp"]["gradient_vs_baseline"]["cohen_d"] for k in fusions]
    fwd_p = [f[k]["cmp"]["forward_vs_baseline"]["p"] for k in fusions]
    x = np.arange(len(fusions)); w = 0.36
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    ax.axhline(0, color="k", lw=0.6)
    b1 = ax.bar(x-w/2, fwd_d, w, color=C_FWD, label="forward-only")
    ax.bar(x+w/2, grd_d, w, color=C_GRAD, label="gradient-only")
    ax.set_xticks(x); ax.set_xticklabels([s + " fusion" for s in fusions])
    ax.set_ylabel("effect size (Cohen's $d$)")
    ax.set_title("Fusion-mechanism robustness ($\\alpha{=}100$)")
    ax.grid(True, axis="y", alpha=0.3, lw=0.5)
    ax.legend(frameon=False, handlelength=1.4, loc="lower left")
    for xi, dval, pval in zip(x, fwd_d, fwd_p):
        tag = "n.s." if pval > 0.05 else ("$d=" + f"{dval:.1f}$")
        ax.annotate(tag, (xi-w/2, dval), textcoords="offset points", xytext=(0, -9 if dval<0 else 3),
                    ha="center", fontsize=6, color=C_FWD)
    fig.savefig(os.path.join(OUT, "fig_fusion.pdf")); plt.close(fig)
    print("fusion fwd_d:", [round(v,2) for v in fwd_d], "p:", [round(p,3) for p in fwd_p])

# ---------------------------------------------------------------- Fig: UTD
def fig_utd():
    u = load("utd_mhad/utd_train_scale_analysis.json")
    order = ["d1_i10", "d1_i5", "d1_i1", "d5_i1", "d10_i1"]
    labels = ["inert.\n$\\times$10", "inert.\n$\\times$5", "base", "depth\n$\\times$5", "depth\n$\\times$10"]
    dom = [u[k]["dominance"] for k in order]
    gain = [u[k]["gain"] for k in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    ax.axhline(0, color="k", lw=0.6)
    ax.bar(x, dom, 0.6, color=C_A, label="dominance (depth$-$inertial)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("dominance", color=C_A)
    ax.tick_params(axis="y", labelcolor=C_A)
    ax.set_ylim(-0.7, 0.05)
    ax2 = ax.twinx()
    ax2.plot(x, gain, "o--", color=C_B, label="multimodal gain")
    ax2.set_ylabel("multimodal gain", color=C_B)
    ax2.tick_params(axis="y", labelcolor=C_B)
    ax2.set_ylim(0.0, 0.45)
    ax.set_title("UTD-MHAD utility-asymmetric scaling")
    fig.savefig(os.path.join(OUT, "fig_utd.pdf")); plt.close(fig)
    print("utd dom:", [round(v,3) for v in dom])

if __name__ == "__main__":
    # Figures included in the 6-page manuscript (Fig. 1 is inline TikZ):
    fig_disentangle(); fig_fusion(); fig_utd()
    # fig_phase() and fig_cremad() remain available (data-sourced) but are not
    # included in the 6-page version; their results are reported in text/tables.
    print("ALL FIGURES WRITTEN to", OUT)
    for fn in sorted(os.listdir(OUT)):
        print("  ", fn, os.path.getsize(os.path.join(OUT, fn)), "bytes")
