r"""Generate the two paper figures (vector PDF, colorblind-safe).

Output figures (written to research_paper/figures/, used by research_paper/main.tex):
  * research_paper/figures/fig_synth.pdf  -> Fig. 2 (\includegraphics{fig_synth.pdf})
  * research_paper/figures/fig_real.pdf   -> Fig. 3 (\includegraphics{fig_real.pdf})
(main.tex sets \graphicspath{{figures/}}, so it references them by basename.)

Input data (read-only):
  * results/phase2_mechanism/phase2_primary_seeds10.csv, phase2_collapse_table.csv   (Fig. 2)
  * results/phase2_mechanism/phase2_cremad_probe.csv, results/phase3_cremad/phase3_controls.csv  (Fig. 3)

Run from the repo root:  python research_paper/scripts/make_paper_figs.py
"""
import pandas as pd, numpy as np, math, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
FIG = os.path.join(R, "research_paper/figures"); os.makedirs(FIG, exist_ok=True)
# colorblind-safe (Okabe-Ito)
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#444444"


def paired(a, b):
    dd = np.asarray(b, float) - np.asarray(a, float); sd = dd.std(ddof=1)
    return dd.mean(), 1.96 * sd / math.sqrt(len(dd))


# ---------- FIG 1: synthetic (collapse + decoupled agreement) ----------
sw = pd.read_csv(os.path.join(R, "results/phase2_mechanism/phase2_primary_seeds10.csv"))
sw = sw[~sw.diverged.astype(bool)]
ct = pd.read_csv(os.path.join(R, "results/phase2_mechanism/phase2_collapse_table.csv"))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
MARK = {"forward": "o", "gradient": "s", "ordinary": "^", "lrmatch": "x"}
for opt, col in [("adam", BLUE), ("sgd", ORANGE)]:
    for arm in ["forward", "gradient", "ordinary"]:
        g = ct[(ct.opt == opt) & (ct.arm == arm)]
        a1.plot(g.m, g.dom_shift, MARK[arm], color=col, ms=5, ls="-", lw=0.9,
                label=f"{opt} {arm}")
a1.set_xscale("log"); a1.axhline(0, color="k", lw=0.5)
a1.set_xlabel(r"predicted multiplier $m$"); a1.set_ylabel(r"dominance shift $\Delta\mathcal{D}$")
a1.set_title("(a) multiplier organizes the effect", fontsize=8)
a1.legend(fontsize=5.3, ncol=2, loc="lower left", framealpha=0.9); a1.grid(alpha=.25)
# (b) decoupled metrics agree (synthetic Adam forward alpha=100)
def base(col): return sw[(sw.optimizer == "adam") & (sw.arm == "baseline") & (sw.alpha == 1.0)].sort_values("seed")[col].values
def arm(col): return sw[(sw.optimizer == "adam") & (sw.arm == "forward") & (sw.alpha == 100.0)].sort_values("seed")[col].values
labs = ["masked\ndom.", "mean-abl\ndom.", "Shapley$_A$", "probe$_A$"]
cols = ["dominance", "meanabl_dominance", "shapley_a", "probe_a"]
vals = [paired(base(c), arm(c)) for c in cols]
x = np.arange(len(labs))
a2.bar(x, [v[0] for v in vals], yerr=[v[1] for v in vals], capsize=3,
       color=[GREY, GREEN, GREEN, BLUE], width=0.6)
a2.axhline(0, color="k", lw=0.5); a2.set_xticks(x); a2.set_xticklabels(labs, fontsize=6.5)
a2.set_ylabel("shift vs baseline"); a2.set_title("(b) synthetic: all metrics agree", fontsize=8)
a2.grid(alpha=.25, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_synth.pdf")); plt.close(fig)
print("saved ->", os.path.join(FIG, "fig_synth.pdf"), " (Fig. 2)")
print("   decoupled shifts:", [f"{c}:{v[0]:+.3f}" for c, v in zip(cols, vals)])

# ---------- FIG 2: real data (masked misleads + positive control) ----------
cre = pd.read_csv(os.path.join(R, "results/phase2_mechanism/phase2_cremad_probe.csv"))
cre = cre[~cre.diverged.astype(bool)]
ctl = pd.read_csv(os.path.join(R, "results/phase3_cremad/phase3_controls.csv"))
ctl = ctl[~ctl.get("diverged", False).astype(bool)]
fig, (b1, b2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
def cb(col): return cre[cre.arm == "baseline"].sort_values("seed")[col].values
def cf(col): return cre[cre.arm == "forward"].sort_values("seed")[col].values
vals = [paired(cb(c), cf(c)) for c in ["dominance", "meanabl_dominance", "shapley_audio", "probe_audio"]]
x = np.arange(4)
b1.bar(x, [v[0] for v in vals], yerr=[v[1] for v in vals], capsize=3,
       color=[GREY, GREEN, GREEN, BLUE], width=0.6)
b1.axhline(0, color="k", lw=0.5); b1.set_xticks(x)
b1.set_xticklabels(["masked\ndom.", "mean-abl\ndom.", "Shapley", "probe"], fontsize=6.5)
b1.set_ylabel("shift vs baseline"); b1.set_title("(a) CREMA-D: masked misleads", fontsize=8)
b1.grid(alpha=.25, axis="y")
for i, v in enumerate(vals):
    b1.text(i, v[0] + (0.012 if v[0] >= 0 else -0.03), f"{v[0]:+.2f}", ha="center", fontsize=6)
# (b) positive control
n = ctl[ctl.control == "noise"]
lv = sorted(n.level.unique())
for col, c, m in [("probe_audio", BLUE, "o"), ("shapley_audio", GREEN, "s")]:
    mu = [n[n.level == l][col].mean() for l in lv]
    b2.plot(lv, mu, m + "-", color=c, ms=5, label=col.replace("_audio", "$_{audio}$"))
b2.set_xlabel("audio input noise level"); b2.set_ylabel("metric value")
b2.set_title("(b) positive control: metrics detect\nreal audio degradation", fontsize=8)
b2.legend(fontsize=6.5); b2.grid(alpha=.25)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_real.pdf")); plt.close(fig)
print("saved ->", os.path.join(FIG, "fig_real.pdf"), " (Fig. 3)")
print("   CREMA shifts:", [f"{v[0]:+.3f}" for v in vals])
