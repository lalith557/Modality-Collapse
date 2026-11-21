"""Read the FROZEN result artifacts and write the values the website's pathway pages show.

The site's hand-transcribed values (website/src/data/research.ts) are tested against
this file by website/src/data/evidence.test.ts, so a transcription error or a value
shown under the wrong dataset/optimizer label fails the test suite. Nothing is
recomputed from seeds here: values are read from the analysis JSONs written by the
original experiment scripts (paired t-tests, Cohen's d_z over the 5 seeds).

Writes website/src/data/evidence.verified.json.
Usage: python scripts/export_site_evidence.py [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "website" / "src" / "data" / "evidence.verified.json"
ALPHA = "100.0"


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def arm_cmp(c: dict) -> dict:
    return {"dDom": c["mean_diff"], "d": c["cohen_d"], "p": c["p"]}


def decomposition(rel: str, label: str) -> dict:
    a = load(rel)["by_alpha"][ALPHA]
    cfg_rel = rel.replace("_analysis.json", "_config.json")
    cfg = load(cfg_rel)
    return {
        "label": label, "source": rel, "seeds": len(cfg["seeds"]), "alpha": 100,
        "forward": arm_cmp(a["cmp"]["forward_vs_baseline"]),
        "gradient": arm_cmp(a["cmp"]["gradient_vs_baseline"]),
        "ordinary": arm_cmp(a["cmp"]["ordinary_vs_baseline"]),
    }


def ratios(rel: str, by: tuple[str, ...], label: str) -> dict:
    a = load(rel)
    for k in by:
        a = a[k]
    return {"label": label, "source": rel + " :: " + "/".join(by), "arm": "gradient-only", "alpha": 100,
            "rG": a["gradient"]["grad_ratio"], "rTheta": a["gradient"]["update_ratio"],
            "baselineRG": a["baseline"]["grad_ratio"], "baselineRTheta": a["baseline"]["update_ratio"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    fr_rel = "results/fusion_robustness/fusion_robustness_analysis.json"
    fr = load(fr_rel)["by_fusion"]
    fr_cfg = load("results/fusion_robustness/fusion_robustness_config.json")
    out = {
        "generated_by": "scripts/export_site_evidence.py",
        "note": "Read from frozen result artifacts; do not edit by hand.",
        "decomposition": {
            "adamSynthetic": decomposition("results/disentangle/disentangle_analysis.json", "Adam · synthetic"),
            "sgdSynthetic": decomposition("results/sgd_disentangle/sgd_disentangle_analysis.json", "SGD · synthetic"),
            "adamCremad": decomposition("results/cremad_disentangle/cremad_disentangle_analysis.json", "Adam · CREMA-D"),
        },
        "ratios": {
            # The 5-seed Adam decomposition run did not log the update ratio; the fusion-robustness
            # run repeats it exactly (identical seeds/config, identical dominance shifts) and does.
            "adamSynthetic": ratios(fr_rel, ("by_fusion", "sum"), "Adam · synthetic (sum fusion)"),
            "sgdSynthetic": ratios("results/sgd_disentangle/sgd_disentangle_analysis.json", ("by_alpha", ALPHA),
                                   "SGD · synthetic"),
            "adamCremad": ratios("results/cremad_disentangle/cremad_disentangle_analysis.json", ("by_alpha", ALPHA),
                                 "Adam · CREMA-D"),
        },
        "fusion": {
            "source": fr_rel, "optimizer": fr_cfg["optimizer"], "alpha": fr_cfg["alpha"],
            "seeds": len(fr_cfg["seeds"]), "dataset": "synthetic",
            "mechanisms": {
                k: {
                    "forward": arm_cmp(v["cmp"]["forward_vs_baseline"]),
                    "gradient": arm_cmp(v["cmp"]["gradient_vs_baseline"]),
                    "ordinaryShift": v["ordinary"]["shift_vs_baseline"],
                    "ordinaryVsForward": arm_cmp(v["cmp"]["ordinary_vs_forward"]),
                    "gradientArmRG": v["gradient"]["grad_ratio"],
                    "gradientArmRTheta": v["gradient"]["update_ratio"],
                    "gateValuesLogged": False,
                } for k, v in fr.items()
            },
        },
    }
    # sanity: the Adam ratio source run reproduces the 5-seed decomposition exactly
    dec = out["decomposition"]["adamSynthetic"]
    fsum = out["fusion"]["mechanisms"]["sum"]
    assert abs(dec["forward"]["dDom"] - fsum["forward"]["dDom"]) < 1e-12
    assert abs(dec["gradient"]["dDom"] - fsum["gradient"]["dDom"]) < 1e-12

    text = json.dumps(out, indent=1, ensure_ascii=False) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            sys.exit(f"{OUT} is stale: re-run scripts/export_site_evidence.py")
        print("evidence.verified.json is up to date")
        return
    OUT.write_text(text, encoding="utf-8")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
