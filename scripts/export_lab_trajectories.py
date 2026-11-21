"""Per-epoch training curves for the Experiment Lab, read from the frozen result artifacts.

Every value is a logged measurement from the forward-vs-gradient decomposition runs
(sum fusion, 5 seeds, evaluated on the VALIDATION split after every epoch):

  synthetic · Adam : results/disentangle/disentangle_trajectories.csv         (alpha 1, 10, 100; 40 epochs)
  synthetic · SGD  : results/sgd_disentangle/sgd_disentangle_trajectories.csv (alpha 1, 10, 100; 40 epochs)
  CREMA-D   · Adam : results/cremad_disentangle/cremad_disentangle_trajectories.csv (alpha 10, 100; 30 epochs)

Metrics kept: val_dominance (masked-accuracy dominance D = acc_A - acc_B) and val_gain
(multimodal gain). Per-seed series are stored so the Lab can show the selected seed;
means/SDs are computed in the browser from these exact values. Nothing is smoothed,
interpolated or modelled. Writes website/src/data/trajectories.generated.json.
Usage: python scripts/export_lab_trajectories.py [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "website" / "src" / "data" / "trajectories.generated.json"
SOURCES = {
    "synthetic_adam": "results/disentangle/disentangle_trajectories.csv",
    "synthetic_sgd": "results/sgd_disentangle/sgd_disentangle_trajectories.csv",
    "cremad_adam": "results/cremad_disentangle/cremad_disentangle_trajectories.csv",
}
METRICS = ["val_dominance", "val_gain"]
ARMS = ["baseline", "forward", "gradient", "ordinary"]


def build() -> dict:
    series: dict = {}
    for key, rel in SOURCES.items():
        df = pd.read_csv(ROOT / rel)
        seeds = sorted(int(s) for s in df["seed"].unique())
        epochs = int(df["epoch"].max())
        alphas = sorted(int(a) for a in df["alpha"].unique())
        data: dict = {}
        for a in alphas:
            data[str(a)] = {}
            for arm in ARMS:
                sub = df[(df["alpha"] == a) & (df["arm"] == arm)]
                if sub.empty:
                    continue
                data[str(a)][arm] = {}
                for m in METRICS:
                    per_seed = []
                    for s in seeds:
                        run = sub[sub["seed"] == s].sort_values("epoch")
                        # a diverged run stops early; keep exactly what was logged (shorter list)
                        per_seed.append([round(float(v), 4) for v in run[m].tolist()])
                    data[str(a)][arm][m] = per_seed
        series[key] = {"source": rel, "epochs": epochs, "alphas": alphas, "seeds": seeds, "data": data}
    return {
        "generated_by": "scripts/export_lab_trajectories.py",
        "note": "Logged per-epoch validation metrics of the decomposition runs (sum fusion). Not smoothed or modelled.",
        "metrics": METRICS,
        "series": series,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the committed JSON is stale")
    args = ap.parse_args()
    text = json.dumps(build(), separators=(",", ":"), ensure_ascii=False) + "\n"
    if args.check:
        cur = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if cur != text:
            print(f"STALE: {OUT.relative_to(ROOT)} differs from the result artifacts; re-run this script.")
            sys.exit(1)
        print("trajectories.generated.json matches the frozen artifacts")
        return
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(text) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
