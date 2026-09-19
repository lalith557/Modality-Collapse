"""Compensation metrics (shared). Quantifies how an injected data-scale ratio is
absorbed into the fusion-boundary rho over training. Consistent with the E3
analysis (which keeps its own inline copy; this module is used by E4 onward)."""
from __future__ import annotations

import math


def logrho_by_epoch(history: list[dict]) -> dict[int, float]:
    return {int(r["epoch"]): math.log(r["rho"]) for r in history}


def compensation_from_history(history: list[dict], log_scale: float, epochs: int) -> dict:
    """log_scale = log(alpha_A/alpha_V). Not applicable when log_scale ~ 0."""
    lr = logrho_by_epoch(history)
    initial = lr.get(0, lr[min(lr)])
    final = lr[max(lr)]
    out = {
        "logrho_init": initial,
        "logrho_ep5": lr.get(5, float("nan")),
        "logrho_ep10": lr.get(10, float("nan")),
        "logrho_final": final,
        "residual_abs_logrho": abs(final),
    }
    if abs(log_scale) < 1e-9:
        out.update(comp_applicable=False, comp_ratio_final=float("nan"),
                   compensation_achieved=float("nan"), time_to_compensation=float("nan"))
        return out
    ttc = next((e for e in sorted(lr) if abs(lr[e]) < 0.5 * abs(log_scale)), epochs + 1)
    out.update(
        comp_applicable=True,
        comp_ratio_init=initial / log_scale,
        comp_ratio_final=final / log_scale,           # ~0 = full compensation, ~1 = none
        compensation_achieved=1.0 - abs(final) / abs(log_scale),
        time_to_compensation=ttc,
    )
    return out
