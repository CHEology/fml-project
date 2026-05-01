"""Salary band blending: combine candidate-conditioned neural prediction
with the role-anchored retrieval band into one quantile envelope."""

from __future__ import annotations


def blend_neural_with_retrieved(
    neural: dict[str, int], retrieved: dict[str, int]
) -> dict[str, int]:
    n_q50 = float(neural["q50"])
    r_q10, r_q50, r_q90 = (float(retrieved[k]) for k in ("q10", "q50", "q90"))
    # Hard sanity bounds: neural q50 cannot escape role band by more than 30%.
    new_q50 = max(r_q10 * 0.70, min(r_q90 * 1.30, n_q50))
    blended_q50 = 0.65 * new_q50 + 0.35 * r_q50
    # Scale neural's relative spread to match the role's q10..q90 width.
    spread_ratio = max(1.0, r_q90 - r_q10) / max(
        1.0, float(neural["q90"]) - float(neural["q10"])
    )
    keys = ("q10", "q25", "q50", "q75", "q90")
    out = {
        key: int(round(blended_q50 + (float(neural[key]) - n_q50) * spread_ratio, -3))
        for key in keys
    }
    last = -float("inf")
    for key in keys:
        if out[key] < last:
            out[key] = int(last)
        last = out[key]
    return out
