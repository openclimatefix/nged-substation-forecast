# ruff: noqa: E501
"""Score every arm on cold nights and the other cuts, with month-resampled intervals.

The score is the quantile score: twice the mean pinball loss over 11 quantiles, in points of
effective capacity. A contrast is the arm's total minus the reference's total over a subset, as a
percentage of the reference's total, with the period's months resampled as whole blocks (paired).
Cold night: ensemble-mean forecast temperature under 2 degrees C, local 17:00 to 08:00.

    uv run --no-sync python ml-experiments/00192-*/e00192_score.py
"""

import json
from typing import Final

import e00192_arms as arms_module
import e00192_common as common
import e00192_data as data
import e00192_score_lib as lib
import numpy as np
import polars as pl

DECISION: Final[tuple[str, ...]] = common.DECISION_SITES
SCORE_DIR = common.WORK_DIR / "report"
HEADLINE_LEVEL: Final[float] = 97.5
"""Two planned contrasts (T and ST against A), each read at 100 - 5 / 2 per cent."""
NON_HEADLINE_LEVEL: Final[float] = 95.0
LEAD_CUTS: Final[dict[str, tuple[float, float]]] = {
    "under 24 h": (0.0, 24.0),
    "24 to 72 h": (24.0, 72.0),
    "72 to 168 h": (72.0, 168.0),
}


def subsets(*, frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return the named boolean row masks of every scored subset (decision substations, usable rows)."""
    usable = data.usable_rows(frame=frame) & frame["site"].is_in(list(DECISION)).to_numpy()
    cold = frame["cold_night"].to_numpy() & usable
    evening = frame["evening_to_morning"].to_numpy() & usable
    t_mean = frame["t_mean"].to_numpy()
    lead = frame["lead_hours"].to_numpy()
    site = frame["site"].to_numpy()
    kind = frame["kind"].to_numpy()
    local = frame["valid_time"].dt.convert_time_zone("Europe/London")
    winter_evening = (
        usable
        & local.dt.month().is_in([11, 12, 1, 2]).to_numpy()
        & ((frame["local_hour"].to_numpy() >= 17) & (frame["local_hour"].to_numpy() < 23))
    )
    coldest_edge = float(np.nanpercentile(t_mean[evening], 5.0))
    out: dict[str, np.ndarray] = {
        "cold nights": cold,
        "coldest 5% of evening-to-morning rows": evening & (t_mean <= coldest_edge),
        "all rows": usable,
        "non-cold rows": usable & ~cold,
        "daylight": usable & (kind == "daylight"),
        "night (sun below horizon)": usable & (kind == "night"),
        "winter evenings (Nov to Feb, 17:00 to 23:00)": winter_evening,
        "evening-to-morning, 2 to 6 C": evening & (t_mean >= 2.0) & (t_mean < 6.0),
    }
    for label, (low, high) in LEAD_CUTS.items():
        out[f"cold nights, {label}"] = cold & (lead >= low) & (lead < high)
    for name in DECISION:
        out[f"cold nights, {name}"] = cold & (site == name)
    out["_coldest_edge"] = np.array([coldest_edge])
    return out


def score_period(*, period: str) -> dict:
    """Return a period's tables: per subset and arm, scores, contrasts, and calibration."""
    frame = pl.read_parquet(arms_module.ARM_DIR / f"{period}_frame.parquet")
    arms = {
        arm: np.load(arms_module.ARM_DIR / f"{period}_{arm}.npy").astype(np.float64)
        for arm in arms_module.ARMS
    }
    metrics = {arm: lib.row_metrics(frame=frame, quantiles=q) for arm, q in arms.items()}
    month = frame["month"].to_numpy()
    masks = subsets(frame=frame)
    out: dict = {"n_rows": frame.height, "coldest_edge_c": float(masks.pop("_coldest_edge")[0])}
    out["subsets"] = {}
    for name, mask in masks.items():
        if mask.sum() == 0:
            continue
        entry: dict = {"rows": int(mask.sum())}
        for arm in arms_module.ARMS:
            m = {k: v[mask] for k, v in metrics[arm].items()}
            row: dict = {k: float(v.mean()) for k, v in m.items()}
            if arm != "a":
                level = HEADLINE_LEVEL if arm in ("t", "st") else NON_HEADLINE_LEVEL
                row["change_pct"] = lib.percent_change(
                    month=month[mask],
                    arm=m["qs"],
                    reference=metrics["a"]["qs"][mask],
                    level=level,
                    seed=192,
                )
                if arm in ("t", "st"):
                    row["change_vs_mr_pct"] = lib.percent_change(
                        month=month[mask],
                        arm=m["qs"],
                        reference=metrics["mr"]["qs"][mask],
                        level=NON_HEADLINE_LEVEL,
                        seed=192,
                    )
            entry[arm] = row
        out["subsets"][name] = entry
    return out


def main() -> None:
    """Score every period and write `report.json`."""
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    report = {period: score_period(period=period) for period in ("tuning", "replicate", "scored")}
    (SCORE_DIR / "report.json").write_text(json.dumps(report, indent=1, default=float))
    for period in ("replicate", "scored"):
        print(period, "coldest 5% edge", report[period]["coldest_edge_c"])
        for name in (
            "cold nights",
            "coldest 5% of evening-to-morning rows",
            "non-cold rows",
            "all rows",
            "daylight",
            "night (sun below horizon)",
        ):
            entry = report[period]["subsets"][name]
            line = [
                f"{name}: n={entry['rows']}, A qs={entry['a']['qs']:.4f} in95={entry['a']['inside90']:.3f} b95={entry['a']['below95']:.3f}"
            ]
            for arm in ("t", "st", "m", "mr", "tp", "tg", "stg"):
                c = entry[arm]["change_pct"]
                line.append(
                    f"{arm} {c['point']:+.2f}% [{c['lower']:+.2f},{c['upper']:+.2f}] b95={entry[arm]['below95']:.3f}"
                )
            print("  ", "; ".join(line))


if __name__ == "__main__":
    main()
