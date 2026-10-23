# ruff: noqa: E501
"""The real record's logged load shifts: LOGGED (logged onset, end known at once) against K0.

Descriptive only. Experiment 208 stored the scores of 8 arms on the rows of every logged shift at
four substations. Each shift's phase, as experiment 208 fixed it, is BEFORE_ALARM, DURING, AFTER
(target within 96 half-hours of the end) or TAIL. The target is observed power, because the real
record has no separate normal-arrangement series. The scores hold five stored quantiles, so twCRPS
is replaced by the pinball loss at the 0.95 and 0.99 levels. Intervals resample whole logged shifts.

    uv run --no-sync python e00283_real.py
"""

from __future__ import annotations

import json
from typing import Final

import e00283_report as rep
import numpy as np
import polars as pl
from e00283_common import WORK_DIR

SRC: Final = WORK_DIR.parent / "00208-switching-value-of-a-detector-by-phase-and-quality" / "real"
SITES: Final[tuple[str, ...]] = ("D5", "D10", "D14", "D16")


def pinball(*, actual: np.ndarray, q: np.ndarray, level: float) -> np.ndarray:
    """Pinball loss at one level, per row."""
    error = actual - q
    return np.maximum(level * error, (level - 1.0) * error)


def main() -> None:
    """Print and store the real-record table."""
    frames = []
    for site in SITES:
        scores = pl.read_parquet(SRC / f"event_scores_{site}.parquet").filter(
            pl.col("arm").is_in(["K0", "LOGGED"])
        )
        rows = pl.read_parquet(SRC / f"event_rows_{site}.parquet")
        wide = scores.pivot(
            on="arm",
            index=["row", "actual", "lead_hours"],
            values=["qs", "q50", "q95", "q99", "mean_forecast"],
            separator="__",
        )
        frames.append(
            rows.join(wide, on="row").with_columns(
                site=pl.lit(site), unit=pl.lit(site) + pl.col("card")
            )
        )
    df = pl.concat(frames)
    actual = df["actual"].to_numpy()
    metrics = {}
    for arm in ("K0", "LOGGED"):
        metrics[arm] = {
            "qs": df[f"qs__{arm}"].to_numpy(),
            "ae_median": np.abs(df[f"q50__{arm}"].to_numpy() - actual),
            "ae_mean": np.abs(df[f"mean_forecast__{arm}"].to_numpy() - actual),
            "pin95": pinball(actual=actual, q=df[f"q95__{arm}"].to_numpy(), level=0.95),
            "pin99": pinball(actual=actual, q=df[f"q99__{arm}"].to_numpy(), level=0.99),
        }
    units = np.unique(df["unit"].to_numpy(), return_inverse=True)[1] + 1
    phase = df["phase"].to_numpy()
    rng = np.random.default_rng(283)
    out = []
    groups = {
        "all shift rows": np.ones(len(phase), dtype=bool),
        **{p: phase == p for p in ("BEFORE_ALARM", "DURING", "AFTER", "TAIL")},
    }
    for name, mask in groups.items():
        for metric in metrics["K0"]:
            gain = metrics["K0"][metric] - metrics["LOGGED"][metric]
            res = rep.interval(
                gain=gain[mask],
                base=metrics["K0"][metric][mask],
                weight=np.ones(mask.sum()),
                units=units[mask],
                rng=rng,
            )
            out.append(
                {
                    "group": name,
                    "metric": metric,
                    "n_shifts": len(np.unique(units[mask])),
                    **res,
                }
            )
            print(
                f"{name:16s} {metric:10s} n={res['n']:.0f} shifts={len(np.unique(units[mask]))} mean {res['mean']:+.3f} ({res['mean_lo']:+.3f}, {res['mean_hi']:+.3f}) pct {res['pct']:+.1f} ({res['pct_lo']:+.1f}, {res['pct_hi']:+.1f})"
            )
    (rep.RESULTS / "real.json").write_text(json.dumps(out, indent=2, default=float))
    print("logged shifts:", len(np.unique(units)))


if __name__ == "__main__":
    rep.RESULTS.mkdir(parents=True, exist_ok=True)
    main()
