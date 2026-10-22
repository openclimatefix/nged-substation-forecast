"""Experiment 185, Part 3 scoring: the hindsight covariate against the stored arm D.

Reads the replayed predictions of `e00185_hindsight.py` and experiment 155's stored arm D.
Scores the five decision substations over the scored year outside hand-filtered switching
events, in the D+1 band (lead 15 to 39 hours) and over leads 0.5 to 168 hours. The quantile score
is twice the mean pinball loss over the 11 quantiles, in points of effective capacity. Intervals
resample whole months.
"""

import json
from datetime import UTC, datetime
from typing import Final

import e00185_common as common
import e00185_days as days_module
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_month_totals
from studies.tail_calibration import tail_calibration_table

VARIANTS: Final[tuple[str, ...]] = ("control", "national_demand", "price")
BANDS: Final[dict[str, tuple[float, float]]] = {"d_plus_1": (15.0, 39.0), "all_leads": (0.5, 168.0)}
SCORED_FIRST: Final[datetime] = datetime(2025, 7, 2, tzinfo=UTC)
SCORED_LAST: Final[datetime] = datetime(2026, 6, 30, 23, 59, tzinfo=UTC)


def read(*, variant: str, band: tuple[float, float]) -> pl.DataFrame:
    """Return a variant's arm D rows for the decision substations in a lead band."""
    frames = [
        pl.read_parquet(common.WORK_DIR / "dp_preds" / f"{variant}_d_scored_{site}.parquet")
        .with_columns(site=pl.lit(site))
        .filter(
            pl.col("actual").is_finite(),
            pl.col("lead_hours") >= band[0],
            pl.col("lead_hours") < band[1],
            pl.col("valid_time") >= SCORED_FIRST,
            pl.col("valid_time") <= SCORED_LAST,
            ~pl.col("switching_mask"),
        )
        for site in common.DECISION_SITES
    ]
    return (
        pl.concat(frames)
        .with_columns(
            score=days_module.pinball_score_expr(),
            mean_abs_error=(pl.col("actual") - pl.col("mean")).abs(),
            median_minus_mean=pl.col("q50") - pl.col("mean"),
        )
        .sort(["site", "power_fcst_init_time", "valid_time"])
    )


def mean_interval(*, values: np.ndarray, months: np.ndarray) -> list[float]:
    """Return the mean and its 95% month-block interval."""
    labels, index = np.unique(months, return_inverse=True)
    sums = np.bincount(index, weights=values, minlength=len(labels))
    counts = np.bincount(index, minlength=len(labels)).astype(float)
    low, high = bootstrap_month_totals(month_sums=sums[None, :], month_counts=counts)
    return [float(values.mean()), low, high]


def main() -> None:
    """Score every variant, band, and substation, and write the report."""
    report: dict = {}
    for band_name, band in BANDS.items():
        frames = {v: read(variant=v, band=band) for v in VARIANTS}
        control = frames["control"]
        months = control["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
        entry: dict = {"n_rows": control.height}
        entry["absolute_score"] = {
            v: {
                "pooled": float(f["score"].mean()),  # ty: ignore[invalid-argument-type]
                "per_site": {
                    s: float(f.filter(pl.col("site") == s)["score"].mean())  # ty: ignore[invalid-argument-type]
                    for s in common.DECISION_SITES
                },
            }
            for v, f in frames.items()
        }
        for v in VARIANTS[1:]:
            f = frames[v]
            assert f.height == control.height
            for key in ("valid_time", "power_fcst_init_time", "time_series_id"):
                assert (f[key] == control[key]).all()
            diff = (f["score"] - control["score"]).to_numpy().astype(float)
            mae = (f["mean_abs_error"] - control["mean_abs_error"]).to_numpy().astype(float)
            per_site = {}
            for s in common.DECISION_SITES:
                rows = (control["site"] == s).to_numpy()
                per_site[s] = mean_interval(values=diff[rows], months=months[rows])
            leave_one_out = {}
            for s in common.DECISION_SITES:
                rows = (control["site"] != s).to_numpy()
                leave_one_out[s] = mean_interval(values=diff[rows], months=months[rows])
            entry[f"contrast_{v}_minus_control"] = {
                "leave_one_site_out_quantile_score": leave_one_out,
                "quantile_score": mean_interval(values=diff, months=months),
                "mean_forecast_abs_error": mean_interval(values=mae, months=months),
                "per_site_quantile_score": per_site,
            }
        entry["median_minus_mean_per_site"] = {
            v: {
                s: float(f.filter(pl.col("site") == s)["median_minus_mean"].mean())  # ty: ignore[invalid-argument-type]
                for s in common.DECISION_SITES
            }
            for v, f in frames.items()
        }
        entry["tail_calibration"] = {
            v: tail_calibration_table(frame=f).to_dicts() for v, f in frames.items()
        }
        report[band_name] = entry
    # Control against the stored arm D: must match to 1e-4.
    stored = pl.concat(
        [
            pl.read_parquet(common.PREDS_DIR / f"d_scored_{s}.parquet").with_columns(site=pl.lit(s))
            for s in common.DECISION_SITES
        ]
    )
    control = pl.concat(
        [
            pl.read_parquet(common.WORK_DIR / "dp_preds" / f"control_d_scored_{s}.parquet")
            for s in common.DECISION_SITES
        ]
    )
    joined = control.join(stored, on=["time_series_id", "run", "valid_time"], suffix="_stored")
    report["control_matches_stored_d"] = {
        "rows": joined.height,
        "rows_stored": stored.height,
        "max_abs_diff": max(
            float((joined[c] - joined[f"{c}_stored"]).abs().max())  # ty: ignore[invalid-argument-type]
            for c in ("mean", "q05", "q50", "q95", "q99")
        ),
    }
    out = common.WORK_DIR / "score_report.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
