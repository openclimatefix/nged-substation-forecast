"""Print the per-farm UKV coverage and wind-speed checks of experiment 00054.

Counts, for each farm, the complete UKV runs in the training window and among the validation runs;
compares UKV's mean 10 m wind speed with ENS's control member on the validation year at the scored
cases' valid times; and counts the placebo's groups of one row, whose rows keep their own values.

    uv run python ml-experiments/00054-product-trained-xgboost-wind-farms/e00054_checks.py
"""

import json
from datetime import timedelta

import e00054_common as common
import e00054_fit as fit
import polars as pl


def main() -> None:
    """Print the checks as JSON, farms by anonymous label."""
    sites = common.site_table()
    stores, first = fit.open_stores()
    cells = fit.farm_cells(group=first, sites=sites)
    label = dict(zip(sites["time_series_id"], sites["site"], strict=True))
    cases = pl.read_parquet(common.REFERENCE_PATHS["ref_best_guess"]).filter(
        (pl.col("technology") == "wind") & (pl.col("seed") == 0)
    )
    runs_val = sorted(cases["power_fcst_init_time"].dt.truncate("1d").unique().to_list())
    runs_train = [
        common.TRAIN_START + timedelta(days=i)
        for i in range((common.LAST_TRAINING_VALID.date() - common.TRAIN_START.date()).days + 1)
    ]
    ukv = fit.read_ukv(stores=stores, cells=cells, runs=runs_train + runs_val)
    out: dict[str, object] = {}
    for series, site in label.items():
        part = ukv.filter(pl.col("time_series_id") == series)
        runs = set(part["nwp_init_time"].unique().to_list())
        out[site] = {
            "training_runs_complete": sum(r in runs for r in runs_train),
            "validation_runs_complete": sum(r in runs for r in runs_val),
        }
    scored = cases.filter(
        common.hours_after_run(
            valid_time=pl.col("valid_time"), power_fcst_init_time=pl.col("power_fcst_init_time")
        )
        >= common.SCORED_FROM_HOURS
    ).select("time_series_id", "power_fcst_init_time", "valid_time")
    scored = scored.with_columns(nwp_init_time=pl.col("power_fcst_init_time").dt.truncate("1d"))
    joined = scored.join(ukv, on=["time_series_id", "nwp_init_time", "valid_time"], how="inner")
    ens = pl.concat(
        [
            pl.read_parquet(
                path,
                columns=[
                    "time_series_id",
                    "valid_time",
                    "nwp_init_time",
                    "ensemble_member",
                    "wind_speed_10m",
                ],
            ).filter(pl.col("ensemble_member") == 0)
            for path in sorted(common.ENS_VALIDATION_DIR.glob("init=*.parquet"))
        ]
    )
    both = joined.join(ens, on=["time_series_id", "valid_time", "nwp_init_time"], how="inner")
    out["validation_mean_speed_10m_m_s"] = {
        "ukv": float(both["ukv_speed_10m"].to_numpy().mean()),
        "ens_control": float(both["wind_speed_10m"].to_numpy().mean()),
        "cases": both.height,
    }
    keyed = ukv.with_columns(
        month=pl.col("valid_time").dt.strftime("%Y-%m"),
        hour=pl.col("valid_time").dt.hour(),
        minute=pl.col("valid_time").dt.minute(),
        lead_day=(pl.col("valid_time") - pl.col("nwp_init_time")).dt.total_hours() // 24,
    )
    groups = keyed.group_by("time_series_id", "month", "hour", "minute", "lead_day").len()
    out["placebo_groups_of_one_row"] = int((groups["len"] == 1).sum())
    out["placebo_groups"] = groups.height
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
