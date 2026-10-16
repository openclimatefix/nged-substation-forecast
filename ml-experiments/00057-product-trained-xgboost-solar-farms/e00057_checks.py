"""Print the UKV coverage, time-convention, and plausibility checks of experiment 00057.

For each solar farm, counts the UKV runs dropped in the training window and among the validation
runs, and the runs lost per variable. Then compares the hour-of-day profile of UKV's shortwave
flux with the ENS control member's shortwave and with the farm's power over capacity at the scored
cases' valid times, checks that UKV's shortwave is not negative and is near zero with the sun below
the horizon, and counts the placebo's groups of one row, whose rows keep their own values.

    uv run python ml-experiments/00057-product-trained-xgboost-solar-farms/e00057_checks.py
"""

import json
from datetime import timedelta

import e00057_common as common
import e00057_fit as fit
import numpy as np
import polars as pl

from studies import solar

SUN_BELOW_ZENITH_DEGREES: float = 95.0


def main() -> None:
    """Print the checks as JSON, farms by anonymous label."""
    sites = common.site_table()
    stores, first = fit.open_stores()
    cells = fit.farm_cells(group=first, sites=sites)
    label = dict(zip(sites["time_series_id"], sites["site"], strict=True))
    cases = pl.read_parquet(common.REFERENCE_PATHS["ref_best_guess"]).filter(
        (pl.col("technology") == "solar") & (pl.col("seed") == 0)
    )
    runs_val = sorted(cases["power_fcst_init_time"].dt.truncate("1d").unique().to_list())
    runs_train = [
        common.TRAIN_START + timedelta(days=i)
        for i in range((common.LAST_TRAINING_VALID.date() - common.TRAIN_START.date()).days + 1)
    ]
    ukv = fit.read_ukv(stores=stores, cells=cells, runs=runs_train + runs_val)
    out: dict[str, object] = {}
    for series, site in label.items():
        runs = set(ukv.filter(pl.col("time_series_id") == series)["nwp_init_time"].unique())
        out[site] = {
            "training_runs_complete": sum(r in runs for r in runs_train),
            "training_runs": len(runs_train),
            "validation_runs_complete": sum(r in runs for r in runs_val),
            "validation_runs": len(runs_val),
        }
    out["ukv_shortwave_negative_rows"] = int((ukv["ukv_shortwave"] < 0).sum())
    out["ukv_ranges"] = {
        name: [float(ukv[name].to_numpy().min()), float(ukv[name].to_numpy().max())]
        for name in common.UKV_SOLAR
    }
    # Hour-of-day profiles (UTC) at the scored cases' valid times.
    scored = cases.filter(
        common.hours_after_run(
            valid_time=pl.col("valid_time"), power_fcst_init_time=pl.col("power_fcst_init_time")
        )
        >= common.SCORED_FROM_HOURS
    ).select("time_series_id", "power_fcst_init_time", "valid_time", "actual_norm")
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
                    "shortwave",
                ],
            ).filter(pl.col("ensemble_member") == 0)
            for path in sorted(common.ENS_VALIDATION_DIR.glob("init=*.parquet"))
        ]
    )
    both = joined.join(ens, on=["time_series_id", "valid_time", "nwp_init_time"], how="inner")
    profile = (
        both.with_columns(hour=pl.col("valid_time").dt.hour())
        .group_by("hour")
        .agg(
            ukv_shortwave=pl.col("ukv_shortwave").mean(),
            ens_shortwave=pl.col("shortwave").mean(),
            power_over_capacity=pl.col("actual_norm").mean(),
        )
        .sort("hour")
    )
    out["hour_of_day_profile_utc"] = profile.to_dicts()
    night, night_native = [], []
    for row in sites.iter_rows(named=True):
        part = both.filter(pl.col("time_series_id") == row["time_series_id"])
        angle = solar.zenith(
            stamps=part["valid_time"], latitude=row["latitude"], longitude=row["longitude"]
        )
        below = angle > SUN_BELOW_ZENITH_DEGREES
        night.append(part["ukv_shortwave"].to_numpy()[below])
        hours = (part["valid_time"] - part["nwp_init_time"]).dt.total_hours().to_numpy()
        night_native.append(
            part["ukv_shortwave"].to_numpy()[below & (hours <= common.HOURLY_LAST_LEAD)]
        )
    out["ukv_shortwave_max_with_sun_below_horizon"] = float(np.concatenate(night).max())
    out["ukv_shortwave_max_with_sun_below_horizon_hourly_leads"] = float(
        np.concatenate(night_native).max()
    )
    out["ukv_mean_shortwave_w_m2"] = float(both["ukv_shortwave"].to_numpy().mean())
    out["ens_control_mean_shortwave"] = float(both["shortwave"].to_numpy().mean())
    out["scored_cases_with_ukv_and_ens"] = both.height
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
