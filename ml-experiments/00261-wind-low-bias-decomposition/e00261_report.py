"""Score the stage tables of `e00261_decompose.py`: PIT shares, median shifts, and wind-speed bias.

The probability-integral-transform (PIT) share of a stage is the share of rows where the observed
power is strictly below that stage's forecast median; a calibrated median gives 0.5. Intervals
resample the 12 calendar months of the scored year (by the run's month), 2,000 times; the farms
are not resampled. Every number the write-up quotes is printed here and written to `report.json`.

    uv run --no-sync python ml-experiments/00261-*/e00261_report.py
"""

import json
from collections.abc import Callable
from typing import Final

import e00261_common as common
import numpy as np
import polars as pl

STAGES: Final[tuple[tuple[str, str], ...]] = (
    ("Oracle: ERA5 wind", "q50_oracle"),
    ("Ensemble-mean wind", "q50_mean_wind"),
    ("51-member mixture (R1)", "q50_stored_r1"),
    ("XGBoost 51-member mixture (R0)", "q50_stored_r0"),
)
EXPECTED_STAGES: Final[tuple[tuple[str, str], ...]] = (
    ("ERA5 wind", "e_oracle"),
    ("Ensemble-mean wind", "e_mean_wind"),
    ("Mean of 51 members", "e_members_mean"),
    ("Median of 51 members", "e_members_median"),
)
SPEED_EDGES: Final[tuple[float, ...]] = (0, 3, 5, 8, 12, 16, 20, 40)
"""Edges of the ERA5 100 m speed bins in m/s; the first and last are not used as cut points."""
SECTORS: Final[int] = 8
RESAMPLES: Final[int] = 2000
SEED: Final[int] = 0
SHIFT_POINTS: Final[float] = common.POINTS_PER_FRACTION


def load(name: str) -> pl.DataFrame:
    """Return one set's stage table with the lead band, speed bin, and direction sector."""
    frame = pl.read_parquet(common.WORK_DIR / f"cases_{name}.parquet")
    angle = (pl.arctan2(pl.col("era5_dir_sin"), pl.col("era5_dir_cos")).degrees() + 360.0) % 360.0
    edges = np.array(SPEED_EDGES[1:-1])
    labels = np.array(
        ["(-inf, 3]", "(3, 5]", "(5, 8]", "(8, 12]", "(12, 16]", "(16, 20]", "(20, inf]"]
    )
    bins = np.digitize(frame["era5_speed_100m"].to_numpy(), edges, right=True)
    return frame.with_columns(
        speed_bin=pl.Series(labels[bins]),
        band=pl.when(pl.col("lead_day") <= 2)
        .then(pl.lit("days 0-2"))
        .when(pl.col("lead_day") <= 6)
        .then(pl.lit("days 3-6"))
        .otherwise(pl.lit("days 7+")),
        sector=((angle + 180.0 / SECTORS) // (360.0 / SECTORS)).cast(pl.Int32) % SECTORS,
        speed_bias_100m=pl.col("ens_speed_100m") - pl.col("era5_speed_100m"),
        speed_bias_10m=pl.col("ens_speed_10m") - pl.col("era5_speed_10m"),
        month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
    )


def mean_of(series: pl.Series) -> float:
    """Return the mean of `series` as a float."""
    return float(np.asarray(series.to_numpy(), dtype=np.float64).mean())


def share(*, frame: pl.DataFrame, column: str) -> float:
    """Return the share of rows whose observed power is strictly below the column's median."""
    return mean_of(frame["actual_norm"] < frame[column])


def stage_shares(*, frame: pl.DataFrame) -> dict[str, float]:
    """Return every stage's share on `frame`."""
    return {label: share(frame=frame, column=column) for label, column in STAGES}


def month_bootstrap(
    *, frame: pl.DataFrame, statistic: Callable[[pl.DataFrame], float]
) -> tuple[float, float]:
    """Return the 2.5% and 97.5% points of `statistic` over resampled months of `frame`."""
    months = sorted(frame["month"].unique().to_list())
    by_month = {m: frame.filter(pl.col("month") == m) for m in months}
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(RESAMPLES):
        picked = rng.choice(len(months), size=len(months))
        draws.append(statistic(pl.concat([by_month[months[i]] for i in picked])))
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def shifts(*, frame: pl.DataFrame) -> dict[str, float]:
    """Return mean median shifts in points of capacity, with the speed-bias conversion.

    `total` is the oracle median minus the final R1 median. `weather` is the oracle minus the
    ensemble-mean-wind median, `mixing` the ensemble-mean-wind median minus the mixture median.
    `slope_implied` is the core curve's slope at the ERA5 wind times the 100 m speed bias.
    """
    total = mean_of(frame["q50_oracle"] - frame["q50_stored_r1"])
    weather = mean_of(frame["q50_oracle"] - frame["q50_mean_wind"])
    mixing = mean_of(frame["q50_mean_wind"] - frame["q50_stored_r1"])
    implied = -mean_of(frame["slope_per_m_s"] * frame["speed_bias_100m"])
    return {
        "total": float(total) * SHIFT_POINTS,
        "weather": float(weather) * SHIFT_POINTS,
        "mixing": float(mixing) * SHIFT_POINTS,
        "slope_implied": float(implied) * SHIFT_POINTS,
        "weather_share": float(weather / total),
        "mixing_share": float(mixing / total),
        "slope_implied_share": float(implied / total),
    }


def main() -> None:
    """Print and write the tables the write-up quotes."""
    report: dict = {}
    for name in ("scored", "hindcast"):
        frame = load(name)
        kept = frame.filter(~pl.col("masked"))
        entry: dict = {"rows_all": frame.height, "rows_kept": kept.height}
        for label, subset in (("kept", kept), ("all", frame)):
            entry[f"share_{label}"] = stage_shares(frame=subset)
            entry[f"share_{label}_by_farm"] = {
                farm: stage_shares(frame=subset.filter(pl.col("site") == farm))
                for farm in sorted(subset["site"].unique())
            }
            entry[f"share_{label}_by_band"] = {
                band: stage_shares(frame=subset.filter(pl.col("band") == band))
                for band in ("days 0-2", "days 3-6", "days 7+")
            }
            entry[f"share_{label}_by_lead_day"] = {
                str(day): stage_shares(frame=subset.filter(pl.col("lead_day") == day))
                for day in range(11)
            }
        entry["shift_kept"] = shifts(frame=kept)
        entry["shift_kept_by_farm"] = {
            farm: shifts(frame=kept.filter(pl.col("site") == farm))
            for farm in sorted(kept["site"].unique())
        }
        entry["shift_kept_by_band"] = {
            band: shifts(frame=kept.filter(pl.col("band") == band))
            for band in ("days 0-2", "days 3-6", "days 7+")
        }
        entry["expected_power_kept"] = {
            label: mean_of(kept[column]) * SHIFT_POINTS for label, column in EXPECTED_STAGES
        }
        entry["speed_bias_by_speed_bin"] = {
            row["speed_bin"]: {
                "rows": row["len"],
                "bias_100m": row["bias_100m"],
                "bias_10m": row["bias_10m"],
                "slope": row["slope"],
            }
            for row in kept.group_by("speed_bin")
            .agg(
                pl.len(),
                bias_100m=pl.col("speed_bias_100m").mean(),
                bias_10m=pl.col("speed_bias_10m").mean(),
                slope=pl.col("slope_per_m_s").mean(),
            )
            .iter_rows(named=True)
        }
        entry["speed_bias_by_sector"] = {
            int(row["sector"]): row["bias_100m"]
            for row in kept.group_by("sector")
            .agg(bias_100m=pl.col("speed_bias_100m").mean())
            .iter_rows(named=True)
        }
        entry["speed_bias_by_band"] = {
            row["band"]: row["bias_100m"]
            for row in kept.group_by("band")
            .agg(bias_100m=pl.col("speed_bias_100m").mean())
            .iter_rows(named=True)
        }
        entry["speed_bias_mean"] = mean_of(kept["speed_bias_100m"])
        if name == "scored":
            entry["interval_kept"] = {
                label: month_bootstrap(
                    frame=kept, statistic=lambda f, c=column: share(frame=f, column=c)
                )
                for label, column in STAGES
            }
            entry["interval_shift"] = {
                key: month_bootstrap(frame=kept, statistic=lambda f, k=key: shifts(frame=f)[k])
                for key in ("total", "weather", "mixing", "slope_implied_share", "weather_share")
            }
        report[name] = entry
    path = common.WORK_DIR / "report.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
