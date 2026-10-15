"""The stage 1 variant S1 of experiment 00027: does PV_Live at the valid time help stage 1a?

For each GSP and seed, this refits experiment 00005's final ERA5-trained stage 1a (window from
2023-09, fit to before 2025-06-17) twice: with its usual columns, and with PV_Live's level over
capacity at the valid time as one more column. It scores each model's point prediction at observed
ERA5 weather over the validation year (July 2025 to June 2026). The observed PV_Live value at the
valid time is not known at the forecast origin, so this says only whether stage 1a understands the
GSP's solar better.

Output: `work/s1/errors.parquet` with one row per series, seed, arm, and valid time.

    uv run python ml-experiments/00027-pvlive-lags-in-stage2-gsp-demand/e00027_stage1.py
"""

import json
from datetime import UTC, date, datetime, timedelta
from typing import Final

import e00027_common as common
import holidays
import numpy as np
import polars as pl
import xgboost as xgb
from studies.solar import azimuth, zenith

DERIVED: Final[tuple[str, ...]] = (
    "solar_zenith",
    "solar_azimuth",
    "hour_local",
    "day_of_week_local",
    "covid_indicator",
    "days_to_nearest_holiday",
    "days_to_christmas",
)
"""Stage 1a columns computed from the timestamp and the location, not read from a weather file."""
NO_CALENDAR_DROPS: Final[tuple[str, ...]] = (
    "hour_local",
    "day_of_week_local",
    "days_to_nearest_holiday",
    "days_to_christmas",
)
"""The calendar-position columns the no-calendar control removes."""
LOCKDOWN_START: Final[datetime] = datetime(2020, 3, 23, tzinfo=UTC)
LOCKDOWN_END: Final[datetime] = datetime(2021, 7, 19, tzinfo=UTC)
"""The 0/1 lockdown indicator is 1 from the first to the last of these days, inclusive."""
HOLIDAY_YEARS: Final[range] = range(2018, 2029)
HOLIDAY_CLIP_DAYS: Final[int] = 7
CHRISTMAS_CLIP_DAYS: Final[int] = 14


def holiday_gaps() -> pl.DataFrame:
    """Return, for each calendar date, the signed days to the nearest holiday and to Christmas.

    Returns:
        A frame with `date`, `days_to_nearest_holiday` (positive before the holiday, negative after
        it, clipped at 7 days either way, from the `holidays` package's England calendar), and
        `days_to_christmas` (the same sign, to 25 December, clipped at 14 days either way).
    """
    calendar = holidays.country_holidays("GB", subdiv="ENG", years=HOLIDAY_YEARS)
    holiday_dates = np.array(sorted(calendar), dtype="datetime64[D]")
    days = pl.date_range(
        date(HOLIDAY_YEARS.start, 1, 1),
        date(HOLIDAY_YEARS.stop - 1, 12, 31),
        eager=True,
    )
    gaps = (holiday_dates[None, :] - days.to_numpy()[:, None]).astype("timedelta64[D]").astype(int)
    nearest = gaps[np.arange(days.len()), np.abs(gaps).argmin(axis=1)]
    christmas = np.array(
        [np.datetime64(f"{year}-12-25") for year in HOLIDAY_YEARS], dtype="datetime64[D]"
    )
    to_christmas = (
        (christmas[None, :] - days.to_numpy()[:, None]).astype("timedelta64[D]").astype(int)
    )
    nearest_christmas = to_christmas[np.arange(days.len()), np.abs(to_christmas).argmin(axis=1)]
    return pl.DataFrame(
        {
            "date": days,
            "days_to_nearest_holiday": np.clip(nearest, -HOLIDAY_CLIP_DAYS, HOLIDAY_CLIP_DAYS),
            "days_to_christmas": np.clip(
                nearest_christmas, -CHRISTMAS_CLIP_DAYS, CHRISTMAS_CLIP_DAYS
            ),
        }
    )


def derived_columns(*, valid_time: pl.Series, latitude: float, longitude: float) -> pl.DataFrame:
    """Return the timestamp-and-location columns of stage 1a for period-ending UTC stamps.

    The sun position is at the midpoint of the half-hour ending at each stamp. The local hour,
    weekday, and holiday distance use the stamp in Europe/London time.

    Args:
        valid_time: Period-ending UTC stamps.
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        One row per stamp with the `DERIVED` columns, float32 (the holiday distance in whole days).
    """
    midpoint = valid_time.dt.offset_by("-15m")
    local = valid_time.dt.convert_time_zone("Europe/London")
    frame = pl.DataFrame(
        {
            "solar_zenith": zenith(stamps=midpoint, latitude=latitude, longitude=longitude),
            "solar_azimuth": azimuth(stamps=midpoint, latitude=latitude, longitude=longitude),
            "hour_local": local.dt.hour().cast(pl.Float64)
            + local.dt.minute().cast(pl.Float64) / 60.0,
            "day_of_week_local": local.dt.weekday().cast(pl.Float64),
            "covid_indicator": (
                (valid_time >= LOCKDOWN_START) & (valid_time < LOCKDOWN_END + timedelta(days=1))
            ).cast(pl.Float64),
            "date": local.dt.date(),
        }
    )
    return (
        frame.join(holiday_gaps(), on="date", how="left", validate="m:1")
        .select(DERIVED)
        .cast(dict.fromkeys(DERIVED, pl.Float32))
    )


def with_derived(*, frame: pl.DataFrame, latitude: float, longitude: float) -> pl.DataFrame:
    """Add the `DERIVED` columns to a frame with a `valid_time` column, replacing any it has.

    Args:
        frame: Rows with a UTC `valid_time`.
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        The frame with the `DERIVED` columns, computed once per distinct valid time.
    """
    times = frame["valid_time"].unique().sort()
    table = derived_columns(valid_time=times, latitude=latitude, longitude=longitude).with_columns(
        valid_time=times
    )
    return frame.drop(DERIVED, strict=False).join(
        table, on="valid_time", how="left", validate="m:1"
    )


def load_features(*, series: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return each GSP's stage 1a input columns on the dense grid, from the sweep's frozen set."""
    columns: list[str] = json.loads(common.FEATURE_SET_PATH.read_text())["columns"]
    weather = [c for c in columns if c not in DERIVED]
    grid = pl.DataFrame({"valid_time": common.grid_times()})
    era5 = pl.read_parquet(common.ERA5_PATH)
    out = {}
    for row in series.iter_rows(named=True):
        joined = grid.join(
            era5.filter(pl.col("time_series_id") == row["time_series_id"]).select(
                "valid_time", *weather
            ),
            on="valid_time",
            how="left",
            validate="1:1",
        ).sort("valid_time")
        derived = with_derived(
            frame=joined.select("valid_time"),
            latitude=row["latitude"],
            longitude=row["longitude"],
        )
        out[row["site"]] = (
            joined.select(weather)
            .with_columns(derived.select(DERIVED))
            .select(columns)
            .to_numpy()
            .astype(np.float32)
        )
    return out


def main() -> None:
    """Fit both stage 1a variants for every GSP and seed and write their validation errors."""
    series = common.series_table()
    power = common.load_power(series=series)
    pv = common.load_pv_live(series=series)
    features = load_features(series=series)
    start = int(common.grid_index(times=common.TRAIN_START))
    cutoff = int(common.grid_index(times=common.FINAL_TRAIN_CUTOFF))
    first = int(common.grid_index(times=common.VALIDATION_START))
    last = int(common.grid_index(times=common.HOLDOUT))
    times = common.grid_times()
    pieces = []
    for site in series["site"]:
        for seed in common.SEEDS:
            for arm in ("without", "with_pv_live"):
                x = features[site]
                if arm == "with_pv_live":
                    x = np.concatenate([x, pv[site]["level"][:, None]], axis=1)
                rows = np.arange(start, cutoff)
                rows = rows[np.isfinite(power[site][rows])]
                model = xgb.train(
                    {**common.XGBOOST_PARAMETERS, "seed": seed},
                    xgb.DMatrix(x[rows], label=power[site][rows]),
                    num_boost_round=common.N_ROUNDS,
                )
                index = np.arange(first, last)
                index = index[np.isfinite(power[site][index])]
                prediction = model.predict(xgb.DMatrix(x[index]))
                pieces.append(
                    pl.DataFrame(
                        {
                            "site": site,
                            "seed": seed,
                            "arm": arm,
                            "valid_time": times.gather(index),
                            "error": (prediction - power[site][index]).astype(np.float32),
                            "pv_live": pv[site]["level"][index],
                        }
                    )
                )
                print(site, seed, arm, flush=True)
    out_dir = common.WORK_DIR / "s1"
    out_dir.mkdir(parents=True, exist_ok=True)
    pl.concat(pieces).write_parquet(out_dir / "errors.parquet")


if __name__ == "__main__":
    main()
