"""The solar offset of experiment 00087: calibrated sunshine anomaly, weights, and capacity.

The offset is the base margin that experiment 00064's arm `am` gives XGBoost at the five Primary
series with a clean embedded-solar capacity (`OFFSET_SITES`). For a row at lead band `b` and local
month `m` the offset, as a fraction of the series' effective capacity, is

    w(site, b, season(m)) * capacity_mw / effective_capacity_mw * anomaly

where `anomaly` is the ensemble-mean ENS sunshine anomaly, calibrated by the lead-band slope of
experiment 00046 (`slope * (u_mean - clim_ens)`, where `u_mean` is the ensemble mean of the unit PV
shape of the ENS shortwave radiation), and `w` is the weight of experiment 00064's table (variant
`all`, zero from the 10-day band on and outside April to September).

Sources, each copied here because an experiment may not import another:

- `e00074_inputs.py` and `e00074_fit.py` (experiment 00074): the unit PV shape chain
  (`u_from_half_hour_ghi`, `read_run_shortwave`, `member_mean_u`), the anomaly table
  (`anomaly_table`, from `shape_table`), the weight lookup (`weights_per_row`), and the base margin
  (`margin_for`, here `margin`).
- Experiment 00074's inputs stop at the run of 2026-05-31 and valid time 2026-06-01 (its last
  monthly refit). This script rebuilds the ensemble-mean `u` of all 365 daily runs of 2025-07-01 to
  2026-06-30 over their whole 351-hour horizon, so the training rows of every weekly refit (the
  last is 2026-06-23) and every validation run have an anomaly. `--check` compares the rebuilt
  anomalies with experiment 00074's `train_shape.parquet` and with experiment 00046's validation
  anomalies of the 73 runs of `fold_a_every5d_v1`.

Differences from experiment 00074, for the result's deviation list:

- Experiment 00074 refits on 365 days at 150 rounds and `eta` 0.15 (P-roll). BG2 uses the offset
  on the BG2 recipe (whole history, half-life 0.5 years, 500 rounds, `eta` 0.05).
- The training base margin uses the held capacity of experiment 00064 (fitted before 2025-07-01).
  The prediction base margin of a validation run uses the capacity vintage of experiment 00078:
  the season-year constant re-estimated on the weeks before the Monday of the run's week less one
  week. Experiment 00078's table holds the 73 runs of `fold_a_every5d_v1`, and every week of the
  year has at least one of them, so the vintage of any daily run is the table's vintage of the same
  `origin_week`. The two runs of Monday 2026-06-29 and Tuesday 2026-06-30 start a week the table
  lacks and take the vintage of the last `origin_week` it holds. Ablation A6 predicts with the held
  capacity instead; only the prediction changes.
- Weights come from experiment 00064's table, fitted before 2025-07-01, and are not refitted.

    uv run python ml-experiments/00087-*/e00087_offset.py --part u_mean [--workers 4] [--limit N]
    uv run python ml-experiments/00087-*/e00087_offset.py --part anomaly [--check]
    uv run python ml-experiments/00087-*/e00087_offset.py --part capacity
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common
import numpy as np
import polars as pl
import pvlib
from ml_core.features._nwp import _upsample_nwp_to_half_hourly
from studies.pv_physics import PlantParameters, SunAndSky, plant_power_mw, solar_position
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OFFSET_SITES: Final[tuple[str, ...]] = ("D9", "D10", "D11", "D14", "D16")
"""The five Primaries whose embedded-solar capacity is clean (experiment 00064's capacity table)."""
E46_SOLAR_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00046-embedded-solar-share-of-primary-forecast-error"
    / "work"
    / "solar"
)
"""CAMS and ENS solar inputs, climatologies, and calibration slopes of experiment 00046."""
E64_WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00064-pvdis-5-net-demand-as-demand-minus-forecast-solar" / "work"
)
CAPACITY_FILE: Final[Path] = E64_WORK_DIR / "capacity.parquet"
"""Experiment 00064's held capacity per offset series, fitted before 2025-07-01."""
WEIGHTS_FILE: Final[Path] = E64_WORK_DIR / "weights.parquet"
"""Experiment 00064's weights per series, lead band, and season, fitted before 2025-07-01."""
VINTAGES_FILE: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00078-pvdis-5-current-capacity-vintages"
    / "work"
    / "capacity_vintages.parquet"
)
"""Experiment 00078's real-time capacity vintages of the 73 runs of `fold_a_every5d_v1`."""
E74_TRAIN_SHAPE_FILE: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00074-an1-same-time-analogues-vs-explicit-solar-forecast-demand"
    / "work"
    / "inputs"
    / "train_shape.parquet"
)
"""Experiment 00074's calibrated shapes of its training rows, for `--check` only."""
OFFSET_DIR: Final[Path] = common.WORK_DIR / "offset"
U_MEAN_DIR: Final[Path] = OFFSET_DIR / "u_mean"
"""One `init=<date>.parquet` per daily run: the ensemble-mean unit PV shape at the offset series."""
ANOMALY_FILE: Final[Path] = OFFSET_DIR / "anomaly.parquet"
"""Calibrated anomaly of every (series, run, valid time) that a training or validation row needs."""
SHORTWAVE_SOURCE: Final[str] = "downward_short_wave_radiation_flux_surface"
HORIZON_HOURS: Final[int] = 351
SLOPE_TOLERANCE: Final[float] = 1e-6
NO_SKILL_BAND: Final[int] = 4
"""Lead bands from this index on (10+ days) get a weight of zero, as in experiment 00064."""
SEASON_OF_MONTH: Final[dict[int, str]] = {
    4: "Apr-Jun",
    5: "Apr-Jun",
    6: "Apr-Jun",
    7: "Jul-Sep",
    8: "Jul-Sep",
    9: "Jul-Sep",
}
"""The two seasons of a fitted weight; a month outside April to September has no weight (zero)."""
E45_FIRST_WEEK_START: Final[date] = date(2019, 9, 2)
"""Monday of week 0 in experiment 00045's weekly index, which experiment 00078's vintages use."""
FLEET: Final[PlantParameters] = PlantParameters(
    tilt_deg=35.0, azimuth_deg=180.0, dc_capacity_mw=1.15, ac_capacity_mw=1.0
)
"""The generic fleet whose output per MW of AC capacity is the unit PV shape `u`."""
ARCHIVE_DIR: Final[Path] = common.ARCHIVE_DIR


def lead_band(*, lead_hours: np.ndarray) -> np.ndarray:
    """Return the index into `common.LEAD_BANDS` of each lead time."""
    edges = np.array([low for _, low, _ in common.LEAD_BANDS[1:]])
    return np.searchsorted(edges, lead_hours, side="right")


def offset_series() -> pl.DataFrame:
    """Return the five offset series' rows of `common.series_table()`.

    Returns:
        The rows of `OFFSET_SITES`, each a Primary.

    Raises:
        ValueError: If a site is missing or is not a Primary.
    """
    table = common.series_table().filter(pl.col("site").is_in(list(OFFSET_SITES)))
    if table.height != len(OFFSET_SITES) or set(table["substation_type"]) != {"Primary"}:
        msg = f"the offset sites {OFFSET_SITES} must be five Primary series"
        raise ValueError(msg)
    held = set(pl.read_parquet(CAPACITY_FILE)["site"].to_list())
    if held != set(OFFSET_SITES):
        msg = f"{CAPACITY_FILE} holds {sorted(held)}, not the offset sites"
        raise ValueError(msg)
    return table


def u_from_half_hour_ghi(
    *, stamps: pl.Series, ghi: np.ndarray, latitude: float, longitude: float
) -> np.ndarray:
    """Return the unit PV shape of half-hour mean irradiance, the chain CAMS goes through.

    The sun is at the midpoint of the half-hour ending at each stamp, the Erbs correlation splits
    the irradiance, and the generic fleet turns it into power per MW of AC capacity.

    Args:
        stamps: Distinct period-ending UTC stamps, shape (n,).
        ghi: Half-hour mean global horizontal irradiance in W/m2, shape (k, n); NaN where unknown.
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        `u`, shape (k, n), NaN where `ghi` is NaN.
    """
    copies = ghi.shape[0]
    midpoints = stamps.dt.offset_by("-15m")
    zenith, sun_azimuth = solar_position(stamps=midpoints, latitude=latitude, longitude=longitude)
    day = midpoints.dt.ordinal_day().to_numpy()
    unknown = ~np.isfinite(ghi)
    flat = np.clip(np.where(unknown, 0.0, ghi), 0.0, None).ravel().astype(np.float64)
    split = pvlib.irradiance.erbs(
        ghi=flat, zenith=np.tile(zenith, copies), datetime_or_doy=np.tile(day, copies)
    )
    sky = SunAndSky(
        half_hour_end_time=np.tile(stamps.to_numpy(), copies),
        ghi_w_m2=flat,
        dni_w_m2=np.nan_to_num(np.asarray(split["dni"]).astype(np.float64)),
        dhi_w_m2=np.nan_to_num(np.asarray(split["dhi"]).astype(np.float64)),
        zenith_deg=np.tile(zenith, copies),
        sun_azimuth_deg=np.tile(sun_azimuth, copies),
    )
    shape = plant_power_mw(sky=sky, parameters=FLEET).reshape(copies, -1)
    return np.where(unknown, np.nan, shape)


def read_run_shortwave(*, run: datetime, series: pl.DataFrame) -> pl.DataFrame:
    """Return one archive run's half-hourly `shortwave` for every member at every series' cell.

    Args:
        run: The 00:00 UTC initialisation time.
        series: `offset_series()`, with `h3_res_5`, `latitude`, `longitude`.

    Returns:
        Columns `time_series_id`, `power_fcst_init_time`, `valid_time`, `ensemble_member`, and
        `shortwave` (W/m2, half-hour mean ending at `valid_time`), for valid times after the
        forecast origin (the run plus 9 hours) and at most 351 hours later.
    """
    stamp = f"{run:%Y-%m-%d} 00:00:00.000000".replace(" ", "%20").replace(":", "%3A")
    native = (
        pl.scan_parquet(ARCHIVE_DIR / f"init_time={stamp}" / "*.parquet")
        .select("h3_index", "ensemble_member", "valid_time", SHORTWAVE_SOURCE)
        .filter(pl.col("h3_index").is_in(series["h3_res_5"].unique().to_list()))
        .join(
            series.lazy().select("time_series_id", "h3_res_5"),
            left_on="h3_index",
            right_on="h3_res_5",
            how="inner",
        )
        .drop("h3_index")
        .with_columns(nwp_init_time=pl.lit(run).cast(pl.Datetime("us", "UTC")))
    )
    coordinates = series.lazy().select("time_series_id", "latitude", "longitude")
    half_hourly = _upsample_nwp_to_half_hourly(native, solar_coordinates=coordinates).collect()
    origin = common.origin_of(run=run)
    return (
        half_hourly.rename({SHORTWAVE_SOURCE: "shortwave"})
        .with_columns(power_fcst_init_time=pl.lit(origin).cast(pl.Datetime("us", "UTC")))
        .filter(
            (pl.col("valid_time") > pl.col("power_fcst_init_time"))
            & (
                pl.col("valid_time")
                <= pl.col("power_fcst_init_time") + pl.duration(hours=HORIZON_HOURS)
            )
        )
        .select(*common.KEYS, "ensemble_member", "shortwave")
    )


def member_mean_u(*, rows: pl.DataFrame, series_row: dict) -> pl.DataFrame:
    """Return the ensemble-mean ENS `u` of every valid time of one series and one run.

    Args:
        rows: One series' rows of one run with `valid_time`, `ensemble_member`, and `shortwave`,
            every member holding every valid time.
        series_row: The series' `latitude` and `longitude`.

    Returns:
        Columns `valid_time` and `u_mean`, sorted by valid time.

    Raises:
        ValueError: If a member lacks a valid time.
    """
    ordered = rows.sort("valid_time", "ensemble_member")
    times = ordered["valid_time"].unique(maintain_order=True)
    members = ordered["ensemble_member"].unique().sort()
    if ordered.height != times.len() * members.len():
        msg = "every member must hold every valid time"
        raise ValueError(msg)
    ghi = ordered["shortwave"].to_numpy().astype(np.float64).reshape(times.len(), members.len()).T
    u = u_from_half_hour_ghi(
        stamps=times,
        ghi=ghi,
        latitude=series_row["latitude"],
        longitude=series_row["longitude"],
    )
    return pl.DataFrame({"valid_time": times, "u_mean": u.mean(axis=0).astype(np.float32)})


def u_mean_path(*, run: datetime) -> Path:
    """Return the file holding one run's ensemble-mean `u` at the offset series."""
    return U_MEAN_DIR / f"init={run:%Y-%m-%d}.parquet"


def build_run(*, run: datetime) -> str:
    """Write one run's ensemble-mean `u` for the five offset series, unless the file exists.

    Args:
        run: The 00:00 UTC initialisation time.

    Returns:
        A progress line.

    Raises:
        ValueError: If a series has no rows for the run.
    """
    target = u_mean_path(run=run)
    if target.exists():
        return f"{target.name}: exists"
    started = time.monotonic()
    series = offset_series()
    rows = read_run_shortwave(run=run, series=series)
    origin = common.origin_of(run=run)
    pieces = []
    for row in series.iter_rows(named=True):
        own = rows.filter(pl.col("time_series_id") == row["time_series_id"])
        if own.is_empty():
            msg = f"{row['site']} has no shortwave rows for the run {run:%Y-%m-%d}"
            raise ValueError(msg)
        pieces.append(
            member_mean_u(rows=own, series_row=row).with_columns(
                site=pl.lit(row["site"]),
                power_fcst_init_time=pl.lit(origin).cast(pl.Datetime("us", "UTC")),
            )
        )
    table = pl.concat(pieces).select("site", "power_fcst_init_time", "valid_time", "u_mean")
    common.atomic_write(frame=table, target=target)
    return f"{target.name}: {table.height:,} rows in {time.monotonic() - started:.1f} s"


def build_u_means(*, workers: int, limit: int | None) -> None:
    """Build (or resume) the ensemble-mean `u` files of the 365 daily runs."""
    runs = common.run_dates(subset="all")[:limit]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(build_run, run=run) for run in runs]
        for future in futures:
            print(future.result(), flush=True)


def anomaly_table() -> pl.DataFrame:
    """Return the calibrated ensemble-mean anomaly of every training and validation row.

    The rows are those of experiment 00046's training runs (the control member's runs to
    2025-06-30, whose `u_mean` experiment 00046 stored) and the 365 rebuilt runs. The climatology
    and the lead-band slopes are experiment 00046's, fitted on data to 2025-06-30.

    Returns:
        Columns `time_series_id`, `power_fcst_init_time`, `valid_time`, and `anomaly` (the lead-band
        slope times the ensemble-mean ENS anomaly; a missing value counts as zero), Float64.
    """
    old = pl.concat(
        [
            pl.read_parquet(path).select("site", "power_fcst_init_time", "valid_time", "u_mean")
            for path in sorted((E46_SOLAR_DIR / "ens_train").glob("init=*.parquet"))
        ]
    )
    new = pl.concat([pl.read_parquet(path) for path in sorted(U_MEAN_DIR.glob("init=*.parquet"))])
    ens = pl.concat([old, new]).filter(pl.col("site").is_in(list(OFFSET_SITES)))
    lead = ((ens["valid_time"] - ens["power_fcst_init_time"]).dt.total_minutes() / 60.0).to_numpy()
    stamp = pl.col("valid_time")
    climatology = pl.read_parquet(E46_SOLAR_DIR / "climatology.parquet").select(
        "site", "month", "hh", "clim_ens"
    )
    slopes = pl.read_parquet(E46_SOLAR_DIR / "calibration.parquet").select("site", "band", "slope")
    series = offset_series().select("site", "time_series_id")
    return (
        ens.with_columns(
            month=stamp.dt.convert_time_zone("Europe/London").dt.month().cast(pl.Int8),
            hh=(stamp.dt.hour() * 2 + stamp.dt.minute() // 30).cast(pl.Int8),
            band=pl.Series(lead_band(lead_hours=lead)),
        )
        .join(climatology, on=["site", "month", "hh"], how="left", validate="m:1")
        .join(slopes, on=["site", "band"], how="left", validate="m:1")
        .join(series, on="site", how="left", validate="m:1")
        .with_columns(
            anomaly=(pl.col("slope") * (pl.col("u_mean") - pl.col("clim_ens")))
            .fill_nan(0.0)
            .fill_null(0.0)
        )
        .select("time_series_id", "power_fcst_init_time", "valid_time", "anomaly")
    )


def check_anomaly(*, table: pl.DataFrame) -> None:
    """Raise unless the rebuilt anomalies equal experiment 00074's and experiment 00046's.

    Two comparisons: every row of experiment 00074's `train_shape.parquet` (valid times to
    2026-06-01) must be in `table` with an equal anomaly, and the validation anomaly of each of the
    73 runs of `fold_a_every5d_v1` that experiment 00046 stored must match, member by member.

    Args:
        table: `anomaly_table()`.

    Raises:
        ValueError: If a row is missing or differs by more than `SLOPE_TOLERANCE`.
    """
    keys = list(common.KEYS)
    other = pl.read_parquet(
        E74_TRAIN_SHAPE_FILE, columns=[*keys, "anomaly_mean_calibrated"]
    ).filter(pl.col("time_series_id").is_in(table["time_series_id"].unique().implode()))
    joined = other.join(table, on=keys, how="left", validate="1:1")
    missing = joined["anomaly"].null_count()
    gap = float(np.abs((joined["anomaly"] - joined["anomaly_mean_calibrated"]).to_numpy()).max())
    print(
        f"check 1 (experiment 00074, {other.height:,} rows to {other['valid_time'].max()}): "
        f"{missing} rows missing, largest difference {gap:.2e}"
    )
    if missing or gap > SLOPE_TOLERANCE:
        msg = "the rebuilt anomalies do not reproduce experiment 00074's train_shape"
        raise ValueError(msg)
    ids = dict(zip(offset_series()["site"], offset_series()["time_series_id"], strict=True))
    worst, compared = 0.0, 0
    for path in sorted((E46_SOLAR_DIR / "ens_val").glob("init=*.parquet")):
        stored = (
            pl.read_parquet(path, columns=["site", "valid_time", "anomaly_mean_calibrated"])
            .filter(pl.col("site").is_in(list(OFFSET_SITES)))
            .unique(["site", "valid_time"])
            .with_columns(time_series_id=pl.col("site").replace_strict(ids, return_dtype=pl.Int32))
        )
        origin = common.origin_of(
            run=datetime.fromisoformat(path.stem.removeprefix("init=")).replace(tzinfo=UTC)
        )
        mine = table.filter(pl.col("power_fcst_init_time") == origin)
        both = stored.join(mine, on=["time_series_id", "valid_time"], how="left", validate="1:1")
        if both["anomaly"].null_count():
            msg = f"{path.name}: rebuilt anomalies miss {both['anomaly'].null_count()} rows"
            raise ValueError(msg)
        difference = (both["anomaly"] - both["anomaly_mean_calibrated"]).to_numpy()
        worst = max(worst, float(np.abs(difference).max()))
        compared += both.height
    print(
        f"check 2 (experiment 00046 validation, {compared:,} rows): largest difference {worst:.2e}"
    )
    if worst > SLOPE_TOLERANCE:
        msg = "the rebuilt validation anomalies do not reproduce experiment 00046's"
        raise ValueError(msg)


def weights_per_row(
    *, weights: pl.DataFrame, site: str, band: np.ndarray, month: np.ndarray
) -> np.ndarray:
    """Return each row's fitted weight of experiment 00064: zero outside April to September.

    Args:
        weights: Experiment 00064's weights table, the `all` variant.
        site: The series label.
        band: Each row's lead-band index.
        month: Each row's local calendar month.

    Returns:
        The applied-shift slope of the row's lead band and season, zero in the 10+ day band.

    Raises:
        ValueError: If the site does not hold one weight per lead band and season.
    """
    cells = weights.filter(pl.col("site") == site)
    if cells.height != len(common.LEAD_BANDS) * 2:
        msg = f"{site} has {cells.height} weight cells, expected {len(common.LEAD_BANDS) * 2}"
        raise ValueError(msg)
    frame = (
        pl.DataFrame({"band": band.astype(np.int64), "month": month.astype(np.int64)})
        .with_columns(
            season=pl.col("month").replace_strict(
                SEASON_OF_MONTH, default=None, return_dtype=pl.String
            )
        )
        .join(
            cells.select(pl.col("band").cast(pl.Int64), "season", "slope_applied"),
            on=["band", "season"],
            how="left",
            validate="m:1",
            maintain_order="left",
        )
    )
    slope = frame["slope_applied"].fill_null(0.0).to_numpy()
    return np.where(band >= NO_SKILL_BAND, 0.0, slope)


def monday_week_index(*, day: date) -> int:
    """Return the week index (experiment 00045's) of the Monday on or before `day`."""
    monday = day - timedelta(days=day.weekday())
    return (monday - E45_FIRST_WEEK_START).days // 7


def vintage_capacity_table() -> pl.DataFrame:
    """Return experiment 00078's season-year capacity (MW) per offset series and `origin_week`.

    Returns:
        Columns `site`, `origin_week`, and `capacity_mw`, one row per pair.

    Raises:
        ValueError: If one `origin_week` holds two different capacities for a series, or the
            table's `origin_week` is not the week index of its `run_date` minus one.
    """
    table = pl.read_parquet(VINTAGES_FILE).filter(pl.col("site").is_in(list(OFFSET_SITES)))
    expected = table["run_date"].map_elements(
        lambda day: monday_week_index(day=day) - 1, return_dtype=pl.Int64
    )
    if (expected != table["origin_week"]).any():
        msg = "the vintage table's origin_week is not the run's week index minus one"
        raise ValueError(msg)
    per_week = table.group_by("site", "origin_week").agg(
        capacity_mw=pl.col("season_year_mw").first(), distinct=pl.col("season_year_mw").n_unique()
    )
    if (per_week["distinct"] > 1).any():
        msg = "a vintage origin_week holds two different capacities"
        raise ValueError(msg)
    return per_week.select("site", "origin_week", "capacity_mw").sort("site", "origin_week")


@dataclass(frozen=True)
class SiteOffset:
    """One offset series' weights and capacities.

    Attributes:
        site: The series label.
        weights: Experiment 00064's weights table, the `all` variant.
        effective_capacity_mw: The series' effective capacity, the label's denominator.
        held_capacity_mw: Experiment 00064's capacity, fitted before 2025-07-01.
        vintage_mw: The capacity vintage per `origin_week`, from experiment 00078.
    """

    site: str
    weights: pl.DataFrame
    effective_capacity_mw: float
    held_capacity_mw: float
    vintage_mw: dict[int, float]

    def scale_of_run(self, *, origin: datetime, basis: str) -> float:
        """Return capacity over effective capacity for one run.

        Args:
            origin: The run's `power_fcst_init_time`.
            basis: `held` for experiment 00064's capacity, `vintage` for experiment 00078's.

        Returns:
            The factor that turns an anomaly into the offset.

        Raises:
            ValueError: If `basis` is unknown, or the table holds no vintage at or before the run's
                `origin_week`. A run after the table's last `origin_week` takes the last vintage.
        """
        if basis == "held":
            return self.held_capacity_mw / self.effective_capacity_mw
        if basis != "vintage":
            msg = f"unknown capacity basis {basis!r}"
            raise ValueError(msg)
        wanted = monday_week_index(day=origin.date()) - 1
        earlier = [week for week in self.vintage_mw if week <= wanted]
        if not earlier:
            msg = f"{self.site} has no capacity vintage at or before origin_week {wanted}"
            raise ValueError(msg)
        usable = max(earlier)
        return self.vintage_mw[usable] / self.effective_capacity_mw


def site_offset(*, site: str) -> SiteOffset:
    """Load one offset series' weights and capacities.

    Args:
        site: One of `OFFSET_SITES`.

    Returns:
        The series' `SiteOffset`.
    """
    row = offset_series().filter(pl.col("site") == site).row(0, named=True)
    vintages = vintage_capacity_table().filter(pl.col("site") == site)
    return SiteOffset(
        site=site,
        weights=pl.read_parquet(WEIGHTS_FILE).filter(pl.col("variant") == "all"),
        effective_capacity_mw=float(row["effective_capacity_mw"]),
        held_capacity_mw=float(
            pl.read_parquet(CAPACITY_FILE).filter(pl.col("site") == site)["capacity_mw"][0]
        ),
        vintage_mw=dict(zip(vintages["origin_week"], vintages["capacity_mw"], strict=True)),
    )


def offset_shift(
    *, offset: SiteOffset, frame: pl.DataFrame, basis: str, anomaly: np.ndarray
) -> np.ndarray:
    """Return `w * scale * anomaly` for each row, as a fraction of effective capacity.

    Args:
        offset: The series' offset parameters.
        frame: Rows with `valid_time` and `power_fcst_init_time`.
        basis: `held` or `vintage` (see `SiteOffset.scale_of_run`). Training rows use `held`.
        anomaly: The rows' calibrated ensemble-mean anomaly.

    Returns:
        A float64 vector, zero in the 10+ day band and outside April to September.
    """
    lead = (
        (frame["valid_time"] - frame["power_fcst_init_time"]).dt.total_minutes() / 60.0
    ).to_numpy()
    month = frame["valid_time"].dt.convert_time_zone("Europe/London").dt.month().to_numpy()
    weight = weights_per_row(
        weights=offset.weights, site=offset.site, band=lead_band(lead_hours=lead), month=month
    )
    if basis == "held":
        scale = np.full(frame.height, offset.held_capacity_mw / offset.effective_capacity_mw)
    else:
        origins = frame["power_fcst_init_time"]
        per_origin = {
            o: offset.scale_of_run(origin=o, basis=basis) for o in origins.unique().to_list()
        }
        scale = origins.replace_strict(per_origin, return_dtype=pl.Float64).to_numpy()
    return weight * scale * anomaly.astype(np.float64)


def margin(*, level: float, shift: np.ndarray) -> np.ndarray:
    """Return the base margin, shape (rows, 11): the training labels' median plus the offset.

    Args:
        level: The median of the refit's training labels (fractions of effective capacity).
        shift: The rows' offsets from `offset_shift`.

    Returns:
        A float32 array with the same value in each of the 11 quantile columns.
    """
    return np.tile((level + shift)[:, None], (1, len(common.LEVELS))).astype(np.float32)


def attach_anomaly(*, frame: pl.DataFrame, table: pl.DataFrame) -> np.ndarray:
    """Return the anomaly of each row of `frame`, in the frame's order.

    Args:
        frame: Rows keyed by `common.KEYS` for one series.
        table: `anomaly_table()` filtered to the series.

    Returns:
        A float64 vector.

    Raises:
        ValueError: If a row has no anomaly.
    """
    joined = frame.select(*common.KEYS).join(
        table, on=list(common.KEYS), how="left", validate="m:1", maintain_order="left"
    )
    if joined["anomaly"].null_count():
        msg = f"{joined['anomaly'].null_count()} rows have no solar anomaly"
        raise ValueError(msg)
    return joined["anomaly"].to_numpy()


def main() -> None:
    """Build the requested part."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("u_mean", "anomaly", "capacity"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="build only the first N runs")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.part == "u_mean":
        build_u_means(workers=args.workers, limit=args.limit)
    elif args.part == "anomaly":
        table = anomaly_table()
        print(
            f"anomaly table: {table.height:,} rows, latest valid time {table['valid_time'].max()}"
        )
        if args.check:
            check_anomaly(table=table)
        common.atomic_write(frame=table, target=ANOMALY_FILE)
    else:
        print(vintage_capacity_table().pivot(on="site", index="origin_week", values="capacity_mw"))
        for site in OFFSET_SITES:
            offset = site_offset(site=site)
            print(
                f"{site}: held {offset.held_capacity_mw:.3f} MW, "
                f"vintage {min(offset.vintage_mw.values()):.3f} to "
                f"{max(offset.vintage_mw.values()):.3f} MW"
            )


if __name__ == "__main__":
    main()
