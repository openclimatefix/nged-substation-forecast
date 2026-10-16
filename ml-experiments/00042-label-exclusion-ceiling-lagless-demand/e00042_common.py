"""Paths, series, grids, labels, exclusion masks, and random blocks shared by experiment 00042.

The experiment asks whether leaving labelled switching out of the training target lowers the
lagless demand model's error
(`docs/ml-experiments/00042-label-exclusion-ceiling-lagless-demand/result.md`). Grid, series, and
feature helpers are copied from experiment 00033's `e00033_common.py`; an experiment may not import
another experiment.

Running this file writes `work/exclusion_report.md`: the event counts and the removed-row and
removed-weight shares of every exclusion, printed before any fit is scored.

    uv run python ml-experiments/00042-*/e00042_common.py
"""

import json
from datetime import UTC, date, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Final, Literal

import holidays
import numpy as np
import polars as pl
from studies.power import scan_power
from studies.solar import azimuth, zenith
from studies.sources import ML_EXPERIMENTS_DATA_DIR, study_dir_for

EXPERIMENT_ID: Final[int] = 42
SLUG: Final[str] = "label-exclusion-ceiling-lagless-demand"
ROW_SET_ID: Final[str] = "fold_a_every5d_v1"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
N_RANDOM_DRAWS: Final[int] = 6
"""Random draws per label set; draw `i` is fitted with model seed `i % len(SEEDS)`."""

SourceType = Literal["era5", "ens"]
LabelSetType = Literal["trusted", "allvis"]
SOURCES: Final[tuple[SourceType, ...]] = ("era5", "ens")
LABEL_SETS: Final[tuple[LabelSetType, ...]] = ("trusted", "allvis")

WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables and predictions; private and never committed."""
FITS_DIR: Final[Path] = WORK_DIR / "fits"
VAL_INPUTS_DIR: Final[Path] = WORK_DIR / "val_inputs"
ARMS_DIR_00013: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00013-primary-demand-against-stored-baselines" / "work" / "arms"
)
"""Experiment 00013's stored `climatology` and `manual_heuristic` tables, read only."""
LABELS_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00041-sw2-0a-labeller-gate" / "real" / "labelled_set.parquet"
)
"""Experiment 00041's labelled set, read only."""

DEMAND_DIR: Final[Path] = study_dir_for(study="stage1_pilot") / "demand"
"""Experiment 00003's inputs, read only."""
SERIES_PATH: Final[Path] = DEMAND_DIR / "inputs" / "series.parquet"
ERA5_PATH: Final[Path] = DEMAND_DIR / "inputs" / "era5_features.parquet"
OBSERVED_PATH: Final[Path] = DEMAND_DIR / "inputs" / "observed.parquet"
VALIDATION_DIR: Final[Path] = DEMAND_DIR / "ens_every5d" / "val"
ENS_TRAIN_PATH: Final[Path] = DEMAND_DIR / "ens_every5d" / "train.parquet"
FEATURE_SET_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00005-lag-placement-stage1b-vs-stage2-demand"
    / "work"
    / "feature_set.json"
)
"""Experiment 00006's final Primary column list, as experiment 00005 froze it; read only."""

DROPPED_SITES: Final[frozenset[str]] = frozenset({"D7", "D20"})
SUBSTATION_TYPES: Final[frozenset[str]] = frozenset({"Primary", "BSP", "GSP"})
GRID_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
GRID_END: Final[datetime] = datetime(2026, 7, 31, 23, 30, tzinfo=UTC)
STEP: Final[timedelta] = timedelta(minutes=30)
LAST_TRAINING_VALID: Final[datetime] = datetime(2025, 6, 30, 23, 30, tzinfo=UTC)
VALIDATION_START: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
HOLDOUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
TRAINING_WINDOW_START: Final[dict[SourceType, datetime]] = {
    "era5": datetime(2022, 1, 1, tzinfo=UTC),
    "ens": datetime(2024, 4, 1, tzinfo=UTC),
}
"""First training valid time (inclusive) of each source."""
LEVELS: Final[np.ndarray] = np.array([0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95])
LEVEL_COLUMNS: Final[tuple[str, ...]] = (
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
)
ATOM_MASSES: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (LEVELS[:-1] + LEVELS[1:]) / 2, [1.0]])
)
"""Probability mass of each level: the interval between the mid-points to its neighbours."""
KEYS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")
MEDIAN: Final[int] = 4

DERIVED: Final[tuple[str, ...]] = (
    "solar_zenith",
    "solar_azimuth",
    "hour_local",
    "day_of_week_local",
    "covid_indicator",
    "days_to_nearest_holiday",
    "days_to_christmas",
)
LOCKDOWN_START: Final[datetime] = datetime(2020, 3, 23, tzinfo=UTC)
LOCKDOWN_END: Final[datetime] = datetime(2021, 7, 19, tzinfo=UTC)
HOLIDAY_YEARS: Final[range] = range(2018, 2029)
HOLIDAY_CLIP_DAYS: Final[int] = 7
CHRISTMAS_CLIP_DAYS: Final[int] = 14
XGBOOST_PARAMETERS: Final[dict[str, object]] = {
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "colsample_bynode": 1.0,
    "objective": "reg:quantileerror",
    "quantile_alpha": LEVELS,
    "tree_method": "hist",
    "device": "cuda",
    "nthread": 2,
}
"""Experiment 00033's N1 parameters with the slot's 2 threads per worker."""
N_ROUNDS: Final[int] = 500
RECENCY_HALF_LIFE_DAYS: Final[float] = 2 * 365.25
LEAD_STEPS: Final[int] = 702
"""Half-hourly leads per daily run in the training layout: 0.5 h to 351 h."""
RUN_HOUR: Final[int] = 9
"""Hour (UTC) of the daily training runs' forecast initialisation."""

TRUSTED_MIN_SHARE: Final[float] = 0.25
TRUSTED_MIN_DURATION: Final[timedelta] = timedelta(days=1)
TRUSTED_MAX_MISSING: Final[float] = 0.5
LOAD_HISTORY: Final[timedelta] = timedelta(days=365)
MAX_PLACEMENT_ATTEMPTS: Final[int] = 20_000
PLACEMENT_TIERS: Final[tuple[tuple[float, bool], ...]] = ((0.10, True), (0.25, True), (0.25, False))
"""Placement rules tried in order: (allowed relative gap in finite readings removed, whether the
block must start in the calendar month of the real event's onset)."""


def feature_columns() -> list[str]:
    """Return stage 1a's input columns from the frozen sweep set (weather, then derived)."""
    columns: list[str] = json.loads(FEATURE_SET_PATH.read_text())["columns"]
    return columns


def input_columns() -> list[str]:
    """Return the lagless model's input columns in matrix order: the features, then lead hours."""
    return [*feature_columns(), "lead_hours"]


def series_table() -> pl.DataFrame:
    """Return the 20 substation series with label, group, capacity, coordinates, and H3 cell.

    The coordinates never leave this process.

    Returns:
        One row per series with `time_series_id`, `site` (D1 to D22), `group`, `substation_type`,
        `latitude`, `longitude`, `h3_res_5`, and `effective_capacity_mw`.

    Raises:
        ValueError: If the table does not hold 20 series or a series is not a substation.
    """
    table = pl.read_parquet(SERIES_PATH)
    metadata = pl.read_parquet(
        study_dir_for(study="quick_wins_fixed_pipeline_1156")
        / "frames"
        / "combined"
        / "fold_a"
        / "metadata.parquet",
        columns=["time_series_id", "substation_type", "h3_res_5"],
    )
    kept = (
        table.filter(~pl.col("site").is_in(DROPPED_SITES))
        .join(metadata, on="time_series_id", how="left", validate="1:1")
        .with_columns(pl.col("substation_type").cast(pl.String))
    )
    if kept.height != 20 or not set(kept["substation_type"].to_list()) <= SUBSTATION_TYPES:
        msg = "expected the 20 Primary, BSP, and GSP series"
        raise ValueError(msg)
    return kept.sort(pl.col("site").str.slice(1).cast(pl.Int32))


def grid_times() -> pl.Series:
    """Return the dense half-hourly grid every per-series array is indexed on."""
    return pl.datetime_range(
        GRID_START, GRID_END, interval="30m", time_unit="us", time_zone="UTC", eager=True
    )


@cache
def grid_months() -> np.ndarray:
    """Return the calendar month (1 to 12) of every grid stamp."""
    return grid_times().dt.month().to_numpy()


def grid_index(*, times: datetime | pl.Series) -> int | np.ndarray:
    """Return the grid index of one timestamp, or of a series of timestamps."""
    if isinstance(times, datetime):
        return int((times - GRID_START) / STEP)
    return ((times.dt.epoch("us") - int(GRID_START.timestamp() * 1e6)) // 1_800_000_000).to_numpy()


def take(*, values: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Return `values[index]`, NaN where the index falls off either end of the grid."""
    inside = (index >= 0) & (index < values.size)
    out = np.full(index.shape, np.nan, dtype=np.float32)
    out[inside] = values[index[inside]]
    return out


def holiday_gaps() -> pl.DataFrame:
    """Return, for each calendar date, the signed days to the nearest holiday and to Christmas."""
    calendar = holidays.country_holidays("GB", subdiv="ENG", years=HOLIDAY_YEARS)
    holiday_dates = np.array(sorted(calendar), dtype="datetime64[D]")
    days = pl.date_range(
        date(HOLIDAY_YEARS.start, 1, 1), date(HOLIDAY_YEARS.stop - 1, 12, 31), eager=True
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
            "days_to_christmas": np.where(
                np.abs(nearest_christmas) <= CHRISTMAS_CLIP_DAYS, nearest_christmas, np.nan
            ),
        }
    )


def derived_columns(*, valid_time: pl.Series, latitude: float, longitude: float) -> pl.DataFrame:
    """Return the timestamp-and-location columns of stage 1a for period-ending UTC stamps."""
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
        frame.join(holiday_gaps(), on="date", how="left", validate="m:1", maintain_order="left")
        .select(DERIVED)
        .cast(dict.fromkeys(DERIVED, pl.Float32))
    )


def with_derived(*, frame: pl.DataFrame, latitude: float, longitude: float) -> pl.DataFrame:
    """Add the `DERIVED` columns to a frame with a `valid_time` column, replacing any it has."""
    times = frame["valid_time"].unique().sort()
    table = derived_columns(valid_time=times, latitude=latitude, longitude=longitude).with_columns(
        valid_time=times
    )
    return frame.drop(DERIVED, strict=False).join(
        table, on="valid_time", how="left", validate="m:1", maintain_order="left"
    )


class SeriesArrays:
    """One series' arrays on the dense half-hourly grid.

    Attributes:
        features: Stage 1a's input columns from ERA5 and the calendar, float32, one row per grid
            half-hour, in `feature_columns()` order.
        power: Observed power over effective capacity, NaN where unobserved.
    """

    def __init__(self, *, features: np.ndarray, power: np.ndarray) -> None:
        """Store the two arrays."""
        self.features = features
        self.power = power


def load_series_arrays(*, series: pl.DataFrame) -> dict[str, SeriesArrays]:
    """Return each series' ERA5 features, calendar, and observed power on the dense grid."""
    columns = feature_columns()
    weather = [c for c in columns if c not in DERIVED]
    grid = pl.DataFrame({"valid_time": grid_times()})
    era5 = pl.read_parquet(ERA5_PATH)
    observed = pl.read_parquet(OBSERVED_PATH, columns=["time_series_id", "valid_time", "power"])
    out: dict[str, SeriesArrays] = {}
    for row in series.iter_rows(named=True):
        joined = (
            grid.join(
                era5.filter(pl.col("time_series_id") == row["time_series_id"]).select(
                    "valid_time", *weather
                ),
                on="valid_time",
                how="left",
                validate="1:1",
            )
            .join(
                observed.filter(pl.col("time_series_id") == row["time_series_id"]).select(
                    "valid_time", pl.col("power") / row["effective_capacity_mw"]
                ),
                on="valid_time",
                how="left",
                validate="1:1",
            )
            .sort("valid_time")
        )
        derived = with_derived(
            frame=joined.select("valid_time"),
            latitude=row["latitude"],
            longitude=row["longitude"],
        )
        matrix = (
            joined.select(weather)
            .with_columns(derived.select(DERIVED))
            .select(columns)
            .to_numpy()
            .astype(np.float32)
        )
        out[row["site"]] = SeriesArrays(
            features=matrix, power=joined["power"].to_numpy().astype(np.float32)
        )
    return out


def pinball_quantile_score(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return twice the mass-weighted mean pinball loss of the 9 levels, per case."""
    difference = actual[:, None] - quantiles
    pinball = np.where(difference >= 0, LEVELS * difference, (LEVELS - 1) * difference)
    return 2.0 * (pinball * ATOM_MASSES).sum(axis=1)


# ---------------------------------------------------------------------------------------------
# Training rows
# ---------------------------------------------------------------------------------------------


def training_layout_era5() -> pl.DataFrame:
    """Return the run layout of the ERA5-trained source: daily runs, 702 half-hourly leads each.

    The layout matches the ENS control member's training rows (a run every day at 09:00 UTC,
    leads 0.5 h to 351 h) and extends it back so that valid times start at 2022-01-01.

    Returns:
        Columns `power_fcst_init_time`, `valid_time` (both UTC) and `lead_hours` (Float64), for
        valid times from the source's first training stamp to `LAST_TRAINING_VALID`.
    """
    first_run = TRAINING_WINDOW_START["era5"].replace(hour=RUN_HOUR) - timedelta(days=16)
    last_run = LAST_TRAINING_VALID.replace(hour=RUN_HOUR, minute=0)
    runs = pl.datetime_range(first_run, last_run, interval="1d", time_zone="UTC", eager=True)
    leads = np.arange(1, LEAD_STEPS + 1) * 0.5
    frame = (
        pl.DataFrame({"power_fcst_init_time": runs})
        .join(pl.DataFrame({"lead_hours": leads}), how="cross")
        .with_columns(
            valid_time=pl.col("power_fcst_init_time")
            + pl.duration(minutes=(pl.col("lead_hours") * 60).cast(pl.Int64))
        )
    )
    return frame.filter(
        (pl.col("valid_time") >= TRAINING_WINDOW_START["era5"])
        & (pl.col("valid_time") <= LAST_TRAINING_VALID)
    )


def training_frame(*, source: SourceType, row: dict, arrays: SeriesArrays) -> pl.DataFrame:
    """Return one series' training rows for a source: valid time, lead, label, and inputs.

    The ERA5 source takes every weather column from ERA5 at the valid time and uses the run
    layout of `training_layout_era5`. The ENS source takes the ENS control member's rows as stored.
    Rows without finite observed power are dropped.

    Args:
        source: Which source's rows to build.
        row: The series' `series_table()` row.
        arrays: The series' grid arrays.

    Returns:
        Columns `valid_time`, `label` (power over effective capacity, Float32),
        then `input_columns()` (Float32), one row per run and valid time.
    """
    columns = input_columns()
    if source == "era5":
        layout = training_layout_era5()
        index = np.asarray(grid_index(times=layout["valid_time"]))
        label = take(values=arrays.power, index=index)
        features = arrays.features[index]
        frame = layout.select("valid_time", "lead_hours").with_columns(
            pl.Series("label", label),
            *[
                pl.Series(name, features[:, position])
                for position, name in enumerate(feature_columns())
            ],
        )
    else:
        stored = (
            pl.scan_parquet(ENS_TRAIN_PATH)
            .filter(
                (pl.col("time_series_id") == row["time_series_id"])
                & (pl.col("valid_time") <= LAST_TRAINING_VALID)
                & (pl.col("valid_time") > pl.col("power_fcst_init_time"))
            )
            .collect()
        )
        stored = with_derived(frame=stored, latitude=row["latitude"], longitude=row["longitude"])
        frame = stored.select(
            "valid_time", *feature_columns(), "power", "power_fcst_init_time"
        ).with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0,
            label=pl.col("power") / row["effective_capacity_mw"],
        )
    return (
        frame.filter(pl.col("label").is_finite())
        .select("valid_time", "label", *columns)
        .with_columns(pl.col("label", *columns).cast(pl.Float32))
    )


def recency_weights(*, valid_time: pl.Series) -> np.ndarray:
    """Return the half-life weights of training rows, anchored at `LAST_TRAINING_VALID`."""
    age_days = (LAST_TRAINING_VALID - valid_time).dt.total_seconds().to_numpy() / 86_400
    return 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)


# ---------------------------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------------------------


def _as_series_id(value: str | float | None) -> int | None:
    """Return a label file's series reference (a string, an int, or missing) as an int."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return int(value)


@cache
def labelled_rows() -> pl.DataFrame:
    """Return every real labelled interval of any visibility, with the series it names.

    Returns:
        One row per labelled row whose kind starts with `real_` and that has an onset and an end:
        `card_id`, `visibility`, `onset_utc`, `end_utc`, and `series` (list of `time_series_id`).
    """
    labels = pl.read_parquet(LABELS_PATH).filter(
        pl.col("kind").str.starts_with("real_") & pl.col("onset_utc").is_not_null()
    )
    rows = []
    for record in labels.iter_rows(named=True):
        named = [record["side_a"], record["side_b"], record["donor"]]
        named += [item["series"] for item in json.loads(record["receivers"])]
        ids = sorted({i for i in map(_as_series_id, named) if i is not None})
        rows.append(
            {
                "card_id": record["card_id"],
                "visibility": record["visibility"],
                "onset_utc": record["onset_utc"],
                "end_utc": record["end_utc"],
                "series": ids,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "card_id": pl.String,
            "visibility": pl.String,
            "onset_utc": pl.Datetime("us", "UTC"),
            "end_utc": pl.Datetime("us", "UTC"),
            "series": pl.List(pl.Int64),
        },
    )


def receiver_load_shares(*, labels: pl.DataFrame) -> dict[str, float]:
    """Return each event's moved load as a share of its receivers' mean load before onset.

    The share is the sum over receivers of `share * moved_mw_p50 / mean power of the receiver over
    the 365 days before onset`, so the stored `moved_share_of_receiver` (missing for most events)
    is not used. An event with no receiver, or a receiver without readings, has no share (NaN).

    Args:
        labels: Experiment 00041's labelled rows (the `real_operation` rows that are visible).

    Returns:
        `card_id` mapped to the share.
    """
    receiver_ids = {
        int(i)
        for text in labels["receivers"].to_list()
        for item in json.loads(text)
        if (i := _as_series_id(item["series"])) is not None
    }
    power = (
        scan_power()
        .filter(pl.col("time_series_id").is_in(sorted(receiver_ids)))
        .select("time_series_id", "time", "power")
        .collect()
    )
    shares: dict[str, float] = {}
    for record in labels.iter_rows(named=True):
        total, found = 0.0, False
        for item in json.loads(record["receivers"]):
            series_id = _as_series_id(item["series"])
            onset = record["onset_utc"]
            history = power.filter(
                (pl.col("time_series_id") == series_id)
                & (pl.col("time") >= onset - LOAD_HISTORY)
                & (pl.col("time") < onset)
            )["power"].to_numpy()
            mean_load = float(history.mean()) if history.size else float("nan")
            if not mean_load > 0:
                continue
            total += float(item["share"]) * record["moved_mw_p50"] / mean_load
            found = True
        shares[record["card_id"]] = total if found else float("nan")
    return shares


@cache
def visible_events() -> pl.DataFrame:
    """Return the visible real operations with the recomputed share and the label-set flags.

    Returns:
        One row per `real_operation` event labelled `conserved` or `one_sided`: `card_id`,
        `visibility`, `onset_utc`, `end_utc`, `duration_days`, `series` (list of `time_series_id`),
        `moved_share` (recomputed), `stored_donor_range_share` (experiment 00041's), `max_missing`
        (`max_side_missing_share`), and `trusted` (a flag).
    """
    labels = pl.read_parquet(LABELS_PATH).filter(
        (pl.col("kind") == "real_operation")
        & pl.col("visibility").is_in(["conserved", "one_sided"])
    )
    shares = receiver_load_shares(labels=labels)
    named = labelled_rows().select("card_id", "series")
    events = (
        labels.join(named, on="card_id", how="left", validate="1:1")
        .with_columns(
            moved_share=pl.col("card_id").replace_strict(shares, return_dtype=pl.Float64),
            duration_days=(pl.col("end_utc") - pl.col("onset_utc")).dt.total_seconds() / 86_400,
        )
        .select(
            "card_id",
            "visibility",
            "onset_utc",
            "end_utc",
            "duration_days",
            "series",
            "moved_share",
            stored_donor_range_share=pl.col("moved_share_of_donor_range"),
            max_missing=pl.col("max_side_missing_share"),
        )
    )
    return events.with_columns(
        trusted=(
            pl.col("moved_share").is_not_nan()
            & (pl.col("moved_share") >= TRUSTED_MIN_SHARE)
            & (pl.col("duration_days") > TRUSTED_MIN_DURATION.total_seconds() / 86_400)
            & (pl.col("max_missing") <= TRUSTED_MAX_MISSING)
        ).fill_null(value=False)
    )


def events_of(*, label_set: LabelSetType) -> pl.DataFrame:
    """Return the events of a label set."""
    events = visible_events()
    return events.filter(pl.col("trusted")) if label_set == "trusted" else events


# ---------------------------------------------------------------------------------------------
# Masks on the grid
# ---------------------------------------------------------------------------------------------


def interval_indices(*, onset: datetime, end: datetime) -> tuple[int, int]:
    """Return the grid indices `[first, last)` of the half-hours that overlap `[onset, end]`.

    A grid stamp `t` labels the half-hour ending at `t`, so the half-hour overlaps the interval
    when `t > onset` and `t - 30 min < end`.
    """
    first = (onset - GRID_START) // STEP + 1
    last = -((GRID_START - end) // STEP) + 1
    return int(first), int(last)


def window_indices(*, source: SourceType) -> tuple[int, int]:
    """Return the grid indices `[first, last)` of a source's training window."""
    return (
        int(grid_index(times=TRAINING_WINDOW_START[source])),
        int(grid_index(times=LAST_TRAINING_VALID)) + 1,
    )


def mask_of(
    *, intervals: list[tuple[datetime, datetime]], clip: tuple[int, int] | None
) -> np.ndarray:
    """Return a grid mask of the half-hours overlapping any interval, clipped to `clip`."""
    mask = np.zeros(grid_months().size, dtype=bool)
    for onset, end in intervals:
        first, last = interval_indices(onset=onset, end=end)
        if clip is not None:
            first, last = max(first, clip[0]), min(last, clip[1])
        if last > first:
            mask[first:last] = True
    return mask


def real_exclusion_mask(
    *, source: SourceType, label_set: LabelSetType, series_id: int
) -> np.ndarray:
    """Return the grid mask of a label set's training exclusion for one series."""
    events = events_of(label_set=label_set).filter(pl.col("series").list.contains(series_id))
    intervals = list(zip(events["onset_utc"].to_list(), events["end_utc"].to_list(), strict=True))
    return mask_of(intervals=intervals, clip=window_indices(source=source))


def switching_mask(*, series_id: int) -> np.ndarray:
    """Return the grid mask of the half-hours inside a visible labelled interval of the series.

    Every `conserved` or `one_sided` real operation that names the series counts, whatever its
    size, duration, or missing share. Scoring uses the part of the mask in the validation year.
    """
    events = visible_events().filter(pl.col("series").list.contains(series_id))
    intervals = list(zip(events["onset_utc"].to_list(), events["end_utc"].to_list(), strict=True))
    return mask_of(intervals=intervals, clip=None)


@cache
def observed_mask(*, series_id: int) -> np.ndarray:
    """Return a grid mask of the half-hours where the series has a finite observed power."""
    observed = pl.read_parquet(OBSERVED_PATH, columns=["time_series_id", "valid_time", "power"])
    own = observed.filter((pl.col("time_series_id") == series_id) & pl.col("power").is_finite())
    mask = np.zeros(grid_months().size, dtype=bool)
    index = np.asarray(grid_index(times=own["valid_time"]))
    mask[index[(index >= 0) & (index < mask.size)]] = True
    return mask


def blocks_of(*, mask: np.ndarray) -> list[tuple[int, int]]:
    """Return the `(start, length)` of every maximal run of true values in a mask."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(int(s), int(e - s)) for s, e in zip(starts, ends, strict=True)]


def _closest_placement(
    *,
    counts: np.ndarray,
    first: int,
    length: int,
    wanted: int,
    blocked: np.ndarray,
    generator: np.random.Generator,
) -> int:
    """Return a start whose block is clear of `blocked` and removes nearly `wanted` readings.

    Picks at random among the (up to) 100 clear starts whose finite count is closest to `wanted`.

    Raises:
        RuntimeError: If no clear start exists.
    """
    cleared = np.concatenate([[0], np.cumsum(blocked)])
    starts = np.arange(first, first + counts.size)
    clear = (cleared[starts + length] - cleared[starts]) == 0
    if not clear.any():
        msg = f"no clear block of {length} half-hours exists"
        raise RuntimeError(msg)
    order = np.argsort(np.abs(counts[clear] - wanted), kind="stable")[:100]
    return int(generator.choice(starts[clear][order]))


def block_onset_months(
    *, source: SourceType, label_set: LabelSetType, series_id: int, blocks: list[tuple[int, int]]
) -> list[int]:
    """Return, per block, the calendar month of the earliest real event onset overlapping it."""
    events = events_of(label_set=label_set).filter(pl.col("series").list.contains(series_id))
    spans = [
        (interval_indices(onset=o, end=e), o.month)
        for o, e in zip(events["onset_utc"].to_list(), events["end_utc"].to_list(), strict=True)
    ]
    months = []
    for start, length in blocks:
        overlapping = [
            (o_month, first)
            for (first, last), o_month in spans
            if first < start + length and last > start
        ]
        months.append(min(overlapping, key=lambda item: item[1])[0])
    return months


def draw_random_mask(
    *, source: SourceType, label_set: LabelSetType, series_id: int, draw: int
) -> np.ndarray:
    """Return a random exclusion mask with the real exclusion's block lengths.

    Each maximal block of the real mask (clipped to the training window; overlapping events merge
    into one block) is placed at a random start inside the window, in the calendar month of the
    onset of the earliest real event that overlaps the block. A placed block overlaps no other
    placed block and no labelled interval of any visibility that names the series, lies wholly
    inside the window, and removes a number of finite observed readings within 10% of the real
    block's (two-sided), widened to 25% only when no placement meets 10%, and then (last
    resort, reported on the console) allowed in any month. Rejection sampling uses
    a generator seeded by the draw number and the series.

    Args:
        source: The training source, which sets the window.
        label_set: Which label set's blocks to copy.
        series_id: The series' `time_series_id`.
        draw: The draw number, 0 to `N_RANDOM_DRAWS - 1`.

    Returns:
        A boolean grid mask.

    Raises:
        RuntimeError: If a block cannot be placed within `MAX_PLACEMENT_ATTEMPTS` attempts at
            either tolerance.
    """
    real = real_exclusion_mask(source=source, label_set=label_set, series_id=series_id)
    first, last = window_indices(source=source)
    labelled = labelled_rows().filter(pl.col("series").list.contains(series_id))
    forbidden = mask_of(
        intervals=list(
            zip(labelled["onset_utc"].to_list(), labelled["end_utc"].to_list(), strict=True)
        ),
        clip=None,
    )
    placed = np.zeros_like(real)
    generator = np.random.default_rng([draw, series_id])
    months = grid_months()
    finite = observed_mask(series_id=series_id).astype(np.int64)
    cumulative = np.concatenate([[0], np.cumsum(finite)])
    blocks = blocks_of(mask=real)
    onset_months = block_onset_months(
        source=source, label_set=label_set, series_id=series_id, blocks=blocks
    )
    order = sorted(range(len(blocks)), key=lambda k: (-blocks[k][1], blocks[k][0]))
    for k in order:
        start, length = blocks[k]
        wanted = int(cumulative[start + length] - cumulative[start])
        candidates = np.flatnonzero(months[first : last - length + 1] == onset_months[k]) + first
        anywhere = np.arange(first, last - length + 1)
        for tier, (tolerance, same_month) in enumerate(PLACEMENT_TIERS):
            pool = candidates if same_month else anywhere
            for _ in range(MAX_PLACEMENT_ATTEMPTS):
                position = int(generator.choice(pool))
                stop = position + length
                count = int(cumulative[stop] - cumulative[position])
                if (
                    abs(count - wanted) <= tolerance * wanted
                    and not forbidden[position:stop].any()
                    and not placed[position:stop].any()
                ):
                    placed[position:stop] = True
                    if tier == len(PLACEMENT_TIERS) - 1:
                        print(
                            f"series {series_id} draw {draw}: block of {length} half-hours "
                            "placed in any month",
                            flush=True,
                        )
                    break
            else:
                continue
            break
        else:
            position = _closest_placement(
                counts=cumulative[first + length : last + 1]
                - cumulative[first : last - length + 1],
                first=first,
                length=length,
                wanted=wanted,
                blocked=forbidden | placed,
                generator=generator,
            )
            placed[position : position + length] = True
            achieved = int(cumulative[position + length] - cumulative[position])
            print(
                f"series {series_id} draw {draw}: no placement within 25% of the real block's "
                f"{wanted} finite readings; used the closest ({achieved}) in any month",
                flush=True,
            )
    return placed


def arm_names() -> list[str]:
    """Return every arm name: `none`, the two real arms, and the random draws of each label set."""
    random_arms = [
        f"rand_{label_set}_d{draw}" for label_set in LABEL_SETS for draw in range(N_RANDOM_DRAWS)
    ]
    return ["none", *LABEL_SETS, *random_arms]


def arm_label_set(*, arm: str) -> LabelSetType | None:
    """Return the label set an arm excludes by, or None for the no-exclusion arm."""
    name = arm.split("_")[1] if arm.startswith("rand_") else arm
    if name == "trusted":
        return "trusted"
    if name == "allvis":
        return "allvis"
    return None


def arm_seeds(*, arm: str) -> tuple[int, ...]:
    """Return the model seeds an arm is fitted with: all three, or a random draw's one seed."""
    if arm.startswith("rand_"):
        return (int(arm.rsplit("_d", 1)[1]) % len(SEEDS),)
    return SEEDS


@cache
def exclusion_mask(*, source: SourceType, arm: str, series_id: int) -> np.ndarray:
    """Return an arm's grid mask of excluded training half-hours for one series."""
    label_set = arm_label_set(arm=arm)
    if label_set is None:
        return np.zeros(grid_months().size, dtype=bool)
    if arm.startswith("rand_"):
        return draw_random_mask(
            source=source,
            label_set=label_set,
            series_id=series_id,
            draw=int(arm.rsplit("_d", 1)[1]),
        )
    return real_exclusion_mask(source=source, label_set=label_set, series_id=series_id)


@cache
def affected_series(*, source: SourceType, label_set: LabelSetType) -> list[int]:
    """Return the series whose training rows the label set removes in a source's window."""
    ids = series_table()["time_series_id"].to_list()
    return [
        i for i in ids if real_exclusion_mask(source=source, label_set=label_set, series_id=i).any()
    ]


def fit_path(
    *, source: SourceType, arm: str, seed: int, series_id: int, root: Path = FITS_DIR
) -> Path:
    """Return the file one (source, arm, seed, series) fit writes."""
    return root / source / f"{arm}_seed{seed}" / f"{series_id}.parquet"


# ---------------------------------------------------------------------------------------------
# Exclusion report
# ---------------------------------------------------------------------------------------------


def _markdown(*, frame: pl.DataFrame) -> list[str]:
    """Return a frame as markdown table lines."""
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    for record in frame.iter_rows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in record]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def event_count_table() -> pl.DataFrame:
    """Return the visible events in each source's window, by label set."""
    rows = []
    for source in SOURCES:
        first, last = window_indices(source=source)
        window_start = TRAINING_WINDOW_START[source]
        for label_set in LABEL_SETS:
            events = events_of(label_set=label_set)
            inside, days, long_, late, ours = 0, 0.0, 0, 0, 0
            ids = set(series_table()["time_series_id"].to_list())
            for record in events.iter_rows(named=True):
                a, b = interval_indices(onset=record["onset_utc"], end=record["end_utc"])
                a, b = max(a, first), min(b, last)
                if b <= a:
                    continue
                inside += 1
                days += (b - a) / 48
                long_ += int(record["duration_days"] > 1)
                late += int(record["end_utc"] > LAST_TRAINING_VALID)
                ours += int(bool(ids & set(record["series"])))
            rows.append(
                {
                    "source": source,
                    "window_start": str(window_start.date()),
                    "label_set": label_set,
                    "events": inside,
                    "longer_than_1_day": long_,
                    "event_days_clipped": round(days, 1),
                    "end_after_window": late,
                    "naming_one_of_20_series": ours,
                }
            )
    return pl.DataFrame(rows)


def removal_table() -> pl.DataFrame:
    """Return, per source, label set, and affected series, the removed row and weight shares.

    Random columns give the mean over the draws and their range.
    """
    series = series_table()
    arrays = load_series_arrays(series=series)
    rows = []
    for source in SOURCES:
        for row in series.iter_rows(named=True):
            ids = {ls: affected_series(source=source, label_set=ls) for ls in LABEL_SETS}
            if row["time_series_id"] not in set(ids["trusted"]) | set(ids["allvis"]):
                continue
            frame = training_frame(source=source, row=row, arrays=arrays[row["site"]])
            index = np.asarray(grid_index(times=frame["valid_time"]))
            weight = recency_weights(valid_time=frame["valid_time"])
            for label_set in LABEL_SETS:
                if row["time_series_id"] not in ids[label_set]:
                    continue
                arms = [label_set] + [f"rand_{label_set}_d{d}" for d in range(N_RANDOM_DRAWS)]
                removed = []
                for arm in arms:
                    mask = exclusion_mask(source=source, arm=arm, series_id=row["time_series_id"])
                    dropped = mask[index]
                    removed.append((dropped.mean(), weight[dropped].sum() / weight.sum()))
                real_rows, real_weight = removed[0]
                finite = observed_mask(series_id=row["time_series_id"])
                real_finite = int(
                    finite[
                        exclusion_mask(
                            source=source, arm=label_set, series_id=row["time_series_id"]
                        )
                    ].sum()
                )
                ratios = [
                    finite[
                        exclusion_mask(source=source, arm=arm, series_id=row["time_series_id"])
                    ].sum()
                    / max(real_finite, 1)
                    for arm in arms[1:]
                ]
                random_rows = np.array([r for r, _ in removed[1:]])
                random_weight = np.array([w for _, w in removed[1:]])
                rows.append(
                    {
                        "source": source,
                        "label_set": label_set,
                        "site": row["site"],
                        "rows_removed": float(real_rows),
                        "weight_removed": float(real_weight),
                        "random_rows_mean": float(random_rows.mean()),
                        "random_weight_mean": float(random_weight.mean()),
                        "random_weight_min": float(random_weight.min()),
                        "random_weight_max": float(random_weight.max()),
                        "finite_ratio_min": float(min(ratios)),
                        "finite_ratio_max": float(max(ratios)),
                    }
                )
    return pl.DataFrame(rows)


def consistency_table() -> pl.DataFrame:
    """Return the recomputed share beside experiment 00041's donor-range share, where both exist."""
    return (
        visible_events()
        .filter(
            pl.col("moved_share").is_not_nan() & pl.col("stored_donor_range_share").is_not_nan()
        )
        .select(
            "card_id",
            "visibility",
            "duration_days",
            "moved_share",
            "stored_donor_range_share",
            "trusted",
        )
    )


def main() -> None:
    """Write and print the exclusion report."""
    events = visible_events()
    lines = [
        "### Visible events and trusted events",
        "",
        (
            f"{events.height} visible events, {int(events['trusted'].sum())} trusted; "
            f"{int(events['moved_share'].is_nan().sum())} have no recomputed share "
            "(no receiver, or a receiver without readings)."
        ),
        "",
        "### Events in each training window",
        "",
        *_markdown(frame=event_count_table()),
        "",
        "### Share of an affected series' training rows and recency weight removed",
        "",
        *_markdown(frame=removal_table()),
        "",
        "### Recomputed share against experiment 00041's donor-range share",
        "",
        *_markdown(frame=consistency_table()),
        "",
    ]
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    (WORK_DIR / "exclusion_report.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
