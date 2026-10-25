"""Paths, farm arrays, and scoring shared by the scripts of experiment 00347 (Stages 1 to 3b).

The weather readers, the kept-row mask of experiment 210, and the quantile score come from
experiment 00230's `e00230_common` (run with `PYTHONPATH` pointing at that folder). Farms are
labelled W1 to W3; no farm's coordinates, identifier, or capacity in megawatts leaves this process.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00230_common as c230  # ty: ignore[unresolved-import]
import e00347_plant as plant
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 347
SLUG: Final[str] = "wind-physics-hybrid-plant-then-mdn"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
"""Private outputs: plant parameters, the flagged-blocks table, and the stage reports."""
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / f"{EXPERIMENT_ID:05d}-{SLUG}"
)
"""Figures, written as SVG."""
SPLIT_TIME: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
"""Plants are fitted on valid times before this; the ERA5 hold-out starts here."""
HOLDOUT_END: Final[datetime] = c230.TRAIN_END
FARMS: Final[tuple[str, ...]] = ("W1", "W2", "W3")
HALF_HOURS_PER_DAY: Final[int] = 48
BOOTSTRAP_RESAMPLES: Final[int] = 2_000
SECTOR_COUNT: Final[int] = 8
SECTOR_NAMES: Final[tuple[str, ...]] = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
"""Direction sectors of 45 degrees centred on the compass points (direction wind blows from)."""


ARRAY_FIELDS: Final[tuple[str, ...]] = (
    "time",
    "speed_100m",
    "direction_rad",
    "target",
    "idle_mask",
    "is_train",
    "month",
    "day",
    "speed_10m",
    "temperature_c",
)


@dataclass(frozen=True)
class FarmData:
    """One farm's ERA5 weather and observed power, split at `SPLIT_TIME`.

    Attributes:
        label: Anonymous farm label.
        time: Valid times (UTC) of every row, ascending.
        speed_100m: ERA5 100 m wind speed in m/s.
        direction_rad: ERA5 100 m wind direction in radians (from `atan2(sin, cos)`).
        target: Observed power as a fraction of the farm's effective capacity.
        idle_mask: True where experiment 210's mask drops the row.
        is_train: True for valid times before `SPLIT_TIME`.
        month: The calendar month label of each row, for month-block resampling.
        day: The calendar day index of each row (days since 1970-01-01).
        speed_10m: ERA5 10 m wind speed in m/s.
        temperature_c: ERA5 2 m air temperature in degrees Celsius.
    """

    label: str
    time: np.ndarray
    speed_100m: np.ndarray
    direction_rad: np.ndarray
    target: np.ndarray
    idle_mask: np.ndarray
    is_train: np.ndarray
    month: np.ndarray
    day: np.ndarray
    speed_10m: np.ndarray
    temperature_c: np.ndarray

    def select(self, *, keep: np.ndarray) -> FarmData:
        """Return the rows where `keep` is true."""
        values = {name: getattr(self, name)[keep] for name in ARRAY_FIELDS}
        return FarmData(label=self.label, **values)

    @property
    def weather(self) -> plant.Weather:
        """Return the plant's weather inputs for these rows."""
        return plant.Weather(
            speed_100m=self.speed_100m,
            speed_10m=self.speed_10m,
            temperature_c=self.temperature_c,
            direction_rad=self.direction_rad,
        )

    @property
    def train(self) -> FarmData:
        """Return the training rows."""
        return self.select(keep=self.is_train)

    @property
    def holdout(self) -> FarmData:
        """Return the hold-out rows."""
        return self.select(keep=~self.is_train)


def load_farms() -> dict[str, FarmData]:
    """Return each farm's rows from the first power reading to 2025-06-16, by anonymous label."""
    sites = c230.site_table()
    era5 = c230.era5_rows(sites=sites)
    table = c230.training_table(sites=sites, era5=era5)
    mask = c230.idle_mask().with_columns(idle=pl.lit(value=True))
    table = table.join(mask, on=["time_series_id", "valid_time"], how="left").with_columns(
        pl.col("idle").fill_null(value=False)
    )
    farms = {}
    for row in sites.iter_rows(named=True):
        part = table.filter(pl.col("time_series_id") == row["time_series_id"]).sort("valid_time")
        direction = np.arctan2(
            part["wind_dir_sin_100m"].to_numpy().astype(np.float64),
            part["wind_dir_cos_100m"].to_numpy().astype(np.float64),
        )
        time = part["valid_time"]
        farms[row["site"]] = FarmData(
            label=row["site"],
            time=time.dt.replace_time_zone(None).to_numpy(),
            speed_100m=part["wind_speed_100m"].to_numpy().astype(np.float64),
            direction_rad=direction,
            target=part["target"].to_numpy().astype(np.float64),
            idle_mask=part["idle"].to_numpy(),
            is_train=(time < SPLIT_TIME).to_numpy(),
            month=time.dt.strftime("%Y-%m").to_numpy(),
            day=time.dt.epoch("d").to_numpy().astype(np.int64),
            speed_10m=part["wind_speed_10m"].to_numpy().astype(np.float64),
            temperature_c=part["temperature_c"].to_numpy().astype(np.float64),
        )
    return {label: farms[label] for label in FARMS}


def sector_index(*, direction_rad: np.ndarray) -> np.ndarray:
    """Return the 45 degree sector (0 is north, centred on the compass point) of each direction."""
    degrees = np.degrees(direction_rad) % 360.0
    return (np.floor((degrees + 22.5) / 45.0).astype(np.int64)) % SECTOR_COUNT


def month_block_interval(*, values: np.ndarray, months: np.ndarray, seed: int = 0) -> list[float]:
    """Return the mean of `values` and its 95% interval, resampling whole calendar months."""
    labels = np.unique(months)
    sums = np.array([values[months == m].sum() for m in labels])
    counts = np.array([(months == m).sum() for m in labels])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, labels.size, size=(BOOTSTRAP_RESAMPLES, labels.size))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return [
        float(sums.sum() / counts.sum()),
        *[float(x) for x in np.quantile(means, [0.025, 0.975])],
    ]


def month_block_gap(
    *, losses_a: np.ndarray, losses_b: np.ndarray, months: np.ndarray, seed: int = 0
) -> list[float]:
    """Return the mean of `a - b` and its 95% interval, resampling whole months."""
    return month_block_interval(values=losses_a - losses_b, months=months, seed=seed)


def score_points(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return experiment 33's quantile score per row, in points of capacity."""
    return c230.quantile_score(quantiles=quantiles, actual=actual) * c230.POINTS_PER_FRACTION


def write_json(*, name: str, payload: object) -> Path:
    """Write `payload` as JSON in the private data folder and return the path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / name
    path.write_text(json.dumps(payload, indent=1))
    return path


def read_json(*, name: str) -> dict:
    """Read a JSON file from the private data folder."""
    return json.loads((DATA_DIR / name).read_text())


def write_report(*, name: str, lines: list[str]) -> Path:
    """Write a report section as text in the private data folder and return the path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / name
    path.write_text("\n".join(lines) + "\n")
    return path


def prior_scales() -> dict[str, float]:
    """Return each farm's flat-top share of effective capacity that experiment 230's core fitted.

    These are the prior means of the plant's `scale`. Effective capacity stands in for the
    registered turbine count times the rating, so the prior mean is the share alone.
    """
    cores = json.loads((c230.WORK_DIR / "b1_cores.json").read_text())
    return {label: float(cores[label]["scale_fraction_of_capacity"]) for label in FARMS}
