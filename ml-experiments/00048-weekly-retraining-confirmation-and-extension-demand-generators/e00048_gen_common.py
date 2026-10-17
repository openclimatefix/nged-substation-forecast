"""Generator series, ERA5-trained stage 1a models, mixtures, and spread factors (experiment 00048).

Copied from experiment 00033's `e00033_gen_common.py` and `e00033_fit_generators.py` (themselves
copied from experiment 00020); an experiment may not import another experiment. The ERA5
inputs it reads were built by the stage 1 pilot and are read only.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00048_common as common
import numpy as np
import polars as pl
import xgboost as xgb
from studies.anonymise import LABEL_PERMUTATION_SEED, SITE_LABELS
from studies.power import scan_power
from studies.pv_dataset import pv_sites, wind_sites
from studies.sources import PER_STUDY_DIR

MEMBERS: Final[int] = 51
KEY_COLUMNS: Final[tuple[str, ...]] = common.KEYS
PILOT_DIR: Final[Path] = PER_STUDY_DIR / "stage1_pilot"
ANALYSIS_PATH: Final[Path] = PILOT_DIR / "inputs" / "analysis_features.parquet"
ENS_VALIDATION_DIR: Final[Path] = PILOT_DIR / "inputs" / "ens_features" / "val"
"""The 53 Tuesday runs' generator frames, read only (the `--check` of the frame builder)."""

ERA5_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
FROZEN_TRAIN_END: Final[datetime] = datetime(2025, 6, 17, tzinfo=UTC)
"""The frozen stage 1a trains on valid times before this: 14 days before the first run."""
ERA5_ARRIVAL_DAYS: Final[int] = 5
"""ERA5 reaches the project this many days after its valid time."""
MONTHLY_REFITS: Final[tuple[datetime, ...]] = tuple(
    datetime(2025 + (6 + i) // 12, (6 + i) % 12 + 1, 1, tzinfo=UTC) for i in range(12)
)
"""The twelve monthly refit dates, 2025-07-01 to 2026-06-01."""

FINE_LEVELS: Final[np.ndarray] = np.arange(1, 200) / 200
"""The 199 levels on which each mixture's quantile function is stored."""
LOWER_INDEX: Final[int] = 0
UPPER_INDEX: Final[int] = 8
GEN_LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0 to 24 h", 0.0, 24.0),
    ("1 to 2 days", 24.0, 72.0),
    ("3 to 6 days", 72.0, 168.0),
    ("7 to 10 days", 168.0, 264.0),
    ("11 to 15 days", 264.0, 10_000.0),
)
SHORT_BANDS: Final[frozenset[str]] = frozenset({"0 to 24 h", "1 to 2 days"})
"""The wind spread factor is fitted on these lead bands (0 to 2 days) and on the rest separately."""
MIN_FIT_CASES: Final[int] = 5_000
"""A spread-factor fit needs at least this many non-trivial earlier cases."""
TRIVIAL_LEVEL: Final[float] = 0.01
"""A case is trivial when the uncorrected 95% quantile is at most this fraction of capacity."""
TARGET_COVERAGE: Final[float] = 0.90
SPREAD_GRID: Final[np.ndarray] = np.round(np.arange(0.60, 2.001, 0.05), 2)
SPREAD_TELEMETRY_DELAY: Final = np.timedelta64(210, "m")
"""A case is known to the spread-factor fit 3.5 hours after its valid time."""

SOLAR_FEATURES: Final[tuple[str, ...]] = (
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
    "solar_zenith",
    "solar_azimuth",
)
WIND_FEATURES: Final[tuple[str, ...]] = (
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_dir_sin_10m",
    "wind_dir_cos_10m",
    "wind_dir_sin_100m",
    "wind_dir_cos_100m",
)
FEATURES: Final[dict[str, tuple[str, ...]]] = {"solar": SOLAR_FEATURES, "wind": WIND_FEATURES}
GENERATOR_FRAME_COLUMNS: Final[tuple[str, ...]] = tuple(
    dict.fromkeys([*SOLAR_FEATURES, *WIND_FEATURES])
)
"""The weather and geometry columns a generator frame holds, beside its keys and the member."""

GENERATOR_PARAMETERS: Final[dict[str, object]] = dict(common.QUANTILE_PARAMETERS)
"""Experiment 00002's multi-quantile settings; callers set `nthread`."""
SETTLING_LEVEL_FRACTION: Final[float] = 0.05
SETTLING_RUN_HALF_HOURS: Final[int] = 48
SETTLING_WEEKS: Final[int] = 6
"""The `new_meter_settling` rule, applied as the stage 1 pilot applied it."""
SPEED_BIN_WIDTH_M_S: Final[float] = 0.5


def site_table() -> pl.DataFrame:
    """Return the nine generator series with anonymous label, technology, capacity, and H3 cell.

    Returns:
        One row per series with `time_series_id`, `site` (A to F, W1 to W3), `technology`,
        `latitude`, `longitude`, `effective_capacity_mw`, and `h3_res_5`. The coordinates never
        leave this process.

    Raises:
        ValueError: If the anonymisation helper's labels have changed, or a series has no H3 cell.
    """
    capacity = pl.read_parquet(common.FOLD_A_FRAMES_DIR / "capacity.parquet").select(
        "time_series_id", "effective_capacity_mw"
    )
    cells = pl.read_parquet(
        common.FOLD_A_FRAMES_DIR / "metadata.parquet", columns=["time_series_id", "h3_res_5"]
    )
    solar = pv_sites().select("time_series_id", "site", "latitude", "longitude")
    wind = wind_sites().select("time_series_id", "site", "latitude", "longitude")
    if set(solar["site"].to_list()) != set(SITE_LABELS) or LABEL_PERMUTATION_SEED != 784:
        msg = "the anonymisation helper's solar labels have changed"
        raise ValueError(msg)
    table = (
        pl.concat(
            [
                solar.with_columns(technology=pl.lit("solar")),
                wind.with_columns(technology=pl.lit("wind")),
            ]
        )
        .join(capacity, on="time_series_id", how="left", validate="1:1")
        .join(cells, on="time_series_id", how="left", validate="1:1")
        .sort("technology", "site")
    )
    if table["h3_res_5"].null_count() or table.height != 9:
        msg = "expected the 9 generator series, each with an H3 cell"
        raise ValueError(msg)
    return table


def generator_frame_path(*, run: datetime) -> Path:
    """Return the generator frame of a daily run (`e00048_build_frames.py` writes all 365)."""
    return common.FRAMES_GENERATORS_DIR / f"init={run:%Y-%m-%d}.parquet"


def band_names(*, lead_hours: np.ndarray) -> np.ndarray:
    """Return the generator lead band name of each lead in hours."""
    names = np.full(lead_hours.shape, "none", dtype=object)
    for name, low, high in GEN_LEAD_BANDS:
        names[(lead_hours >= low) & (lead_hours < high)] = name
    return names


def mixture_quantiles(*, member_quantiles: np.ndarray, chunk: int = 20_000) -> np.ndarray:
    """Return each case's mixture quantile function on `FINE_LEVELS`.

    Args:
        member_quantiles: Shape (cases, members, 9) member quantiles, each member's levels sorted.
        chunk: Cases sorted at a time.

    Returns:
        Shape (cases, 199) float32 quantiles.
    """
    cases, members, levels = member_quantiles.shape
    weights = np.tile(common.ATOM_MASSES / members, members)
    out = np.empty((cases, FINE_LEVELS.size), dtype=np.float32)
    for start in range(0, cases, chunk):
        flat = member_quantiles[start : start + chunk].reshape(-1, members * levels)
        order = np.argsort(flat, axis=1, kind="stable")
        values = np.take_along_axis(flat, order, axis=1)
        mass = weights[order]
        midpoint = np.cumsum(mass, axis=1) - mass / 2
        for row in range(flat.shape[0]):
            out[start + row] = np.interp(FINE_LEVELS, midpoint[row], values[row])
    return out


def read_levels(*, fine: np.ndarray) -> np.ndarray:
    """Return the 9 output quantiles of each row of `fine`, by linear interpolation."""
    position = np.clip(common.LEVELS * 200 - 1, 0, FINE_LEVELS.size - 1)
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, FINE_LEVELS.size - 1)
    fraction = (position - low).astype(np.float32)
    return fine[:, low] * (1 - fraction) + fine[:, high] * fraction


def covered(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return whether the observed power lies in the 5% to 95% band, per case."""
    return (actual >= quantiles[:, LOWER_INDEX]) & (actual <= quantiles[:, UPPER_INDEX])


def scale_about_median(*, quantiles: np.ndarray, shift: float, factor: float) -> np.ndarray:
    """Return `median + shift + factor * (q - median)`, clipped to 0 to 1 and sorted."""
    median = quantiles[:, common.MEDIAN : common.MEDIAN + 1]
    scaled = median + shift + factor * (quantiles - median)
    return np.sort(np.clip(scaled, 0.0, 1.0), axis=1)


def fit_spread_factor(*, quantiles: np.ndarray, actual: np.ndarray) -> float:
    """Return the grid factor whose 5% to 95% coverage is nearest `TARGET_COVERAGE`."""
    gaps = [
        abs(
            covered(
                quantiles=scale_about_median(quantiles=quantiles, shift=0.0, factor=float(k)),
                actual=actual,
            ).mean()
            - TARGET_COVERAGE
        )
        for k in SPREAD_GRID
    ]
    return float(SPREAD_GRID[int(np.argmin(gaps))])


def apply_spread(*, cases: pl.DataFrame, mixture: np.ndarray, technology: str) -> np.ndarray:
    """Return the quantiles after the spread factor, fitted month by month on earlier cases.

    The factor for the cases of one origin month is fitted only on cases whose valid time is at
    least 3.5 hours before the month's first origin, so no fit reads an observation from after the
    forecast it corrects. Solar takes one factor per month; wind takes one for leads of up to 2
    days and one for the rest.

    Args:
        cases: One row per case with `valid_time`, `power_fcst_init_time`, `actual_norm` (a fraction
            of capacity, finite), and `band` (a name from `GEN_LEAD_BANDS`).
        mixture: Shape (cases, 9), the pooled mixture's quantiles of each case.
        technology: `solar` or `wind`.

    Returns:
        The scaled quantiles, shape (cases, 9).
    """
    actual = cases["actual_norm"].to_numpy().astype(np.float64)
    valid = cases["valid_time"].to_numpy()
    init = cases["power_fcst_init_time"].to_numpy()
    month = cases["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    short = np.isin(cases["band"].to_numpy(), list(SHORT_BANDS))
    scorable = mixture[:, UPPER_INDEX] > TRIVIAL_LEVEL
    out = mixture.copy()
    groups = [np.ones_like(short)] if technology == "solar" else [short, ~short]
    for month_label in sorted(set(month)):
        in_month = month == month_label
        cutoff = init[in_month].min()
        for group in groups:
            target = in_month & group & scorable
            fit_index = np.flatnonzero(
                (valid <= cutoff - SPREAD_TELEMETRY_DELAY) & group & scorable
            )
            if not target.any() or fit_index.size < MIN_FIT_CASES:
                continue
            factor = fit_spread_factor(quantiles=mixture[fit_index], actual=actual[fit_index])
            out[target] = scale_about_median(quantiles=mixture[target], shift=0.0, factor=factor)
    return out


def full_power(*, sites: pl.DataFrame) -> dict[int, np.ndarray]:
    """Return each series' observed power over capacity on the dense half-hourly grid."""
    grid = pl.DataFrame({"valid_time": common.grid_times()})
    power = (
        scan_power()
        .filter(
            pl.col("time_series_id").is_in(sites["time_series_id"].to_list()),
            pl.col("time") >= ERA5_START,
        )
        .select("time_series_id", pl.col("time").alias("valid_time"), "power")
        .collect()
    )
    out = {}
    for row in sites.iter_rows(named=True):
        joined = grid.join(
            power.filter(pl.col("time_series_id") == row["time_series_id"]).select(
                "valid_time", "power"
            ),
            on="valid_time",
            how="left",
            validate="1:1",
        )
        out[int(row["time_series_id"])] = (
            joined["power"].to_numpy().astype(np.float64) / row["effective_capacity_mw"]
        ).astype(np.float32)
    return out


def observed_training_power(*, sites: pl.DataFrame, train_end: datetime) -> pl.DataFrame:
    """Return cleaned power over capacity from the ERA5 era to the end of training.

    Rows before each series' settling cutoff are dropped, as the stage 1 pilot did.

    Args:
        sites: `site_table()`, or the rows of it a caller needs.
        train_end: Exclusive end of the training window.

    Returns:
        Columns `time_series_id`, `valid_time`, and `target` (a fraction of capacity).
    """
    power = (
        scan_power()
        .filter(
            pl.col("time_series_id").is_in(sites["time_series_id"].to_list()),
            pl.col("time") >= ERA5_START,
            pl.col("time") < train_end,
        )
        .select("time_series_id", pl.col("time").alias("valid_time"), "power")
        .collect()
        .join(sites.select("time_series_id", "effective_capacity_mw"), on="time_series_id")
        .sort("time_series_id", "valid_time")
    )
    live = (
        pl.col("power").abs() > SETTLING_LEVEL_FRACTION * pl.col("effective_capacity_mw")
    ).fill_null(value=False)
    run_end = (
        live.cast(pl.Int32).rolling_sum(window_size=SETTLING_RUN_HALF_HOURS).over("time_series_id")
        == SETTLING_RUN_HALF_HOURS
    )
    cutoffs = (
        power.filter(run_end)
        .group_by("time_series_id")
        .agg(
            cutoff=pl.col("valid_time")
            .min()
            .dt.offset_by(f"-{SETTLING_RUN_HALF_HOURS * 30 - 30}m")
            .dt.offset_by(f"{SETTLING_WEEKS}w")
        )
    )
    return (
        power.join(cutoffs, on="time_series_id", how="left")
        .filter(pl.col("cutoff").is_null() | (pl.col("valid_time") >= pl.col("cutoff")))
        .filter(pl.col("power").is_finite())
        .select(
            "time_series_id", "valid_time", target=pl.col("power") / pl.col("effective_capacity_mw")
        )
    )


def power_curve(*, speed: np.ndarray, power_fraction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a non-decreasing curve of power fraction on 100 m wind speed.

    Args:
        speed: 100 m wind speeds of the training rows, m/s.
        power_fraction: Observed power over capacity of the same rows.

    Returns:
        The bin centres and the running maximum of each bin's mean power fraction, for
        `np.interp`.
    """
    bins = np.floor(speed / SPEED_BIN_WIDTH_M_S).astype(np.int64)
    ordered = (
        pl.DataFrame({"bin": bins, "power": power_fraction})
        .group_by("bin")
        .agg(pl.col("power").mean())
        .sort("bin")
    )
    centres = (ordered["bin"].to_numpy() + 0.5) * SPEED_BIN_WIDTH_M_S
    return centres, np.maximum.accumulate(ordered["power"].to_numpy())


def feature_matrix(*, frame: pl.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """Return a frame's `columns` as a float32 matrix."""
    return frame.select([pl.col(c).cast(pl.Float32) for c in columns]).to_numpy()


class FittedQuantiles:
    """One series' fitted quantile model, with the wind power curve when a base margin is used."""

    def __init__(
        self,
        *,
        rows: pl.DataFrame,
        columns: tuple[str, ...],
        seed: int,
        rounds: int,
        use_margin: bool,
        nthread: int,
    ) -> None:
        """Fit the model; `use_margin` makes the power curve the base margin of all 9 levels.

        Args:
            rows: Training rows with the `columns` and a `target` (a fraction of capacity).
            columns: The input columns.
            seed: The fitting seed.
            rounds: Boosting rounds.
            use_margin: Whether the wind power curve is the base margin.
            nthread: Threads the fit may use.

        Raises:
            ValueError: If there are no training rows, or the base margin did not change the
                predictions.
        """
        if rows.height == 0:
            msg = "no training rows"
            raise ValueError(msg)
        self.columns = columns
        self.curve: tuple[np.ndarray, np.ndarray] | None = None
        features = feature_matrix(frame=rows, columns=columns)
        target = rows["target"].to_numpy()
        margin = None
        if use_margin:
            self.curve = power_curve(
                speed=rows["wind_speed_100m"].to_numpy(), power_fraction=target
            )
            margin = self._margin(speed=rows["wind_speed_100m"].to_numpy())
        matrix = xgb.DMatrix(features, label=target, base_margin=margin)
        self.booster = xgb.train(
            {**GENERATOR_PARAMETERS, "seed": seed, "nthread": nthread},
            matrix,
            num_boost_round=rounds,
        )
        if margin is not None:
            plain = self.booster.predict(xgb.DMatrix(features[:2000]))
            with_margin = self.booster.predict(
                xgb.DMatrix(features[:2000], base_margin=margin[:2000])
            )
            if np.allclose(plain, with_margin):
                msg = "the base margin did not change the predictions"
                raise ValueError(msg)

    def _margin(self, *, speed: np.ndarray) -> np.ndarray:
        if self.curve is None:
            msg = "no power curve"
            raise ValueError(msg)
        centres, curve = self.curve
        return np.tile(np.interp(speed, centres, curve)[:, None], (1, len(common.LEVEL_COLUMNS)))

    def predict(self, *, frame: pl.DataFrame) -> np.ndarray:
        """Return the 9 sorted quantile columns for every row (a fraction of capacity)."""
        margin = (
            None if self.curve is None else self._margin(speed=frame["wind_speed_100m"].to_numpy())
        )
        matrix = xgb.DMatrix(feature_matrix(frame=frame, columns=self.columns), base_margin=margin)
        return np.sort(self.booster.predict(matrix), axis=1)


def pooled_mixture(*, part: pl.DataFrame, member_quantiles: np.ndarray) -> pl.DataFrame:
    """Return the keys and 9 mixture quantiles of every case in `part`.

    Args:
        part: One run's rows for one series, with `ensemble_member` and the key columns.
        member_quantiles: Shape (rows, 9), the member quantiles of each row of `part`.

    Returns:
        One row per case.

    Raises:
        ValueError: If a case lacks a member.
    """
    ordered = (
        part.select(*KEY_COLUMNS, "ensemble_member")
        .with_columns(row=pl.int_range(pl.len()))
        .sort(*KEY_COLUMNS, "ensemble_member")
    )
    if ordered.height % MEMBERS:
        msg = f"a case lacks some of the {MEMBERS} members"
        raise ValueError(msg)
    matrix = member_quantiles[ordered["row"].to_numpy()].reshape(
        -1, MEMBERS, len(common.LEVEL_COLUMNS)
    )
    keys = ordered.gather_every(MEMBERS).select(KEY_COLUMNS)
    if (ordered["ensemble_member"].to_numpy().reshape(-1, MEMBERS) != np.arange(MEMBERS)).any():
        msg = "the members of a case are not 0 to 50"
        raise ValueError(msg)
    mixture = read_levels(fine=mixture_quantiles(member_quantiles=matrix.astype(np.float32)))
    return pl.concat(
        [keys, pl.DataFrame(mixture, schema=list(common.LEVEL_COLUMNS))], how="horizontal"
    )


def refit_serving(*, origin: datetime) -> datetime:
    """Return the monthly refit that serves a run: the latest refit date at or before its origin."""
    earlier = [refit for refit in MONTHLY_REFITS if refit <= origin]
    if not earlier:
        msg = f"no monthly refit at or before {origin}"
        raise ValueError(msg)
    return max(earlier)


def training_end_for(*, refit: datetime) -> datetime:
    """Return the exclusive end of the ERA5 window a monthly refit trains on."""
    return refit - timedelta(days=ERA5_ARRIVAL_DAYS)
