"""Load an exported bundle of the step-5 neural network: weights, standardisers, and constants.

Research exports one bundle per monthly refit. The bundle is a plain directory, described by its
own `README.md`: `manifest.json` (format version, release number, seeds, file list),
`config.json` (every constant of the recipe), `primaries.json` (the 16 Primaries), and one
component folder holding the column order, the standardisers, the offset knots, and each seed's
weights as `seed<N>.npz`. `load_bundle` reads all of it once and returns a frozen `Bundle`, so
nothing downstream reads a file.

A bundle that this package cannot serve raises `ValueError`, because a bundle is our own
deployment artefact: a wrong format version, a recipe this package does not implement, or a
constant that differs from the value the code implements is a packaging bug, not a fault in the
outside world.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
import torch

BUNDLE_FORMAT_VERSION: Final[str] = "2"
"""The `bundle_format_version` this package reads; the manifest stores it as a string."""

STEP5_RECIPE: Final[str] = "demand_network_step5"
"""The manifest's `kind` of the one recipe this package implements."""

NATIVE_ORIGIN_HOUR_UTC: Final[int] = 9
"""The neural network treats the 00Z ENS run as available at 09:00 UTC: the native origin."""

PUBLICATION_DELAY: Final[timedelta] = timedelta(hours=NATIVE_ORIGIN_HOUR_UTC)
"""Native origin minus the ENS run's initialisation time."""

SUMMARY_HORIZON: Final[timedelta] = timedelta(hours=351)
"""How far after the native origin the ENS summary reaches.

The input rows stop at 339 hours, but the daily minimum and maximum temperature columns cover the
whole local day of the last input row, so the summary reaches 12 hours further.
"""

INPUT_HORIZON: Final[timedelta] = timedelta(hours=339)
"""The last input row's lead after the native origin."""

GRID_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
"""Index 0 of research's half-hourly grid, which the lag grid-index columns count from."""

GRID_STEP: Final[timedelta] = timedelta(minutes=30)

CALENDAR_TIMEZONE: Final[str] = "Europe/London"

B1_TRAILING_WINDOW_STEPS: Final[tuple[int, int, int]] = (6, 12, 48)
"""The 3-hour, 6-hour, and 24-hour windows of the B1 trailing means, in half-hours."""

_HOLIDAY_CALENDAR: Final[str] = "holidays.country_holidays('GB', subdiv='ENG')"
"""The calendar the bundle must name: the code implements this one and no other."""


@dataclass(frozen=True)
class LagConfig:
    """The day and week lag rule of `config.json` (`components.step5.lags`)."""

    cap_steps: int
    """Half-hours between the newest lag and the native origin."""
    week_steps: int
    clip_sd: float
    age_scale_hours: float
    missing_age_code_day: float
    missing_age_code_week: float
    std_floor: float


@dataclass(frozen=True)
class B1Config:
    """The constants of the thermal-lag and solar-gain ("B1") columns."""

    cold_snap_threshold_c: float
    cold_snap_cap_days: float
    lookback_steps: int
    min_clear_sky_w_m2: float
    max_clear_sky_index: float
    cams_columns_source: str
    """`"arm drop"` (both previous-day CAMS columns are 0) or `"arm replay"` (real CAMS values)."""


@dataclass(frozen=True)
class CalendarConfig:
    """The holiday and lockdown constants of `config.json` (`shared.holidays`)."""

    nearest_holiday_clip_days: int
    christmas_clip_days: int
    covid_first_day: datetime
    covid_last_day: datetime


@dataclass(frozen=True)
class NetworkConfig:
    """The layout of the 60 input columns and the read-out constants of the neural network."""

    column_slices: dict[str, tuple[int, int]]
    """Group name (`weather`, `spread`, `datetime`, `lead`, `day_lag`, `week_lag`, `b1`) to the
    half-open column range of that group."""
    lag_grid_index_columns: tuple[int, int]
    knot_position_column: int
    n_b1: int
    mixture_components: int
    log_std_clamp: tuple[float, float]
    bisection_range: float
    bisection_steps: int
    expected_weather_positions: tuple[int, ...]
    """Positions in the first 37 columns of the 28 columns that an expected-weather row holds."""
    input_clip_sd: float
    lead_scale_hours: float
    first_knot_month: str
    """`YYYY-MM` of knot 0."""
    n_knots: int


@dataclass(frozen=True)
class Standardisers:
    """The fold's standardisers, as research fitted them."""

    x37_mean: np.ndarray
    """Float32 mean of the 37 lag-free columns."""
    x37_std: np.ndarray
    b1_mean: np.ndarray
    """Float32 mean of the 14 B1 columns."""
    b1_std: np.ndarray
    lag_mean: tuple[float, float]
    """Float64 mean of the day and the week lag value (fraction of effective capacity)."""
    lag_std: tuple[float, float]
    target_mean: np.ndarray
    """Float64 mean of each Primary's target fraction, by `primary_index`."""
    target_std: np.ndarray


@dataclass(frozen=True)
class Bundle:
    """Everything the forecast reads from one exported bundle."""

    root: Path
    recipe: str
    """The component's `kind`; `STEP5_RECIPE` is the only one this package implements."""
    model_release: int
    """0 for a bundle that is not a release of the forecasting model, otherwise its number."""
    research_commit: str
    refit_month: str
    primaries: pl.DataFrame
    """One row per Primary in `primary_index` order: `primary_index` (Int64), `time_series_id`
    (Int32), `h3_index` (Int64), `latitude` and `longitude` (Float64), `effective_capacity_mw`
    and `capacity_mw` (Float64), and `unit` (String, `MW` or `MVA`)."""
    column_names: tuple[str, ...]
    """The 60 input columns in the neural network's order."""
    network: NetworkConfig
    lags: LagConfig
    b1: B1Config
    calendar: CalendarConfig
    standardisers: Standardisers
    quantile_levels: tuple[float, ...]
    """The 13 quantile levels NGED's delivery tables carry."""
    seeds: tuple[int, ...]
    weights: tuple[dict[str, torch.Tensor], ...]
    """Each seed's tensors, keyed by their `state_dict` names, in `seeds` order."""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _check(*, name: str, found: object, expected: object) -> None:
    if found != expected:
        msg = f"bundle {name} is {found!r}; this package implements {expected!r}"
        raise ValueError(msg)


def _check_shared_constants(*, shared: dict[str, Any]) -> None:
    """Raise if a shared constant differs from the value the code implements."""
    _check(
        name="native_origin_hour_utc",
        found=shared["native_origin_hour_utc"],
        expected=NATIVE_ORIGIN_HOUR_UTC,
    )
    _check(
        name="publication_delay_hours",
        found=shared["publication_delay_hours"],
        expected=PUBLICATION_DELAY / timedelta(hours=1),
    )
    _check(
        name="max_lead_hours_ens_summaries",
        found=shared["max_lead_hours_ens_summaries"],
        expected=SUMMARY_HORIZON / timedelta(hours=1),
    )
    _check(
        name="max_lead_hours_trained_and_stored",
        found=shared["max_lead_hours_trained_and_stored"],
        expected=INPUT_HORIZON / timedelta(hours=1),
    )
    _check(
        name="grid_start_utc",
        found=datetime.fromisoformat(shared["grid_start_utc"]),
        expected=GRID_START,
    )
    _check(
        name="grid_step_minutes",
        found=shared["grid_step_minutes"],
        expected=GRID_STEP / timedelta(minutes=1),
    )
    _check(name="calendar_timezone", found=shared["calendar_timezone"], expected=CALENDAR_TIMEZONE)
    _check(
        name="holidays.calendar", found=shared["holidays"]["calendar"], expected=_HOLIDAY_CALENDAR
    )


def _primaries(*, entries: list[dict[str, Any]]) -> pl.DataFrame:
    primaries = pl.DataFrame(
        {
            "primary_index": [entry["primary_index"] for entry in entries],
            "time_series_id": [entry["time_series_id"] for entry in entries],
            "h3_index": [entry["h3_res_5"] for entry in entries],
            "latitude": [entry["latitude"] for entry in entries],
            "longitude": [entry["longitude"] for entry in entries],
            "effective_capacity_mw": [entry["effective_capacity_mw"] for entry in entries],
            "capacity_mw": [entry["capacity_mw"] for entry in entries],
            "unit": [entry["unit"] for entry in entries],
        },
        schema={
            "primary_index": pl.Int64,
            "time_series_id": pl.Int32,
            "h3_index": pl.Int64,
            "latitude": pl.Float64,
            "longitude": pl.Float64,
            "effective_capacity_mw": pl.Float64,
            "capacity_mw": pl.Float64,
            "unit": pl.String,
        },
    )
    if primaries["primary_index"].to_list() != list(range(primaries.height)):
        msg = "primaries.json must list the Primaries in primary_index order from 0"
        raise ValueError(msg)
    return primaries


def _network_config(*, step5: dict[str, Any], n_knots: int) -> NetworkConfig:
    return NetworkConfig(
        column_slices={
            name: (int(bounds[0]), int(bounds[1]))
            for name, bounds in step5["column_slices"].items()
        },
        lag_grid_index_columns=(
            int(step5["lag_grid_index_columns"][0]),
            int(step5["lag_grid_index_columns"][1]),
        ),
        knot_position_column=int(step5["knot_position_column"]),
        n_b1=int(step5["n_b1"]),
        mixture_components=int(step5["mixture_components"]),
        log_std_clamp=(float(step5["log_std_clamp"][0]), float(step5["log_std_clamp"][1])),
        bisection_range=float(step5["bisection_range"]),
        bisection_steps=int(step5["bisection_steps"]),
        expected_weather_positions=tuple(
            int(p) for p in step5["expected_weather"]["x37_positions"]
        ),
        input_clip_sd=float(step5["input_clip_sd"]),
        lead_scale_hours=float(step5["lead_scale_hours"]),
        first_knot_month=str(step5["offset_knots"]["first_knot_month"]),
        n_knots=n_knots,
    )


def _lag_config(*, lags: dict[str, Any]) -> LagConfig:
    return LagConfig(
        cap_steps=int(lags["cap_steps"]),
        week_steps=int(lags["week_steps"]),
        clip_sd=float(lags["clip_sd"]),
        age_scale_hours=float(lags["age_scale_hours"]),
        missing_age_code_day=float(lags["missing_age_code"]["day"]),
        missing_age_code_week=float(lags["missing_age_code"]["week"]),
        std_floor=float(lags["std_floor"]),
    )


def _b1_config(*, b1: dict[str, Any]) -> B1Config:
    _check(
        name="b1_constants.trailing_windows_steps",
        found=b1["trailing_windows_steps"],
        expected=list(B1_TRAILING_WINDOW_STEPS),
    )
    return B1Config(
        cold_snap_threshold_c=float(b1["cold_snap_threshold_c"]),
        cold_snap_cap_days=float(b1["cold_snap_cap_days"]),
        lookback_steps=int(b1["lookback_steps"]),
        min_clear_sky_w_m2=float(b1["min_clear_sky_w_m2"]),
        max_clear_sky_index=float(b1["max_clear_sky_index"]),
        cams_columns_source=str(b1["cams_columns_source"]),
    )


def _calendar_config(*, holidays: dict[str, Any]) -> CalendarConfig:
    return CalendarConfig(
        nearest_holiday_clip_days=int(holidays["nearest_holiday_clip_days"]),
        christmas_clip_days=int(holidays["christmas_clip_days"]),
        covid_first_day=datetime.fromisoformat(holidays["covid_first_day"]).replace(tzinfo=UTC),
        covid_last_day=datetime.fromisoformat(holidays["covid_last_day"]).replace(tzinfo=UTC),
    )


def _standardisers(*, stored: dict[str, Any], n_primaries: int) -> Standardisers:
    target = stored["target"]
    if target["primary_index"] != list(range(n_primaries)):
        msg = "standardisers.json must hold the target statistics in primary_index order"
        raise ValueError(msg)
    return Standardisers(
        x37_mean=np.asarray(stored["x37"]["mean"], dtype=np.float32),
        x37_std=np.asarray(stored["x37"]["std"], dtype=np.float32),
        b1_mean=np.asarray(stored["b1"]["mean"], dtype=np.float32),
        b1_std=np.asarray(stored["b1"]["std"], dtype=np.float32),
        lag_mean=(float(stored["lag"]["day"]["mean"]), float(stored["lag"]["week"]["mean"])),
        lag_std=(float(stored["lag"]["day"]["std"]), float(stored["lag"]["week"]["std"])),
        target_mean=np.asarray(target["mean"], dtype=np.float64),
        target_std=np.asarray(target["std"], dtype=np.float64),
    )


def _load_weights(*, root: Path, files: list[str]) -> tuple[dict[str, torch.Tensor], ...]:
    weights = []
    for relative in files:
        with np.load(root / relative) as arrays:
            weights.append({name: torch.from_numpy(arrays[name]) for name in arrays.files})
    return tuple(weights)


def load_bundle(path: Path) -> Bundle:
    """Read an exported bundle of the step-5 neural network.

    Args:
        path: The bundle directory, holding `manifest.json`.

    Returns:
        The bundle's constants, standardisers, Primaries, and each seed's weights.

    Raises:
        ValueError: If the format version, the recipe, or a constant is one this package does not
            implement.
    """
    manifest = _read_json(path / "manifest.json")
    _check(
        name="bundle_format_version",
        found=manifest["bundle_format_version"],
        expected=BUNDLE_FORMAT_VERSION,
    )
    if len(manifest["components"]) != 1:
        msg = f"expected one component, found {sorted(manifest['components'])}"
        raise ValueError(msg)
    (component_name, component), *_ = manifest["components"].items()
    if component["kind"] != STEP5_RECIPE:
        msg = f"bundle recipe {component['kind']!r} is not implemented; only {STEP5_RECIPE!r} is"
        raise ValueError(msg)
    config = _read_json(path / manifest["shared_files"]["config"])
    shared, step5 = config["shared"], config["components"][component_name]
    _check_shared_constants(shared=shared)
    files = component["files"]
    columns = _read_json(path / files["columns"])
    knots = _read_json(path / files["offset_knots"])
    primaries = _primaries(
        entries=_read_json(path / manifest["shared_files"]["primaries"])["primaries"]
    )
    return Bundle(
        root=path,
        recipe=component["kind"],
        model_release=int(manifest["model_release"]),
        research_commit=str(manifest["research_git"]["commit"]),
        refit_month=str(manifest["refit_month"]),
        primaries=primaries,
        column_names=tuple(column["name"] for column in columns),
        network=_network_config(step5=step5, n_knots=int(knots["n_knots"])),
        lags=_lag_config(lags=step5["lags"]),
        b1=_b1_config(b1=step5["b1_constants"]),
        calendar=_calendar_config(holidays=shared["holidays"]),
        standardisers=_standardisers(
            stored=_read_json(path / files["standardisers"]), n_primaries=primaries.height
        ),
        quantile_levels=tuple(float(level) for level in step5["quantile_levels"]),
        seeds=tuple(int(seed) for seed in component["seeds"]),
        weights=_load_weights(root=path, files=files["weights"]),
    )
