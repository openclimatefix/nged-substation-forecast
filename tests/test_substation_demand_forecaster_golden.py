"""Golden tests: `substation_demand_forecaster` against a research bundle's fixtures.

Skipped unless `PRIMARY_SUBSTATION_BUNDLE_PATH` names a bundle directory. The fixtures hold NGED
telemetry, so they are not committed and the tests run on the workstation only. The tests read the
weather and the cleaned power from production's own `nwp` and `cleaned_power_time_series` Delta
tables (through `get_settings()`), so they test the real read path; the fixtures are the expected
values only.
"""

import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from contracts.settings import get_settings
from nged_data.storage import scan_cleaned_power
from substation_demand_forecaster.bundle import Bundle, load_bundle
from substation_demand_forecaster.ens_summary import SUMMARY_COLUMNS, summarise_runs
from substation_demand_forecaster.features import (
    ExpectedRows,
    InputMatrix,
    build_expected_rows,
    build_input_matrix,
    native_origin,
    standardise_input_matrix,
)
from substation_demand_forecaster.neural_network import QuantileForecast, forecast

BUNDLE_ENV_VAR = "PRIMARY_SUBSTATION_BUNDLE_PATH"
BUNDLE_PATH = os.environ.get(BUNDLE_ENV_VAR)
HISTORY_DAYS = 13
CAMS_COLUMNS = ("cams_prev_day_mean", "cams_prev_day_csi")

pytestmark = pytest.mark.skipif(
    BUNDLE_PATH is None, reason=f"set {BUNDLE_ENV_VAR} to a bundle directory"
)


def _fixture_dirs() -> list[Path]:
    if BUNDLE_PATH is None:
        return []
    return sorted((Path(BUNDLE_PATH) / "fixtures" / "step5").glob("origin=*"))


@dataclass(frozen=True)
class Run:
    fixture: Path
    bundle: Bundle
    nwp_init_time: datetime
    summary: pl.DataFrame
    matrix: InputMatrix
    expected_rows: ExpectedRows
    quantiles: QuantileForecast
    levels: tuple[float, ...]


@pytest.fixture(scope="module")
def bundle() -> Bundle:
    assert BUNDLE_PATH is not None
    return load_bundle(Path(BUNDLE_PATH))


def _level_columns(frame: pl.DataFrame, *, suffix: str) -> list[str]:
    return [c for c in frame.columns if re.fullmatch(rf"q\d\d{suffix}", c)]


def _levels(columns: list[str], *, suffix: str) -> tuple[float, ...]:
    return tuple(int(c.removeprefix("q").removesuffix(suffix)) / 100 for c in columns)


def _build(*, fixture: Path, bundle: Bundle) -> Run:
    day = datetime.fromisoformat(fixture.name.removeprefix("origin=")).replace(tzinfo=UTC)
    settings = get_settings()
    cells = bundle.primaries["h3_index"].unique().to_list()
    nwp = pl.scan_delta(settings.nwp_data_path).filter(
        pl.col("nwp_model_id") == "ECMWF_ENS_0_25_degree",
        pl.col("init_time").is_between(day - timedelta(days=HISTORY_DAYS), day),
        pl.col("h3_index").is_in(cells),
    )
    summary = summarise_runs(nwp, bundle.primaries).collect()
    origin = native_origin(day)
    power = (
        scan_cleaned_power(settings.cleaned_power_time_series_data_path)
        .filter(
            pl.col("time_series_id").is_in(bundle.primaries["time_series_id"].to_list()),
            pl.col("time").is_between(origin - timedelta(days=9), origin),
        )
        .collect()
    )
    matrix = build_input_matrix(
        bundle=bundle, summary=summary, cleaned_power=power, nwp_init_time=day
    )
    if bundle.b1.cams_columns_source != "arm drop":
        # Production has no CAMS columns; this bundle's fixtures hold real values, so feed them in.
        stored = pl.read_parquet(fixture / "input_matrix.parquet")
        keyed = matrix.keys.join(
            stored.select(
                pl.col("primary_index").cast(pl.Int64),
                "valid_time",
                *(f"raw__{c}" for c in CAMS_COLUMNS),
            ),
            on=["primary_index", "valid_time"],
            how="left",
            maintain_order="left",
        )
        raw = matrix.raw.copy()
        for column in CAMS_COLUMNS:
            raw[:, bundle.column_names.index(column)] = keyed[f"raw__{column}"].to_numpy()
        matrix = InputMatrix(
            keys=matrix.keys,
            raw=raw,
            standardised=standardise_input_matrix(
                raw=raw, target_index=matrix.keys["target_grid_index"].to_numpy(), bundle=bundle
            ),
        )
    expected_rows = build_expected_rows(bundle=bundle, summary=summary, nwp_init_time=day)
    outputs = pl.read_parquet(fixture / "outputs_native.parquet")
    levels = _levels(_level_columns(outputs, suffix="_mean"), suffix="_mean")
    quantiles = forecast(
        bundle=bundle,
        matrix=matrix.standardised,
        primary_index=matrix.keys["primary_index"].to_numpy(),
        expected_rows=expected_rows,
        levels=levels,
    )
    return Run(
        fixture=fixture,
        bundle=bundle,
        nwp_init_time=day,
        summary=summary,
        matrix=matrix,
        expected_rows=expected_rows,
        quantiles=quantiles,
        levels=levels,
    )


_RUNS: dict[Path, Run] = {}


@pytest.fixture(params=_fixture_dirs(), ids=lambda path: path.name)
def run(request: pytest.FixtureRequest, bundle: Bundle) -> Run:
    fixture = request.param
    if fixture not in _RUNS:
        _RUNS[fixture] = _build(fixture=fixture, bundle=bundle)
    return _RUNS[fixture]


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    assert np.array_equal(np.isnan(a), np.isnan(b)), "the NaN positions differ"
    both = ~np.isnan(a)
    return float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0


def _max_relative(want: np.ndarray, got: np.ndarray) -> float:
    """Return the largest difference over the larger of the value's size and 1."""
    want, got = want.astype(np.float64), got.astype(np.float64)
    assert np.array_equal(np.isnan(want), np.isnan(got)), "the NaN positions differ"
    finite = ~np.isnan(want)
    if not finite.any():
        return 0.0
    return float((np.abs(got[finite] - want[finite]) / np.maximum(np.abs(want[finite]), 1.0)).max())


def _keyed_matrix(run: Run) -> tuple[pl.DataFrame, pl.DataFrame]:
    stored = pl.read_parquet(run.fixture / "input_matrix.parquet").with_columns(
        pl.col("primary_index").cast(pl.Int64)
    )
    mine = run.matrix.keys.select("primary_index", "valid_time")
    assert stored.select("primary_index", "valid_time").sort(pl.all()).equals(mine.sort(pl.all()))
    return stored.sort("primary_index", "valid_time"), mine


def test_ens_summary(run: Run) -> None:
    stored = pl.read_parquet(run.fixture / "ens_summary.parquet")
    mine = run.summary.filter(pl.col("nwp_init_time") == run.nwp_init_time)
    joined = stored.join(mine, on=["time_series_id", "valid_time"], how="full", suffix="_prod")
    assert joined.height == stored.height == mine.height
    worst = {
        c: _max_abs(joined[c].to_numpy(), joined[f"{c}_prod"].to_numpy()) for c in SUMMARY_COLUMNS
    }
    print("ens_summary max abs diff", max(worst.values()), worst)
    assert max(worst.values()) <= 1e-5, worst


def test_input_matrix_standardised(run: Run) -> None:
    stored, _ = _keyed_matrix(run)
    worst = {}
    for k, name in enumerate(run.bundle.column_names):
        worst[name] = _max_abs(stored[f"std__{name}"].to_numpy(), run.matrix.standardised[:, k])
    print("std max abs diff", max(worst.values()), {k: v for k, v in worst.items() if v > 0})
    assert max(worst.values()) <= 1e-5, {k: v for k, v in worst.items() if v > 1e-5}


def test_input_matrix_raw(run: Run) -> None:
    stored, _ = _keyed_matrix(run)
    worst = {}
    for k, name in enumerate(run.bundle.column_names):
        worst[name] = _max_relative(stored[f"raw__{name}"].to_numpy(), run.matrix.raw[:, k])
    print("raw max relative diff", max(worst.values()), {k: v for k, v in worst.items() if v > 0})
    assert max(worst.values()) <= 1e-6, {k: v for k, v in worst.items() if v > 1e-6}


def test_expected_rows(run: Run) -> None:
    stored = pl.read_parquet(run.fixture / "lag_and_expected_rows.parquet")
    rows = run.expected_rows
    mine = pl.DataFrame(
        {
            "primary_index": rows.primary_index,
            "grid_index": rows.grid_index,
            "run_init_time": pl.Series(rows.run_init_time).cast(pl.Datetime("us", "UTC")),
        }
    ).with_row_index("row")
    joined = stored.with_columns(pl.col("primary_index").cast(pl.Int64)).join(
        mine, on=["primary_index", "grid_index"], how="left", maintain_order="left"
    )
    assert joined["row"].is_not_null().all()
    expected_init = joined["run_origin_time"] - timedelta(hours=9)
    assert (joined["run_init_time"] == expected_init).all()
    index = joined["row"].to_numpy()
    names = [run.bundle.column_names[p] for p in run.bundle.network.expected_weather_positions]
    raw = max(
        _max_relative(joined[f"raw__{n}"].to_numpy(), rows.raw[index, k])
        for k, n in enumerate(names)
    )
    std = max(
        _max_abs(joined[f"std_f16__{n}"].to_numpy(), rows.standardised[index, k].astype(np.float32))
        for k, n in enumerate(names)
    )
    print("expected rows max abs diff raw", raw, "std f16", std)
    assert std <= 1e-5
    assert raw <= 1e-6


def test_outputs_native(run: Run) -> None:
    stored = pl.read_parquet(run.fixture / "outputs_native.parquet").with_columns(
        pl.col("primary_index").cast(pl.Int64)
    )
    order = run.matrix.keys.with_row_index("row").join(
        stored.with_row_index("stored"), on=["primary_index", "valid_time"], how="inner"
    )
    assert order.height == stored.height
    mine, theirs = order["row"].to_numpy(), order["stored"].to_numpy()
    want = stored.select(_level_columns(stored, suffix="_mean")).to_numpy()[theirs]
    worst_mean = float(np.abs(run.quantiles.mean[mine] - want).max())
    worst_seed = 0.0
    for s, seed in enumerate(run.bundle.seeds):
        columns = _level_columns(stored, suffix=f"_s{seed}")
        seed_want = stored.select(columns).to_numpy()[theirs]
        worst_seed = max(
            worst_seed, float(np.abs(run.quantiles.per_seed[s][mine] - seed_want).max())
        )
    print("outputs_native max abs diff seed mean", worst_mean, "per seed", worst_seed)
    assert worst_mean <= 2e-5
    assert worst_seed <= 2e-5


def test_outputs_contract(run: Run) -> None:
    """Megawatts at the 12:00 UTC contract origin, before the MVA zero clamp, on 671 rows."""
    stored = pl.read_parquet(run.fixture / "outputs_contract.parquet")
    contract_origin = run.nwp_init_time + timedelta(hours=12)
    stored = stored.filter(pl.col("valid_time") > contract_origin).with_columns(
        pl.col("time_series_id").cast(pl.Int32)
    )
    assert stored.height == run.bundle.primaries.height * 671
    keys = run.matrix.keys.with_row_index("row").join(
        run.bundle.primaries.select("time_series_id", "capacity_mw"), on="time_series_id"
    )
    joined = stored.join(keys, on=["time_series_id", "valid_time"], how="inner")
    assert joined.height == stored.height
    rows = joined["row"].to_numpy()
    capacity = joined["capacity_mw"].to_numpy()[:, None]
    mine = run.quantiles.mean[rows].astype(np.float64) * capacity
    want = joined.select(_level_columns(joined, suffix="_mw")).to_numpy()
    worst = float((np.abs(mine - want) / capacity).max())
    print("outputs_contract max abs diff over capacity", worst)
    assert worst <= 2e-5
