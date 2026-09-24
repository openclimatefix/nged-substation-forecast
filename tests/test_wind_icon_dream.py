"""Tests for `studies/beam_diffuse_split/wind_icon_dream.py`'s data-cleaning and geometry helpers.

These pin the behaviour a wrong ICON-DREAM-EU build would break silently: `filter_nan_padding` and
`raise_on_duplicate_keys` are the two gates the module docstring calls hard stops, `speed_at_level`
is what picks one of DWD's ten model levels, `wind_direction_degrees` /
`mean_absolute_angle_difference_deg` are the meteorological-convention arithmetic the pre-fit
direction check rests on, and `icon_dream_site_frame` / `icon_dream_common_rows` are the row-build
functions no earlier test covered -- each synthetic fixture below is built so that a wrong level, a
wrong direction convention, a timestamp shift, a skipped `common_rows`, or a removed NaN/duplicate
gate changes an assertion's outcome, not just a hidden intermediate value.
"""

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Final

import polars as pl
import pytest

REPO_ROOT: Final[Path] = Path(__file__).parent.parent
"""The repo root, one level above this `tests/` directory."""

SCRIPT_PATH: Final[Path] = REPO_ROOT / "studies" / "beam_diffuse_split" / "wind_icon_dream.py"
"""The study script under test, imported by path because `studies/` is not an importable package."""


def _load_script() -> ModuleType:
    """Import `wind_icon_dream.py` from its path in `studies/beam_diffuse_split/`.

    Its own directory goes on `sys.path` first, because the script does bare imports (`from
    sources import ...`, `from build_dataset import ...`) that only resolve when it is run
    directly, where Python puts the script's own directory there itself.

    Returns:
        The imported module.
    """
    sys.path.insert(0, str(SCRIPT_PATH.parent))
    try:
        spec = importlib.util.spec_from_file_location("wind_icon_dream", SCRIPT_PATH)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPT_PATH.parent))
    return module


wind_icon_dream = _load_script()


def test_filter_nan_padding_drops_nan_rows() -> None:
    """A row holding not-a-number in a value column is dropped."""
    frame = pl.DataFrame(
        {"cell_id": [1, 2, 3], "value": pl.Series([1.0, float("nan"), 3.0], dtype=pl.Float32)}
    )

    filtered = wind_icon_dream.filter_nan_padding(frame=frame, value_columns=["value"])

    assert filtered["cell_id"].to_list() == [1, 3]


def test_filter_nan_padding_keeps_rows_with_no_nan() -> None:
    """A frame with no not-a-number value passes through unchanged."""
    frame = pl.DataFrame({"cell_id": [1, 2], "value": pl.Series([1.0, 2.0], dtype=pl.Float32)})

    filtered = wind_icon_dream.filter_nan_padding(frame=frame, value_columns=["value"])

    assert filtered.height == 2


def test_duplicate_key_counts_finds_a_repeated_key() -> None:
    """A key held by two rows is reported with `n_rows` 2."""
    frame = pl.DataFrame(
        {"cell_id": [1, 1, 2], "time": ["t0", "t0", "t0"], "value": [1.0, 2.0, 3.0]}
    )

    duplicates = wind_icon_dream.duplicate_key_counts(frame=frame, key_columns=["cell_id", "time"])

    assert duplicates.height == 1
    assert duplicates.row(0, named=True)["cell_id"] == 1
    assert duplicates.row(0, named=True)["n_rows"] == 2


def test_duplicate_key_counts_empty_when_every_key_is_unique() -> None:
    """A frame with no repeated key reports no duplicates."""
    frame = pl.DataFrame(
        {"cell_id": [1, 2, 3], "time": ["t0", "t0", "t0"], "value": [1.0, 2.0, 3.0]}
    )

    duplicates = wind_icon_dream.duplicate_key_counts(frame=frame, key_columns=["cell_id", "time"])

    assert duplicates.is_empty()


def test_raise_on_duplicate_keys_raises_on_a_repeated_key() -> None:
    """A repeated key raises `ValueError`, naming the count."""
    frame = pl.DataFrame({"cell_id": [1, 1], "time": ["t0", "t0"], "value": [1.0, 2.0]})

    with pytest.raises(ValueError, match="1 keys"):
        wind_icon_dream.raise_on_duplicate_keys(
            frame=frame, key_columns=["cell_id", "time"], label="test.parquet"
        )


def test_raise_on_duplicate_keys_passes_on_unique_keys() -> None:
    """A frame with every key unique raises nothing."""
    frame = pl.DataFrame({"cell_id": [1, 2], "time": ["t0", "t0"], "value": [1.0, 2.0]})

    wind_icon_dream.raise_on_duplicate_keys(
        frame=frame, key_columns=["cell_id", "time"], label="test.parquet"
    )


def test_speed_at_level_keeps_only_the_named_level() -> None:
    """Selecting level 72 drops every other level's rows and the `model_level` column."""
    frame = pl.DataFrame(
        {"model_level": [70, 72, 74], "cell_id": [1, 1, 1], "value": [1.0, 2.0, 3.0]}
    )

    selected = wind_icon_dream.speed_at_level(frame=frame, level=72)

    assert selected["value"].to_list() == [2.0]
    assert "model_level" not in selected.columns


def test_speed_at_level_raises_on_an_unserved_level() -> None:
    """A level DWD does not serve raises `ValueError` rather than silently returning no rows."""
    frame = pl.DataFrame({"model_level": [72], "cell_id": [1], "value": [1.0]})

    with pytest.raises(ValueError, match="not one of the served levels"):
        wind_icon_dream.speed_at_level(frame=frame, level=1)


@pytest.mark.parametrize(
    ("u", "v", "expected_degrees"),
    [
        (0.0, -5.0, 0.0),  # blowing due south: a wind from the north
        (-5.0, 0.0, 90.0),  # blowing due west: a wind from the east
        (0.0, 5.0, 180.0),  # blowing due north: a wind from the south
        (5.0, 0.0, 270.0),  # blowing due east: a wind from the west
    ],
)
def test_wind_direction_degrees_matches_meteorological_convention(
    u: float, v: float, expected_degrees: float
) -> None:
    """`u=0, v=-5` (blowing south) reads as a wind from the north, meteorological 0 degrees, and so
    on around the compass."""
    frame = pl.DataFrame({"u": [u], "v": [v]})

    direction = frame.select(
        wind_icon_dream.wind_direction_degrees(u=pl.col("u"), v=pl.col("v")).alias("direction")
    )["direction"][0]

    assert direction == pytest.approx(expected_degrees)


def test_wind_direction_degrees_wraps_into_0_360() -> None:
    """The direction never falls outside [0, 360), whatever quadrant `u` and `v` sit in."""
    frame = pl.DataFrame({"u": [-3.0, 3.0, -3.0], "v": [-4.0, 4.0, 4.0]})

    direction = frame.select(
        wind_icon_dream.wind_direction_degrees(u=pl.col("u"), v=pl.col("v")).alias("direction")
    )["direction"]

    assert (direction >= 0.0).all()
    assert (direction < 360.0).all()


def test_mean_absolute_angle_difference_deg_is_zero_for_identical_angles() -> None:
    """Two identical angle series differ by zero."""
    a = pl.Series([10.0, 200.0, 350.0])

    difference = wind_icon_dream.mean_absolute_angle_difference_deg(a_deg=a, b_deg=a)

    assert difference == pytest.approx(0.0)


def test_mean_absolute_angle_difference_deg_wraps_across_0_360() -> None:
    """350 degrees and 10 degrees differ by 20 degrees, not 340: the true separation wraps.

    A version that did not wrap the difference into [-180, 180] before taking its absolute value
    would report 340 degrees here instead of 20, which is the bug this test exists to catch.
    """
    a = pl.Series([350.0])
    b = pl.Series([10.0])

    difference = wind_icon_dream.mean_absolute_angle_difference_deg(a_deg=a, b_deg=b)

    assert difference == pytest.approx(20.0)


def test_mean_absolute_angle_difference_deg_is_symmetric() -> None:
    """Swapping the two series gives the same mean absolute difference."""
    a = pl.Series([10.0, 90.0, 200.0])
    b = pl.Series([350.0, 100.0, 190.0])

    forward = wind_icon_dream.mean_absolute_angle_difference_deg(a_deg=a, b_deg=b)
    backward = wind_icon_dream.mean_absolute_angle_difference_deg(a_deg=b, b_deg=a)

    assert forward == pytest.approx(backward)


_T0: Final[datetime] = datetime(2025, 3, 1, tzinfo=UTC)
"""The first hour every synthetic frame below builds from."""

_HOURS: Final[tuple[int, ...]] = tuple(range(7))
"""Seven hours, so `icon_dream_common_rows`'s inner join has one hour to drop as a zero hour."""

_CELLS: Final[tuple[int, ...]] = (1, 2)
"""Two cells; `icon_dream_cells` below always resolves site `W1` to cell 1."""

_UV_BY_LEVEL: Final[dict[int, tuple[float, float]]] = {
    72: (-5.0, -5.0),  # hub level: a wind from 45 degrees, so a degrees/radians mix-up is visible
    73: (-8.0, 0.0),  # a different u from level 72's, so reading this level for u is visible
    71: (5.0, 0.0),
    74: (0.0, 5.0),
}
"""Each level's own `(u, v)`, distinct enough that reading the wrong level changes the direction."""


def _level_speed(*, level: int, hour: int, cell: int) -> float:
    """Return a synthetic level speed with `level` embedded, so a level mix-up is never masked."""
    return level + hour / 10 + cell * 100


def _level_rows(*, kind: str) -> pl.DataFrame:
    """Build one synthetic multi-level ICON-DREAM-EU variable, plus one colliding NaN-padded row.

    Args:
        kind: `"ws"`, `"u"`, or `"v"`.

    Returns:
        `valid_time`, `model_level`, `cell_id`, and that variable's own value column.
    """
    rows = []
    for hour in _HOURS:
        for level, (u, v) in _UV_BY_LEVEL.items():
            for cell in _CELLS:
                value = {"ws": _level_speed(level=level, hour=hour, cell=cell), "u": u, "v": v}[
                    kind
                ]
                rows.append((_T0 + timedelta(hours=hour), level, cell, value))
    # cfgrib-style NaN padding colliding with a real key, as filter_nan_padding exists to drop.
    rows.append((_T0, 72, 1, float("nan")))
    column = {"ws": "ws_m_s", "u": "u_m_s", "v": "v_m_s"}[kind]
    return pl.DataFrame(
        rows,
        schema={
            "valid_time": pl.Datetime("ns"),
            "model_level": pl.Int64,
            "cell_id": pl.Int64,
            column: pl.Float32,
        },
        orient="row",
    )


def _surface_rows(*, kind: str) -> pl.DataFrame:
    """Build one synthetic single-level (10 m) ICON-DREAM-EU variable, plus one colliding NaN row.

    Args:
        kind: `"ws"`, `"u"`, or `"v"`.

    Returns:
        `valid_time`, `cell_id`, and that variable's own value column.
    """
    rows = [
        (
            _T0 + timedelta(hours=hour),
            cell,
            {"ws": 10 + hour / 10 + cell * 100, "u": 0.0, "v": 1.0}[kind],
        )
        for hour in _HOURS
        for cell in _CELLS
    ]
    rows.append((_T0, 1, float("nan")))
    column = {"ws": "ws_10m_m_s", "u": "u_10m_m_s", "v": "v_10m_m_s"}[kind]
    return pl.DataFrame(
        rows,
        schema={"valid_time": pl.Datetime("ns"), "cell_id": pl.Int64, column: pl.Float32},
        orient="row",
    )


@pytest.fixture
def synthetic_icon_dream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> pl.DataFrame:
    """Write synthetic ICON-DREAM-EU parquets under `tmp_path`, and point the module at them.

    `icon_dream_cells` is replaced so the fixture needs no real cell-centre cache: site `W1`
    always resolves to cell 1.

    Args:
        tmp_path: pytest's per-test scratch directory.
        monkeypatch: pytest's monkeypatch fixture.

    Returns:
        The one-row wind roster, `site="W1"`.
    """
    for kind, name in (
        ("ws", wind_icon_dream.WS_FILE),
        ("u", wind_icon_dream.U_FILE),
        ("v", wind_icon_dream.V_FILE),
    ):
        _level_rows(kind=kind).write_parquet(tmp_path / name)
    for kind, name in (
        ("ws", wind_icon_dream.WS_10M_FILE),
        ("u", wind_icon_dream.U_10M_FILE),
        ("v", wind_icon_dream.V_10M_FILE),
    ):
        _surface_rows(kind=kind).write_parquet(tmp_path / name)
    monkeypatch.setattr(wind_icon_dream, "ICON_DREAM_DIR", tmp_path)
    monkeypatch.setattr(
        wind_icon_dream,
        "icon_dream_cells",
        lambda *, sites, cell_ids: pl.DataFrame(
            {"site": ["W1"], "cell_id": [1], "distance_km": [1.0]}
        ),
    )
    return pl.DataFrame({"site": ["W1"]})


def test_icon_dream_site_frame_reads_the_hub_level_and_converts_direction(
    synthetic_icon_dream: pl.DataFrame,
) -> None:
    """The hub speed comes from level 72's own `u`, `v` and `ws`, with the direction in radians.

    Catches, each by a different assertion: reading the hub speed or the hub direction's `u` from
    the wrong level (`HUB_LEVEL` changed, or `u` read from level 73), skipping the
    degrees-to-radians conversion before `.sin()`, a one-hour timestamp shift, and the NaN filter
    and the duplicate gate both being removed (which would duplicate rows through the joins).
    """
    sites = synthetic_icon_dream

    frame = wind_icon_dream.icon_dream_site_frame(sites=sites).sort("time")

    assert frame.height == len(_HOURS)
    hub, sin_c, cos_c, surface = wind_icon_dream._wind_columns(product=wind_icon_dream.PRODUCT)
    hours = [
        round((t.replace(tzinfo=None) - _T0.replace(tzinfo=None)).total_seconds() / 3600)
        for t in frame["time"]
    ]
    assert hours == list(_HOURS)
    assert frame[hub].to_list() == pytest.approx(
        [_level_speed(level=72, hour=hour, cell=1) for hour in hours]
    )
    assert frame[surface].to_list() == pytest.approx([10 + hour / 10 + 100 for hour in hours])
    expected_direction_deg = wind_icon_dream.wind_direction_degrees(
        u=pl.lit(_UV_BY_LEVEL[72][0]), v=pl.lit(_UV_BY_LEVEL[72][1])
    )
    expected_sin, expected_cos = pl.select(
        sin=expected_direction_deg.radians().sin(), cos=expected_direction_deg.radians().cos()
    ).row(0)
    assert frame[sin_c].to_list() == pytest.approx([expected_sin] * frame.height, abs=1e-6)
    assert frame[cos_c].to_list() == pytest.approx([expected_cos] * frame.height, abs=1e-6)
    float_columns = [c for c in frame.columns if frame.schema[c] in (pl.Float32, pl.Float64)]
    assert not any(frame[c].is_nan().any() for c in float_columns)


def test_icon_dream_common_rows_drops_the_zero_hour_and_matches_on_time(
    synthetic_icon_dream: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`common_rows` still runs: the zero-half-hour row is dropped, not silently kept.

    Catches `common_rows` being skipped, which would keep all seven hours (or drop the wrong one)
    instead of the six the `has_zero_half_hour` flag marks for removal.

    Args:
        synthetic_icon_dream: The wind roster, with ICON-DREAM-EU parquets written.
        monkeypatch: pytest's monkeypatch fixture.
    """
    sites = synthetic_icon_dream
    zero_hour = 3
    base = pl.DataFrame(
        {
            "site": ["W1"] * len(_HOURS),
            "time": [_T0 + timedelta(hours=hour) for hour in _HOURS],
            "power_mw": [float(hour) for hour in _HOURS],
            "effective_capacity_mw": [10.0] * len(_HOURS),
            "has_zero_half_hour": [hour == zero_hour for hour in _HOURS],
        },
        schema_overrides={"time": pl.Datetime("us", "UTC")},
    )
    monkeypatch.setattr(wind_icon_dream, "joined", lambda *, sites: base)
    monkeypatch.setattr(
        wind_icon_dream, "with_eras", lambda *, frame: frame.with_columns(fold=pl.lit(0))
    )

    rows = wind_icon_dream.icon_dream_common_rows(sites=sites).sort("time")

    assert sorted(t.hour for t in rows["time"]) == [h for h in _HOURS if h != zero_hour]
    hub = wind_icon_dream._wind_columns(product=wind_icon_dream.PRODUCT)[0]
    assert rows[hub].to_list() == pytest.approx(
        [_level_speed(level=72, hour=t.hour, cell=1) for t in rows["time"]]
    )
