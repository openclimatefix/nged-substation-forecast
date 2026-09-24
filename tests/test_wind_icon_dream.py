"""Tests for `studies/beam_diffuse_split/wind_icon_dream.py`'s data-cleaning and geometry helpers.

These pin the behaviour a wrong ICON-DREAM-EU build would break silently: `filter_nan_padding` and
`raise_on_duplicate_keys` are the two gates the module docstring calls hard stops, `speed_at_level`
is what picks one of DWD's ten model levels, and `wind_direction_degrees` /
`mean_absolute_angle_difference_deg` are the meteorological-convention arithmetic the pre-fit
direction check rests on.
"""

import importlib.util
import sys
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
