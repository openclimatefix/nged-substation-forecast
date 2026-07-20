"""Unit tests for ``geo.h3`` grid-weight computation.

These pin down the load-bearing invariant (every H3 parent's child proportions sum to 1), the core
nearest-grid-centre snapping, the four documented ``ValueError`` guards on
``compute_h3_grid_weights``, and the boundary wrapper's real cell generation plus its empty-boundary
error path.
"""

from collections.abc import Callable
from typing import Final

import h3.api.basic_int as h3
import polars as pl
import polars_h3 as plh3
import pytest
import shapely
from geo.h3 import compute_h3_grid_weights, compute_h3_grid_weights_for_boundary

_EXPECTED_COLUMNS: Final[frozenset[str]] = frozenset(
    {"h3_index", "nwp_lat", "nwp_lon", "proportion"}
)


def _res5_cells() -> list[int]:
    """Seven contiguous resolution-5 H3 cells over central Great Britain."""
    center = h3.latlng_to_cell(52.5, -1.5, 5)
    return sorted(h3.grid_disk(center, 1))


def test_grid_weights_invariant() -> None:
    cells = _res5_cells()
    weights = compute_h3_grid_weights(nwp_grid_size_degrees=0.25, h3_index=cells)

    # Output carries exactly the schema columns, one or more rows per input parent...
    assert set(weights.columns) == _EXPECTED_COLUMNS
    assert set(weights["h3_index"].to_list()) == set(cells)

    # ...each parent's child proportions sum to 1 (the area-weighting invariant)...
    per_parent = weights.group_by("h3_index").agg(pl.col("proportion").sum())
    max_deviation = per_parent.select((pl.col("proportion") - 1.0).abs().max()).item()
    assert max_deviation < 1e-5

    # ...and every proportion is a genuine fraction in (0, 1].
    bounds = weights.select(lo=pl.col("proportion").min(), hi=pl.col("proportion").max())
    assert bounds["lo"].item() > 0
    assert bounds["hi"].item() <= 1.0 + 1e-6


def test_grid_weights_snap_to_nearest_grid_centre() -> None:
    """Pins the core snapping: each child H3 cell is binned to its *nearest* NWP grid centre.

    An independent nearest-rounding oracle reproduces the exact set of produced grid points. This
    catches a regression to floor-snapping (or no snapping at all) -- neither of which the
    sum-to-one arithmetic identity in ``test_grid_weights_invariant`` would notice.
    """
    grid = 0.25
    cells = _res5_cells()
    produced = compute_h3_grid_weights(nwp_grid_size_degrees=grid, h3_index=cells)

    # Oracle: round each child centroid to the nearest grid point (round, not the production floor).
    oracle = (
        pl.DataFrame({"h3_index": cells}, schema={"h3_index": pl.UInt64})
        .with_columns(child=plh3.cell_to_children("h3_index", 7))  # default child_res == h3_res + 2
        .explode("child", empty_as_null=True)
        .select(
            nwp_lat=(plh3.cell_to_lat("child") / grid).round() * grid,
            nwp_lon=(plh3.cell_to_lng("child") / grid).round() * grid,
        )
        .unique()
    )
    expected_points = set(
        zip(oracle["nwp_lat"].to_list(), oracle["nwp_lon"].to_list(), strict=True)
    )
    produced_points = set(
        zip(
            produced["nwp_lat"].cast(pl.Float64).to_list(),
            produced["nwp_lon"].cast(pl.Float64).to_list(),
            strict=True,
        )
    )
    assert produced_points == expected_points


def test_grid_weights_preserve_geographic_orientation() -> None:
    """A known point maps to NWP grid cells at its own (lat, lon) — no lat/lon swap.

    This is the geographic complement to the ``dynamical_data`` orientation test: that test proves
    ``convert`` preserves the value↔(lat, lon) pairing through the join; *this* one proves the
    (lat, lon) labels themselves are geographically right, i.e. that
    ``compute_h3_grid_weights`` snaps each cell to grid points at the cell's true location.

    Two well-separated Great Britain landmarks pin it down. Absolute check: Edinburgh (~56°N, ~3°W)
    must map to ``nwp_lat`` near +56 and ``nwp_lon`` near -3 — a lat/lon swap would send it to
    (lat -3, lon +56), in the Indian Ocean, and is caught here. ``compute_h3_grid_weights`` fills
    ``nwp_lat`` from ``cell_to_lat`` and ``nwp_lon`` from ``cell_to_lng`` directly — it has no
    axis-flip or transpose code path — so the bug this actually guards is that swap (verified by
    mutation: exchanging the two ``cell_to_*`` calls fails this test). The relative check below (north
    keeps the larger latitude, west the smaller longitude) is cheap defence-in-depth against a future
    refactor that introduces axis handling.

    See the orientation-coverage table in
    <https://openclimatefix.github.io/nged-substation-forecast/architecture/testing/#nwp-grid-h3-orientation-coverage>
    for how this test sits alongside the ``dynamical_data`` layers.
    """
    grid = 0.25
    edinburgh_lat, edinburgh_lon = 55.95, -3.19  # far north
    lands_end_lat, lands_end_lon = 50.07, -5.71  # far south-west
    edinburgh_cell = h3.latlng_to_cell(edinburgh_lat, edinburgh_lon, 6)
    lands_end_cell = h3.latlng_to_cell(lands_end_lat, lands_end_lon, 6)

    weights = compute_h3_grid_weights(
        nwp_grid_size_degrees=grid, h3_index=[edinburgh_cell, lands_end_cell]
    )
    edinburgh = weights.filter(pl.col("h3_index") == edinburgh_cell)
    lands_end = weights.filter(pl.col("h3_index") == lands_end_cell)
    edinburgh_lats, edinburgh_lons = edinburgh["nwp_lat"].to_list(), edinburgh["nwp_lon"].to_list()
    lands_end_lats, lands_end_lons = lands_end["nwp_lat"].to_list(), lands_end["nwp_lon"].to_list()

    # Absolute: each landmark's grid points sit within ~1° of the landmark itself. This is what a
    # lat/lon swap breaks (it would put Edinburgh's nwp_lat near -3, not +56).
    assert 55.0 <= min(edinburgh_lats) and max(edinburgh_lats) <= 57.0
    assert -4.0 <= min(edinburgh_lons) and max(edinburgh_lons) <= -2.0
    assert 49.0 <= min(lands_end_lats) and max(lands_end_lats) <= 51.0
    assert -7.0 <= min(lands_end_lons) and max(lands_end_lons) <= -4.0

    # Relative: north stays north (larger lat), west stays west (smaller lon).
    assert min(edinburgh_lats) > max(lands_end_lats)
    assert max(lands_end_lons) < min(edinburgh_lons)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: compute_h3_grid_weights(0.25, []), "empty"),
        (lambda: compute_h3_grid_weights(0.0, _res5_cells()), "strictly positive"),
        (
            lambda: compute_h3_grid_weights(
                0.25, [h3.latlng_to_cell(52.5, -1.5, 4), h3.latlng_to_cell(52.5, -1.5, 5)]
            ),
            "same resolution",
        ),
        (lambda: compute_h3_grid_weights(0.25, _res5_cells(), child_h3_res=5), "strictly greater"),
    ],
)
def test_grid_weights_validation_errors(call: Callable[[], object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


def test_for_boundary_generates_weights() -> None:
    boundary = shapely.box(-2.0, 52.0, -1.0, 53.0)  # A 1x1-degree box over central GB.
    # An explicit child_h3_res (rather than the h3_res + 2 default) exercises the caller-supplied
    # sampling-resolution path through to compute_h3_grid_weights.
    weights = compute_h3_grid_weights_for_boundary(
        boundary=boundary, nwp_grid_size_degrees=0.25, h3_res=5, child_h3_res=8
    )
    assert set(weights.columns) == _EXPECTED_COLUMNS
    assert weights.height > 0


def test_for_boundary_empty_raises() -> None:
    # A sub-degree box at resolution 0 (cells ~1000km across) contains no cell centre, so
    # h3.geo_to_cells returns no cells and the wrapper raises.
    tiny_box = shapely.box(-1.5001, 52.5, -1.5, 52.5001)
    with pytest.raises(ValueError, match="No H3 cells"):
        compute_h3_grid_weights_for_boundary(
            boundary=tiny_box, nwp_grid_size_degrees=0.25, h3_res=0
        )
