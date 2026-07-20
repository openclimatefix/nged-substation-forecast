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
