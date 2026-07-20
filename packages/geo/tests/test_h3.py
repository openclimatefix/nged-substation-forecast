"""Unit tests for ``geo.h3`` grid-weight computation.

These pin down the load-bearing invariant (every H3 parent's child proportions sum to 1), the four
documented ``ValueError`` guards on ``compute_h3_grid_weights``, and the boundary wrapper's real
cell generation plus its empty-boundary error path.
"""

from collections.abc import Callable

import h3.api.basic_int as h3
import polars as pl
import pytest
import shapely
from geo.h3 import compute_h3_grid_weights, compute_h3_grid_weights_for_boundary

_EXPECTED_COLUMNS: frozenset[str] = frozenset({"h3_index", "nwp_lat", "nwp_lon", "proportion"})


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
    weights = compute_h3_grid_weights_for_boundary(
        boundary=boundary, nwp_grid_size_degrees=0.25, h3_res=5
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
