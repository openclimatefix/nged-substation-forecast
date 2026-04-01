import polars as pl
import pytest
from geo.h3 import compute_h3_grid_weights


def test_compute_h3_grid_weights_empty():
    df = pl.DataFrame(schema={"h3_index": pl.UInt64})
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        compute_h3_grid_weights(df, grid_size=0.25)


def test_compute_h3_grid_weights_coarser_child_res_fails_validation():
    # Res 5 cell
    df = pl.DataFrame({"h3_index": [0x85194AD7FFFFFFF]}, schema={"h3_index": pl.UInt64})
    # Try child_res 4 (coarser than res 5)
    # This should fail with a ValueError
    with pytest.raises(ValueError, match="child_res .* must be strictly greater than h3_res"):
        compute_h3_grid_weights(df, grid_size=0.25, child_res=4)


def test_compute_h3_grid_weights_single_cell():
    # London res 5 cell
    df = pl.DataFrame({"h3_index": [0x85194AD7FFFFFFF]}, schema={"h3_index": pl.UInt64})
    result = compute_h3_grid_weights(df, grid_size=0.25, child_res=7)
    assert not result.is_empty()
    # Check that proportions sum to 1.0
    assert result["proportion"].sum() == pytest.approx(1.0)
    # Check that total is correct (7^2 = 49 children for 2 levels difference)
    assert result["total"][0] == 49
    assert result["len"].sum() == 49
