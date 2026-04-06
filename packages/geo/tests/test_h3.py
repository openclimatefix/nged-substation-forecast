import polars as pl
import pytest
from geo.h3 import compute_h3_grid_weights


def test_compute_h3_grid_weights_invalid_grid_size():
    """Test that grid_size validation raises ValueError for non-positive values."""
    df = pl.DataFrame({"h3_index": [0x85194AD7FFFFFFF]}, schema={"h3_index": pl.UInt64})
    with pytest.raises(ValueError, match="grid_size must be strictly positive"):
        compute_h3_grid_weights(df, grid_size=0)
    with pytest.raises(ValueError, match="grid_size must be strictly positive"):
        compute_h3_grid_weights(df, grid_size=-0.25)
