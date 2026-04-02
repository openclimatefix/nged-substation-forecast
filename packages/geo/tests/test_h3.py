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


def test_compute_h3_grid_weights_longitude_wrap_around():
    """Test that longitude normalization correctly handles values near the anti-meridian."""
    # H3 cell near the anti-meridian (179.9 longitude)
    h3_index = 0x857EB50FFFFFFFF
    df = pl.DataFrame({"h3_index": [h3_index]}, schema={"h3_index": pl.UInt64})

    # Use a grid size that might push it over 180 if not normalized
    grid_size = 0.25
    result = compute_h3_grid_weights(df, grid_size=grid_size, child_res=7)

    # Check that all nwp_lng are within [-180, 180)
    assert (result["nwp_lng"] >= -180).all()
    assert (result["nwp_lng"] < 180).all()

    # Specifically check if any value was wrapped
    # If lng is 179.9 and grid_size is 0.25:
    # (179.9 + 0.125) / 0.25 = 180.025 / 0.25 = 720.1
    # floor(720.1) * 0.25 = 180.0
    # (180.0 + 180) % 360 - 180 = 360 % 360 - 180 = 0 - 180 = -180.0
    assert -180.0 in result["nwp_lng"].to_list()
