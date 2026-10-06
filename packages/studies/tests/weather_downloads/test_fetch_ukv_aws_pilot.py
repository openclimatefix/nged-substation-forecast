"""Tests for the pure helpers of `studies/weather_downloads/fetch_ukv_aws_pilot.py`.

No test touches the network or `data/`. Each is built to fail on the bug it exists for: a packed
or filled variable stored undecoded, an object key built with the wrong valid time, a crop
rectangle with rows and columns swapped, and a listing that loops forever.
"""

import importlib
from types import ModuleType
from xml.etree import ElementTree

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("fsspec")
pytest.importorskip("aiohttp")
pyproj = pytest.importorskip("pyproj")


@pytest.fixture(scope="module")
def pilot() -> ModuleType:
    """Import the script as a module."""
    return importlib.import_module("fetch_ukv_aws_pilot")


def test_decode_values_applies_packing_and_fill(pilot: ModuleType) -> None:
    raw = np.array([0, 10, -999], dtype=np.int16)
    decoded = pilot.decode_values(
        raw=raw, attributes={"scale_factor": "0.5", "add_offset": "100", "_FillValue": "-999"}
    )
    assert decoded[:2].tolist() == [100.0, 105.0]
    assert np.isnan(decoded[2])
    assert decoded.dtype == np.float32


def test_decode_values_leaves_unpacked_floats_alone(pilot: ModuleType) -> None:
    raw = np.array([1.5, 2.5], dtype=np.float32)
    assert pilot.decode_values(raw=raw, attributes={}).tolist() == [1.5, 2.5]


def test_decode_values_rejects_strings(pilot: ModuleType) -> None:
    with pytest.raises(TypeError):
        pilot.decode_values(raw=np.array(["a"]), attributes={})


def test_wanted_keys_builds_the_valid_time_and_lead_into_each_name(pilot: ModuleType) -> None:
    run = "20261002T2300Z"
    expected = (
        "uk-deterministic-2km/20261002T2300Z/"
        "20261003T0200Z-PT0003H00M-temperature_at_screen_level.nc"
    )
    available = {expected: 1}
    found, missing = pilot.wanted_keys(run=run, available=available)
    assert found == [(expected, 3, "temperature_at_screen_level")]
    assert len(missing) == len(pilot.LEADS_HOURS) * len(pilot.ALL_FILES) - 1


def test_crop_rectangle_rows_follow_y_and_columns_follow_x(pilot: ModuleType) -> None:
    crs = pyproj.CRS.from_proj4("+proj=laea +lat_0=54.9 +lon_0=-2.5 +x_0=0 +y_0=0 +R=6371229")
    x = np.arange(-100_000.0, 100_000.0, 2000.0)
    y = np.arange(-50_000.0, 50_000.0, 2000.0)
    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x[60], y[20])
    margin = 0.01
    rectangle = pilot.compute_crop_rectangle(
        x=x, y=y, crs=crs, box_bounds=(lat - margin, lat + margin, lon - margin, lon + margin)
    )
    assert rectangle.row_start <= 20 < rectangle.row_stop
    assert rectangle.col_start <= 60 < rectangle.col_stop


def test_crop_rectangle_raises_when_the_box_is_off_the_grid(pilot: ModuleType) -> None:
    crs = pyproj.CRS.from_proj4("+proj=laea +lat_0=54.9 +lon_0=-2.5 +R=6371229")
    axis = np.arange(-1000.0, 1000.0, 2000.0)
    with pytest.raises(RuntimeError):
        pilot.compute_crop_rectangle(x=axis, y=axis, crs=crs, box_bounds=(0.0, 1.0, 0.0, 1.0))


def test_truncated_listing_without_a_token_raises(pilot: ModuleType) -> None:
    namespace = pilot._S3_NAMESPACE
    page = ElementTree.fromstring(
        f'<R xmlns="{namespace[1:-1]}"><IsTruncated>true</IsTruncated></R>'
    )
    with pytest.raises(RuntimeError):
        pilot._next_token(page=page)
