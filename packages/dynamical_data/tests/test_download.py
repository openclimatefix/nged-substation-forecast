"""Tests for ``dynamical_data.ecmwf_ens.download``.

The only external dependency is ``dynamical_catalog.open`` (the Dynamical.org network call),
which is patched with ``monkeypatch`` to return an in-memory synthetic dataset. Everything else
runs offline.
"""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import patito as pt
import pytest
import xarray as xr
from contracts.geo_schemas import H3GridWeights
from dynamical_data.ecmwf_ens import download

_INIT_TIME = datetime(2024, 1, 1, tzinfo=UTC)
_INIT_NP = np.datetime64("2024-01-01T00:00:00", "ns")


def _patch_catalog(monkeypatch: pytest.MonkeyPatch, ds: xr.Dataset) -> None:
    monkeypatch.setattr(download.dynamical_catalog, "open", lambda *args, **kwargs: ds)


# --------------------------------------------------------------------------------------------------
# open_ecmwf_ens_run
# --------------------------------------------------------------------------------------------------


def test_open_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    make_ens_dataset: Callable[..., xr.Dataset],
    default_h3_grid: pt.DataFrame[H3GridWeights],
) -> None:
    ds = make_ens_dataset(init_time=_INIT_TIME, init_time_as_dim=True)
    _patch_catalog(monkeypatch, ds)

    result = download.open_ecmwf_ens_run(_INIT_TIME, default_h3_grid)

    assert set(result.data_vars) == set(download._ECMWF_ENS_VARS_TO_DOWNLOAD)
    # init_time was selected down to a scalar coordinate.
    assert result["init_time"].values == _INIT_NP
    # The grid bounds cover both grid points, so nothing is sliced away.
    assert result.latitude.size == 2
    assert result.longitude.size == 2


def test_open_accepts_non_utc_timezone(
    monkeypatch: pytest.MonkeyPatch,
    make_ens_dataset: Callable[..., xr.Dataset],
    default_h3_grid: pt.DataFrame[H3GridWeights],
) -> None:
    ds = make_ens_dataset(init_time=_INIT_TIME, init_time_as_dim=True)
    _patch_catalog(monkeypatch, ds)

    # 2024-01-01 00:00 UTC expressed in UTC-5.
    est = timezone(timedelta(hours=-5))
    nwp_init_time = datetime(2023, 12, 31, 19, 0, tzinfo=est)

    result = download.open_ecmwf_ens_run(nwp_init_time, default_h3_grid)

    assert result["init_time"].values == _INIT_NP


def test_open_raises_on_empty_h3_grid(
    make_h3_grid: Callable[..., pt.DataFrame[H3GridWeights]],
) -> None:
    empty = make_h3_grid(h3_index=[], nwp_lat=[], nwp_lon=[], proportion=[])
    with pytest.raises(ValueError, match="empty"):
        download.open_ecmwf_ens_run(_INIT_TIME, empty)


def test_open_raises_on_naive_datetime(
    default_h3_grid: pt.DataFrame[H3GridWeights],
) -> None:
    naive = datetime(2024, 1, 1)
    with pytest.raises(ValueError, match="timezone aware"):
        download.open_ecmwf_ens_run(naive, default_h3_grid)


def test_open_raises_when_run_not_available(
    monkeypatch: pytest.MonkeyPatch,
    make_ens_dataset: Callable[..., xr.Dataset],
    default_h3_grid: pt.DataFrame[H3GridWeights],
) -> None:
    ds = make_ens_dataset(init_time=_INIT_TIME, init_time_as_dim=True)
    _patch_catalog(monkeypatch, ds)

    missing = datetime(2025, 6, 1, tzinfo=UTC)
    with pytest.raises(download.NwpRunNotYetAvailable):
        download.open_ecmwf_ens_run(missing, default_h3_grid)


def test_open_raises_on_empty_coords(
    monkeypatch: pytest.MonkeyPatch,
    make_ens_dataset: Callable[..., xr.Dataset],
    default_h3_grid: pt.DataFrame[H3GridWeights],
) -> None:
    ds = make_ens_dataset(init_time=_INIT_TIME, init_time_as_dim=True).isel(longitude=slice(0, 0))
    _patch_catalog(monkeypatch, ds)

    with pytest.raises(ValueError, match="empty longitude or latitude"):
        download.open_ecmwf_ens_run(_INIT_TIME, default_h3_grid)


@pytest.mark.parametrize("longitudes", [(-1.0, 200.0), (-200.0, -1.0)])
def test_open_raises_on_longitude_out_of_range(
    monkeypatch: pytest.MonkeyPatch,
    make_ens_dataset: Callable[..., xr.Dataset],
    default_h3_grid: pt.DataFrame[H3GridWeights],
    longitudes: tuple[float, float],
) -> None:
    # Both bounds are guarded: above +180 and below -180 must each raise.
    ds = make_ens_dataset(init_time=_INIT_TIME, longitudes=longitudes, init_time_as_dim=True)
    _patch_catalog(monkeypatch, ds)

    with pytest.raises(ValueError, match=r"\[-180, 180\]"):
        download.open_ecmwf_ens_run(_INIT_TIME, default_h3_grid)


def test_open_raises_on_no_spatial_overlap(
    monkeypatch: pytest.MonkeyPatch,
    make_ens_dataset: Callable[..., xr.Dataset],
    make_h3_grid: Callable[..., pt.DataFrame[H3GridWeights]],
) -> None:
    ds = make_ens_dataset(init_time=_INIT_TIME, init_time_as_dim=True)
    _patch_catalog(monkeypatch, ds)

    # A grid whose bounds sit entirely outside the dataset's lat/lon coverage.
    far = make_h3_grid(
        h3_index=[1, 1],
        nwp_lat=[10.0, 9.75],
        nwp_lon=[100.0, 100.25],
        proportion=[0.5, 0.5],
    )
    with pytest.raises(ValueError, match="No spatial overlap"):
        download.open_ecmwf_ens_run(_INIT_TIME, far)


# --------------------------------------------------------------------------------------------------
# download_ecmwf_ens_data
# --------------------------------------------------------------------------------------------------


def test_download_computes_all_variables(
    make_ens_dataset: Callable[..., xr.Dataset],
) -> None:
    ds = make_ens_dataset()  # post-open shape (init_time scalar coordinate)

    result = download.download_ecmwf_ens_data(ds)

    assert set(result.data_vars) == set(download._ECMWF_ENS_VARS_TO_DOWNLOAD)
    np.testing.assert_array_equal(result["temperature_2m"].values, ds["temperature_2m"].values)


# --------------------------------------------------------------------------------------------------
# _calc_slice_for_lat_or_lng
# --------------------------------------------------------------------------------------------------


def test_calc_slice_ascending() -> None:
    ds = xr.Dataset(coords={"longitude": [0.0, 1.0, 2.0]})
    assert download._calc_slice_for_lat_or_lng("longitude", ds, 0.5, 1.5) == slice(0.5, 1.5)


def test_calc_slice_descending() -> None:
    ds = xr.Dataset(coords={"latitude": [2.0, 1.0, 0.0]})
    assert download._calc_slice_for_lat_or_lng("latitude", ds, 0.5, 1.5) == slice(1.5, 0.5)


def test_calc_slice_raises_when_min_equals_max() -> None:
    ds = xr.Dataset(coords={"latitude": [2.0, 1.0, 0.0]})
    with pytest.raises(ValueError, match="cannot be equal"):
        download._calc_slice_for_lat_or_lng("latitude", ds, 1.0, 1.0)


def test_calc_slice_raises_on_single_value_coord() -> None:
    ds = xr.Dataset(coords={"latitude": [1.0]})
    with pytest.raises(ValueError, match="multiple values"):
        download._calc_slice_for_lat_or_lng("latitude", ds, 0.5, 1.5)


# --------------------------------------------------------------------------------------------------
# raise_if_instantaneous_slices_empty
# --------------------------------------------------------------------------------------------------

_SHAPE = (3, 2, 2, 2)  # (lead_time, ensemble_member, latitude, longitude), as in the factory.
_INSTANTANEOUS = sorted(download.ECMWF_ENS_INSTANTANEOUS_VARS)


def _nan_where(
    *, lead_idx: int, member_idx: int, lat_idx: int | slice, lon_idx: int | slice
) -> np.ndarray:
    values = np.ones(_SHAPE, dtype=np.float32)
    values[lead_idx, member_idx, lat_idx, lon_idx] = np.nan
    return values


@pytest.mark.parametrize("variable", _INSTANTANEOUS)
@pytest.mark.parametrize("lead_idx", [0, 2])
def test_empty_slice_of_any_instantaneous_variable_raises(
    make_ens_dataset: Callable[..., xr.Dataset], variable: str, lead_idx: int
) -> None:
    values = _nan_where(lead_idx=lead_idx, member_idx=1, lat_idx=slice(None), lon_idx=slice(None))
    ds = make_ens_dataset(var_values={variable: values})

    with pytest.raises(download.NwpRunNotYetAvailable, match=variable):
        download.raise_if_instantaneous_slices_empty(ds)


def test_slice_with_one_valid_grid_point_does_not_raise(
    make_ens_dataset: Callable[..., xr.Dataset],
) -> None:
    """Scattered NaN, including a whole NaN row or column, is not an unfinished run."""
    row_nan = _nan_where(lead_idx=1, member_idx=0, lat_idx=0, lon_idx=slice(None))
    column_nan = _nan_where(lead_idx=1, member_idx=0, lat_idx=slice(None), lon_idx=0)
    ds = make_ens_dataset(var_values={"temperature_2m": row_nan, "pressure_surface": column_nan})

    download.raise_if_instantaneous_slices_empty(ds)


@pytest.mark.parametrize(
    "variable",
    ["precipitation_surface", "categorical_precipitation_type_surface"],
)
def test_empty_slice_of_a_variable_that_may_be_empty_does_not_raise(
    make_ens_dataset: Callable[..., xr.Dataset], variable: str
) -> None:
    """De-accumulated variables are empty at lead-0, and the categorical one before 2024-11-13."""
    values = _nan_where(lead_idx=0, member_idx=0, lat_idx=slice(None), lon_idx=slice(None))
    ds = make_ens_dataset(var_values={variable: values})

    download.raise_if_instantaneous_slices_empty(ds)


def test_complete_dataset_does_not_raise(make_ens_dataset: Callable[..., xr.Dataset]) -> None:
    download.raise_if_instantaneous_slices_empty(make_ens_dataset())
