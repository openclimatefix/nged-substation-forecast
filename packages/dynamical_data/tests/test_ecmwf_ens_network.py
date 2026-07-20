"""Network-gated end-to-end test against the *real* Dynamical.org ECMWF ENS catalog.

Every test here is marked ``network`` and is therefore **deselected by default** (see the
``-m "not network"`` in the root ``pyproject.toml``); a plain ``uv run pytest`` — local dev and the
per-PR CI — never touches the network. Run these explicitly with ``uv run pytest -m network``
(nightly CI only).

Why this exists — the "shared-convention blind spot". Every other ``dynamical_data`` test runs on
the synthetic ``xr.Dataset`` built by ``conftest.py``, which *encodes our assumptions* about the
real catalog: dimension order, descending latitude, longitude in [-180, 180], coordinate names,
dtypes, and physical units. If Dynamical ever delivers latitude ascending, longitude in [0, 360],
or temperature in Kelvin, the offline tests stay green because the fixture and the code share the
same (now-wrong) assumption. Only a run against real data can catch that — which is what this does.

See also the offline orientation test
``test_convert_to_polars.py::test_convert_maps_each_grid_point_to_its_own_lat_lon`` (proves
``convert`` preserves the value↔lat/lon pairing) and
``geo/tests/test_h3.py::test_grid_weights_preserve_geographic_orientation`` (proves the H3→lat/lon
labels are geographically right). This test composes both against the genuine dataset.
"""

from datetime import datetime, timedelta, timezone

import pytest
import shapely
from contracts.weather_schemas import Nwp
from dynamical_data.ecmwf_ens import download
from dynamical_data.ecmwf_ens.convert_to_polars import (
    convert_nwp_xarray_dataset_to_polars_dataframe as convert,
)
from geo.h3 import compute_h3_grid_weights_for_boundary


def _recent_init_time() -> datetime:
    """A 00:00 UTC ECMWF ENS run old enough to be reliably published (three days ago)."""
    three_days_ago = datetime.now(timezone.utc) - timedelta(days=3)
    return three_days_ago.replace(hour=0, minute=0, second=0, microsecond=0)


@pytest.mark.network
def test_real_ecmwf_ens_pipeline_conventions_match_offline_fixtures() -> None:
    """Open → download → convert on real data; assert the conventions the offline fixtures assume.

    A small box near Edinburgh keeps the download tiny. The assertions target exactly the
    structural conventions the synthetic fixture bakes in, plus a physical-range sanity check on
    temperature that would expose a Kelvin/°C unit mismatch even though ``Nwp.validate`` alone
    would also reject it (Kelvin ~288 is far outside the schema's [-100, 100] °C bounds).
    """
    boundary = shapely.box(-3.5, 55.5, -3.0, 56.0)  # ~0.5° box over Edinburgh
    h3_grid = compute_h3_grid_weights_for_boundary(
        boundary=boundary, nwp_grid_size_degrees=0.25, h3_res=5, child_h3_res=8
    )

    sliced = download.open_ecmwf_ens_run(_recent_init_time(), h3_grid)

    # --- Convention checks on the raw catalog data (the "blind spot") ---
    lats = sliced.latitude.values
    lons = sliced.longitude.values
    assert lats[0] > lats[-1], "latitude must be descending, as the offline fixture assumes"
    assert lons.min() >= -180.0 and lons.max() <= 180.0, "longitude must be in [-180, 180]"
    # The slice must land on the requested box, not its lat/lon transpose.
    assert 55.0 <= lats.min() and lats.max() <= 56.5
    assert -3.6 <= lons.min() and lons.max() <= -2.9

    downloaded = download.download_ecmwf_ens_data(sliced)
    df = convert(ds=downloaded, h3_grid=h3_grid)

    # --- Whole-pipeline checks on real values ---
    Nwp.validate(df)  # dtype/bounds/unit contract on genuine data
    assert df.height > 0
    # GB near-surface air temperature is physically in this band; Kelvin would blow it.
    temperatures = df["temperature_2m"].to_list()
    assert min(temperatures) >= -30.0
    assert max(temperatures) <= 45.0
