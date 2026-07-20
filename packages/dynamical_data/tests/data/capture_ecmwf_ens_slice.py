"""Capture a tiny real ECMWF ENS slice for the hermetic ``test_ecmwf_ens_cached.py`` test.

Run from the uv workspace root:

    uv run python packages/dynamical_data/tests/data/capture_ecmwf_ens_slice.py

It downloads a single ~0.5° box over Edinburgh for one init_time, one lead time, and one ensemble
member from the real Dynamical.org ECMWF ENS catalog, then writes it to
``ecmwf_ens_real_slice.nc`` beside this script. The point of committing a real slice (rather than a
synthetic one) is that it carries the *genuine* conventions — descending latitude, longitude in
[-180, 180], dimension order, dtypes, and physical units — that the offline synthetic fixtures only
assume. ECMWF open data permits redistribution (attribution: ECMWF, via Dynamical.org).

The file is written as netCDF3 (``engine="scipy"``) so that neither capture nor test needs a
netCDF4/h5netcdf dependency; ``ensemble_member`` is downcast to int32 for netCDF3 compatibility.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import shapely
from dynamical_data.ecmwf_ens import download
from geo.h3 import compute_h3_grid_weights_for_boundary

_OUT = Path(__file__).parent / "ecmwf_ens_real_slice.nc"
_BOUNDARY = shapely.box(-3.5, 55.5, -3.0, 56.0)  # ~0.5° box over Edinburgh


def main() -> None:
    init_time = (datetime.now(timezone.utc) - timedelta(days=3)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    h3_grid = compute_h3_grid_weights_for_boundary(
        boundary=_BOUNDARY, nwp_grid_size_degrees=0.25, h3_res=5, child_h3_res=8
    )
    sliced = download.open_ecmwf_ens_run(init_time, h3_grid)

    # Trim to a single lead time and a single ensemble member; keep the full (small) lat/lon box.
    tiny = sliced.isel(lead_time=slice(0, 1), ensemble_member=slice(0, 1)).load()
    tiny = tiny.assign_coords(ensemble_member=tiny.ensemble_member.astype(np.int32))

    # Strip provenance attrs (some are dicts that netCDF3 cannot serialize); the test reads only
    # structure and values, and xarray regenerates CF datetime encoding on write regardless.
    for variable in tiny.variables.values():
        variable.attrs = {}
    tiny.attrs = {}

    tiny.to_netcdf(_OUT, engine="scipy")
    print(f"Wrote {_OUT} ({_OUT.stat().st_size} bytes)")
    print(f"  latitude={tiny.latitude.values}")
    print(f"  longitude={tiny.longitude.values}")
    print(f"  init_time={tiny.init_time.values}  valid_time={tiny.valid_time.values}")


if __name__ == "__main__":
    main()
