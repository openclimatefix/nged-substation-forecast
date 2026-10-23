"""Gate P0b: the irradiance the engine builds from CAMS agrees with the cache's CAMS values.

Feeds the cache's CAMS irradiance for April to June 2026 through `SunGeometry.pv_shape_and_ghi`
(the step every driven cache uses) and compares the result with the input, per UTC hour of day.

    uv run --no-sync python ml-experiments/00289-*/e00289_gate_stamping.py
"""

import e00289_common as common
import numpy as np
from studies.dp_ens_weather import SunGeometry

SITE = "D5"


def main() -> None:
    """Write `work/gate_stamping.json`."""
    cache = common.load_cache()
    stamps = cache.end_times
    start = int(np.searchsorted(stamps, np.datetime64("2026-04-01T00:30:00")))
    stop = int(np.searchsorted(stamps, np.datetime64("2026-07-01T00:30:00")))
    stop -= (stop - start) % 2
    block = slice(start, stop)
    features = cache.sites[SITE]
    lat, lon = features.table.latitude, features.table.longitude
    sun = SunGeometry.build(end_times=stamps[block], latitude=lat, longitude=lon)
    ghi = features.ghi_w_m2[block]
    _, flux = sun.pv_shape_and_ghi(ghi_w_m2=ghi[None, :])
    flux = flux[0]
    both = np.isfinite(flux) & np.isfinite(ghi)
    hours = (
        (stamps[block] - stamps[block].astype("datetime64[D]")) / np.timedelta64(30, "m")
    ).astype(int)
    profile = {
        int(h): float(np.mean(flux[both & (hours == h)] - ghi[both & (hours == h)]))
        for h in range(48)
        if (both & (hours == h)).any()
    }
    result = {
        "mean_abs_difference_w_m2": float(np.mean(np.abs(flux[both] - ghi[both]))),
        "mean_difference_w_m2": float(np.mean(flux[both] - ghi[both])),
        "max_hour_profile_abs_w_m2": float(max(abs(v) for v in profile.values())),
        "mean_ghi_w_m2": float(np.mean(ghi[both])),
        "rows": int(both.sum()),
    }
    common.write_json_atomic(
        path=common.WORK_DIR / "gate_stamping.json", content={**result, "profile": profile}
    )
    print(result)


if __name__ == "__main__":
    main()
