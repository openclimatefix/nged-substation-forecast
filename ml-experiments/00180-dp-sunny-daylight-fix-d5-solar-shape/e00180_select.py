"""Arm P's selection: choose each substation's tilt and azimuth on data before 2024-04-01.

For each of the 25 cells (tilt 15 to 55 degrees, azimuth 135 to 225 degrees, DC:AC 1.15) the shape
is
scaled by one number per substation, so that its sum over the daylight half-hours of
2023-04-01 to 2024-03-31 on CAMS irradiance equals the fleet shape's. T-all is then fitted at the
quarterly cuts 2023-04-01, 2023-07-01, 2023-10-01, and 2024-01-01 and scored on the sunny daylight
half-hours of the quarter after each cut (the top third of the quarter-window's daylight half-hours
by the fleet shape). The cell with the lowest mean absolute error replaces the fleet shape only if
it is at least 1% lower than the fleet cell's. D9 has no pinned-capacity vintage before 2025-04, so
it keeps the fleet shape.

    E00180_ARM=sel uv run --no-sync python ml-experiments/00180-*/e00180_select.py
"""

import json
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from itertools import product
from typing import Final

import e00180_common as common
import e00180_tall as tall
import numpy as np
from studies.dp_ens_weather import SunGeometry, replace_site
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_pinned_capacity import read_vintages
from studies.pv_physics import PlantParameters

SELECT_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
TILTS: Final[tuple[float, ...]] = (15.0, 25.0, 35.0, 45.0, 55.0)
AZIMUTHS: Final[tuple[float, ...]] = (135.0, 157.5, 180.0, 202.5, 225.0)
CUTS: Final[tuple[datetime, ...]] = (
    datetime(2023, 4, 1, tzinfo=UTC),
    datetime(2023, 7, 1, tzinfo=UTC),
    datetime(2023, 10, 1, tzinfo=UTC),
    datetime(2024, 1, 1, tzinfo=UTC),
)
SCORE_ENDS: Final[tuple[datetime, ...]] = (*CUTS[1:], datetime(2024, 4, 1, tzinfo=UTC))
"""The quarter scored after each cut ends at the next cut."""
WINDOW_START: Final[datetime] = datetime(2023, 4, 1, tzinfo=UTC)
WINDOW_END: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
MARGIN: Final[float] = 0.01
"""A candidate must beat the fleet cell's error by this share to replace the fleet shape."""
SUN_UP: Final[float] = 0.01


def shape_of(*, cache: FeatureCache, site: str, plant: PlantParameters) -> np.ndarray:
    """Return the unit PV shape of a plant over the whole grid from the cache's CAMS irradiance."""
    features = cache.sites[site]
    n_even = len(cache.end_times) // 2 * 2
    sun = SunGeometry.build(
        end_times=cache.end_times[:n_even],
        latitude=features.table.latitude,
        longitude=features.table.longitude,
    )
    shape, _ = sun.pv_shape_and_ghi(ghi_w_m2=features.ghi_w_m2[None, :n_even], plant=plant)
    return np.concatenate([shape[0], np.zeros(len(cache.end_times) - n_even)])


def _job(site: str) -> dict:
    cache = common.load_cache()
    vintages = read_vintages()
    features = cache.sites[site]
    ends = cache.end_times
    fleet = shape_of(cache=cache, site=site, plant=common.FLEET)
    in_window = (ends > naive_utc(moment=WINDOW_START)) & (ends <= naive_utc(moment=WINDOW_END))
    daylight = in_window & np.isfinite(fleet) & (fleet > SUN_UP)
    quarters = []
    for cut, end in zip(CUTS, SCORE_ENDS, strict=True):
        start, stop = naive_utc(moment=cut), naive_utc(moment=end)
        rows = np.flatnonzero(
            (ends > start) & (ends <= stop) & np.isfinite(fleet) & (fleet > SUN_UP)
        )
        edge = np.quantile(fleet[rows], 2 / 3)
        quarters.append(rows[(fleet[rows] >= edge) & np.isfinite(features.y[rows])])
    surface = {}
    for tilt, azimuth in product(TILTS, AZIMUTHS):
        raw = PlantParameters(
            tilt_deg=tilt,
            azimuth_deg=azimuth,
            dc_capacity_mw=common.FLEET.dc_capacity_mw,
            ac_capacity_mw=common.FLEET.ac_capacity_mw,
        )
        candidate = shape_of(cache=cache, site=site, plant=raw)
        scale = float(fleet[daylight].sum() / candidate[daylight].sum())
        shape = candidate * scale
        variant = replace_site(cache=cache, site=site, pv_unit_shape=shape)
        errors = []
        for cut, rows in zip(CUTS, quarters, strict=True):
            fit = tall.fit_tall(
                cache=variant,
                vintages=vintages,
                site=site,
                cut=cut,
                penalty_weight=common.T_ALL_LAMBDA,
            )
            ok = rows[np.isfinite(shape[rows])]
            errors.append(
                np.abs(features.y[ok] - tall.predict_tall(cache=variant, fit=fit, index=ok))
            )
        surface[f"{tilt:g}_{azimuth:g}"] = {
            "tilt_deg": tilt,
            "azimuth_deg": azimuth,
            "scale": scale,
            "mae_points": float(np.concatenate(errors).mean()),
        }
    return {"site": site, "surface": surface, "n_rows": [len(q) for q in quarters]}


def main() -> None:
    """Select the cells, write `work/select.json`, and add arm `p` to the arm specs."""
    with ProcessPoolExecutor(max_workers=len(SELECT_SITES)) as pool:
        results = list(pool.map(_job, SELECT_SITES))
    chosen = {}
    for result in results:
        surface = result["surface"]
        best = min(surface.values(), key=lambda cell: cell["mae_points"])
        fleet = surface["35_180"]
        better = best["mae_points"] <= fleet["mae_points"] * (1 - MARGIN)
        result["fleet_mae"] = fleet["mae_points"]
        result["best"] = best
        result["changed"] = bool(better)
        if better:
            chosen[result["site"]] = {
                "tilt_deg": best["tilt_deg"],
                "azimuth_deg": best["azimuth_deg"],
                "scale": best["scale"],
            }
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    common.write_json_atomic(
        path=common.WORK_DIR / "select.json", content={r["site"]: r for r in results}
    )
    specs = json.loads(common.SPEC_PATH.read_text()) if common.SPEC_PATH.exists() else {}
    specs["p"] = chosen
    common.write_json_atomic(path=common.SPEC_PATH, content=specs)
    for result in results:
        print(
            result["site"],
            "changed" if result["changed"] else "kept fleet",
            round(result["fleet_mae"], 3),
            result["best"],
            flush=True,
        )


if __name__ == "__main__":
    main()
