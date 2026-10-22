"""Learn the solar export cap at each monthly cut from data before the cut; write arm specs.

Two arms learn the cap, and a third softens the hand-set clip:

- `p` (profile): `studies.dp_gam.profile_clip` fits the convex model once for each ceiling share
  from 0.40 to 1.00 and keeps the share with the lowest penalised training objective, with the
  capacity multiplier fixed at 1. The training rows end at the cut.
- `e` (envelope): the solar output that the unclipped fit implies on the clear-sky half-hours of
  the 12 months before the cut is `xθ - y`, the fitted demand model minus the metered net power.
  The ceiling is the `ENVELOPE_QUANTILE` quantile of that output, over the capacity held after the
  cut. A clear-sky half-hour has a clear-sky index of 0.8 or more and a unit PV shape of at least
  half of the window's peak.
- `sat`: the hand-set 60% ceiling with a clip four times softer than the default.

The hand-set 60% clip (`k60`) and the as-of choice are experiment 214's, read from its stored
forecasts. Writes `work/arm_specs.json` and `work/learned_caps.json`.

    systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00247-*/e00247_learn.py
"""

import json
from typing import Final

import e00247_common as common
import e00247_tall as tall
import numpy as np
from studies.dp_ens_weather import SunGeometry
from studies.dp_gam import pinned_solar_points, profile_clip
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D16")
ENVELOPE_QUANTILE: Final[float] = 0.90
ENVELOPE_MONTHS_DAYS: Final[int] = 365
CLEAR_SKY_INDEX: Final[float] = 0.8
PEAK_SHARE: Final[float] = 0.5
MIN_ROWS: Final[int] = 200
SOFT_BETA: Final[float] = 0.5
SHARE_RANGE: Final[tuple[float, float]] = (0.3, 1.0)


def envelope_share(*, cache, site: str, fit: tall.TallFit, vintages) -> tuple[float, int]:  # noqa: ANN001
    """Return the envelope ceiling share at a fit's cut and the number of clear-sky half-hours."""
    features = cache.sites[site]
    cut = np.datetime64(fit.cut.replace(tzinfo=None))
    ends = cache.end_times
    window = (ends > cut - np.timedelta64(ENVELOPE_MONTHS_DAYS, "D")) & (ends <= cut)
    index = np.flatnonzero(window)
    sun = SunGeometry.build(
        end_times=ends[: len(ends) // 2 * 2],
        latitude=features.table.latitude,
        longitude=features.table.longitude,
    )
    clear = np.concatenate([sun.clear_sky_w_m2, np.zeros(len(ends) - len(sun.clear_sky_w_m2))])
    index = index[np.isfinite(features.y[index]) & (clear[index] > 0)]
    csi = features.ghi_w_m2[index] / clear[index]
    unit = features.pv_unit_shape[index]
    keep = (csi >= CLEAR_SKY_INDEX) & (unit >= PEAK_SHARE * np.nanmax(unit))
    index = index[keep]
    if len(index) < MIN_ROWS:
        return 1.0, len(index)
    fitted = tall.predict_tall(cache=cache, fit=fit, index=index)
    pv = pinned_solar_points(
        cache=cache, features=features, pinned=fit.pinned, index=index, clip=None
    )
    implied = fitted + pv - features.y[index]
    share = float(np.quantile(implied, ENVELOPE_QUANTILE)) / float(fit.pinned.held_points)
    return float(np.clip(share, *SHARE_RANGE)), len(index)


def main() -> None:
    """Write the learned ceiling share of every site and cut, and the arm specs."""
    cache = common.load_cache()
    vintages = read_vintages()
    learned: dict = {}
    for site in SITES:
        features = cache.sites[site]
        fits = tall.load_control_fits(site=site)
        profile_table, envelope_table, counts = {}, {}, {}
        for fit in fits:
            cut = fit.cut
            key = cut.strftime("%Y-%m-%d")
            pinned = pinned_capacity(vintages=vintages, table=features.table, cut=cut)
            profile = profile_clip(
                cache=cache,
                site=site,
                spec=common.base_spec(),
                cut=cut,
                window=common.base_window(),
                season_years=common.SEASON_YEARS,
                pinned=pinned,
                multipliers=(1.0,),
            )
            profile_table[key] = float(profile.best.p_max_share)
            envelope_table[key], counts[key] = envelope_share(
                cache=cache, site=site, fit=fit, vintages=vintages
            )
            print(
                site,
                key,
                profile_table[key],
                round(envelope_table[key], 3),
                counts[key],
                flush=True,
            )
        learned[site] = {"p": profile_table, "e": envelope_table, "n_clear": counts}
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    (common.WORK_DIR / "learned_caps.json").write_text(json.dumps(learned, indent=1))
    specs: dict = {"p": {}, "e": {}, "sat": {}}
    for site in SITES:
        specs["p"][site] = {"clip": True, "clip_table": learned[site]["p"]}
        specs["e"][site] = {"clip": True, "clip_table": learned[site]["e"]}
        specs["sat"][site] = {"clip": True, "clip_share": 0.6, "beta": SOFT_BETA}
    common.write_json_atomic(path=common.SPEC_PATH, content=specs)


if __name__ == "__main__":
    main()
