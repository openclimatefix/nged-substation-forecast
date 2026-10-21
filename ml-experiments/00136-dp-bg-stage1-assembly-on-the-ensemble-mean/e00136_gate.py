"""Control P2 of experiment 136: T-all with a huge difference penalty reproduces `gamk`.

As the difference penalty grows, T-all's pieces are forced equal and the arm becomes B0, which is
what experiment 111's `gamk` fit is. The control fits T-all at that penalty at a site's monthly cut
and compares its prediction with the stored `gamk` fit's on the same rows. The difference must be
under 1e-3 points (the ridge on each piece is B0's ridge divided by the number of pieces, so the
agreement is close and not exact).

    uv run python ml-experiments/00136-*/e00136_gate.py
"""

import pickle

import e00136_common as common
import e00136_tall as tall
import numpy as np
from studies.dp_gam import GamFit, predict_gam
from studies.dp_pinned_capacity import read_vintages

HUGE_PENALTY = 1e9
TOLERANCE_POINTS = 1e-3


def main() -> None:
    """Compare at the scored year's first cut, at each scored substation."""
    cache = common.load_cache()
    vintages = read_vintages()
    out = {}
    for site in common.SCORED_SITES:
        stored: list[GamFit] = pickle.loads((common.FITS_DIR / "gamk" / f"{site}.pkl").read_bytes())
        fit = next(f for f in stored if f.cut == common.SCORED_FIRST_RUN)
        tall_fit = tall.fit_tall(
            cache=cache,
            vintages=vintages,
            site=site,
            cut=common.SCORED_FIRST_RUN,
            penalty_weight=HUGE_PENALTY,
        )
        features = cache.sites[site]
        usable = np.flatnonzero(
            np.isfinite(features.smoothed_temperature_c).all(axis=1)
            & np.isfinite(features.pv_unit_shape)
            & np.isfinite(features.temperature_c)
            & (cache.end_times > np.datetime64("2025-07-01T00:00", "us"))
        )[:20000]
        difference = tall.predict_tall(cache=cache, fit=tall_fit, index=usable) - predict_gam(
            cache=cache, fit=fit, index=usable
        )
        out[site] = {
            "max_abs_difference_points": float(np.abs(difference).max()),
            "rows": len(usable),
        }
    out["passed"] = all(
        v["max_abs_difference_points"] < TOLERANCE_POINTS
        for v in out.values()
        if isinstance(v, dict)
    )
    common.write_json_atomic(path=common.WORK_DIR / "gate" / "p2.json", content=out)
    print(out)


if __name__ == "__main__":
    main()
