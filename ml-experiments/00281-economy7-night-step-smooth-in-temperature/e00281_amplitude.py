"""Print each decision substation's fitted step amplitude against temperature, from the `e2` fit.

The amplitude at a 24-hour smoothed temperature `T` is the sum over the four knots of the hinge
coefficient times `max(0, knot - T) / HEATING_SCALE_C`, in points of the substation's rated demand,
for the fit at the cut 2026-06-01 (the last monthly cut of the scored year). The prediction was that
the amplitude is larger below 3 C than at 3 C at 4 of 5 substations.

    uv run --no-sync python ml-experiments/00281-*/e00281_amplitude.py
"""

from datetime import UTC, datetime
from typing import Final

import e00281_common as common
import e00281_design as design
import e00281_tall as tall
import numpy as np
from studies.dp_gam import HEATING_SCALE_C
from studies.dp_pinned_capacity import read_vintages

CUT: Final[datetime] = datetime(2026, 6, 1, tzinfo=UTC)
TEMPERATURES_C: Final[tuple[float, ...]] = (-3.0, 0.0, 3.0, 6.0, 9.0, 12.0, 15.5)


def main() -> None:
    """Fit `e2` at each decision substation and write the amplitude curve."""
    cache = common.load_cache()
    vintages = read_vintages()
    knots = np.array(design.STEP_KNOTS_C)
    out: dict[str, dict] = {}
    colder = 0
    for site in common.DECISION_SITES:
        fit = tall.fit_tall(
            cache=cache,
            vintages=vintages,
            site=site,
            cut=CUT,
            penalty_weight=common.T_ALL_LAMBDA,
            variant="e2",
        )
        coefficients = fit.fit.theta[-len(knots) :]
        curve = {
            t: float(np.sum(coefficients * np.clip(knots - t, 0.0, None)) / HEATING_SCALE_C)
            for t in TEMPERATURES_C
        }
        colder += int(curve[0.0] > curve[3.0])
        out[site] = {"coefficients": coefficients.tolist(), "amplitude_by_temperature_c": curve}
        print(site, {t: round(v, 2) for t, v in curve.items()})
    out["summary"] = {"larger_at_0c_than_3c": colder}
    print(f"amplitude at 0 C above amplitude at 3 C at {colder} of 5 substations")
    common.write_json_atomic(path=common.WORK_DIR / "amplitude.json", content=out)


if __name__ == "__main__":
    main()
