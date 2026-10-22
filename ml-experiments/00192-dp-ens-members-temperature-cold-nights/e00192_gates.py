"""The re-registered known-answer gate G3' of experiment 192, run on a fake ensemble.

The fake ensemble is the CAMS irradiance of six scored runs at D10 times 51 equally spaced
multipliers from 0.6 to 1.4, as in experiment 184's gate G3.

- G3' (the gate): on every daylight row, the members' responses are non-increasing in the members'
  own unit PV shape. The response is a fitted solar coefficient times the difference between the
  member's unit PV shape and the ensemble mean's, so it must follow the shape, whatever
  irradiance does. Solar output lowers demand, hence non-increasing.
- Wiring checks, as in experiment 184: the multiplier-1 member's response is zero, and responses
  at night are zero.
- Diagnostic (not a gate): the response is non-increasing in the irradiance multiplier, which
  experiment 184's G3 required. The count is split at a solar elevation of 17 degrees.

    uv run --no-sync python ml-experiments/00192-*/e00192_gates.py
"""

import json
from dataclasses import replace

import e00192_common as common
import e00192_members as members
import e00192_tall as tall
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.dp_ens_weather import Calibration, EnsSiteRun

GATE_SITE = "D10"
N_FAKE_RUNS = 6
MULTIPLIERS = np.linspace(0.6, 1.4, members.N_MEMBERS)
TOLERANCE = 1e-9
LOW_SUN_ELEVATION_DEG = 17.0


def main() -> None:
    """Run G3' and write `gates.json`."""
    torch.set_num_threads(1)
    cache = common.load_cache()
    setup = members.load_context(site=GATE_SITE, cache=cache)
    fits = tall.load_fits(cadence="monthly", site=GATE_SITE)
    features = cache.sites[GATE_SITE]
    scored = setup.indices(periods={"scored"})[10 : 10 + N_FAKE_RUNS]
    fake = replace(setup.with_calibration(calibration=Calibration.identity()))
    counts = {
        "daylight_rows": 0,
        "shape_monotone_rows": 0,
        "irradiance_monotone_rows": 0,
        "irradiance_monotone_rows_elevation_17_and_over": 0,
        "daylight_rows_elevation_17_and_over": 0,
        "night_nonzero": 0,
        "multiplier_one_member_max_abs_response": 0.0,
    }
    middle = int(np.argmin(np.abs(MULTIPLIERS - 1.0)))
    for index in scored:
        run = setup.runs[index]
        n = run.shortwave_w_m2.shape[1]
        rows = np.arange(run.origin_position + 1, run.origin_position + 1 + n)
        truth = np.nan_to_num(features.ghi_w_m2[rows])
        temperature = np.tile(np.nan_to_num(features.temperature_c[rows]), (members.N_MEMBERS, 1))
        fake_run = EnsSiteRun(
            origin_position=run.origin_position,
            temperature_c=temperature,
            shortwave_w_m2=MULTIPLIERS[:, None] * truth[None, :],
        )
        runs = list(fake.runs)
        runs[index] = fake_run
        full = members.compute_run(
            cache=cache, setup=replace(fake, runs=runs), fits=fits, index=index
        )
        delta, shape = full["delta_m"], full["shape_m"]
        daylight = full["clear"] > members.MIN_CLEAR_SKY_W_M2 * 10
        order = np.argsort(shape[daylight], axis=1, kind="stable")
        by_shape = np.take_along_axis(delta[daylight], order, axis=1)
        counts["shape_monotone_rows"] += int(
            (np.diff(by_shape, axis=1) <= TOLERANCE).all(axis=1).sum()
        )
        counts["daylight_rows"] += int(daylight.sum())
        by_irradiance = np.diff(delta[daylight], axis=1) <= TOLERANCE
        high_sun = full["elevation"][daylight] >= LOW_SUN_ELEVATION_DEG
        counts["irradiance_monotone_rows"] += int(by_irradiance.all(axis=1).sum())
        counts["irradiance_monotone_rows_elevation_17_and_over"] += int(
            by_irradiance.all(axis=1)[high_sun].sum()
        )
        counts["daylight_rows_elevation_17_and_over"] += int(high_sun.sum())
        counts["night_nonzero"] += int((np.abs(delta[full["clear"] <= 0.0]) > TOLERANCE).sum())
        counts["multiplier_one_member_max_abs_response"] = max(
            counts["multiplier_one_member_max_abs_response"], float(np.abs(delta[:, middle]).max())
        )
    counts["g3_prime_passes"] = counts["shape_monotone_rows"] == counts["daylight_rows"]
    common.write_json_atomic(path=common.WORK_DIR / "gates.json", content=counts)
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
