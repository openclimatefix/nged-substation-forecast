"""Known-answer gate G4 of experiment 192, run on a fake ensemble.

The fake ensemble is the ERA5 temperature of six winter scored runs at D10 plus 51 equally spaced
offsets from -4 to +4 K, with the true irradiance in every member. The gate reads night rows only,
so the PV cell-temperature derating does not enter.

- G4 (the gate): on every row where the true temperature is under 8 degrees C at night, the member's
  response is non-increasing in the member's temperature offset (colder weather raises demand),
  and strictly decreasing between the two extreme members.
- Wiring checks: the zero-offset member's response is zero, and every response is zero on rows
  where the true temperature is above 21 degrees C, so that the heating hinge (15.5 C) is
  inactive.

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
OFFSETS_K = np.linspace(-4.0, 4.0, members.N_MEMBERS)
TOLERANCE = 1e-9
COLD_C = 12.0
WARM_C = 17.0


def main() -> None:
    """Run G4 and write `gates.json`."""
    torch.set_num_threads(1)
    cache = common.load_cache()
    setup = members.load_context(site=GATE_SITE, cache=cache)
    fits = tall.load_fits(cadence="monthly", site=GATE_SITE)
    features = cache.sites[GATE_SITE]
    winter = [i for i in setup.indices(periods={"scored"}) if setup.origins[i].month in (12, 1, 2)]
    scored = winter[:: len(winter) // N_FAKE_RUNS][:N_FAKE_RUNS]
    fake = replace(setup.with_calibration(calibration=Calibration.identity()))
    counts = {
        "cold_rows": 0,
        "monotone_rows": 0,
        "extremes_ordered_rows": 0,
        "warm_rows": 0,
        "warm_nonzero": 0,
        "zero_offset_member_max_abs_response": 0.0,
    }
    middle = int(np.argmin(np.abs(OFFSETS_K)))
    for index in scored:
        run = setup.runs[index]
        n = run.shortwave_w_m2.shape[1]
        rows = np.arange(run.origin_position + 1, run.origin_position + 1 + n)
        truth_t = np.nan_to_num(features.temperature_c[rows])
        truth_ghi = np.nan_to_num(features.ghi_w_m2[rows])
        fake_run = EnsSiteRun(
            origin_position=run.origin_position,
            temperature_c=truth_t[None, :] + OFFSETS_K[:, None],
            shortwave_w_m2=np.tile(truth_ghi, (members.N_MEMBERS, 1)),
        )
        runs = list(fake.runs)
        runs[index] = fake_run
        full = members.compute_run(
            cache=cache, setup=replace(fake, runs=runs), fits=fits, index=index
        )
        delta = full["delta_t"]
        truth_t = full["t_mean"]
        n = len(truth_t)
        # The trailing means reach back before the origin, so test the rows 72 hours after it.
        settled = np.arange(n) >= 144
        night = full["elevation"] < 0.0
        cold = (truth_t < COLD_C - 4.0) & settled & night
        warm = (truth_t > WARM_C + 4.0) & settled & night
        counts["cold_rows"] += int(cold.sum())
        counts["monotone_rows"] += int(
            (np.diff(delta[cold], axis=1) <= TOLERANCE).all(axis=1).sum()
        )
        counts["extremes_ordered_rows"] += int((delta[cold][:, 0] > delta[cold][:, -1]).sum())
        counts["warm_rows"] += int(warm.sum())
        counts["warm_nonzero"] += int((np.abs(delta[warm]) > TOLERANCE).any(axis=1).sum())
        counts["zero_offset_member_max_abs_response"] = max(
            counts["zero_offset_member_max_abs_response"], float(np.abs(delta[:, middle]).max())
        )
    counts["g4_passes"] = (
        counts["monotone_rows"] == counts["cold_rows"]
        and counts["extremes_ordered_rows"] == counts["cold_rows"]
    )
    common.write_json_atomic(path=common.WORK_DIR / "gates.json", content=counts)
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
