# ruff: noqa: E501
"""The known-answer gates G1 to G4 of experiment 184, run on the stored deltas and a fake ensemble.

- G1: the ensemble-mean-driven T-all forecast plus the filter's level and anomaly equals
  experiment 136's stored mean (read from the `g1` column that `e00184_members.py` stored).
- G2: the unit PV shape built by `SunGeometry.pv_shape_and_ghi` from the cache's irradiance equals
  the cache's own unit PV shape (through-origin slope per substation).
- G3: a fake ensemble of CAMS irradiance times 51 equally spaced multipliers gives responses that
  fall with the multiplier, are zero at night, and are zero for the member at multiplier 1.
- G4: the spread of the clear-sky index across members, raw against calibrated, by clear-sky band.

    uv run --no-sync python ml-experiments/00184-*/e00184_gates.py
"""

import json
from dataclasses import replace

import e00184_common as common
import e00184_data as data
import e00184_members as members
import e00184_score as score
import e00184_tall as tall
import numpy as np
import polars as pl
from studies.dp_ens_weather import Calibration, EnsSiteRun, SunGeometry

GATE_SITE = "D10"
G2_START = np.datetime64("2025-01-01T00:30:00", "us")
G2_DAYS = 90
N_FAKE_RUNS = 6
MULTIPLIERS = np.linspace(0.6, 1.4, members.N_MEMBERS)


def gate_one() -> dict:
    """Return the largest absolute reproduction gap per site and period."""
    out = {}
    for site in common.SCORED_SITES:
        for period in members.PERIODS:
            with np.load(members.out_path(site=site, period=period)) as stored:
                out[f"{site}_{period}"] = float(np.nanmax(np.abs(stored["g1"])))
    return out


def gate_two(*, cache) -> dict:  # noqa: ANN001
    """Return the through-origin slope and the largest gap of the unit PV shape per substation."""
    out = {}
    start = int(np.searchsorted(cache.end_times, G2_START))
    span = slice(start, start + 48 * G2_DAYS)
    for site in common.SCORED_SITES:
        features = cache.sites[site]
        sun = SunGeometry.build(
            end_times=cache.end_times[span],
            latitude=features.table.latitude,
            longitude=features.table.longitude,
        )
        shape, _ = sun.pv_shape_and_ghi(ghi_w_m2=features.ghi_w_m2[span][None, :])
        built, stored = shape[0], features.pv_unit_shape[span]
        keep = np.isfinite(built) & np.isfinite(stored) & (stored > data.SUN_UP_UNIT_SHAPE)
        out[site] = {
            "slope": float((built[keep] * stored[keep]).sum() / (stored[keep] ** 2).sum()),
            "max_abs_gap": float(np.abs(built[keep] - stored[keep]).max()),
            "mean_stored": float(stored[keep].mean()),
            "rows": int(keep.sum()),
        }
    return out


def gate_three(*, cache) -> dict:  # noqa: ANN001
    """Run a fake ensemble through the member pass and return what it must satisfy."""
    setup = members.load_context(site=GATE_SITE, cache=cache)
    fits = tall.load_fits(cadence="monthly", site=GATE_SITE)
    features = cache.sites[GATE_SITE]
    scored = setup.indices(periods={"scored"})[10 : 10 + N_FAKE_RUNS]
    identity = setup.with_calibration(calibration=Calibration.identity())
    fake_runs = []
    for index in scored:
        run = setup.runs[index]
        n = run.shortwave_w_m2.shape[1]
        rows = np.arange(run.origin_position + 1, run.origin_position + 1 + n)
        truth = np.nan_to_num(features.ghi_w_m2[rows])
        temperature = np.tile(np.nan_to_num(features.temperature_c[rows]), (members.N_MEMBERS, 1))
        fake_runs.append(
            EnsSiteRun(
                origin_position=run.origin_position,
                temperature_c=temperature,
                shortwave_w_m2=MULTIPLIERS[:, None] * truth[None, :],
            )
        )
    fake = replace(identity, runs=[*identity.runs])
    checks = {
        "monotone_rows": 0,
        "daylight_rows": 0,
        "physical_rows": 0,
        "physical_monotone_rows": 0,
        "night_nonzero": 0,
        "mean_member_gap": 0.0,
        "std_rel_gap": 0.0,
    }
    for index, run in zip(scored, fake_runs, strict=True):
        runs = list(fake.runs)
        runs[index] = run
        full = members.compute_run(
            cache=cache, setup=replace(fake, runs=runs), fits=fits, index=index
        )
        delta = full["delta_m"]
        # The fake members have no meaningful temperature, so only the solar response is checked.
        clear = full["clear"]
        daylight = clear > members.MIN_CLEAR_SKY_W_M2 * 10
        ordered = np.diff(delta[daylight], axis=1) <= 1e-9
        checks["monotone_rows"] += int(ordered.all(axis=1).sum())
        # Post hoc: rows where the fake index stays physical (at most 1.1 at the 1.4 multiplier).
        physical = daylight & (MULTIPLIERS.max() * np.nan_to_num(full["csi_cal"], nan=9.0) <= 1.1)
        ordered_physical = np.diff(delta[physical], axis=1) <= 1e-9
        checks["physical_rows"] += int(physical.sum())
        checks["physical_monotone_rows"] += int(ordered_physical.all(axis=1).sum())
        checks["daylight_rows"] += int(daylight.sum())
        checks["night_nonzero"] += int((np.abs(delta[clear <= 0.0]) > 1e-9).sum())
        middle = int(np.argmin(np.abs(MULTIPLIERS - 1.0)))
        checks["mean_member_gap"] = max(
            checks["mean_member_gap"], float(np.abs(delta[:, middle]).max())
        )
        # z_std of the fake ensemble equals the multiplier spread times the mean member's index.
        expected = MULTIPLIERS.std() * full["csi_cal"]
        keep = daylight & np.isfinite(expected) & (expected > 0.05)
        checks["std_rel_gap"] = max(
            checks["std_rel_gap"], float(np.max(np.abs(full["z_std"][keep] / expected[keep] - 1.0)))
        )
    return checks


def gate_four() -> dict:
    """Return the median standard deviation of the clear-sky index across members, raw and calibrated."""
    cache = common.load_cache()
    frame, _, _ = data.load_period(period="scored", cache=cache)
    frame = score.label_columns(frame=frame)
    day = frame.filter(
        pl.col("usable") & (pl.col("kind") == "daylight") & pl.col("csi_band").is_not_null()
    )
    table = (
        day.group_by("csi_band")
        .agg(
            rows=pl.len(),
            raw_std=pl.col("raw_std").median(),
            calibrated_std=pl.col("z_std").median(),
            raw_mean_index=pl.col("csi_raw").median(),
            calibrated_mean_index=pl.col("csi_cal").median(),
        )
        .sort("csi_band")
    )
    return {row["csi_band"]: row for row in table.to_dicts()}


def main() -> None:
    """Run the gates and write `gates.json`."""
    cache = common.load_cache()
    report = {
        "g1_max_abs_gap_points": gate_one(),
        "g2_unit_shape": gate_two(cache=cache),
        "g3_fake_ensemble": gate_three(cache=cache),
        "g4_spread_by_clear_sky_band": gate_four(),
    }
    common.write_json_atomic(path=common.WORK_DIR / "gates.json", content=report)
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
