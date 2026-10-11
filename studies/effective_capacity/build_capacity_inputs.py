"""Build the record, the planted-loss draws, and the windows the effective-capacity study uses.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1143>. The
question is whether a time-varying capacity estimate beats one fixed number per plant when forecasting
solar power, and which estimation technique is most promising. `docs/studies/forecasts/effective-capacity.md`
holds the design and the results.

**The record.** For each of the 6 plants, a day-by-hour grid from 2019-09-01 to 2026-06-30 of the
cleaned hourly power (multi-day zero runs, meter spikes, the commissioning ramp, and the export-capped
hours removed, as the lag study cleans it), the clear-sky irradiance, the CAMS satellite irradiance,
and the unit-capacity output that a physics model gives for each. Each plant's tilt and azimuth are
fitted once with `studies.pv_fit.fit_plant` on data before 2024-12-01 (for the ANM plant, the 2 months
after its commissioning ramp ends). A day is labelled by the date of its hours' midpoints.

**The checks.** The build raises if the lag frame does not hold 29,445 rows, if any input is at or
after the final-test start, if the static power differs from the lag frame's capacity column, if an
event cannot be placed, or if the power with the export-capped hours left in differs from the cleaned
power at a plant with no export cap.

**Outputs**, under `<output-root>/inputs/`: `record.npz`, `orientation.json`, `events.json`, and
`manifest.json`.

Run it with `uv run python studies/effective_capacity/build_capacity_inputs.py` (add `--smoke` to
write under `_smoke/`).
"""

import argparse
import json
import logging
import sys
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from capacity_paths import (
    CONTROL_MONTHS_PER_PLANT,
    CONTROL_SEED,
    CONTROL_SIZE,
    DERATED_PLAN_PATH,
    DRAW_SEEDS,
    FALSE_MOVEMENT_SEED,
    FALSE_MOVEMENTS_PER_PLANT,
    LAG_FRAME_PATH,
    LAG_FRAME_ROWS,
    OUTPUT_RATIOS_PATH,
    claim_root,
    event_to_json,
    inputs_dir,
    save_record,
    sha256_of,
    write_json_atomic,
)
from contracts.config_schemas import load_cv_config
from studies.baselines import hourly_clear_sky
from studies.capacity_estimators import CapacityRecord
from studies.capacity_recovery import (
    PlantedEvent,
    calendar_month_spans,
    draw_control_steps,
    draw_events,
    false_movement_windows,
)
from studies.commissioning import SETTLED_OUTPUT_FROM, drop_commissioning_ramp
from studies.export_cap import with_export_cap
from studies.guards import refuse_to_overwrite
from studies.power import CV_CONFIG_PATH
from studies.pv_dataset import (
    MIN_CAMS_RELIABILITY,
    drop_outages_and_spikes,
    pv_sites,
    read_cams,
    solar_hourly_power,
)
from studies.pv_fit import fit_plant
from studies.pv_physics import PlantParameters, half_hour_sun_and_sky, plant_power_mw

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("build_capacity_inputs")

FIRST_DAY: Final[date] = date(2019, 9, 1)
"""The first day of the record, which is the first day CAMS covers."""

ORIENTATION_END: Final[datetime] = datetime(2024, 12, 1, tzinfo=UTC)
"""Each plant's orientation is fitted on data before this."""

UNCLIPPED_AC_MW: Final[float] = 1.0e9
"""An alternating-current limit so high that the unit-capacity output is never clipped."""


def final_test_cutoff() -> datetime:
    """Return midnight UTC at the start of `final_test_start`, which no study may read.

    Returns:
        The cut-off.
    """
    return datetime.combine(load_cv_config(CV_CONFIG_PATH).final_test_start, time.min, tzinfo=UTC)


def hourly_grid_ends(*, days: int) -> pl.Series:
    """Return every hour end from 01:00 on the first day to 23:00 on the last day.

    Args:
        days: The number of days in the record.

    Returns:
        A UTC datetime series of `24 * days - 1` hour ends. The hour ending at midnight after the
        last day is not in it, because it is at or after the final-test cut-off.
    """
    first = datetime.combine(FIRST_DAY, time(1), tzinfo=UTC)
    return pl.datetime_range(
        first, first + timedelta(hours=24 * days - 2), interval="1h", eager=True, time_zone="UTC"
    )


def to_cube(*, values: np.ndarray, days: int) -> np.ndarray:
    """Reshape an hourly series on the grid into (days, 24), padding the one missing last hour.

    Args:
        values: A series of `24 * days - 1` values.
        days: The number of days.

    Returns:
        An array of shape (days, 24), NaN in the padded entry.
    """
    return np.concatenate([values, [np.nan]]).reshape(days, 24)


def cleaned_power(*, aware: bool) -> pl.DataFrame:
    """Return NGED's cleaned hourly power as the lag study's lag source reads it.

    Args:
        aware: Whether to remove the hours the export cap held down. `False` leaves them in, which
            is what an estimator that ignores active network management would read.

    Returns:
        `site`, `time`, and `power_mw`, before the final-test cut-off.
    """
    sites = pv_sites()
    hourly = solar_hourly_power(sites=sites)
    cleaned = drop_outages_and_spikes(power=hourly, sites=sites).select("site", "time", "power_mw")
    capped = with_export_cap(dataset=drop_commissioning_ramp(dataset=cleaned))
    kept = capped.filter(~pl.col("constrained")) if aware else capped
    return kept.filter(pl.col("time") < final_test_cutoff()).select("site", "time", "power_mw")


def site_cube(
    *, table: pl.DataFrame, site: str, column: str, grid: pl.Series, days: int
) -> np.ndarray:
    """Place one site's hourly column on the day-by-hour grid.

    Args:
        table: Rows with `site`, `time`, and the column.
        site: The plant's label.
        column: The column to place.
        grid: `hourly_grid_ends`' result.
        days: The number of days.

    Returns:
        An array of shape (days, 24), NaN where the table has no row.
    """
    mine = table.filter(pl.col("site") == site).select("time", column)
    joined = pl.DataFrame({"time": grid}).join(mine, on="time", how="left")
    return to_cube(values=joined[column].cast(pl.Float64).to_numpy(), days=days)


def unit_output(
    *,
    ghi_w_m2: np.ndarray,
    grid: pl.Series,
    parameters: PlantParameters,
    latitude: float,
    longitude: float,
) -> np.ndarray:
    """Return the hourly output of a plant of unit direct-current capacity under a given sky.

    Args:
        ghi_w_m2: Hourly global horizontal irradiance on the grid, with NaN where missing.
        grid: The hour ends.
        parameters: The plant, with unit direct-current capacity.
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        The hourly mean of the half-hourly output, NaN where the irradiance is.
    """
    sky = half_hour_sun_and_sky(
        hourly=pl.DataFrame({"time": grid, "ghi_w_m2": np.nan_to_num(ghi_w_m2)}),
        latitude=latitude,
        longitude=longitude,
    )
    half_hourly = plant_power_mw(sky=sky, parameters=parameters)
    hourly = half_hourly.reshape(-1, 2).mean(axis=1)
    return np.where(np.isfinite(ghi_w_m2), hourly, np.nan)


def fit_orientation(
    *,
    ghi_w_m2: np.ndarray,
    power_mw: np.ndarray,
    grid: pl.Series,
    latitude: float,
    longitude: float,
    site: str,
) -> PlantParameters:
    """Fit a plant's orientation on its data before `ORIENTATION_END`.

    Args:
        ghi_w_m2: Hourly CAMS irradiance on the grid.
        power_mw: The plant's hourly power on the grid, NaN where missing.
        grid: The hour ends.
        latitude: Degrees north.
        longitude: Degrees east.
        site: The plant's label, for the error message.

    Returns:
        The fitted plant (tilt, azimuth, and capacities).

    Raises:
        RuntimeError: If the fit finds too few usable half-hours.
    """
    sky = half_hour_sun_and_sky(
        hourly=pl.DataFrame({"time": grid, "ghi_w_m2": np.nan_to_num(ghi_w_m2)}),
        latitude=latitude,
        longitude=longitude,
    )
    output = np.repeat(power_mw, 2)
    stamps = sky.half_hour_end_time.astype("datetime64[us]")
    early = stamps < np.datetime64(ORIENTATION_END.replace(tzinfo=None), "us")
    usable = early & np.repeat(np.isfinite(ghi_w_m2), 2)
    fit = fit_plant(sky=sky, output_mw=output, orientation="free", usable=usable)
    if fit is None:
        msg = f"the orientation fit for plant {site} found too few usable half-hours"
        raise RuntimeError(msg)
    _LOG.info(
        "plant %s: tilt %.1f, azimuth %.1f, DC %.2f MW, AC %.2f MW, %d half-hours",
        site,
        fit.parameters.tilt_deg,
        fit.parameters.azimuth_deg,
        fit.parameters.dc_capacity_mw,
        fit.parameters.ac_capacity_mw,
        fit.half_hours_fitted,
    )
    return fit.parameters


def build_record() -> tuple[CapacityRecord, np.ndarray, dict[str, dict[str, float]]]:
    """Build the record, the ANM-blind power, and each plant's fitted orientation.

    Returns:
        The record, the power with the export-capped hours left in, and the orientation fits by
        site.

    Raises:
        ValueError: If an input is at or after the cut-off, or the ANM-blind power differs from the
            cleaned power at a plant with no export-cap record.
    """
    sites = pv_sites().sort("site")
    labels = sites["site"].to_list()
    last_day = final_test_cutoff().date() - timedelta(days=1)
    days = (last_day - FIRST_DAY).days + 1
    grid = hourly_grid_ends(days=days)
    if grid.max() >= final_test_cutoff():
        msg = "the hour grid reaches the final-test start"
        raise ValueError(msg)
    aware = cleaned_power(aware=True)
    blind = cleaned_power(aware=False)
    clear = hourly_clear_sky(sites=sites, first=grid.min(), last=grid.max())
    cams = read_cams(min_reliability=MIN_CAMS_RELIABILITY).filter(
        pl.col("time") < final_test_cutoff()
    )
    capped_sites = set(
        blind.join(aware, on=["site", "time"], how="anti")["site"].unique().to_list()
    )
    power, power_blind, clear_cube, ghi_cube, cams_shape, clear_shape = ([] for _ in range(6))
    orientation: dict[str, dict[str, float]] = {}
    for row in sites.iter_rows(named=True):
        site, latitude, longitude = row["site"], row["latitude"], row["longitude"]
        power.append(site_cube(table=aware, site=site, column="power_mw", grid=grid, days=days))
        power_blind.append(
            site_cube(table=blind, site=site, column="power_mw", grid=grid, days=days)
        )
        if site not in capped_sites and not np.array_equal(
            power[-1], power_blind[-1], equal_nan=True
        ):
            msg = f"plant {site} has no export-cap record, yet its ANM-blind power differs"
            raise ValueError(msg)
        clear_cube.append(
            site_cube(table=clear, site=site, column="clear_sky_w_m2", grid=grid, days=days)
        )
        ghi_cube.append(site_cube(table=cams, site=site, column="ghi_w_m2", grid=grid, days=days))
        ghi_flat = ghi_cube[-1].reshape(-1)[:-1]
        clear_flat = clear_cube[-1].reshape(-1)[:-1]
        power_flat = power[-1].reshape(-1)[:-1]
        fitted = fit_orientation(
            ghi_w_m2=ghi_flat,
            power_mw=power_flat,
            grid=grid,
            latitude=latitude,
            longitude=longitude,
            site=site,
        )
        orientation[site] = {
            "tilt_deg": fitted.tilt_deg,
            "azimuth_deg": fitted.azimuth_deg,
            "dc_capacity_mw": fitted.dc_capacity_mw,
            "ac_capacity_mw": fitted.ac_capacity_mw,
            "static_p99_mw": float(row["effective_capacity_mw"]),
        }
        unit = PlantParameters(
            tilt_deg=fitted.tilt_deg,
            azimuth_deg=fitted.azimuth_deg,
            dc_capacity_mw=1.0,
            ac_capacity_mw=UNCLIPPED_AC_MW,
        )
        cams_shape.append(
            to_cube(
                values=unit_output(
                    ghi_w_m2=ghi_flat,
                    grid=grid,
                    parameters=unit,
                    latitude=latitude,
                    longitude=longitude,
                ),
                days=days,
            )
        )
        clear_shape.append(
            to_cube(
                values=unit_output(
                    ghi_w_m2=clear_flat,
                    grid=grid,
                    parameters=unit,
                    latitude=latitude,
                    longitude=longitude,
                ),
                days=days,
            )
        )
    record = CapacityRecord(
        sites=tuple(labels),
        dates=np.datetime64(FIRST_DAY) + np.arange(days),
        power_mw=np.stack(power),
        clear_sky_w_m2=np.nan_to_num(np.stack(clear_cube)),
        cams_ghi_w_m2=np.stack(ghi_cube),
        cams_shape=np.stack(cams_shape),
        clear_shape=np.stack(clear_shape),
        static_p99_mw=sites["effective_capacity_mw"].cast(pl.Float64).to_numpy(),
    )
    return record, np.stack(power_blind), orientation


def check_against_lag_frame(*, record: CapacityRecord) -> pl.DataFrame:
    """Check the record against the lag frame and return the frame.

    Args:
        record: The record.

    Returns:
        The lag frame.

    Raises:
        ValueError: If the frame's row count differs from 29,445, a row is at or after the
            final-test start, or a plant's static power differs from the frame's capacity column.
    """
    frame = pl.read_parquet(LAG_FRAME_PATH)
    if frame.height != LAG_FRAME_ROWS:
        msg = f"the lag frame holds {frame.height} rows, not {LAG_FRAME_ROWS}"
        raise ValueError(msg)
    if frame["time"].max() >= final_test_cutoff():
        msg = "the lag frame reaches the final-test start"
        raise ValueError(msg)
    capacity = frame.group_by("site").agg(capacity=pl.col("effective_capacity_mw").first())
    for site, expected in capacity.iter_rows():
        index = record.sites.index(site)
        if abs(record.static_p99_mw[index] - float(expected)) > 1e-6:
            msg = f"plant {site}'s static power {record.static_p99_mw[index]} differs from the frame's {expected}"
            raise ValueError(msg)
    return frame


def build_events(*, record: CapacityRecord, frame: pl.DataFrame) -> dict[str, object]:
    """Draw the events, the control, and the false-movement windows.

    Args:
        record: The record.
        frame: The lag frame.

    Returns:
        The JSON payload of `events.json`.
    """
    plan = json.loads(DERATED_PLAN_PATH.read_text())
    low_plant = record.sites.index(plan["site"])
    low_months = list(plan["months"])
    spans = calendar_month_spans(record=record)
    reserved = [PlantedEvent(low_plant, "step", 0.3, *spans[month]) for month in low_months]
    settled = {
        record.sites.index(site): int((np.datetime64(when.date()) - record.dates[0]).astype(int))
        for site, when in SETTLED_OUTPUT_FROM.items()
    }
    anm_plant = next(iter(settled))
    draws = [
        draw_events(
            record=record,
            seed=seed,
            reserved=reserved,
            earliest_first_day=settled,
            may_have_fewer=list(settled),
        )
        for seed in DRAW_SEEDS
    ]
    scored_months = sorted(frame["month"].unique().to_list())
    control = draw_control_steps(
        record=record,
        months_per_plant=CONTROL_MONTHS_PER_PLANT,
        size=CONTROL_SIZE,
        scored_months=scored_months,
        barred={low_plant: low_months},
        seed=CONTROL_SEED,
    )
    ratios = pl.read_parquet(OUTPUT_RATIOS_PATH).select("site", "month", "ratio")
    windows = false_movement_windows(
        ratios=ratios, record=record, per_plant=FALSE_MOVEMENTS_PER_PLANT, seed=FALSE_MOVEMENT_SEED
    )
    return {
        "draws": [[event_to_json(event=e) for e in draw] for draw in draws],
        "control": [event_to_json(event=e) for e in control],
        "reserved": [event_to_json(event=e) for e in reserved],
        "false_movement_windows": [list(window) for window in windows],
        "scored_months": scored_months,
        "low_plant": low_plant,
        "low_months": low_months,
        "anm_plant": anm_plant,
    }


def main() -> int:
    """Build the study's inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Write under _smoke/, not a result.")
    arguments = parser.parse_args()
    root = claim_root(smoke=arguments.smoke)
    target = inputs_dir(root=root)
    target.mkdir(parents=True, exist_ok=True)
    outputs = [
        target / name for name in ("record.npz", "orientation.json", "events.json", "manifest.json")
    ]
    refuse_to_overwrite(paths=outputs)
    record, power_blind, orientation = build_record()
    frame = check_against_lag_frame(record=record)
    try:
        events = build_events(record=record, frame=frame)
    except ValueError:
        # Keep the record for diagnosis, under a name that no later run mistakes for a result.
        save_record(
            path=target / "record_failed_events.npz", record=record, power_anm_blind=power_blind
        )
        raise
    save_record(path=outputs[0], record=record, power_anm_blind=power_blind)
    write_json_atomic(path=outputs[1], payload=orientation)
    write_json_atomic(path=outputs[2], payload=events)
    write_json_atomic(
        path=outputs[3],
        payload={
            "smoke": arguments.smoke,
            "lag_frame_sha256": sha256_of(path=LAG_FRAME_PATH),
            "lag_frame_rows": frame.height,
            "days": len(record.dates),
            "sites": list(record.sites),
            "valid_day_share": None,
        },
    )
    _LOG.info("wrote %s", target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
