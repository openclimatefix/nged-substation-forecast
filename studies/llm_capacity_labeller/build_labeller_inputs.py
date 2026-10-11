"""Build the fresh draws of planted losses, the synthetic null record, and the export-cap overlay.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1147>. The
first capacity study's record and events are read, never written (`labeller_paths`).

**Outputs under `<root>/built/`:**

- `fresh_draws.json`: two draws of planted losses (seeds 11441 and 11442), each one event of every
  shape at every plant, drawn with the first study's footprint rules and placed around the real
  low-output months and the active-network-management plant's settled-output date.
- `null_power.npz`: the synthetic null record's power
  (`studies.labeller_scoring.synthetic_null_record`).
- `cap_overlay.npz`: the export cap, in megawatts, and whether it binds, per hour of the plant that
  has one.
- `manifest.json`: the seeds and the counts, printed for the report. It counts the daylight
  binding hours that have a reading and how many of them the record removed.

Run it with `uv run python studies/llm_capacity_labeller/build_labeller_inputs.py`; `--smoke`
writes under `_smoke/`.
"""

import argparse
import datetime as dt
import logging
import sys
from typing import Final

import numpy as np
import polars as pl
from labeller_paths import (
    FRESH_DRAW_SEEDS,
    NULL_SEED,
    built_dir,
    claim_root,
    event_to_json,
    load_first_events,
    load_first_record,
    save_cap_overlay,
    write_json_atomic,
)
from studies.capacity_estimators import HOURS_PER_DAY, LOW_SUN_CLEAR_SKY_W_M2
from studies.capacity_evidence import CapOverlay
from studies.capacity_recovery import draw_events
from studies.commissioning import SETTLED_OUTPUT_FROM
from studies.export_cap import with_export_cap
from studies.guards import refuse_to_overwrite
from studies.labeller_scoring import synthetic_null_record

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("build_labeller_inputs")


def cap_overlay_for(*, site: str, first_day: dt.date, days: int) -> CapOverlay:
    """Read the export cap of one plant onto the record's hourly grid.

    Args:
        site: The plant's label.
        first_day: The record's first date.
        days: The number of days.

    Returns:
        The overlay: the cap where one was recorded (NaN elsewhere) and whether it binds. Hour `h`
        of a day ends at `h + 1` o'clock UTC, the record's convention.
    """
    start = dt.datetime.combine(first_day, dt.time(0), tzinfo=dt.UTC)
    times = [
        start + dt.timedelta(days=day, hours=hour + 1)
        for day in range(days)
        for hour in range(HOURS_PER_DAY)
    ]
    frame = pl.DataFrame({"site": [site] * len(times), "time": times}).with_columns(
        pl.col("time").dt.cast_time_unit("us")
    )
    capped = with_export_cap(dataset=frame)
    shape = (days, HOURS_PER_DAY)
    cap = capped["cap_mw"].fill_null(float("nan")).to_numpy().reshape(shape)
    return CapOverlay(cap_mw=cap, binding=capped["constrained"].to_numpy().reshape(shape))


def main() -> int:
    """Build the inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Write under _smoke/, not a result.")
    arguments = parser.parse_args()
    root = claim_root(smoke=arguments.smoke)
    target = built_dir(root=root)
    names = ("fresh_draws.json", "null_power.npz", "cap_overlay.npz", "manifest.json")
    outputs = [target / name for name in names]
    refuse_to_overwrite(paths=outputs)
    target.mkdir(parents=True, exist_ok=True)
    record = load_first_record()
    blind = load_first_record(anm_blind=True)
    first = load_first_events()
    settled = {
        record.sites.index(site): int((np.datetime64(when.date()) - record.dates[0]).astype(int))
        for site, when in SETTLED_OUTPUT_FROM.items()
    }
    draws = [
        draw_events(
            record=record,
            seed=seed,
            reserved=first.reserved,
            earliest_first_day=settled,
            may_have_fewer=list(settled),
        )
        for seed in FRESH_DRAW_SEEDS
    ]
    write_json_atomic(
        path=outputs[0],
        payload={
            "seeds": list(FRESH_DRAW_SEEDS),
            "draws": [[event_to_json(event=e) for e in d] for d in draws],
        },
    )
    null = synthetic_null_record(
        record=record, windows=first.false_movement_windows, seed=NULL_SEED
    )
    np.savez_compressed(outputs[1], power_mw=null.power_mw.astype(np.float32))
    plant = first.anm_plant
    overlay = cap_overlay_for(
        site=record.sites[plant], first_day=record.dates[0].astype(dt.date), days=len(record.dates)
    )
    removed = np.isfinite(blind.power_mw[plant]) & ~np.isfinite(record.power_mw[plant])
    daylight = record.clear_sky_w_m2[plant] >= LOW_SUN_CLEAR_SKY_W_M2
    measured_binding = overlay.binding & daylight & np.isfinite(blind.power_mw[plant])
    stray = int((measured_binding & ~removed).sum())
    if stray:
        _LOG.warning(
            "%d daylight binding hours with a reading are not hours the record removed", stray
        )
    save_cap_overlay(root=root, plant=plant, overlay=overlay)
    write_json_atomic(
        path=outputs[3],
        payload={
            "smoke": arguments.smoke,
            "fresh_draw_seeds": list(FRESH_DRAW_SEEDS),
            "events_per_draw": [len(d) for d in draws],
            "null_seed": NULL_SEED,
            "binding_hours": int(overlay.binding.sum()),
            "daylight_binding_hours_with_a_reading": int(measured_binding.sum()),
            "of_which_not_removed_from_record": stray,
            "hours_removed_from_record": int(removed.sum()),
            "quiet_windows": len(first.false_movement_windows),
        },
    )
    _LOG.info("wrote %s", target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
