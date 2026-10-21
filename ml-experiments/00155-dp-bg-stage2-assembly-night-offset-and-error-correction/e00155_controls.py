"""Controls of experiment 155, run before any contrast is read.

1. **Leakage.** Add 100 points to `actual` on or after each cut in a copy of the frames and rebuild
   the corrections; the shifts applied to the cut's month must not change. For recipe R the actual
   of every row on or after the origin's own UTC day is corrupted and each run's shift is compared.
2. **Injected known answer.** In one copy add a winter-night step (6 points, 01:00 to 08:30 UTC,
   November to March) and in another a daily shape error (3 points, half-hours ending 17:30 to
   19:00 local) to every row's `actual`; the winter offset must move by 6 points on the injected
   half-hours and recipe R's shift by +2.06 points (0.75 of 3 minus the day's mean of 0.25).
3. **Placebo.** The offset with the winter indicator moved six months, scored in the report.

    uv run --no-sync python ml-experiments/00155-*/e00155_controls.py
"""

import json
from datetime import timedelta

import e00155_common as common
import e00155_corrections as corr
import numpy as np
import polars as pl

STEP_POINTS = 6.0
SHAPE_POINTS = 3.0
STEP_MINUTES = tuple(range(60, 9 * 60 + 1, 30))
"""UTC minutes at which a half-hour ends, 01:00 to 08:30 start (ends 01:00 to 09:00)."""


def leakage(*, site: str) -> dict:
    """Corrupt `actual` from each cut on and report the largest change in an applied shift."""
    base = corr.add_calendar(frame=common.read_base(site=site))
    clean_offset, _ = corr.winter_offset_shift(frame=base)
    mean = base["mean"].to_numpy().astype(np.float64)
    clean_r = corr.recipe_r_shift(frame=base, forecast=mean)
    month = base["power_fcst_init_time"].dt.truncate("1mo")
    worst_offset = 0.0
    for cut in sorted(month.unique().to_list())[3:]:
        corrupt = base.with_columns(
            actual=pl.when(pl.col("valid_time") >= cut)
            .then(pl.col("actual") + 100.0)
            .otherwise(pl.col("actual"))
        )
        offset, _ = corr.winter_offset_shift(frame=corrupt)
        this = (month == cut).to_numpy()
        worst_offset = max(worst_offset, float(np.abs(offset[this] - clean_offset[this]).max()))
    # R: corrupt every row from the origin's UTC day on, per run; the run's own shift must not move.
    worst_r = 0.0
    origins = base["power_fcst_init_time"].unique().sort().to_list()
    for origin in origins[200::60]:
        day = origin.replace(hour=0, minute=0, second=0, microsecond=0)
        corrupt = base.with_columns(
            actual=pl.when(pl.col("valid_time") > day)
            .then(pl.col("actual") + 100.0)
            .otherwise(pl.col("actual"))
        )
        shift = corr.recipe_r_shift(frame=corrupt, forecast=mean)
        this = (base["power_fcst_init_time"] == origin).to_numpy()
        worst_r = max(worst_r, float(np.abs(shift[this] - clean_r[this]).max()))
    return {"max_offset_change": worst_offset, "max_r_change": worst_r}


def injected(*, site: str) -> dict:
    """Return the recovered size of the injected winter step and daily shape."""
    base = corr.add_calendar(frame=common.read_base(site=site))
    minute = base["valid_time"].dt.hour().cast(pl.Int32) * 60 + base["valid_time"].dt.minute().cast(
        pl.Int32
    )
    winter_night = (
        (minute.is_in(list(STEP_MINUTES)) | (minute == 0))
        & base["evening"].dt.month().is_in(list(common.WINTER_MONTHS))
    ).to_numpy()
    local_minute = base["valid_time"].dt.convert_time_zone(corr.LONDON).dt.hour().cast(
        pl.Int32
    ) * 60 + base["valid_time"].dt.convert_time_zone(corr.LONDON).dt.minute().cast(pl.Int32)
    shape = local_minute.is_in([17 * 60 + 30, 18 * 60, 18 * 60 + 30, 19 * 60]).to_numpy()
    actual = base["actual"].to_numpy().astype(np.float64)
    stepped = base.with_columns(actual=pl.Series(actual + STEP_POINTS * winter_night))
    shaped = base.with_columns(actual=pl.Series(actual + SHAPE_POINTS * shape))
    clean_offset, _ = corr.winter_offset_shift(frame=base)
    bumped_offset, _ = corr.winter_offset_shift(frame=stepped)
    mean = base["mean"].to_numpy().astype(np.float64)
    clean_r = corr.recipe_r_shift(frame=base, forecast=mean)
    bumped_r = corr.recipe_r_shift(frame=shaped, forecast=mean)
    scored = (base["period"] == "scored").to_numpy()
    late = scored & (base["power_fcst_init_time"] >= common_cut_2025_11()).to_numpy()
    step_rows = late & winter_night
    shape_rows = scored & shape
    return {
        "step_recovered": float(np.mean(bumped_offset[step_rows] - clean_offset[step_rows])),
        "shape_shift": float(np.mean(bumped_r[shape_rows] - clean_r[shape_rows])),
    }


def common_cut_2025_11():  # noqa: ANN201
    """Return the first origin of the cuts the step control reads (2025-11-01 UTC)."""
    from datetime import UTC, datetime

    return datetime(2025, 11, 1, tzinfo=UTC) + timedelta(0)


def main() -> None:
    """Run the controls at every substation and write `work/controls.json`."""
    out = {}
    for site in common.SCORED_SITES:
        out[site] = {"leakage": leakage(site=site), "injected": injected(site=site)}
        print(site, json.dumps(out[site]), flush=True)
    common.write_json_atomic(path=common.WORK_DIR / "controls.json", content=out)


if __name__ == "__main__":
    main()
