"""Post hoc, exploratory: where does the largest night rise fall, by year, on cold BST nights?

Reads winters 2021-22 to 2024-25 only (British Summer Time nights from March 2021 to
October 2025). For each
substation and each calendar year, over cold nights in British Summer Time (daily mean below 8 C),
it finds the half-hour entered by the largest rise between 23:30 and 03:00 on the local clock, and
reports the share of nights where that half-hour is the one a tariff fixed to the local clock would
reach (starting 23:30 or 00:30 local) and the share where it is the one a step fixed to UTC would
reach (starting 01:30 local). `cannot tell` where a year has fewer than 10 such nights.

    uv run python ml-experiments/00123-*/e00123_clock_by_year.py
"""

import json
from datetime import timedelta
from typing import Final
from zoneinfo import ZoneInfo

import e00123_common as common
import e00123_followups as followups
import e00123_report as report
import numpy as np
import polars as pl

LOCAL_FIXED_SLOTS: Final[tuple[int, ...]] = (3, 5)
"""Indices in `followups.SLOTS_LONG` of the half-hours starting 23:30 and 00:30 local."""
UTC_FIXED_SLOT: Final[int] = 7
"""Index of the half-hour starting 01:30 local, where a UTC-fixed 00:30 step falls in BST."""
MIN_NIGHTS: Final[int] = 10


def night_profile(*, lookup: dict, evening: common.date, scale: float) -> np.ndarray | None:
    """Return one night's local-clock profile in points relative to 22:00-23:30, or None."""
    midnight = common.datetime.combine(evening, common.time.min).replace(
        tzinfo=ZoneInfo(common.LOCAL_TIME_ZONE)
    )
    values = [
        lookup.get((midnight + timedelta(minutes=m)).astimezone(common.UTC) + timedelta(minutes=30))
        for m in followups.SLOTS_LONG
    ]
    present = [float(v) for v in values if v is not None]
    if len(present) != len(values):
        return None
    level = sum(present[:4]) / 4.0
    return np.array([(v - level) * scale for v in present])


def main() -> None:
    """Print the by-year shares and write `clock_by_year.json`."""
    infos = common.series_infos()
    table = pl.read_parquet(common.WORK_DIR / "night_tables.parquet")
    power = common.load_power(
        time_series_ids=[infos[s].time_series_id for s in common.FITTED_SERIES]
    )
    end = report.night_end(winters=common.HEADLINE_WINTERS)
    rows = []
    for site in common.FITTED_SERIES:
        info = infos[site]
        own = power.filter(pl.col("time_series_id") == info.time_series_id)
        lookup = dict(zip(own["time"].to_list(), own["power"].to_list(), strict=True))
        nights = table.filter(
            (pl.col("site") == site)
            & (pl.col("season") == "bst_other")
            & ~pl.col("excluded")
            & (pl.col("daily_mean_c") < common.COLD_BST_NIGHT_C)
            & (pl.col("evening") >= common.date(2021, 3, 28))
            & (pl.col("evening") < end)
        )["evening"].to_list()
        by_year: dict[int, list[int]] = {}
        for evening in nights:
            profile = night_profile(
                lookup=lookup, evening=evening, scale=100.0 / info.effective_capacity_mw
            )
            if profile is not None:
                by_year.setdefault(evening.year, []).append(followups.jump_slot(profile=profile))
        for year, slots in sorted(by_year.items()):
            array = np.array(slots)
            rows.append(
                {
                    "site": site,
                    "year": year,
                    "nights": len(slots),
                    "utc_fixed": float((array == UTC_FIXED_SLOT).mean()),
                    "local_fixed": float(np.isin(array, LOCAL_FIXED_SLOTS).mean()),
                }
            )
    (common.WORK_DIR / "clock_by_year.json").write_text(json.dumps(rows, indent=1))
    years = sorted({r["year"] for r in rows})
    print("| Series | " + " | ".join(str(y) for y in years) + " |")
    print("|---|" + "---|" * len(years))
    for site in common.FITTED_SERIES:
        cells = []
        for year in years:
            hit = next((r for r in rows if r["site"] == site and r["year"] == year), None)
            if hit is None:
                cells.append("cannot tell")
            elif int(hit["nights"]) < MIN_NIGHTS:
                cells.append(f"cannot tell ({hit['nights']})")
            else:
                cells.append(f"{hit['utc_fixed']:.0%} / {hit['local_fixed']:.0%} ({hit['nights']})")
        print(f"| {site} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
