"""Post hoc, exploratory readouts of experiment 123 (items 1 to 3 of the follow-up list).

Reads winters 2021-22 to 2024-25 only (BST nights up to 2025-10-25). Prints three tables:

1. Per substation, the share of cold British Summer Time (BST) nights whose largest half-hour rise
   between 23:30 and 03:00 on the local clock falls in the half-hour a local-clock step would
   reach (starting 00:30) or in the half-hour a UTC-fixed step would reach (starting 01:30), with
   the same share on cold winter nights as the reference for how often the step shows as the
   largest rise at all.
2. Per substation, the window shape on cold winter nights (mean demand minus the 22:00-23:30 level,
   minus the same on summer nights): the fraction of the plateau reached at the half-hours starting
   00:00, 00:30, and 01:00; the half-hour of the largest fall between 05:00 and 10:00; and any
   second rise (a rise of at least 30% of the main one, outside two half-hours of it).
3. Per substation, the step by winter, with the change from the first to the last winter and its
   label (`falls`, `rises`, or `cannot tell` where the winters are fewer than 3 or the first and
   last 90% intervals overlap).

    uv run python ml-experiments/00123-*/e00123_followups.py headline
"""

import json
from datetime import timedelta
from zoneinfo import ZoneInfo

import e00123_common as common
import e00123_report as report
import numpy as np
import polars as pl

SLOTS_LONG = tuple(range(22 * 60, 34 * 60, 30))
"""Start minutes from the evening's midnight: 22:00 to 09:30 next morning (24 slots)."""
LOCAL_STEP_SLOT = 5
"""The half-hour starting 00:30 (index 5 in `SLOTS_LONG`), where a local-clock step shows."""
SECOND_RISE_SHARE = 0.3
MIN_WINTERS_FOR_TREND = 3


def slot_clock(*, index: int) -> str:
    """Return the `HH:MM` start of a slot of `SLOTS_LONG`."""
    minute = SLOTS_LONG[index] % 1440
    return f"{minute // 60:02d}:{minute % 60:02d}"


def profiles(*, lookup: dict, evenings: list, scale: float, local: bool) -> np.ndarray:
    """Return the (nights, 24) profile relative to the 22:00-23:30 level, NaN rows dropped."""
    rows = []
    for evening in evenings:
        base = common.datetime.combine(evening, common.time.min)
        zone = ZoneInfo(common.LOCAL_TIME_ZONE) if local else common.UTC
        midnight = base.replace(tzinfo=zone)
        values = [
            lookup.get(
                (midnight + timedelta(minutes=m)).astimezone(common.UTC) + timedelta(minutes=30)
            )
            for m in SLOTS_LONG
        ]
        if any(v is None for v in values):
            continue
        floats = [float(v) for v in values if v is not None]
        level = sum(floats[:4]) / 4.0
        rows.append([(v - level) * scale for v in floats])
    return np.array(rows)


def jump_slot(*, profile: np.ndarray) -> int:
    """Return the `SLOTS_LONG` index of the half-hour entered by the largest rise, 23:30-03:00."""
    increments = np.diff(profile)
    window = increments[2:10]
    return int(np.argmax(window)) + 3


def print_cards(*, result: dict[str, list[dict]]) -> None:
    """Print one markdown card row per substation for the three readouts."""
    print("\n== follow-up card rows (markdown) ==")
    print(
        "| Series | Largest rise at the local-clock step, cold BST nights | At the UTC step, "
        "cold BST nights | Same two shares, cold winter nights | Plateau reached by the "
        "half-hour starting 00:00 / 00:30 | Largest fall 05:00 to 10:00 | Second rise | "
        "Step by winter |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for share, window, year in zip(
        result["clock_share"], result["window"], result["by_year"], strict=True
    ):
        reached = (
            f"{window['reached_at_0000']:.2f} / {window['reached_at_0030']:.2f}"
            if window["reached_at_0000"] is not None
            else "no step"
        )
        print(
            f"| {share['site']} | {share['bst']['at_local_step']:.0%} | "
            f"{share['bst']['at_utc_step']:.0%} | "
            f"{share['winter_reference']['at_local_step']:.0%} / "
            f"{share['winter_reference']['at_utc_step']:.0%} | {reached} | "
            f"{window['largest_fall_starts']} ({window['largest_fall_points']:+.1f}) | "
            f"{window['second_rise']} | {year['label']} ({year['change']:+.1f}) |"
        )


def by_year_rows() -> list[dict]:
    """Return each substation's first and last winter step, the change, and the trend label."""
    numbers = json.loads((common.WORK_DIR / "numbers_headline.json").read_text())
    rows = []
    for site in common.FITTED_SERIES:
        own = sorted(
            (r for r in numbers["per_winter"] if r["site"] == site), key=lambda r: r["winters"]
        )
        first, last = own[0], own[-1]
        if len(own) < MIN_WINTERS_FOR_TREND:
            label = "cannot tell"
        elif last["step_high"] < first["step_low"]:
            label = "falls"
        elif last["step_low"] > first["step_high"]:
            label = "rises"
        else:
            label = "cannot tell"
        rows.append(
            {
                "site": site,
                "winters": len(own),
                "first": first["step"],
                "last": last["step"],
                "change": last["step"] - first["step"],
                "label": label,
            }
        )
    return rows


def main() -> None:
    """Print the three readouts and write `followups.json`."""
    winters = common.HEADLINE_WINTERS
    infos = common.series_infos()
    table = pl.read_parquet(common.WORK_DIR / "night_tables.parquet")
    power = common.load_power(
        time_series_ids=[infos[s].time_series_id for s in common.FITTED_SERIES]
    )
    end = report.night_end(winters=winters)
    result: dict[str, list[dict]] = {"clock_share": [], "window": [], "by_year": []}
    for site in common.FITTED_SERIES:
        info = infos[site]
        own = power.filter(pl.col("time_series_id") == info.time_series_id)
        lookup = dict(zip(own["time"].to_list(), own["power"].to_list(), strict=True))
        scale = 100.0 / info.effective_capacity_mw
        cold = table.filter(
            (pl.col("site") == site)
            & ~pl.col("excluded")
            & (pl.col("daily_mean_c") < 8.0)
            & (pl.col("evening") < end)
        )
        bst = cold.filter(pl.col("season") == "bst_other")["evening"].to_list()
        gmt = cold.filter((pl.col("season") == "gmt") & pl.col("winter").is_in(list(winters)))[
            "evening"
        ].to_list()
        summer = table.filter(
            (pl.col("site") == site)
            & (pl.col("season") == "summer")
            & ~pl.col("excluded")
            & pl.col("evening").dt.year().is_in(list(common.SUMMER_YEARS))
        )["evening"].to_list()
        local_bst = profiles(lookup=lookup, evenings=bst, scale=scale, local=True)
        local_gmt = profiles(lookup=lookup, evenings=gmt, scale=scale, local=True)
        local_summer = profiles(lookup=lookup, evenings=summer, scale=scale, local=True)

        # item 1: where does each night's largest rise fall?
        def shares(matrix: np.ndarray) -> dict[str, float]:
            slots = np.array([jump_slot(profile=row) for row in matrix])
            local_slot = LOCAL_STEP_SLOT + 0  # half-hour starting 00:30 has index 5 in both lists
            return {
                "n": len(slots),
                "at_local_step": float((slots == local_slot).mean()),
                "at_utc_step": float((slots == local_slot + 2).mean()),
            }

        result["clock_share"].append(
            {"site": site, "bst": shares(local_bst), "winter_reference": shares(local_gmt)}
        )
        # item 2: window shape on cold winter nights, minus summer
        excess = local_gmt.mean(axis=0) - local_summer.mean(axis=0)
        plateau = float(excess[7:11].mean())  # half-hours starting 01:00 to 02:30
        drops = np.diff(excess)
        late = drops[SLOTS_LONG.index(5 * 60 + 22 * 60 - 22 * 60 + 0) if False else 14 : 23]
        fall_slot = int(np.argmin(late)) + 15
        increments = np.diff(excess)[2:12]
        main = int(np.argmax(increments)) + 3
        others = [(i + 3, v) for i, v in enumerate(increments) if abs((i + 3) - main) > 2]
        second = max(others, key=lambda item: item[1]) if others else (0, 0.0)
        main_size = float(increments[main - 3])
        result["window"].append(
            {
                "site": site,
                "plateau_points": plateau,
                "reached_at_0000": float(excess[4] / plateau) if plateau > 0 else None,
                "reached_at_0030": float(excess[5] / plateau) if plateau > 0 else None,
                "reached_at_0100": float(excess[6] / plateau) if plateau > 0 else None,
                "largest_fall_starts": slot_clock(index=fall_slot),
                "largest_fall_points": float(late.min()),
                "second_rise": slot_clock(index=second[0])
                if second[1] >= SECOND_RISE_SHARE * main_size
                else "none",
            }
        )
    result["by_year"] = by_year_rows()
    (common.WORK_DIR / "followups.json").write_text(json.dumps(result, indent=1))
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=2, tbl_width_chars=200):
        print("== 1. share of cold nights whose largest rise falls at each clock's step ==")
        print(
            pl.DataFrame(
                [
                    {
                        "site": r["site"],
                        **{f"bst_{k}": v for k, v in r["bst"].items()},
                        **{f"winter_{k}": v for k, v in r["winter_reference"].items()},
                    }
                    for r in result["clock_share"]
                ]
            )
        )
        print("== 2. window shape on cold winter nights ==")
        print(pl.DataFrame(result["window"]))
        print("== 3. step by winter ==")
        print(pl.DataFrame(result["by_year"]))
    print_cards(result=result)


if __name__ == "__main__":
    main()
