r"""Known-answer gate: add a +3 point step to real residuals on one day and follow it.

For each injected day, every reading of the day gets +3 points. Each arm's filter runs on the real
and on the changed stream, and the change in the correction is averaged over the D+1 rows whose
target falls 1 to 5 days after the injected day, as a share of the step. A holiday day is skipped
by K-skip, K-skip+1, and K-zero, so they must carry none of it; K0 must carry some. A control day
that is not a holiday must pass through K-skip as it does through K0. The injected days lie in the
tuning months.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00143-*/e00143_gate.py
"""

import json
from datetime import date
from typing import Final

import e00143_arms as arms
import e00143_common as common
import e00143_frames as frames
import e00143_holidays as hol
import numpy as np

STEP: Final[float] = 3.0
HOLIDAY_DAYS: Final[tuple[date, ...]] = (date(2025, 1, 1), date(2025, 5, 5), date(2025, 5, 26))
CONTROL_DAY: Final[date] = date(2025, 1, 15)
AFTER_CLUSTER_DAY: Final[date] = date(2025, 1, 3)
"""The day after the Christmas cluster ends (2 January); K-skip+1 skips it and K-skip does not."""
AFTER_DAYS: Final[range] = range(1, 6)
MIN_K0_SHARE: Final[float] = 0.10
"""K0 must carry at least this share into the second day after an injected holiday."""
MAX_SKIP_SHARE: Final[float] = 1e-9
CONTROL_TOLERANCE: Final[float] = 0.02
"""K-skip's share on the control day lies within this of K0's."""


def day_number(*, day: date) -> int:
    """Return a date as days since 1970-01-01."""
    return int((np.datetime64(day, "D") - hol.EPOCH).astype(np.int64))


def shares(*, site: str, day: date) -> dict[str, dict[int, float]]:
    """Return each arm's share of the step carried to D+1 targets 1 to 5 days later, at a site."""
    data = frames.get_site(site)
    days = arms.grid_days(data=data)
    stream = data.stream.copy()
    injected = np.isfinite(stream) & (days == day_number(day=day))
    stream[injected] += STEP
    target_day = days[data.position]
    d1 = data.d1_rows()
    out: dict[str, dict[int, float]] = {}
    for arm in arms.ARMS:
        base = arms.arm_correction(data=data, arm=arm).total
        moved = arms.arm_correction(data=data, arm=arm, stream=stream).total
        out[arm] = {}
        for after in AFTER_DAYS:
            rows = d1 & (target_day == day_number(day=day) + after)
            out[arm][after] = (
                float(np.mean(moved[rows] - base[rows]) / STEP) if rows.any() else np.nan
            )
    return out


def holiday_target_change(*, site: str, day: date) -> float:
    """Return K-zero's largest change in correction on holiday-target rows."""
    data = frames.get_site(site)
    days = arms.grid_days(data=data)
    stream = data.stream.copy()
    stream[np.isfinite(stream) & (days == day_number(day=day))] += STEP
    base = arms.arm_correction(data=data, arm="K-zero").total
    moved = arms.arm_correction(data=data, arm="K-zero", stream=stream).total
    holiday_row = np.isin(days[data.position], arms.holiday_numbers())
    return float(np.max(np.abs(moved - base)[holiday_row]))


def structure_checks(*, site: str) -> dict[str, float | int | bool]:
    """Check the masks on the real stream: skipped readings per holiday day, and the early rows.

    Returns the number of holiday days with readings, the smallest and largest count of skipped
    readings on one holiday day (46 to 50 on daylight-saving days), whether K-skip equals K0
    exactly on rows whose cutoff precedes the first skipped reading, and the share of grid days
    that hold a category label.
    """
    data = frames.get_site(site)
    days = arms.grid_days(data=data)
    skipped = np.isfinite(data.stream) & np.isin(days, arms.holiday_numbers())
    counts = np.array([skipped[days == d].sum() for d in np.unique(days[skipped])])
    first = int(np.flatnonzero(skipped)[0])
    cutoff = data.cutoffs(delay_hours=arms.DELAY_HOURS)[data.run_of_row]
    early = cutoff < first
    k0 = arms.arm_correction(data=data, arm="K0").total
    skip = arms.arm_correction(data=data, arm="K-skip").total
    labels = hol.categorise(local_day=np.unique(days), holiday=arms.holiday_numbers())
    return {
        "holiday_days_with_readings": len(counts),
        "fewest_skipped_on_a_day": int(counts.min()),
        "most_skipped_on_a_day": int(counts.max()),
        "early_rows": int(early.sum()),
        "early_rows_identical": bool(np.array_equal(k0[early], skip[early])),
        "labelled_days": int((labels != "").sum()),
        "grid_days": len(labels),
    }


def main() -> None:
    """Run the gate and print and store its numbers."""
    results: dict[str, dict] = {}
    for day in (*HOLIDAY_DAYS, CONTROL_DAY, AFTER_CLUSTER_DAY):
        per_site = {site: shares(site=site, day=day) for site in common.SCORED_SITES}
        median = {
            arm: {
                after: float(np.nanmedian([per_site[s][arm][after] for s in common.DECISION_SITES]))
                for after in AFTER_DAYS
            }
            for arm in arms.ARMS
        }
        worst_skip = {
            arm: float(
                np.nanmax(
                    [abs(per_site[s][arm][a]) for s in common.SCORED_SITES for a in AFTER_DAYS]
                )
            )
            for arm in ("K-skip", "K-skip+1", "K-zero")
        }
        results[day.isoformat()] = {
            "per_site": per_site,
            "median": median,
            "worst_skip": worst_skip,
        }
        print(f"== injected day {day} (share of the +3 step, median of 5 decision sites) ==")
        for arm in arms.ARMS:
            print(f"  {arm}: " + ", ".join(f"day+{a} {median[arm][a]:+.3f}" for a in AFTER_DAYS))
        print(f"  largest |share| in K-skip, K-skip+1, K-zero over all 6 sites: {worst_skip}")
    zero_change = max(
        holiday_target_change(site=site, day=HOLIDAY_DAYS[0]) for site in common.SCORED_SITES
    )
    print(f"K-zero largest change on holiday targets: {zero_change:.2e}")
    conditions = {
        "K0 carries on every injected holiday (day+2)": all(
            results[d.isoformat()]["median"]["K0"][2] >= MIN_K0_SHARE for d in HOLIDAY_DAYS
        ),
        "skip arms carry nothing from a holiday": all(
            max(results[d.isoformat()]["worst_skip"].values()) <= MAX_SKIP_SHARE
            for d in HOLIDAY_DAYS
        ),
        "K-skip matches K0 on the control day": all(
            abs(
                results[CONTROL_DAY.isoformat()]["median"]["K-skip"][a]
                - results[CONTROL_DAY.isoformat()]["median"]["K0"][a]
            )
            <= CONTROL_TOLERANCE
            for a in AFTER_DAYS
        ),
        "K-zero is unchanged on holiday targets": zero_change <= MAX_SKIP_SHARE,
    }
    after = results[AFTER_CLUSTER_DAY.isoformat()]["median"]
    conditions["a step the day after the cluster is carried by K-skip but not K-skip+1"] = bool(
        after["K-skip"][2] >= MIN_K0_SHARE and abs(after["K-skip+1"][2]) <= MAX_SKIP_SHARE
    )
    checks = {site: structure_checks(site=site) for site in common.SCORED_SITES}
    print(json.dumps(checks, indent=1))
    conditions["skipped readings per holiday day are 46 to 50 and early rows are identical"] = all(
        c["fewest_skipped_on_a_day"] >= 46
        and c["most_skipped_on_a_day"] <= 50
        and c["early_rows_identical"]
        for c in checks.values()
    )
    print(json.dumps(conditions, indent=1))
    common.write_json_atomic(
        path=common.WORK_DIR / "gate.json",
        content={"conditions": conditions, "days": results, "structure": checks},
    )


if __name__ == "__main__":
    main()
