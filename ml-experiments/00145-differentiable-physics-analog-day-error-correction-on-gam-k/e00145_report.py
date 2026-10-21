"""Score experiment 145 and print every number its write-up quotes.

Writes `work/results.json` and prints the same numbers as text, which is committed beside the
write-up as `report.txt`. Intervals resample whole calendar months, paired across arms. The five
planned contrasts use 99% intervals (Bonferroni across five at an overall 5%); every other
contrast uses 95%.

    uv run --no-sync python ml-experiments/00145-*/e00145_report.py \
        > docs/ml-experiments/00145-*/report.txt
"""

import json
from datetime import date, timedelta
from typing import Final

import e00145_analog as analog
import e00145_common as common
import e00145_run as run
import e00145_score as score
import numpy as np
import polars as pl
from studies.tail_calibration import tail_calibration_table

PERIODS: Final[tuple[str, ...]] = ("replicate", "scored")
ARMS: Final[tuple[str, ...]] = (
    "K0",
    "R",
    "RB",
    "RB-day-types",
    "ANALOG-k",
    "ANALOG+R",
    "ANALOG-holiday",
    "ANALOG+R-notemp",
    "ANALOG+R-recent",
)
PLANNED: Final[dict[str, tuple[str, str]]] = {
    "C1 ANALOG+R - R": ("ANALOG+R", "R"),
    "C2 ANALOG-k - R": ("ANALOG-k", "R"),
    "C3 ANALOG+R - RB-day-types": ("ANALOG+R", "RB-day-types"),
    "C4 ANALOG-holiday - R": ("ANALOG-holiday", "R"),
    "C5 ANALOG+R - K0": ("ANALOG+R", "K0"),
}
EXTRA: Final[dict[str, tuple[str, str]]] = {
    "ANALOG+R-notemp - ANALOG+R (diagnostic)": ("ANALOG+R-notemp", "ANALOG+R"),
    "ANALOG+R-recent - ANALOG+R (post hoc)": ("ANALOG+R-recent", "ANALOG+R"),
    "ANALOG+R-recent - R (post hoc)": ("ANALOG+R-recent", "R"),
    "ANALOG+R - RB": ("ANALOG+R", "RB"),
    "ANALOG-k - ANALOG+R": ("ANALOG-k", "ANALOG+R"),
    "R - K0": ("R", "K0"),
}
SHOWN: Final[tuple[str, ...]] = ("K0", "R", "ANALOG+R", "ANALOG-holiday")
PLANNED_LEVEL: Final[float] = 99.0
WINTER: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})
WEEKDAYS: Final[tuple[str, ...]] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
COLD_SNAP_C: Final[float] = 3.0
TAIL_LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.95, 0.99)
EVENING: Final[range] = range(16, 20)
D1: Final = score.LEAD_BANDS[0]
ALL_LEADS: Final = score.LEAD_BANDS[6]


def local_columns(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the valid time's local weekday (0 Monday), hour, month, and date."""
    local = pl.col("valid_time").dt.convert_time_zone("Europe/London")
    return frame.with_columns(
        weekday=local.dt.weekday() - 1,
        local_hour=local.dt.hour(),
        month=local.dt.month(),
        local_date=local.dt.date(),
    )


def holiday_flags(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add `holiday_like` (the rule of `e00145_profiles`), `christmas_window`, and `easter`."""
    dates = frame["local_date"]
    first, last = dates.min(), dates.max()
    assert isinstance(first, date)
    assert isinstance(last, date)
    holiday = set(analog.holiday_identities(first=first, last=last))
    from studies.dp_calendar import bank_holiday_names

    names = bank_holiday_names()
    good_fridays = [d for d, n in names.items() if n == "Good Friday"]
    easter = {g + timedelta(days=k) for g in good_fridays for k in range(4)}
    after = {h + timedelta(days=k) for h in holiday for k in (1, 2, 3)} - holiday
    christmas = {
        d
        for d in (first + timedelta(days=i) for i in range((last - first).days + 1))
        if (d.month == 12 and d.day >= 15) or (d.month == 1 and d.day <= 7)
    }
    return frame.with_columns(
        holiday_like=dates.is_in(list(holiday)),
        christmas_window=dates.is_in(list(christmas)),
        easter=dates.is_in(list(easter)),
        after_holiday=dates.is_in(list(after)),
    )


def cold_mask(*, frame: pl.DataFrame, site: str) -> np.ndarray:
    """Return True for rows on a cold-snap day (daily mean under 3 C) at the site."""
    cache = common.load_cache()
    features = cache.sites[site]
    dates = cache.calendar.local_date.astype("datetime64[D]").astype(np.int64)
    unique, inverse = np.unique(dates, return_inverse=True)
    mean = np.bincount(inverse, weights=features.temperature_c) / np.bincount(inverse)
    months = unique.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64) % 12 + 1
    cold_days = unique[(mean < COLD_SNAP_C) & np.isin(months, list(WINTER))]
    position = np.searchsorted(
        cache.end_times, frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    )
    return np.isin(dates[np.minimum(position, len(dates) - 1)], cold_days)


def load_frames(*, period: str, sites: tuple[str, ...]) -> dict[str, pl.DataFrame]:
    """Return every arm's rows for a period."""
    frames = {arm: score.load_arm(arm=arm, period=period, sites=sites) for arm in ARMS}
    return {arm: holiday_flags(frame=local_columns(frame=f)) for arm, f in frames.items()}


def pair_rows(*, frames: dict[str, pl.DataFrame], pair: tuple[str, str]) -> dict[str, pl.DataFrame]:
    """Restrict a pair of arms to the rows both hold."""
    return score.common_rows(arms_by_name={k: frames[k] for k in pair})


def run_contrast(
    *,
    subset: dict[str, pl.DataFrame],
    pair: tuple[str, str],
    level: float = 95.0,
    minimum: int = 300,
) -> dict | None:
    """Return a paired contrast on already-selected rows, or `None` if there are too few rows."""
    if subset[pair[0]].height < minimum:
        return None
    losses = score.stack_losses(arms_by_name=subset)
    out = {
        "qs": score.contrast(losses=losses, treatment=pair[0], reference=pair[1], level=level),
        "mae_median": score.contrast(
            losses=losses, treatment=pair[0], reference=pair[1], metric="abs_median", level=level
        ),
        "absolute": {a: score.mean_of(score.quantile_score(frame=subset[a])) for a in pair},
        "days": int(subset[pair[0]].select(pl.struct("site", "local_date")).n_unique()),
    }
    if "abs_mean" in losses.columns:
        out["mae_mean"] = score.contrast(
            losses=losses, treatment=pair[0], reference=pair[1], metric="abs_mean", level=level
        )
    return out


def breakdowns(*, frames: dict[str, pl.DataFrame], pair: tuple[str, str], level: float) -> dict:
    """Return a contrast at D+1 on each breakdown of the plan."""
    d1 = {
        k: score.lead_filter(frame=f, band=D1)
        for k, f in pair_rows(frames=frames, pair=pair).items()
    }
    out: dict = {}

    def pick(name: str, expression: pl.Expr) -> None:
        out[name] = run_contrast(
            subset={k: f.filter(expression) for k, f in d1.items()}, pair=pair, level=level
        )

    out["all D+1"] = run_contrast(subset=d1, pair=pair, level=level)
    out["all leads"] = run_contrast(
        subset={
            k: score.lead_filter(frame=f, band=ALL_LEADS)
            for k, f in pair_rows(frames=frames, pair=pair).items()
        },
        pair=pair,
        level=level,
    )
    pick("non-holiday rows", ~pl.col("holiday_like"))
    pick("holiday-like days", pl.col("holiday_like"))
    pick("Christmas window", pl.col("christmas_window"))
    pick("Easter", pl.col("easter"))
    pick("1 to 3 days after a holiday-like day", pl.col("after_holiday"))
    for index, name in enumerate(WEEKDAYS):
        pick(f"weekday {name}", (pl.col("weekday") == index) & ~pl.col("holiday_like"))
    winter = pl.col("month").is_in(list(WINTER))
    pick("winter", winter)
    pick("winter evening 16-20", winter & pl.col("local_hour").is_in(list(EVENING)))
    pick("non-winter", ~winter)
    cold = {}
    for key, f in d1.items():
        flags = np.zeros(f.height, dtype=bool)
        for site in common.SCORED_SITES:
            where = (f["site"] == site).to_numpy()
            if where.any():
                flags[where] = cold_mask(frame=f.filter(pl.col("site") == site), site=site)
        cold[key] = f.filter(pl.Series(flags))
    out["cold-snap days"] = run_contrast(subset=cold, pair=pair, level=level, minimum=200)
    cold_night = {
        k: f.filter(pl.col("local_hour").is_in([22, 23, 0, 1, 2, 3, 4, 5, 6]))
        for k, f in cold.items()
    }
    out["cold-snap nights"] = run_contrast(subset=cold_night, pair=pair, level=level, minimum=200)
    return out


def lead_bands_and_leave_out(
    *, frames: dict[str, pl.DataFrame], pair: tuple[str, str], level: float
) -> dict:
    """Return a contrast per lead band, and at D+1 with each decision substation left out."""
    shared = pair_rows(frames=frames, pair=pair)
    out: dict = {"bands": {}, "leave_one_site_out": {}}
    for band in score.LEAD_BANDS:
        out["bands"][band[0]] = run_contrast(
            subset={k: score.lead_filter(frame=f, band=band) for k, f in shared.items()},
            pair=pair,
            level=level,
        )
    d1 = {k: score.lead_filter(frame=f, band=D1) for k, f in shared.items()}
    for site in common.DECISION_SITES:
        out["leave_one_site_out"][site] = run_contrast(
            subset={k: f.filter(pl.col("site") != site) for k, f in d1.items()},
            pair=pair,
            level=level,
        )
    return out


def tails(*, frames: dict[str, pl.DataFrame]) -> dict:
    """Return coverage and pinball loss at the tail levels per arm at D+1."""
    shared = score.common_rows(arms_by_name={a: frames[a] for a in SHOWN})
    return {
        arm: tail_calibration_table(
            frame=score.lead_filter(frame=f, band=D1), levels=TAIL_LEVELS
        ).to_dicts()
        for arm, f in shared.items()
    }


def mean_qs(*, frame: pl.DataFrame, band: tuple[str, float, float]) -> float:
    """Return the mean quantile score of a frame's rows in a lead band."""
    return score.mean_of(score.quantile_score(frame=score.lead_filter(frame=frame, band=band)))


def interval(entry: dict | None) -> str:
    """Return a contrast as `estimate [lower, upper] sites better`."""
    if entry is None:
        return "too few rows"
    q = entry["qs"]
    text = (
        f"{q['estimate']:+.4f} [{q['lower']:+.4f}, {q['upper']:+.4f}] "
        f"{q['sites_better']}/{q['n_sites']} sites, rows {q['rows']:,}, "
        f"site-days {entry['days']:,}; MAE median {entry['mae_median']['estimate']:+.4f}"
    )
    if "mae_mean" in entry:
        text += f", MAE mean {entry['mae_mean']['estimate']:+.4f}"
    return text


def print_cold_night_coverage(*, frames: dict[str, pl.DataFrame]) -> None:
    """Print the share of readings at or below q95 and q99 on cold-snap nights at D+1."""
    print("cold-snap nights (22:00 to 06:00, daily mean under 3 C) at D+1: coverage of q95, q99")
    for arm in SHOWN:
        d1 = score.lead_filter(frame=frames[arm], band=D1)
        night = d1.filter(pl.col("local_hour").is_in([22, 23, 0, 1, 2, 3, 4, 5, 6]))
        flags = np.zeros(night.height, dtype=bool)
        for site in common.SCORED_SITES:
            where = (night["site"] == site).to_numpy()
            if where.any():
                flags[where] = cold_mask(frame=night.filter(pl.col("site") == site), site=site)
        rows = night.filter(pl.Series(flags))
        if rows.height == 0:
            continue
        print(
            f"    {arm}: rows {rows.height:,}, q95 {(rows['actual'] <= rows['q95']).mean():.3f}, "
            f"q99 {(rows['actual'] <= rows['q99']).mean():.3f}"
        )


def print_history_counts() -> None:  # noqa: PLR0912
    """Print candidate counts, the holiday paths, and the cold-day match quality at site D10."""
    site = "D10"
    days = run.site_days(site=site)
    spec = run.chosen_spec(family="ANALOG+R")
    dates = [analog.EPOCH_DATE + timedelta(days=int(t)) for t in days.targets]
    first_scored = date(2025, 7, 3)
    for label, low, high in (
        ("tuning and replicate", date(2024, 7, 1), date(2025, 6, 30)),
        ("scored", first_scored, date(2026, 6, 30)),
    ):
        counts, prior = [], []
        for index, day in enumerate(dates):
            if not (low <= day <= high) or days.identity[index]:
                continue
            chosen = analog.analog_days(days=days, run=index, spec=spec)
            pool = analog.analog_days(
                days=days,
                run=index,
                spec=analog.AnalogSpec(kappa=1.0, k=10_000, weeks=spec.weeks),
            )
            if pool is not None:
                counts.append(len(pool))
                prior.append(int((days.targets[index] - days.targets[pool] > 300).sum()))
            del chosen
        print(
            f"{site} {label}: non-holiday targets with an analog pool {len(counts)}, mean pool "
            f"{np.mean(counts):.1f}, mean prior-year candidates {np.mean(prior):.1f}"
        )
    holiday_spec = run.HOLIDAY_SPEC
    scored = [
        i for i, d in enumerate(dates) if d >= first_scored and days.identity[i] and days.usable[i]
    ]
    identity_path = fallback_path = none = 0
    for index in scored:
        found = analog.analog_days(days=days, run=index, spec=holiday_spec)
        if found is None:
            none += 1
            continue
        if all(days.identity[j] == days.identity[index] for j in found):
            identity_path += 1
        else:
            fallback_path += 1
    print(
        f"{site} scored holiday-like targets {len(scored)} (the scored runs hold fewer): identity "
        f"path {identity_path}, fallback path {fallback_path}, no set {none}"
    )
    gaps = []
    for index, day in enumerate(dates):
        if day < first_scored or days.identity[index] or days.temperature_c[index] >= COLD_SNAP_C:
            continue
        found = analog.analog_days(days=days, run=index, spec=spec)
        if found is not None:
            gaps.append(float(np.abs(days.temperature_c[found] - days.temperature_c[index]).mean()))
    print(
        f"{site} cold scored days {len(gaps)}: mean absolute temperature gap to the chosen "
        f"analogs {np.mean(gaps):.2f} C (all-days gap on non-cold days for comparison below)"
    )
    warm = []
    for index, day in enumerate(dates):
        if day < first_scored or days.identity[index] or days.temperature_c[index] < COLD_SNAP_C:
            continue
        found = analog.analog_days(days=days, run=index, spec=spec)
        if found is not None:
            warm.append(float(np.abs(days.temperature_c[found] - days.temperature_c[index]).mean()))
    print(f"{site} other scored days {len(warm)}: mean gap {np.mean(warm):.2f} C")


def main() -> None:  # noqa: PLR0912
    """Score everything and print."""
    tuned = json.loads((common.WORK_DIR / "tuned.json").read_text())
    gate = json.loads((common.WORK_DIR / "gate.json").read_text())
    print("== Tuning (pooled D+1 mean absolute error of the corrected median, tuning rows) ==")
    print(f"chosen: {tuned['chosen']}")
    for family, name in tuned["chosen"].items():
        print(f"  {family}: {name} {tuned['pooled_mae'][name]:.4f}")
    for key in (
        "K0",
        "R-exact",
        "R-kappa0.5",
        "R-kappa1",
        "R-kappa1.25",
        "ANALOG-holiday",
        "ANALOG-k-ref",
        "ANALOG+R-ref",
    ):
        print(f"  {key}: {tuned['pooled_mae'][key]:.4f}")
    print("== Gate: median share of the injected shift recovered, and the lookahead check ==")
    print(json.dumps(gate["median_recovery"], indent=1))
    print(
        "G0 largest profile change when the target day and the origin's day are injected: "
        f"{max(max(r['g0_max_change'].values()) for r in gate['sites'])}"
    )
    print_history_counts()
    results: dict = {}
    for period in PERIODS:
        frames = load_frames(period=period, sites=common.DECISION_SITES)
        results[period] = {"planned": {}, "extra": {}}
        print(f"== {period}: planned contrasts (99%), then breakdowns ==")
        for label, pair in PLANNED.items():
            block = breakdowns(frames=frames, pair=pair, level=PLANNED_LEVEL)
            results[period]["planned"][label] = block
            extra = lead_bands_and_leave_out(frames=frames, pair=pair, level=PLANNED_LEVEL)
            results[period]["planned"][label]["lead_bands"] = extra
            for name, entry in extra["bands"].items():
                print(f"{period} | {label} | lead band {name} | {interval(entry)}")
            for name, entry in extra["leave_one_site_out"].items():
                print(f"{period} | {label} | leave out {name} | {interval(entry)}")
            for name, entry in block.items():
                if name == "lead_bands":
                    continue
                print(f"{period} | {label} | {name} | {interval(entry)}")
                if name == "all D+1" and entry:
                    print(f"    absolute {entry['absolute']}")
        print(f"== {period}: other contrasts (95%) ==")
        for label, pair in EXTRA.items():
            block = breakdowns(frames=frames, pair=pair, level=95.0)
            results[period]["extra"][label] = block
            for name in (
                "all D+1",
                "all leads",
                "non-holiday rows",
                "winter",
                "winter evening 16-20",
            ):
                print(f"{period} | {label} | {name} | {interval(block[name])}")
        d9 = load_frames(period=period, sites=("D9",))
        print(
            f"{period} | D9 absolute D+1: "
            + ", ".join(f"{a} {mean_qs(frame=f, band=D1):.3f}" for a, f in d9.items())
        )
        shared = score.common_rows(arms_by_name=frames)
        print(f"{period} | absolute on rows every arm holds:")
        for arm, f in shared.items():
            print(
                f"    {arm}: D+1 {mean_qs(frame=f, band=D1):.3f}, "
                f"all leads {mean_qs(frame=f, band=ALL_LEADS):.3f}"
            )
        print(f"{period} | tails at D+1 (coverage gap pts, pinball):")
        for arm, rows in tails(frames=frames).items():
            print(
                f"    {arm}: "
                + ", ".join(
                    f"q{round(r['level'] * 100):02d} {100 * r['gap']:+.2f} {r['pinball']:.4f}"
                    for r in rows
                )
            )
        print(f"{period} | median minus mean forecast at D+1, per site:")
        for arm in SHOWN:
            d1 = score.lead_filter(frame=frames[arm], band=D1)
            grouped = (
                d1.group_by("site").agg(gap=(pl.col("q50") - pl.col("mean")).mean()).sort("site")
            )
            print(f"    {arm}: " + ", ".join(f"{s} {g:+.3f}" for s, g in grouped.iter_rows()))
        print_cold_night_coverage(frames=frames)
    common.write_json_atomic(path=common.WORK_DIR / "results.json", content=results)


if __name__ == "__main__":
    main()
