"""Score experiment 134 and print every number its write-up quotes.

Writes `work/results.json` and prints the same numbers as text, which is committed beside the
write-up as `report.txt`. Intervals resample whole calendar months, paired across arms. The five
planned contrasts use 99% intervals (Bonferroni across five at an overall 5%); every other
contrast uses 95%.

    uv run --no-sync python ml-experiments/00134-*/e00134_report.py \
        > docs/ml-experiments/00134-*/report.txt
"""

import json
from datetime import date, timedelta
from typing import Final

import e00134_common as common
import e00134_profiles as profiles
import e00134_score as score
import numpy as np
import polars as pl
from studies.tail_calibration import tail_calibration_table

PERIODS: Final[tuple[str, ...]] = ("replicate", "scored")
ARMS: Final[tuple[str, ...]] = ("K0", "R", "Rh", "L1", "RW", "RB", "RB-day-types")
PLANNED: Final[dict[str, tuple[str, str]]] = {
    "C1 RB - R": ("RB", "R"),
    "C2 RW - R": ("RW", "R"),
    "C3 RB - K0": ("RB", "K0"),
    "C4 RB - L1": ("RB", "L1"),
    "C5 RB-day-types - RB": ("RB-day-types", "RB"),
}
EXTRA: Final[dict[str, tuple[str, str]]] = {
    "C6 RB - Rh (diagnostic)": ("RB", "Rh"),
    "Rh - R": ("Rh", "R"),
    "R - K0": ("R", "K0"),
    "RB-day-types - R": ("RB-day-types", "R"),
}
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
    """Add `holiday_like` (the rule of `e00134_profiles`), `christmas_window`, and `easter`."""
    dates = frame["local_date"]
    first, last = dates.min(), dates.max()
    assert isinstance(first, date)
    assert isinstance(last, date)
    holiday = profiles.holiday_like_days(first=first, last=last)
    from studies.dp_calendar import bank_holiday_names

    names = bank_holiday_names()
    good_fridays = [d for d, n in names.items() if n == "Good Friday"]
    easter = {g + timedelta(days=k) for g in good_fridays for k in range(4)}
    christmas = {
        d
        for d in (first + timedelta(days=i) for i in range((last - first).days + 1))
        if (d.month == 12 and d.day >= 15) or (d.month == 1 and d.day <= 7)
    }
    return frame.with_columns(
        holiday_like=dates.is_in(list(holiday)),
        christmas_window=dates.is_in(list(christmas)),
        easter=dates.is_in(list(easter)),
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


def copy7_frames(*, period: str, sites: tuple[str, ...]) -> dict[str, pl.DataFrame]:
    """Return COPY7 beside K0 and RB on the rows where the reading 7 days earlier exists."""
    frames = load_frames(period=period, sites=sites)
    cache = common.load_cache()
    values = {}
    for site in sites:
        features = cache.sites[site]
        present = features.present
        y = features.y
        valid = frames["K0"].filter(pl.col("site") == site)["valid_time"]
        position = np.searchsorted(cache.end_times, valid.dt.replace_time_zone(None).to_numpy())
        earlier = position - 7 * 48
        ok = (earlier >= 0) & present[np.maximum(earlier, 0)]
        values[site] = np.where(ok, y[np.maximum(earlier, 0)], np.nan)
    pieces = []
    base = frames["K0"]
    for site in sites:
        part = base.filter(pl.col("site") == site)
        value = values[site].astype(np.float32)
        pieces.append(
            part.with_columns(
                **{q: pl.Series(value) for q in score.LEVEL_COLUMNS},
                mean=pl.Series(value),
            )
        )
    copy = pl.concat(pieces).filter(pl.col("q50").is_finite())
    return {"COPY7": copy, "K0": frames["K0"], "RB": frames["RB"], "R": frames["R"]}


def tails(*, frames: dict[str, pl.DataFrame]) -> dict:
    """Return coverage and pinball loss at the tail levels per arm at D+1."""
    shared = score.common_rows(arms_by_name={a: frames[a] for a in ("K0", "R", "Rh", "RB")})
    return {
        arm: tail_calibration_table(
            frame=score.lead_filter(frame=f, band=D1), levels=TAIL_LEVELS
        ).to_dicts()
        for arm, f in shared.items()
    }


def mean_qs(*, frame: pl.DataFrame, band: tuple[str, float, float]) -> float:
    """Return the mean quantile score of a frame's rows in a lead band."""
    return mean_qs(frame=frame, band=band)


def interval(entry: dict | None) -> str:
    """Return a contrast as `estimate [lower, upper] sites better`."""
    if entry is None:
        return "too few rows"
    q = entry["qs"]
    text = (
        f"{q['estimate']:+.4f} [{q['lower']:+.4f}, {q['upper']:+.4f}] "
        f"{q['sites_better']}/{q['n_sites']} sites, rows {q['rows']:,}; MAE median "
        f"{entry['mae_median']['estimate']:+.4f}"
    )
    if "mae_mean" in entry:
        text += f", MAE mean {entry['mae_mean']['estimate']:+.4f}"
    return text


def main() -> None:  # noqa: PLR0912
    """Score everything and print."""
    tuned = json.loads((common.WORK_DIR / "tuned.json").read_text())
    gate = json.loads((common.WORK_DIR / "gate.json").read_text())
    print("== Tuning (pooled D+1 mean absolute error of the corrected median, tuning rows) ==")
    print(f"chosen: {tuned['chosen']}")
    for family, name in tuned["chosen"].items():
        print(f"  {family}: {name} {tuned['pooled_mae'][name]:.4f}")
    for key in ("K0", "R-exact", "RB-ref", "RB-ref-placebo"):
        print(f"  {key}: {tuned['pooled_mae'][key]:.4f}")
    print(
        "  RB grid (kappa 0.75): "
        + ", ".join(
            f"{k}={v:.4f}"
            for k, v in sorted(tuned["pooled_mae"].items())
            if k.startswith("RB_") and k.endswith("k0.75")
        )
    )
    print("== Gate: median share of the injected shift recovered ==")
    print(json.dumps(gate["median_recovery"], indent=1))
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
        for arm in ("K0", "R", "Rh", "RB"):
            d1 = score.lead_filter(frame=frames[arm], band=D1)
            grouped = (
                d1.group_by("site").agg(gap=(pl.col("q50") - pl.col("mean")).mean()).sort("site")
            )
            print(f"    {arm}: " + ", ".join(f"{s} {g:+.3f}" for s, g in grouped.iter_rows()))
        copy = copy7_frames(period=period, sites=common.DECISION_SITES)
        copy = score.common_rows(arms_by_name=copy)
        print(
            f"{period} | COPY7 subset (rows with a reading 7 days earlier): "
            + ", ".join(
                f"{a} D+1 {mean_qs(frame=f, band=D1):.3f} "
                f"all leads {mean_qs(frame=f, band=ALL_LEADS):.3f}"
                for a, f in copy.items()
            )
        )
    common.write_json_atomic(path=common.WORK_DIR / "results.json", content=results)


if __name__ == "__main__":
    main()
