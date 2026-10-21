"""Find D8's multi-day drops, and describe their recurrence, neighbours, closures, and alarms.

Reads the half-hourly panel that the false-alarm analysis stored (`arrays.pkl`: metered power, the
frozen convex model's expected power, and ERA5 temperature per series on experiment 130's grid) and
the RO-NC alarms of experiment 130 arm B. Writes private tables under `WORK_DIR` and prints every
number the write-up quotes into `printed_report.txt`.

Run: OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00159-*/e00159_drops.py
"""

from __future__ import annotations

import pickle
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from scipy import stats
from studies.dp_calendar import bank_holiday_names
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00159-switching-d8-multi-day-drops-recurrence-and-report"
)
"""Private tables; never committed."""
FALSE_ALARM_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "briefs" / "sw_false_alarms" / "work"
LOGGED_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00037-switching-first-look" / "intervals.parquet"
)
GRID_START: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
D8_SERIES: Final[int] = 26
NEIGHBOUR_SERIES: Final[tuple[int, ...]] = (3, 16, 17)
"""D8's connected metered neighbours in experiment 130's adjacency."""
NEIGHBOUR_LABELS: Final[dict[int, str]] = {3: "D3", 16: "D14", 17: "D5"}
PLACEBO_FROM_STAMP: Final[int] = 546 * 48
"""2023-07-01, the month before which D8 has one drop; placebo times and windows start here."""
LONG_CLOSURE_JOB: Final[str] = "21D5583"
"""The logged job on D8 and D14 that overlaps 12 of the drops of 2025."""
HALF_HOURS_PER_DAY: Final[int] = 48
SMOOTHING_WINDOW: Final[int] = 48
"""The residual is smoothed with a centred rolling median of 24 hours."""
DEPTH_THRESHOLD_MW: Final[float] = 1.5
MERGE_GAP: Final[int] = 24
"""Runs separated by under 12 hours are merged."""
MIN_DURATION: Final[int] = 2 * HALF_HOURS_PER_DAY
NEIGHBOUR_WINDOW: Final[int] = 12
"""The neighbour's 6-hour median before and after the drop's start."""
STEP_FRACTIONS: Final[tuple[float, ...]] = (0.25, 0.5, 0.75)
LOG_PAD: Final[int] = 2 * HALF_HOURS_PER_DAY
ALARM_NEAR: Final[int] = 12
"""An alarm within 6 hours of a drop's start or end."""
N_PLACEBO: Final[int] = 5000
SEED: Final[int] = 0
REPORT: list[str] = []


def say(text: str = "") -> None:
    """Print a line and keep it for `printed_report.txt`."""
    print(text, flush=True)
    REPORT.append(text)


def wilson(*, k: int, n: int) -> tuple[float, float]:
    """95% Wilson interval for a share."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.96
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (float(centre - half), float(centre + half))


def share(*, k: int, n: int) -> str:
    """Format `k` of `n` with its Wilson interval."""
    low, high = wilson(k=k, n=n)
    return f"{k} of {n} ({k / n:.2f}; {low:.2f} to {high:.2f})"


def find_drops(
    *, residual: np.ndarray, threshold_mw: float = DEPTH_THRESHOLD_MW
) -> list[tuple[int, int]]:
    """Runs of the smoothed residual below -`DEPTH_THRESHOLD_MW`, merged, of at least 2 days."""
    smooth = (
        pl.Series(residual)
        .rolling_median(SMOOTHING_WINDOW, min_samples=SMOOTHING_WINDOW // 2, center=True)
        .to_numpy()
    )
    below = np.isfinite(smooth) & (smooth < -threshold_mw)
    edges = np.flatnonzero(np.diff(np.r_[0, below.astype(int), 0]) != 0).reshape(-1, 2)
    merged: list[list[int]] = []
    for first, last in edges:
        if merged and first - merged[-1][1] < MERGE_GAP:
            merged[-1][1] = int(last)
        else:
            merged.append([int(first), int(last)])
    return [(a, b) for a, b in merged if b - a >= MIN_DURATION]


def stamp_to_local(*, stamps: np.ndarray) -> pl.DataFrame:
    """Local (Europe/London) date, weekday (0 = Monday), hour of day, and month per stamp."""
    utc = pl.Series(
        [GRID_START + timedelta(minutes=30 * int(s)) for s in stamps],
        dtype=pl.Datetime(time_zone="UTC"),
    )
    local = utc.dt.convert_time_zone("Europe/London")
    return pl.DataFrame(
        {
            "date": local.dt.date(),
            "weekday": local.dt.weekday() - 1,
            "hour": local.dt.hour() + local.dt.minute() / 60,
            "month": local.dt.month(),
            "utc": utc,
        }
    )


def step_up(*, power: np.ndarray, at: int) -> float:
    """A neighbour's 6-hour median power after `at` minus its 6-hour median before."""
    before = power[at - NEIGHBOUR_WINDOW : at]
    after = power[at : at + NEIGHBOUR_WINDOW]
    if np.isfinite(before).sum() < 6 or np.isfinite(after).sum() < 6:
        return float("nan")
    return float(np.nanmedian(after) - np.nanmedian(before))


def closures_near(*, first: int, last: int) -> list[str]:
    """Logged closures touching D8 whose interval lies within 24 hours of [first, last)."""
    logged = pl.read_parquet(LOGGED_PATH).filter(
        (pl.col("side_a") == str(D8_SERIES)) | (pl.col("side_b") == str(D8_SERIES))
    )
    pad = 48
    hits = logged.filter(
        (pl.col("first_stamp") < last + pad) & (pl.col("last_stamp_exclusive") > first - pad)
    )
    return [f"{row['sheet']} {row['switch']}" for row in hits.iter_rows(named=True)]


def main() -> None:  # noqa: PLR0912, PLR0915
    """Build the drop table and every figure's input."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    arrays = pickle.loads((FALSE_ALARM_DIR / "arrays.pkl").read_bytes())
    d8 = arrays[D8_SERIES]
    power, expected, residual, temperature = (
        d8["power"],
        d8["expected"],
        d8["resid"],
        d8["temp"],
    )
    n_stamps = len(power)
    drops = find_drops(residual=residual)
    say(f"D8 grid: {n_stamps} half-hours, {np.isfinite(power).sum()} with power")
    for threshold in (1.0, 2.0, 2.5):
        count = len(find_drops(residual=residual, threshold_mw=threshold))
        say(f"Sensitivity: threshold -{threshold} MW gives {count} drops of 2 days or longer")
    say(f"Drops (smoothed residual below -{DEPTH_THRESHOLD_MW} MW, >= 2 days): {len(drops)}")

    firsts = np.array([a for a, _ in drops])
    lasts = np.array([b for _, b in drops])
    start_info = stamp_to_local(stamps=firsts)
    end_info = stamp_to_local(stamps=lasts - 1)
    names = bank_holiday_names()
    day_offsets = np.array([-1, 0, 1])

    def near_holiday(days: list[date]) -> list[bool]:
        return [any((d + timedelta(days=int(o))) in names for o in day_offsets) for d in days]

    rows: list[dict[str, object]] = []
    for k, (first, last) in enumerate(drops):
        inside = slice(first, last)
        depth = float(np.nanmedian(expected[inside] - power[inside]))
        steps = {s: step_up(power=arrays[s]["power"], at=first) for s in NEIGHBOUR_SERIES}
        end_steps = {s: step_up(power=arrays[s]["power"], at=last) for s in NEIGHBOUR_SERIES}
        best = max(steps, key=lambda s: -np.inf if np.isnan(steps[s]) else steps[s])
        closures = closures_near(first=first, last=last)
        rows.append(
            {
                "drop": k + 1,
                "first_stamp": first,
                "last_stamp": last,
                "start_date": start_info["date"][k],
                "end_date": end_info["date"][k],
                "depth_mw": depth,
                "duration_days": (last - first) / HALF_HOURS_PER_DAY,
                "start_weekday": int(start_info["weekday"][k]),
                "start_hour": float(start_info["hour"][k]),
                "end_weekday": int(end_info["weekday"][k]),
                "end_hour": float(end_info["hour"][k]),
                "month": int(start_info["month"][k]),
                "median_power_mw": float(np.nanmedian(power[inside])),
                "temperature_c": float(np.nanmean(temperature[inside])),
                "best_neighbour": NEIGHBOUR_LABELS[best],
                "best_neighbour_step_mw": steps[best],
                **{f"step_{NEIGHBOUR_LABELS[s]}_mw": steps[s] for s in NEIGHBOUR_SERIES},
                **{f"end_step_{NEIGHBOUR_LABELS[s]}_mw": end_steps[s] for s in NEIGHBOUR_SERIES},
                "closures": "; ".join(closures),
            }
        )
    table = pl.DataFrame(rows)
    table = table.with_columns(
        near_holiday=pl.Series(near_holiday(table["start_date"].to_list())),
        near_holiday_end=pl.Series(near_holiday(table["end_date"].to_list())),
    )
    table.write_parquet(WORK_DIR / "drops.parquet")

    # ---- summary statistics ----------------------------------------------------------------
    n = table.height
    say(
        f"\nDepth (median expected minus metered, MW): median {table['depth_mw'].median():.2f}, "
        f"range {table['depth_mw'].min():.2f} to {table['depth_mw'].max():.2f}, "
        f"mean {table['depth_mw'].mean():.2f}, sd {table['depth_mw'].std():.2f}"
    )
    say(
        f"Duration (days): median {table['duration_days'].median():.1f}, "
        f"range {table['duration_days'].min():.1f} to {table['duration_days'].max():.1f}"
    )
    days_inside = float(table["duration_days"].sum())
    total_days = n_stamps / HALF_HOURS_PER_DAY
    say(
        f"Total days inside drops: {days_inside:.1f} of {total_days:.0f} "
        f"({days_inside / total_days:.3f})"
    )
    say(f"First drop: {table['start_date'].min()}; last: {table['start_date'].max()}")
    per_year = table.group_by(table["start_date"].dt.year().alias("year")).len().sort("year")
    years = dict(zip(per_year["year"].to_list(), per_year["len"].to_list(), strict=True))
    say(f"Drops per start year: {years}")
    after_mid_2023 = table.filter(pl.col("start_date") >= date(2023, 7, 1)).height
    say(f"Drops starting on or after 2023-07-01: {after_mid_2023} of {n}")

    # What the power does inside a drop: the share of half-hours spent low.
    inside_mask = np.zeros(n_stamps, dtype=bool)
    for first, last in drops:
        inside_mask[first:last] = True
    after_first = np.arange(n_stamps) >= drops[0][0]
    valid = np.isfinite(power)
    for label, mask in (
        ("inside drops", inside_mask),
        ("outside drops, after the first", ~inside_mask & after_first),
    ):
        use = mask & valid
        say(
            f"Half-hours {label}: below 1 MW {np.mean(power[use] < 1.0):.3f}, below 3 MW "
            f"{np.mean(power[use] < 3.0):.3f}, median power {np.median(power[use]):.2f} MW, "
            f"median expected {np.nanmedian(expected[use]):.2f} MW"
        )

    # Weekday and hour of day.
    for label, col, bins in (
        ("start weekday", "start_weekday", 7),
        ("end weekday", "end_weekday", 7),
    ):
        counts = np.bincount(table[col].to_numpy(), minlength=bins)
        expected_counts = np.full(bins, n / bins)
        chi = stats.chisquare(counts, expected_counts)
        say(f"{label} counts Mon..Sun {counts.tolist()}, chi-squared p = {chi.pvalue:.3f}")
    for label, col in (("start hour", "start_hour"), ("end hour", "end_hour")):
        counts = np.bincount((table[col].to_numpy() // 6).astype(int), minlength=4)
        chi = stats.chisquare(counts, np.full(4, n / 4))
        say(
            f"{label} counts by 6-hour block (0-6, 6-12, 12-18, 18-24) {counts.tolist()}, "
            f"chi-squared p = {chi.pvalue:.3f}"
        )

    # Spacing between starts.
    gaps = np.diff(table["first_stamp"].to_numpy()) / HALF_HOURS_PER_DAY
    say(
        f"Spacing between starts (days): median {np.median(gaps):.1f}, "
        f"quartiles {np.percentile(gaps, 25):.1f} and {np.percentile(gaps, 75):.1f}, "
        f"range {gaps.min():.1f} to {gaps.max():.0f}"
    )
    after_mask = table["start_date"].to_numpy() >= np.datetime64("2023-07-01")
    late_gaps = np.diff(table["first_stamp"].to_numpy()[after_mask]) / HALF_HOURS_PER_DAY
    say(
        f"Spacing since 2023-07-01 (days): median {np.median(late_gaps):.1f}, "
        f"quartiles {np.percentile(late_gaps, 25):.1f} and {np.percentile(late_gaps, 75):.1f}"
    )

    # Depth similarity.
    depth = table["depth_mw"].to_numpy()
    say(
        f"Depth coefficient of variation {depth.std(ddof=1) / depth.mean():.2f}; "
        f"depth against start time Spearman rho "
        f"{stats.spearmanr(table['first_stamp'].to_numpy(), depth).statistic:.2f} "
        f"(p = {stats.spearmanr(table['first_stamp'].to_numpy(), depth).pvalue:.3f})"
    )

    # Bank holidays: share of drops starting within 1 day of one, against the share of days.
    all_days = pl.date_range(date(2022, 1, 1), date(2026, 6, 30), "1d", eager=True).to_list()
    base_holiday = float(np.mean(near_holiday(all_days)))
    k_holiday = int(table["near_holiday"].sum())
    say(
        f"Drop starts within 1 day of a bank holiday: {share(k=k_holiday, n=n)}; "
        f"share of all days within 1 day of one {base_holiday:.3f}; binomial p = "
        f"{stats.binomtest(k_holiday, n, base_holiday).pvalue:.3f}"
    )
    k_end = int(table["near_holiday_end"].sum())
    say(f"Drop ends within 1 day of a bank holiday: {share(k=k_end, n=n)}")

    # Season and temperature.
    month_days = np.array([sum(1 for d in all_days if d.month == m) for m in range(1, 13)])
    month_counts = np.bincount(table["month"].to_numpy(), minlength=13)[1:]
    chi = stats.chisquare(month_counts, n * month_days / month_days.sum())
    say(
        f"Drop starts by calendar month Jan..Dec {month_counts.tolist()}; chi-squared "
        f"p = {chi.pvalue:.3f}"
    )
    season_counts = {
        "winter (Dec-Feb)": int(month_counts[[11, 0, 1]].sum()),
        "spring (Mar-May)": int(month_counts[[2, 3, 4]].sum()),
        "summer (Jun-Aug)": int(month_counts[[5, 6, 7]].sum()),
        "autumn (Sep-Nov)": int(month_counts[[8, 9, 10]].sum()),
    }
    say(f"Drop starts by season: {season_counts}")
    # Temperature anomaly: drop mean against the mean of every day of the same calendar month.
    all_local = stamp_to_local(stamps=np.arange(0, n_stamps, 1))
    month_mean = {
        m: float(np.nanmean(temperature[(all_local["month"] == m).to_numpy()]))
        for m in range(1, 13)
    }
    anomaly = np.array(
        [t - month_mean[m] for t, m in zip(table["temperature_c"], table["month"], strict=True)]
    )
    rng = np.random.default_rng(SEED)
    boot = [rng.choice(anomaly, size=n).mean() for _ in range(5000)]
    say(
        f"Mean temperature during a drop minus the month's mean: {anomaly.mean():+.2f} C "
        f"(95% interval {np.percentile(boot, 2.5):+.2f} to {np.percentile(boot, 97.5):+.2f})"
    )

    # ---- neighbour check, with a placebo ---------------------------------------------------
    steps_best = table["best_neighbour_step_mw"].to_numpy()
    say("\nNeighbour check: best step-up among D3, D14, and D5 in the 6 hours after the start")
    say(
        f"Best neighbour step (MW): median {np.nanmedian(steps_best):+.2f}, "
        f"range {np.nanmin(steps_best):+.2f} to {np.nanmax(steps_best):+.2f}; drop depth median "
        f"{np.median(depth):.2f}"
    )
    in_drop = np.zeros(n_stamps, dtype=bool)
    for first, last in drops:
        in_drop[max(first - 2 * HALF_HOURS_PER_DAY, 0) : last + 2 * HALF_HOURS_PER_DAY] = True
    eligible = np.flatnonzero(~in_drop)
    eligible = eligible[(eligible > NEIGHBOUR_WINDOW) & (eligible < n_stamps - NEIGHBOUR_WINDOW)]
    eligible = eligible[eligible >= PLACEBO_FROM_STAMP]
    # Each placebo time has the time of day of a randomly drawn drop start.
    by_time_of_day = {h: eligible[eligible % 48 == h] for h in range(48)}
    placebo_at = np.array(
        [rng.choice(by_time_of_day[int(firsts[rng.integers(n)] % 48)]) for _ in range(N_PLACEBO)]
    )
    placebo_steps = np.array(
        [
            np.nanmax([step_up(power=arrays[s]["power"], at=int(a)) for s in NEIGHBOUR_SERIES])
            for a in placebo_at
        ]
    )
    placebo_rates: dict[float, float] = {}
    for fraction in STEP_FRACTIONS:
        hit = int((steps_best >= fraction * depth).sum())
        # Placebo: each placebo time takes a drop depth drawn at random.
        depths = rng.choice(depth, size=N_PLACEBO)
        finite = np.isfinite(placebo_steps)
        placebo_rate = float(np.mean(placebo_steps[finite] >= (fraction * depths)[finite]))
        placebo_rates[fraction] = placebo_rate
        say(
            f"Share of drops with a neighbour step-up of at least {fraction:.2f} x depth: "
            f"{share(k=hit, n=n)}; placebo (random times from July 2023, away from any drop, "
            f"same time of day) {placebo_rate:.3f}"
        )
    pl.DataFrame(
        {"fraction": list(placebo_rates), "share": list(placebo_rates.values())}
    ).write_parquet(WORK_DIR / "placebo.parquet")
    n_missing = int(np.isnan(steps_best).sum())
    say(f"Drops with no readable neighbour step: {n_missing}")
    # The step down at the drop's end, with the neighbour.
    end_cols = [f"end_step_{NEIGHBOUR_LABELS[s]}_mw" for s in NEIGHBOUR_SERIES]
    end_down = table.select(pl.min_horizontal(end_cols)).to_numpy().ravel()
    hit_end = int((end_down <= -0.5 * depth).sum())
    say(
        f"Share of drops with a neighbour step down of at least 0.5 x depth at the end: "
        f"{share(k=hit_end, n=n)}"
    )

    # ---- logged closures -------------------------------------------------------------------
    with_closure = int((table["closures"] != "").sum())
    say(f"\nDrops with a logged closure of D8 within 24 hours: {share(k=with_closure, n=n)}")
    durations = table["last_stamp"].to_numpy() - table["first_stamp"].to_numpy()
    placebo_hits = []
    for _ in range(2000):
        length = int(rng.choice(durations))
        while True:
            first = int(rng.choice(eligible[eligible < n_stamps - length]))
            if not in_drop[first : first + length].any():
                break
        placebo_hits.append(bool(closures_near(first=first, last=first + length)))
    say(
        f"Placebo: windows of random drop lengths placed at random times from July 2023 "
        f"that overlap no drop and "
        f"have a logged closure of D8 within 24 hours: {float(np.mean(placebo_hits)):.3f}"
    )
    other_closure = sum(
        any(not token.endswith(LONG_CLOSURE_JOB) for token in row.split("; ") if token)
        for row in table["closures"].to_list()
    )
    say(
        f"Drops with a logged closure other than the long-running job {LONG_CLOSURE_JOB}: "
        f"{share(k=other_closure, n=n)}"
    )
    logged = pl.read_parquet(LOGGED_PATH).filter(
        (pl.col("side_a") == str(D8_SERIES)) | (pl.col("side_b") == str(D8_SERIES))
    )
    say(f"Logged closures touching D8 in total: {logged.height}")

    # ---- alarms ----------------------------------------------------------------------------
    alarms = pl.read_parquet(FALSE_ALARM_DIR / "alarms_flagged.parquet").filter(
        pl.col("series") == D8_SERIES
    )
    unexplained = alarms.filter(pl.col("unexplained"))
    alarm_stamps = unexplained["stamp"].to_numpy()
    say(f"\nRO-NC alarms on D8: {alarms.height}; unexplained: {unexplained.height}")
    inside = near_start = near_end = 0
    offsets: list[dict[str, float]] = []
    for stamp, step in zip(alarm_stamps, unexplained["step_mw"].to_numpy(), strict=True):
        d_start = (stamp - firsts) / 2.0
        d_end = (stamp - lasts) / 2.0
        k_start = int(np.argmin(np.abs(d_start)))
        k_end = int(np.argmin(np.abs(d_end)))
        is_inside = bool(((stamp >= firsts) & (stamp < lasts)).any())
        is_near_start = abs(d_start[k_start]) <= ALARM_NEAR / 2
        is_near_end = abs(d_end[k_end]) <= ALARM_NEAR / 2
        inside += is_inside
        near_start += is_near_start
        near_end += is_near_end
        offsets.append(
            {
                "stamp": float(stamp),
                "step_mw": float(step),
                "hours_from_nearest_start": float(d_start[k_start]),
                "hours_from_nearest_end": float(d_end[k_end]),
                "inside_drop": float(is_inside),
            }
        )
    pl.DataFrame(offsets).write_parquet(WORK_DIR / "alarms_vs_drops.parquet")
    near_either = sum(
        1
        for o in offsets
        if abs(o["hours_from_nearest_start"]) <= 6 or abs(o["hours_from_nearest_end"]) <= 6
    )
    say(f"Unexplained alarms inside a drop: {share(k=inside, n=len(offsets))}")
    say(
        f"Within 6 hours of a drop's start: {share(k=near_start, n=len(offsets))}; of its end: "
        f"{share(k=near_end, n=len(offsets))}; of either: {share(k=near_either, n=len(offsets))}"
    )
    every = alarms["stamp"].to_numpy()
    every_near = sum(
        bool((np.abs(s - firsts) <= ALARM_NEAR).any() or (np.abs(s - lasts) <= ALARM_NEAR).any())
        for s in every
    )
    say(
        f"All {len(every)} RO-NC alarms on D8 within 6 hours of a drop's start or end: {every_near}"
    )
    every_start = sum(bool((np.abs(s - firsts) <= ALARM_NEAR).any()) for s in every)
    say(f"Drops with any RO-NC alarm within 6 hours of the start: {every_start} of {n}")
    drops_with_alarm = sum(bool((np.abs(every - f) <= ALARM_NEAR).any()) for f in firsts)
    say(f"Drops with an alarm of any kind within 6 hours of their start: {drops_with_alarm} of {n}")
    share_of_days = float(in_drop.mean())
    say(f"Share of the grid inside a drop or within 2 days of one: {share_of_days:.3f}")
    jump_size = 0.3 * float(d8["cap"])
    jump_steps = np.zeros(n_stamps, dtype=bool)
    jump_steps[1:] = np.abs(np.diff(power)) >= jump_size
    jump = sum(
        1
        for stamp in alarm_stamps
        if jump_steps[
            max(stamp - 2 * HALF_HOURS_PER_DAY, 0) : stamp + 2 * HALF_HOURS_PER_DAY + 1
        ].any()
    )
    say(
        f"Alarms with a metered half-hour jump of at least 0.3 x capacity ({jump_size:.2f} MW) "
        f"within 24 hours either side: {share(k=jump, n=len(offsets))}"
    )
    all_jump = jump_steps.sum()
    say(
        f"Half-hour jumps of that size on D8 in all history: {all_jump} of "
        f"{np.isfinite(np.diff(power)).sum()}"
    )
    window = 2 * HALF_HOURS_PER_DAY
    csum = np.r_[0, np.cumsum(jump_steps)]
    lo = np.clip(np.arange(n_stamps) - window, 0, n_stamps)
    hi = np.clip(np.arange(n_stamps) + window + 1, 0, n_stamps)
    chance_jump = float(np.mean(csum[hi] - csum[lo] > 0))
    say(f"Share of all half-hours with such a jump within 24 hours either side: {chance_jump:.3f}")
    near_edge = np.zeros(n_stamps, dtype=bool)
    for edge in np.r_[firsts, lasts]:
        near_edge[max(edge - ALARM_NEAR, 0) : edge + ALARM_NEAR + 1] = True
    say(
        f"Share of all half-hours within 6 hours of a drop's start or end: {near_edge.mean():.3f}; "
        f"expected alarms among 52: {52 * near_edge.mean():.1f}; "
        f"among 28: {28 * near_edge.mean():.1f}"
    )
    when = [GRID_START + timedelta(minutes=30 * int(s)) for s in alarm_stamps]
    say(f"Unexplained alarms from {min(when):%Y-%m-%d} to {max(when):%Y-%m-%d}")
    in_window = sum(
        datetime(2023, 8, 1, tzinfo=UTC) <= w < datetime(2024, 10, 1, tzinfo=UTC) for w in when
    )
    say(f"Unexplained alarms between 2023-08-01 and 2024-09-30: {in_window} of {len(when)}")
    say(f"Drops starting in 2025 or later: {int((table['start_date'] >= date(2025, 1, 1)).sum())}")
    long_2025 = sum(
        LONG_CLOSURE_JOB in c and d.year == 2025
        for c, d in zip(table["closures"].to_list(), table["start_date"].to_list(), strict=True)
    )
    say(f"Drops starting in 2025 with the long-running job {LONG_CLOSURE_JOB} near: {long_2025}")
    say(f"Median power of D5 (series 17): {np.nanmedian(arrays[17]['power']):.2f} MW")
    sign = np.sign(unexplained["step_mw"].to_numpy())
    say(f"Alarm sign: {int((sign < 0).sum())} downward and {int((sign > 0).sum())} upward")

    lines = [
        (
            "| Drop | Start | End | Depth (MW) | Duration (days) | Start weekday and hour "
            "| Best neighbour step (MW) | Logged closures within 24 hours |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for row in table.iter_rows(named=True):
        n_closures = len([c for c in row["closures"].split("; ") if c])
        lines.append(
            f"| {row['drop']} | {row['start_date']} | {row['end_date']} | {row['depth_mw']:.1f} "
            f"| {row['duration_days']:.1f} | {weekday_names[row['start_weekday']]} "
            f"{int(row['start_hour']):02d}:{int(row['start_hour'] % 1 * 60):02d} "
            f"| {row['best_neighbour_step_mw']:+.1f} ({row['best_neighbour']}) | {n_closures} |"
        )
    (Path(__file__).parent / "drops_table.md").write_text("\n".join(lines) + "\n")

    (WORK_DIR / "printed_report.txt").write_text("\n".join(REPORT) + "\n")
    (Path(__file__).parent / "printed_report.txt").write_text("\n".join(REPORT) + "\n")


if __name__ == "__main__":
    main()
