"""Step 0 of experiment 124: the closure audit of the bulk supply point (BSP) readings.

The audit asks whether the Primaries, the metered generators, a level, and unmetered solar close
the BSP readings, and where the residual sits.

Run `uv run python ml-experiments/00124-.../e00124_audit.py`. It is descriptive: it writes
`report.md` beside the private working files, and the decision rule for step 1 is applied in the
write-up from the numbers it prints.

**The fit, per BSP and cut, uses only rows before the cut.**

    BSP(t) = a_P * sum(MW Primaries) + b_G * sum(generators) + c * sum(MVA Primaries)
             - q * pv_shape(t) + level + steps(t) + h * heating(t) + daily_shape(half-hour)

with a_P, c, q >= 0 and the steps' positive and negative parts >= 0 under an L1 cost. The
generator coefficient `b_G` has a free sign, because the sign convention of the generator series is
one of the things the audit checks; `a_G = -b_G` is reported, positive when a generator lowers the
reading.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch
from e00124_common import (
    CUTS,
    HEATING_SCALE_C,
    HEATING_THRESHOLD_C,
    HELD_OUT,
    WORK_DIR,
    BspSeries,
    load_bsps,
    load_cache,
    sum_all,
    to_64,
)
from studies.dp_features import FeatureCache
from studies.dp_gam import solve_bounded_quadratic

N_BOOTSTRAP: Final[int] = 200
STEP_PENALTIES: Final[tuple[float, ...]] = (10.0, 100.0, 1000.0)
SHAPE_RIDGE: Final[float] = 1.0
KNOT_DAYS: Final[float] = 30.4375
ROWS_PER_DAY: Final[int] = 48
BLOCK_ROWS: Final[dict[str, int]] = {
    "6 h": 12,
    "24 h": 48,
    "72 h": 144,
    "30 d": 1440,
    "90 d": 4320,
}
WINTER_MONTHS: Final[tuple[int, ...]] = (11, 12, 1, 2, 3)
COLD_SNAP_QUANTILE: Final[float] = 0.05
"""A cold-snap day is a winter day whose mean temperature is in the coldest 5% of the winter days
before the tuning cut. Fixed here, before any residual is read."""
PRIVATE: Final[str] = "PRIVATE: "
"""Marks a report line that stays in the private copy: counts that would reveal the grouping."""
MIN_BLOCKS: Final[int] = 8
"""Fewer complete blocks than this and a block-mean sd is reported as 'cannot tell'."""
MIN_COVERAGE: Final[float] = 0.8
"""A block counts only if at least this share of its rows hold a residual."""
SEEDS: Final[int] = 100
SIZES: Final[tuple[float, ...]] = (0.10, 0.25, 0.40)
PLACEBO_WINDOWS: Final[int] = 1000
DETECTION_SIGMAS: Final[float] = 3.0


@dataclass(frozen=True)
class Design:
    """The columns of one BSP's closure fit on the cache grid."""

    names: tuple[str, ...]
    matrix: np.ndarray
    constrained: np.ndarray
    cost: np.ndarray
    ridge: np.ndarray
    p_column: int


MIN_SERIES_COVERAGE: Final[float] = 0.9
"""A series enters a fit only if it holds a reading in this share of the rows where the BSP does
(set after a first look at the coverage and before any residual was read). A series that fails the
rule stays inside the remainder, and the report counts it."""


@dataclass(frozen=True)
class Sums:
    """The sums of the series that enter one fit, and how many of each were left out."""

    primaries_mw: np.ndarray
    primaries_mva: np.ndarray
    generators: np.ndarray
    gc: np.ndarray
    left_out: dict[str, tuple[int, int]]
    kept_mw: list[np.ndarray]


def select_series(*, cache: FeatureCache, bsp: BspSeries, cut: np.datetime64) -> Sums:
    """Sum the series with enough history before the cut; count the ones left out."""
    n = len(cache.end_times)
    window = np.isfinite(bsp.reading) & (cache.end_times <= cut)
    left_out: dict[str, tuple[int, int]] = {}

    def keep(name: str, series: list[np.ndarray]) -> list[np.ndarray]:
        kept = [
            s
            for s in series
            if window.sum() and np.isfinite(s[window]).mean() >= MIN_SERIES_COVERAGE
        ]
        left_out[name] = (len(series) - len(kept), len(series))
        return kept

    kept_mw = keep("MW Primaries", bsp.mw_list)
    return Sums(
        kept_mw=kept_mw,
        primaries_mw=sum_all(arrays=kept_mw, n=n),
        primaries_mva=sum_all(arrays=keep("MVA Primaries", bsp.mva_list), n=n),
        generators=sum_all(arrays=keep("generation series", bsp.generator_list), n=n),
        gc=sum_all(arrays=bsp.gc_list, n=n),
        left_out=left_out,
    )


def build_design(
    *, cache: FeatureCache, bsp: BspSeries, sums: Sums, cut_day: float, first_day: float
) -> Design:
    """Lay out and build the design matrix over the whole grid (rows are masked later)."""
    n = len(cache.end_times)
    days = cache.end_times.astype("datetime64[s]").astype(np.int64) / 86400.0
    n_knots = int(np.ceil((cut_day - first_day) / KNOT_DAYS)) + 1
    knots = cut_day - KNOT_DAYS * np.arange(n_knots)[::-1]
    after = (days[:, None] >= knots[None, 1:-1]).astype(np.float64)
    hinge = np.clip(HEATING_THRESHOLD_C - bsp.temperature_c, 0.0, None) / HEATING_SCALE_C
    shape = np.zeros((n, ROWS_PER_DAY))
    shape[np.arange(n), cache.calendar.half_hour] = 1.0
    columns = [
        sums.primaries_mw[:, None],
        sums.generators[:, None],
        sums.primaries_mva[:, None],
        -bsp.pv_shape[:, None],
        np.ones((n, 1)),
        hinge[:, None],
        after,
        -after,
        shape,
    ]
    sizes = [c.shape[1] for c in columns]
    names = (
        "a_P",
        "b_G",
        "c",
        "q",
        "level",
        "h",
        *(f"up{i}" for i in range(after.shape[1])),
        *(f"down{i}" for i in range(after.shape[1])),
        *(f"shape{i}" for i in range(ROWS_PER_DAY)),
    )
    constrained = np.zeros(sum(sizes), dtype=bool)
    constrained[[0, 2, 3]] = True
    constrained[6 : 6 + 2 * after.shape[1]] = True
    cost = np.zeros(sum(sizes))
    cost[6 : 6 + 2 * after.shape[1]] = 1.0
    ridge = np.full(sum(sizes), 1e-6)
    ridge[6 + 2 * after.shape[1] :] = SHAPE_RIDGE
    return Design(
        names=names,
        matrix=np.concatenate(columns, axis=1),
        constrained=constrained,
        cost=cost,
        ridge=ridge,
        p_column=0,
    )


def usable_rows(
    *, cache: FeatureCache, bsp: BspSeries, design: Design, cut: np.datetime64
) -> np.ndarray:
    """Return the grid rows before the cut with a finite reading and finite columns."""
    keep = (
        np.isfinite(bsp.reading) & np.isfinite(design.matrix).all(axis=1) & (cache.end_times <= cut)
    )
    return np.flatnonzero(keep)


def fit(
    *,
    x: np.ndarray,
    y: np.ndarray,
    design: Design,
    step_penalty: float,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Solve the penalised, sign-constrained least squares for the rows `x`, `y`."""
    w = np.ones(len(y)) if weights is None else weights
    gram = (x * w[:, None]).T @ x
    rhs = (x * w[:, None]).T @ y
    return solve_gram(gram=gram, rhs=rhs, design=design, step_penalty=step_penalty, n=len(y))


def solve_gram(
    *, gram: np.ndarray, rhs: np.ndarray, design: Design, step_penalty: float, n: int
) -> np.ndarray:
    """Add the ridge and the L1 cost to a Gram system and solve it."""
    cost = design.cost * step_penalty * n / 17532.0
    matrix = gram + np.diag(design.ridge)
    theta, _ = solve_bounded_quadratic(
        matrix=torch.as_tensor(matrix)[None],
        rhs=torch.as_tensor(rhs - 0.5 * cost)[None],
        constrained=torch.as_tensor(design.constrained),
    )
    return theta[0].numpy()


def tune_step_penalty(*, x: np.ndarray, y: np.ndarray, design: Design) -> float:
    """Pick the step penalty with the lowest 4-fold contiguous-block residual variance."""
    folds = np.array_split(np.arange(len(y)), 4)
    best, best_score = STEP_PENALTIES[0], np.inf
    for penalty in STEP_PENALTIES:
        score = 0.0
        for held in folds:
            train = np.setdiff1d(np.arange(len(y)), held)
            theta = fit(x=x[train], y=y[train], design=design, step_penalty=penalty)
            score += float(np.mean((y[held] - x[held] @ theta) ** 2))
        if score < best_score:
            best, best_score = penalty, score
    return best


def block_means(*, residual: np.ndarray, block: int) -> np.ndarray:
    """Return the means of complete blocks with at least `MIN_COVERAGE` of their rows present."""
    n = len(residual) // block
    if n == 0:
        return np.zeros(0)
    cells = residual[: n * block].reshape(n, block)
    present = np.isfinite(cells)
    keep = present.mean(axis=1) >= MIN_COVERAGE
    return np.where(present, cells, 0.0).sum(axis=1)[keep] / present.sum(axis=1)[keep]


def block_sd(*, residual: np.ndarray, block: int) -> float | None:
    """Return the standard deviation of the block means, or `None` for 'cannot tell'."""
    means = block_means(residual=residual, block=block)
    return float(np.std(means, ddof=1)) if len(means) >= MIN_BLOCKS else None


def robust_scale(*, values: np.ndarray) -> float:
    """Return 1.4826 times the median absolute deviation from the median."""
    finite = values[np.isfinite(values)]
    return float(1.4826 * np.median(np.abs(finite - np.median(finite)))) if len(finite) else np.nan


def fmt(value: float | None, digits: int = 2) -> str:
    """Format a number, or 'cannot tell' for `None` or NaN."""
    return "cannot tell" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def percent(share: float | None) -> str:
    """Format a share as a percentage with one decimal, or 'cannot tell'."""
    return fmt(None if share is None else 100 * share, 1)


def residual_on_grid(
    *,
    cache: FeatureCache,
    bsp: BspSeries,
    design: Design,
    theta: np.ndarray,
    start: np.datetime64,
    stop: np.datetime64,
) -> np.ndarray:
    """Return the residual on the grid rows in (start, stop].

    The residual is `NaN` where a column or the reading is missing.
    """
    mask = (cache.end_times > start) & (cache.end_times <= stop)
    out = np.full(int(mask.sum()), np.nan)
    rows = np.flatnonzero(mask)
    ok = np.isfinite(bsp.reading[rows]) & np.isfinite(design.matrix[rows]).all(axis=1)
    out[ok] = bsp.reading[rows][ok] - design.matrix[rows][ok] @ theta
    return out


def band_and_month_table(
    *, cache: FeatureCache, rows: np.ndarray, residual: np.ndarray, lines: list[str], title: str
) -> None:
    """Print the median residual by hour band and by calendar month, and the midday median."""
    from studies.dp_calendar import BAND_NAMES

    band = cache.calendar.band[rows]
    month = cache.end_times[rows].astype("datetime64[M]").astype(int) % 12 + 1
    half_hour = cache.calendar.half_hour[rows]
    ok = np.isfinite(residual)
    by_band = ", ".join(
        f"{name} {np.median(residual[ok & (band == i)]):+.2f}"
        for i, name in enumerate(BAND_NAMES)
        if (ok & (band == i)).sum() > 100
    )
    by_month = ", ".join(
        f"{m} {np.median(residual[ok & (month == m)]):+.2f}"
        for m in range(1, 13)
        if (ok & (month == m)).sum() > 100
    )
    midday = ok & (half_hour >= 20) & (half_hour < 28)
    lines.append(f"- {title}: median residual (MW) by hour band: {by_band}")
    lines.append(f"- {title}: median residual (MW) by calendar month: {by_month}")
    lines.append(
        f"- {title}: median residual at local 10:00 to 14:00: {np.median(residual[midday]):+.2f} MW"
    )


def cold_snap_threshold(*, cache: FeatureCache, bsp: BspSeries) -> float:
    """Return the cold-snap temperature: the 5% quantile of winter-day means before the tuning cut.

    A day is a cold-snap day, in any winter, when its mean temperature is at or below this value.
    The threshold is fixed from rows before the tuning cut and before any residual is read.
    """
    cut = to_64(moment=CUTS["tuning"])
    days = cache.end_times.astype("datetime64[D]")
    month = cache.end_times.astype("datetime64[M]").astype(int) % 12 + 1
    mask = (cache.end_times <= cut) & np.isin(month, WINTER_MONTHS)
    unique = np.unique(days[mask])
    means = np.array([np.nanmean(bsp.temperature_c[mask & (days == d)]) for d in unique])
    return float(np.quantile(means[np.isfinite(means)], COLD_SNAP_QUANTILE))


def winter_table(
    *,
    cache: FeatureCache,
    bsp: BspSeries,
    rows: np.ndarray,
    residual: np.ndarray,
    lines: list[str],
    title: str,
) -> None:
    """Print the residual per winter (November to March), and on cold-snap days."""
    ends = cache.end_times[rows]
    month = ends.astype("datetime64[M]").astype(int) % 12 + 1
    year = ends.astype("datetime64[Y]").astype(int) + 1970
    winter_year = np.where(month >= 11, year + 1, year)
    days = ends.astype("datetime64[D]")
    threshold = cold_snap_threshold(cache=cache, bsp=bsp)
    all_days = cache.end_times.astype("datetime64[D]")
    day_mean = {d: np.nanmean(bsp.temperature_c[all_days == d]) for d in np.unique(days)}
    snaps = {d for d, m in day_mean.items() if m <= threshold}
    is_snap = (
        np.isin(days, np.array(sorted(snaps), dtype="datetime64[D]"))
        if snaps
        else np.zeros(len(rows), bool)
    )
    usable_winters = 0
    for wy in sorted(set(winter_year[np.isin(month, WINTER_MONTHS)])):
        mask = (winter_year == wy) & np.isin(month, WINTER_MONTHS) & np.isfinite(residual)
        n_days = len(np.unique(days[mask]))
        if n_days < 80:
            lines.append(
                f"- {title}, winter ending {wy}: cannot tell "
                f"({n_days} days with a residual, 80 needed)"
            )
            continue
        usable_winters += 1
        sd_24h = block_sd(residual=np.where(mask, residual, np.nan), block=48)
        snap_mask = mask & is_snap
        snap_text = (
            f"cold-snap days ({len(np.unique(days[snap_mask]))}): "
            f"median {np.median(residual[snap_mask]):+.2f} MW"
            if len(np.unique(days[snap_mask])) >= 3
            else "cold-snap days: cannot tell (fewer than 3)"
        )
        lines.append(
            f"- {title}, winter ending {wy}: {n_days} days; "
            f"median {np.median(residual[mask]):+.2f} MW, "
            f"robust scale {robust_scale(values=residual[mask]):.2f} MW, "
            f"24 h block-mean sd {fmt(sd_24h)} MW; {snap_text}"
        )
    lines.append(
        f"- {title}: {usable_winters} independent winters with at least 80 days of residuals"
    )


def prefix_fits(
    *, x: np.ndarray, y: np.ndarray, months: np.ndarray, design: Design, penalty: float
) -> dict[np.datetime64, np.ndarray]:
    """Fit on the rows before each month's start; return the coefficients by month."""
    gram = np.zeros((x.shape[1], x.shape[1]))
    rhs = np.zeros(x.shape[1])
    seen = 0
    fits = {}
    for month in np.unique(months):
        if seen >= 4 * ROWS_PER_DAY * 30:
            fits[month] = solve_gram(
                gram=gram, rhs=rhs, design=design, step_penalty=penalty, n=seen
            )
        in_month = months == month
        gram = gram + x[in_month].T @ x[in_month]
        rhs = rhs + x[in_month].T @ y[in_month]
        seen += int(in_month.sum())
    return fits


def false_alarm_rates(
    *, residual: np.ndarray, thresholds: dict[int, float | None], rng: np.random.Generator
) -> dict[int, float | None]:
    """Return the share of placebo windows of `residual` whose mean beats each threshold."""
    finite = np.flatnonzero(np.isfinite(residual))
    out: dict[int, float | None] = {}
    for length, threshold in thresholds.items():
        if threshold is None or len(finite) < 2 * length:
            out[length] = None
            continue
        starts = rng.integers(finite[0], finite[-1] - length, size=PLACEBO_WINDOWS)
        means = np.array(
            [
                np.nanmean(residual[s : s + length])
                if np.isfinite(residual[s : s + length]).mean() >= MIN_COVERAGE
                else np.nan
                for s in starts
            ]
        )
        out[length] = float(np.mean(np.abs(means[np.isfinite(means)]) > threshold))
    return out


def window_detections(
    *,
    xw: np.ndarray,
    y_window: np.ndarray,
    delta: np.ndarray,
    gram0: np.ndarray,
    rhs0: np.ndarray,
    pre_fit: np.ndarray | None,
    theta: np.ndarray,
    design: Design,
    penalty: float,
    n: int,
    threshold: float,
) -> dict[str, bool]:
    """Say which fits show an injected error `delta` in the window rows `xw`, `y_window`.

    The injection adds `delta` to the first design column over the window, which changes the Gram
    system by a rank-two update.
    """
    v = xw.T @ delta
    gram = gram0.copy()
    gram[0, :] += v
    gram[:, 0] += v
    gram[0, 0] += float(delta @ delta)
    rhs = rhs0.copy()
    rhs[0] += float(delta @ y_window)
    refit = solve_gram(gram=gram, rhs=rhs, design=design, step_penalty=penalty, n=n)
    xw_new = xw.copy()
    xw_new[:, 0] += delta
    fits = {"refit": refit, "in-sample clean fit": theta}
    if pre_fit is not None:
        fits["pre-injection fit"] = pre_fit
    return {
        name: abs(float(np.mean(y_window - xw_new @ fitted))) > threshold
        for name, fitted in fits.items()
    }


def injection_audit(
    *,
    cache: FeatureCache,
    bsp: BspSeries,
    sums: Sums,
    design: Design,
    index: np.ndarray,
    theta: np.ndarray,
    penalty: float,
    cut: np.datetime64,
    out_of_sample: np.ndarray,
    lines: list[str],
) -> dict[str, float]:
    """Inject metering errors into the included MW Primaries; count detections.

    A window is detected if the absolute mean of the residual over it exceeds 3 block-mean standard
    deviations of the clean residual for that window length. Three fits are read: the refit, which
    includes the injected series (the reading the decision rule uses); the clean fit trained on all
    pre-cut rows, including the uninjected rows of the window itself (in-sample, so it flatters
    detection); and a fit trained on the rows before the window's month only. Each scale error
    is injected with both signs, in the size of the stated share of the Primary's median load, and
    the lower detection rate of the two signs is reported. The false-alarm rate comes from placebo
    windows of the in-sample residual and of the frozen fit's residual over the year after the cut.
    """
    x, y = design.matrix[index], bsp.reading[index]
    gram0, rhs0 = x.T @ x, x.T @ y
    grid_clean = np.full(len(cache.end_times), np.nan)
    grid_clean[index] = y - x @ theta
    last_year = index[cache.end_times[index] > cut - np.timedelta64(365, "D")]
    kept = [id(s) for s in (sums.kept_mw)]
    series = [s for s in bsp.mw_list if id(s) in kept]
    cells = [("stuck 24 h", 48, "stuck", 0.0)] + [
        (f"scale {name} {int(100 * s)}%", rows, "scale", s)
        for name, rows in (("30 d", 1440), ("90 d", 4320))
        for s in SIZES
    ]
    rng = np.random.default_rng(1)
    lengths = (48, 1440, 4320)
    sds = {L: block_sd(residual=grid_clean[index[0] : index[-1] + 1], block=L) for L in lengths}
    thresholds = {L: None if v is None else DETECTION_SIGMAS * v for L, v in sds.items()}
    placebo = false_alarm_rates(residual=grid_clean, thresholds=thresholds, rng=rng)
    placebo_out = false_alarm_rates(residual=out_of_sample, thresholds=thresholds, rng=rng)
    months = cache.end_times[index].astype("datetime64[M]")
    pre = prefix_fits(x=x, y=y, months=months, design=design, penalty=penalty)
    position = {g: i for i, g in enumerate(index)}
    results: dict[str, float] = {}
    for label, length, kind, size in cells:
        threshold = thresholds[length]
        if threshold is None:
            lines.append(f"- injected {label}: cannot tell (too few blocks for its threshold)")
            continue
        hits = {"refit": [0, 0], "in-sample clean fit": [0, 0], "pre-injection fit": [0, 0]}
        counts = {"refit": [0, 0], "in-sample clean fit": [0, 0], "pre-injection fit": [0, 0]}
        signs = (1.0, -1.0) if kind == "scale" else (1.0,)
        for s in series:
            median_load = float(np.nanmedian(s[index]))
            for _ in range(SEEDS):
                for _try in range(20):
                    start = int(rng.choice(last_year[: max(len(last_year) - length, 1)]))
                    window = np.arange(start, start + length)
                    present = np.array([g in position for g in window])
                    if present.mean() >= MIN_COVERAGE and np.isfinite(s[window[present]]).all():
                        break
                else:
                    continue
                rows = window[present]
                local = np.array([position[g] for g in rows])
                xw = x[local]
                month = cache.end_times[start].astype("datetime64[M]")
                for sign_number, sign in enumerate(signs):
                    delta = (
                        sign * size * median_load * np.ones(len(rows))
                        if kind == "scale"
                        else s[start - 1 if np.isfinite(s[start - 1]) else rows[0]] - s[rows]
                    )
                    detected = window_detections(
                        xw=xw,
                        y_window=y[local],
                        delta=delta,
                        gram0=gram0,
                        rhs0=rhs0,
                        pre_fit=pre.get(month),
                        theta=theta,
                        design=design,
                        penalty=penalty,
                        n=len(y),
                        threshold=threshold,
                    )
                    for name, hit in detected.items():
                        hits[name][sign_number] += hit
                        counts[name][sign_number] += 1
        rates = {
            k: min(h[s] / counts[k][s] for s in range(len(signs)))
            for k, h in hits.items()
            if min(counts[k][: len(signs)]) > 0
        }
        if "refit" not in rates:
            lines.append(f"- injected {label}: cannot tell (no window with enough complete rows)")
            continue
        results.update({f"{label} {k}": v for k, v in rates.items()})
        in_sample = percent(placebo[length])
        out_sample = percent(placebo_out[length])
        lines.append(
            f"- injected {label}: detected "
            + ", ".join(f"{100 * v:.0f}% with the {k}" for k, v in rates.items())
            + f" ({counts['refit'][0]} injections per sign; false-alarm rate {in_sample}% "
            f"in-sample, {out_sample}% in the year after the cut)"
        )
    return results


def bootstrap_interval(
    *,
    cache: FeatureCache,
    index: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    design: Design,
    penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the 10th and 90th percentiles of four coefficients from a month-block bootstrap."""
    months = cache.end_times[index].astype("datetime64[M]")
    unique = np.unique(months)
    month_gram = {
        m: (x[months == m].T @ x[months == m], x[months == m].T @ y[months == m]) for m in unique
    }
    generator = np.random.default_rng(0)
    draws = []
    for _ in range(N_BOOTSTRAP):
        picked = generator.choice(unique, size=len(unique))
        gram = sum(month_gram[m][0] for m in picked)
        rhs = sum(month_gram[m][1] for m in picked)
        draws.append(
            solve_gram(gram=gram, rhs=rhs, design=design, step_penalty=penalty, n=len(y))[:4]
        )
    low, high = np.percentile(np.array(draws), [10, 90], axis=0)
    return low, high


def scored_tables(
    *,
    cache: FeatureCache,
    bsp: BspSeries,
    design: Design,
    theta: np.ndarray,
    cut64: np.datetime64,
    lines: list[str],
) -> None:
    """Print the hour-band, month, and winter tables for the training rows and the scored year."""
    rows_all = np.arange(len(cache.end_times))
    full = residual_on_grid(
        cache=cache,
        bsp=bsp,
        design=design,
        theta=theta,
        start=np.datetime64("1900-01-01", "us"),
        stop=to_64(moment=HELD_OUT),
    )
    inside = (cache.end_times <= cut64) & np.isfinite(full)
    band_and_month_table(
        cache=cache,
        rows=rows_all[inside],
        residual=full[inside],
        lines=lines,
        title="in-sample",
    )
    winter_table(
        cache=cache,
        bsp=bsp,
        rows=rows_all[inside],
        residual=full[inside],
        lines=lines,
        title="in-sample",
    )
    after_cut = (cache.end_times > cut64) & np.isfinite(full)
    if after_cut.sum() > ROWS_PER_DAY * 60:
        band_and_month_table(
            cache=cache,
            rows=rows_all[after_cut],
            residual=full[after_cut],
            lines=lines,
            title="scored year",
        )
        winter_table(
            cache=cache,
            bsp=bsp,
            rows=rows_all[after_cut],
            residual=full[after_cut],
            lines=lines,
            title="scored year",
        )


def report_cut(
    *, cache: FeatureCache, bsp: BspSeries, cut_name: str, lines: list[str]
) -> tuple[dict[str, float] | None, dict[str, float]]:
    """Fit one BSP at one cut and print its closure residual; return the facts and injections."""
    cut = CUTS[cut_name]
    cap = bsp.effective_capacity_mw
    injections: dict[str, float] = {}
    cut64 = to_64(moment=cut)
    first_day = float(
        cache.end_times[np.flatnonzero(np.isfinite(bsp.reading) & (cache.end_times <= cut64))[0]]
        .astype("datetime64[s]")
        .astype(np.int64)
        / 86400.0
    )
    cut_day = float(cut64.astype("datetime64[s]").astype(np.int64) / 86400.0)
    sums = select_series(cache=cache, bsp=bsp, cut=cut64)
    design = build_design(cache=cache, bsp=bsp, sums=sums, cut_day=cut_day, first_day=first_day)
    index = usable_rows(cache=cache, bsp=bsp, design=design, cut=cut64)
    x, y = design.matrix[index], bsp.reading[index]
    n_years = len(index) / (ROWS_PER_DAY * 365.25)
    lines.append("")
    lines.append(
        f"#### Cut {cut_name} ({cut.date().isoformat()}): "
        f"{len(index)} training rows ({n_years:.1f} years)"
    )
    lines.append(
        PRIVATE
        + "- left out for too little history (their power stays in the remainder): "
        + ", ".join(f"{a} of {b} {k}" for k, (a, b) in sums.left_out.items())
    )
    if n_years < 1.0:
        lines.append("- cannot tell: fewer than one year of complete rows before the cut")
        return None, {}
    penalty = tune_step_penalty(x=x, y=y, design=design)
    theta = fit(x=x, y=y, design=design, step_penalty=penalty)
    low, high = bootstrap_interval(
        cache=cache, index=index, x=x, y=y, design=design, penalty=penalty
    )
    residual_in = y - x @ theta
    scale = robust_scale(values=residual_in)
    lines.append(
        f"- step penalty {penalty:g}; a_P {theta[0]:.2f} ({low[0]:.2f} to {high[0]:.2f}); "
        f"a_G {-theta[1]:.2f} ({-high[1]:.2f} to {-low[1]:.2f}); "
        f"c {theta[2]:.2f} ({low[2]:.2f} to {high[2]:.2f}); "
        f"unmetered PV capacity q {theta[3]:.2f} MW ({low[3]:.2f} to {high[3]:.2f}); "
        f"heating h {theta[5]:.2f}"
    )
    lines.append(
        f"- in-sample residual: robust half-hourly scale {scale:.2f} MW "
        f"= {100 * scale / cap:.1f} points of effective capacity"
    )
    sds = {
        name: block_sd(
            residual=residual_on_grid(
                cache=cache,
                bsp=bsp,
                design=design,
                theta=theta,
                start=cache.end_times[index[0]] - np.timedelta64(1, "us"),
                stop=cut64,
            ),
            block=b,
        )
        for name, b in BLOCK_ROWS.items()
    }
    lines.append(
        "- in-sample standard deviation of the residual's block means (MW): "
        + ", ".join(f"{k} {fmt(v)}" for k, v in sds.items())
    )
    if cut_name == "scored":
        scored_tables(cache=cache, bsp=bsp, design=design, theta=theta, cut64=cut64, lines=lines)
    facts = {
        "scale": scale,
        "a_P": theta[0],
        "a_G": -theta[1],
        "c": theta[2],
        "n_years": n_years,
    }
    # Out-of-sample year after the cut, coefficients frozen, steps held at their last value.
    stop = min(cut.replace(year=cut.year + 1), HELD_OUT)
    out = residual_on_grid(
        cache=cache, bsp=bsp, design=design, theta=theta, start=cut64, stop=to_64(moment=stop)
    )
    out_grid = np.full(len(cache.end_times), np.nan)
    out_grid[(cache.end_times > cut64) & (cache.end_times <= to_64(moment=stop))] = out
    if cut_name in ("tuning", "replicate") and n_years >= 1.5:
        inj = injection_audit(
            cache=cache,
            bsp=bsp,
            sums=sums,
            design=design,
            index=index,
            theta=theta,
            penalty=penalty,
            cut=cut64,
            out_of_sample=out_grid,
            lines=lines,
        )
        injections = inj
    valid = out[np.isfinite(out)]
    if len(valid) >= ROWS_PER_DAY * 90:
        lines.append(
            f"- scored-year residual (frozen fit): median {np.median(valid):.2f} MW, "
            f"robust scale {robust_scale(values=out):.2f} MW "
            f"= {100 * robust_scale(values=out) / cap:.1f} points; block-mean sd (MW): "
            + ", ".join(
                f"{k} {fmt(block_sd(residual=out, block=b))}" for k, b in BLOCK_ROWS.items()
            )
        )
        facts["out_scale"] = robust_scale(values=out)
    else:
        lines.append("- scored-year residual: cannot tell (fewer than 90 days of complete rows)")
    return facts, injections


def report_bsp(
    *, cache: FeatureCache, bsp: BspSeries, lines: list[str]
) -> dict[str, dict[str, float]]:
    """Fit one BSP at the three cuts and print its closure residual; return the fitted facts."""
    cap = bsp.effective_capacity_mw
    median_load = float(np.nanmedian(bsp.reading[cache.end_times <= to_64(moment=CUTS["scored"])]))
    lines.append(f"### Bulk supply point {bsp.index + 1}")
    lines.append("")
    lines.append(f"- effective capacity {cap:.1f} MW; median load {median_load:.1f} MW")
    lines.append(
        PRIVATE
        + f"- {bsp.n_primaries} metered Primaries, {bsp.n_generators} metered generation series "
        "(G-c included)"
    )
    facts: dict[str, dict[str, float]] = {}
    injections: dict[str, dict[str, float]] = {}
    for cut_name in CUTS:
        cut_facts, cut_injections = report_cut(cache=cache, bsp=bsp, cut_name=cut_name, lines=lines)
        if cut_facts is not None:
            facts[cut_name] = cut_facts
        injections[cut_name] = cut_injections
    for cut_name, inj in injections.items():
        facts.setdefault(cut_name, {})
        facts[cut_name].update(inj)
    return facts


def main() -> None:
    """Run the audit for both BSPs and write the report."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache()
    bsps = load_bsps(cache=cache)
    lines = ["# Experiment 124, step 0: the BSP closure audit", ""]
    for bsp in bsps:
        report_bsp(cache=cache, bsp=bsp, lines=lines)
    (WORK_DIR / "audit_report.md").write_text(
        "\n".join(line.removeprefix(PRIVATE) for line in lines) + "\n"
    )
    text = "\n".join(line for line in lines if not line.startswith(PRIVATE)) + "\n"
    (
        Path(__file__).parents[2]
        / "docs"
        / "ml-experiments"
        / Path(__file__).parent.name
        / "audit_report.md"
    ).write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
