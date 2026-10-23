"""Experiment 00233: known-answer capacity-loss detection and the real-series description.

Reads `work/daily.parquet` (made by `e00233_common.daily_table`), builds the relative ratio of
observed to expected power on above-rated half-hours in a trailing window, calibrates the alarm
threshold on 2021-09-01 to 2024-06-30, injects capacity losses into the observed power from random
start dates, runs the drift nulls, and writes `work/detect.json` and the tables the figures read.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00233-*/e00233_detect.py
"""

import json
import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from dataclasses import dataclass  # noqa: E402
from datetime import date  # noqa: E402
from typing import Final  # noqa: E402

import e00233_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

FIRST_DAY: Final[date] = date(2019, 9, 1)
LAST_DAY: Final[date] = date(2026, 6, 30)
REFERENCE_END: Final[date] = date(2024, 7, 1)
CALIBRATION_START: Final[date] = date(2021, 9, 1)
HOLDOUT_SPLIT: Final[date] = date(2023, 7, 1)
START_FIRST: Final[date] = date(2024, 7, 1)
START_LAST: Final[date] = date(2025, 12, 31)
FIT_END: Final[date] = date(2025, 6, 17)
WINDOWS: Final[tuple[int, ...]] = (30, 60, 90)
PRIMARY_WINDOW: Final[int] = 60
VARIANTS: Final[tuple[str, ...]] = ("core", "busy", "const")
"""`core`: above-rated half-hours with the core's expected power; `busy`: half-hours at 20% or more
of the scale; `const`: the `core` half-hours with the farm's scale as the expected power."""
LOSSES: Final[tuple[float, ...]] = (0.10, 0.25, 0.50)
N_STARTS: Final[int] = 30
MAX_DELAY_DAYS: Final[int] = 180
DETECT_WITHIN_DAYS: Final[int] = 120
MIN_BIG_DAYS: Final[int] = 6
BIG_DAY_HALF_HOURS: Final[int] = 4
DRIFT_RANGES: Final[tuple[float, ...]] = (0.09, 0.15, 0.25)
LINEAR_DRIFT_PER_YEAR: Final[float] = 0.10
BOOTSTRAP_RESAMPLES: Final[int] = 5_000
SEED: Final[int] = 233


def calendar() -> list[date]:
    """Return every day from `FIRST_DAY` to `LAST_DAY`."""
    return pl.date_range(FIRST_DAY, LAST_DAY, interval="1d", eager=True).to_list()


def day_index(*, days: list[date], day: date) -> int:
    """Return the position of `day` in `days`."""
    return (day - days[0]).days


def daily_arrays(*, table: pl.DataFrame, farm: str, scale: float, days: list[date]) -> dict:
    """Return the per-day arrays of a farm: observed and expected sums per variant, and counts."""
    part = (
        pl.DataFrame({"day": days})
        .join(table.filter(pl.col("site") == farm), on="day", how="left")
        .fill_null(0)
    )
    n_above = part["n_above"].to_numpy().astype(float)
    return {
        "obs": {
            "core": part["obs_above"].to_numpy().astype(float),
            "busy": part["obs_busy"].to_numpy().astype(float),
            "const": part["obs_above"].to_numpy().astype(float),
        },
        "exp": {
            "core": part["exp_above"].to_numpy().astype(float),
            "busy": part["exp_busy"].to_numpy().astype(float),
            "const": n_above * scale,
        },
        "big": {
            "core": (n_above >= BIG_DAY_HALF_HOURS).astype(float),
            "busy": (part["n_busy"].to_numpy() >= BIG_DAY_HALF_HOURS).astype(float),
            "const": (n_above >= BIG_DAY_HALF_HOURS).astype(float),
        },
    }


def trailing_sum(*, values: np.ndarray, window: int) -> np.ndarray:
    """Return the sum over the `window` days ending at each day (fewer at the start)."""
    cumulative = np.cumsum(values)
    shifted = np.concatenate([np.zeros(window), cumulative[:-window]])
    return cumulative - shifted


def relative_ratio(
    *, obs: np.ndarray, expected: np.ndarray, big: np.ndarray, window: int, reference: float
) -> np.ndarray:
    """Return the relative ratio per day, NaN where the window has too few big days."""
    sum_obs = trailing_sum(values=obs, window=window)
    sum_exp = trailing_sum(values=expected, window=window)
    defined = trailing_sum(values=big, window=window) >= MIN_BIG_DAYS
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(defined & (sum_exp > 0), sum_obs / sum_exp / reference, np.nan)


def reference_level(*, obs: np.ndarray, expected: np.ndarray, days: list[date]) -> float:
    """Return the sum of observed over expected power before `REFERENCE_END`."""
    end = day_index(days=days, day=REFERENCE_END)
    return float(obs[:end].sum() / expected[:end].sum())


def drift_factor(
    *, days: list[date], drift_range: float | None, linear: float | None
) -> np.ndarray:
    """Return the per-day multiplier of a drift that starts at 1 on `START_FIRST`."""
    since = np.array([(d - START_FIRST).days for d in days], dtype=float)
    factor = np.ones(len(days))
    after = since >= 0
    if drift_range is not None:
        factor[after] = 1 - 0.5 * drift_range * (1 - np.cos(2 * np.pi * since[after] / 365.25))
    if linear is not None:
        factor[after] = 1 - linear * since[after] / 365.25
    return factor


def first_delay(*, alarm: np.ndarray, quiet: np.ndarray, start: int) -> float:
    """Return days from `start` to the first alarm that the un-injected run lacks, else infinity."""
    stop = min(start + MAX_DELAY_DAYS, alarm.size)
    hits = np.flatnonzero(alarm[start:stop] & ~quiet[start:stop])
    return float(hits[0]) if hits.size else float("inf")


def bootstrap_months(
    *, delays: np.ndarray, month_labels: np.ndarray, rng: np.random.Generator
) -> dict:
    """Return the median delay and the share detected within `DETECT_WITHIN_DAYS`, with intervals.

    Resamples whole calendar months of start dates, because starts in one month overlap in time.
    """
    months = np.unique(month_labels)
    groups = [delays[month_labels == m] for m in months]
    medians, shares = [], []
    for _ in range(BOOTSTRAP_RESAMPLES):
        picked = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        medians.append(np.median(picked))
        shares.append((picked <= DETECT_WITHIN_DAYS).mean())
    point_median = float(np.median(delays))
    return {
        "median_delay_days": point_median,
        "median_interval": [float(np.quantile(medians, q)) for q in (0.025, 0.975)],
        "share_detected_within_120": float((delays <= DETECT_WITHIN_DAYS).mean()),
        "share_interval": [float(np.quantile(shares, q)) for q in (0.025, 0.975)],
        "starts": int(delays.size),
    }


@dataclass(frozen=True)
class FarmSeries:
    """One farm and variant's daily sums, reference level, and the alarm threshold."""

    obs: np.ndarray
    expected: np.ndarray
    big: np.ndarray
    reference: float

    def ratio(self, *, window: int, obs: np.ndarray | None = None) -> np.ndarray:
        """Return the relative ratio, optionally for a modified observed series."""
        return relative_ratio(
            obs=self.obs if obs is None else obs,
            expected=self.expected,
            big=self.big,
            window=window,
            reference=self.reference,
        )


def alarms(*, ratio: np.ndarray, threshold: float) -> np.ndarray:
    """Return whether each day's ratio is below `threshold`; days with no statistic never alarm."""
    return np.nan_to_num(ratio, nan=np.inf) < threshold


def injection_summary(
    *,
    series: FarmSeries,
    window: int,
    threshold: float,
    loss: float,
    start_indices: list[int],
    month_labels: np.ndarray,
    after_fit: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Return delay statistics for one loss size, over the start dates with no prior alarm."""
    quiet = alarms(ratio=series.ratio(window=window), threshold=threshold)
    delays = []
    for start in start_indices:
        if quiet[start - 1]:
            delays.append(np.nan)
            continue
        injected = series.obs.copy()
        injected[start:] *= 1 - loss
        alarm = alarms(ratio=series.ratio(window=window, obs=injected), threshold=threshold)
        delays.append(first_delay(alarm=alarm, quiet=quiet, start=start))
    delays_array = np.array(delays)
    keep = ~np.isnan(delays_array)
    summary = bootstrap_months(delays=delays_array[keep], month_labels=month_labels[keep], rng=rng)
    summary["excluded_already_alarming"] = int((~keep).sum())
    for name, mask in (("before_fit_end", ~after_fit), ("after_fit_end", after_fit)):
        sub = delays_array[keep & mask]
        summary[name] = {
            "starts": int(sub.size),
            "median_delay_days": float(np.median(sub)) if sub.size else None,
            "share_detected_within_120": float((sub <= DETECT_WITHIN_DAYS).mean())
            if sub.size
            else None,
        }
    return summary


def drift_summary(
    *, series: FarmSeries, window: int, threshold: float, days: list[date]
) -> tuple[dict, np.ndarray]:
    """Return alarm counts under each drift null (all, and those the un-injected run lacks)."""
    eval_start = day_index(days=days, day=START_FIRST)
    quiet = alarms(ratio=series.ratio(window=window), threshold=threshold)
    drifts = {
        f"range_{r:.2f}": drift_factor(days=days, drift_range=r, linear=None) for r in DRIFT_RANGES
    }
    drifts["linear_10_per_year"] = drift_factor(
        days=days, drift_range=None, linear=LINEAR_DRIFT_PER_YEAR
    )
    out, nine = {}, np.array([])
    for name, factor in drifts.items():
        ratio = series.ratio(window=window, obs=series.obs * factor)
        out[name] = {
            "alarm_days": int(alarms(ratio=ratio, threshold=threshold)[eval_start:].sum()),
            "new_alarm_days": int(
                (alarms(ratio=ratio, threshold=threshold) & ~quiet)[eval_start:].sum()
            ),
            "min_ratio": float(np.nanmin(ratio[eval_start:])),
        }
        if name == "range_0.09":
            nine = ratio
    return out, nine


def main() -> None:
    """Run the known-answer tests and write `work/detect.json` and the figure tables."""
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    path = common.WORK_DIR / "daily.parquet"
    if not path.exists():
        common.daily_table(sites=common.site_table()).write_parquet(path)
    table = pl.read_parquet(path)
    cores = json.loads(common.CORES_PATH.read_text())
    days = calendar()
    rng = np.random.default_rng(SEED)
    span = (START_LAST - START_FIRST).days
    offsets = np.sort(rng.integers(0, span + 1, N_STARTS))
    start_indices = [day_index(days=days, day=START_FIRST) + int(o) for o in offsets]
    start_dates = [days[i] for i in start_indices]
    month_labels = np.array([d.strftime("%Y-%m") for d in start_dates])
    after_fit = np.array([d >= FIT_END for d in start_dates])
    index = {
        name: day_index(days=days, day=day)
        for name, day in (
            ("cal", CALIBRATION_START),
            ("ref", REFERENCE_END),
            ("hold", HOLDOUT_SPLIT),
        )
    }
    report: dict = {"start_dates": [d.isoformat() for d in start_dates], "farms": {}}
    frames = []
    for farm in sorted(cores):
        arrays = daily_arrays(
            table=table, farm=farm, scale=cores[farm]["scale_fraction_of_capacity"], days=days
        )
        farm_report: dict = {}
        for variant in VARIANTS:
            obs, expected = arrays["obs"][variant], arrays["exp"][variant]
            series = FarmSeries(
                obs=obs,
                expected=expected,
                big=arrays["big"][variant],
                reference=reference_level(obs=obs, expected=expected, days=days),
            )
            for window in WINDOWS:
                ratio = series.ratio(window=window)
                threshold = float(np.nanmin(ratio[index["cal"] : index["ref"]]))
                holdout_threshold = float(np.nanmin(ratio[index["cal"] : index["hold"]]))
                test = ratio[index["hold"] : index["ref"]]
                drift, nine = drift_summary(
                    series=series, window=window, threshold=threshold, days=days
                )
                farm_report[f"{variant}/{window}"] = {
                    "reference": series.reference,
                    "threshold": threshold,
                    "holdout_threshold": holdout_threshold,
                    "holdout_alarm_days": int(
                        alarms(ratio=test, threshold=holdout_threshold).sum()
                    ),
                    "holdout_days_defined": int(np.isfinite(test).sum()),
                    "share_days_defined": float(np.isfinite(ratio).mean()),
                    "uninjected_alarm_days_after_2024_07": int(
                        alarms(ratio=ratio, threshold=threshold)[index["ref"] :].sum()
                    ),
                    "injections": {
                        f"{loss:.2f}": injection_summary(
                            series=series,
                            window=window,
                            threshold=threshold,
                            loss=loss,
                            start_indices=start_indices,
                            month_labels=month_labels,
                            after_fit=after_fit,
                            rng=rng,
                        )
                        for loss in LOSSES
                    },
                    "drift": drift,
                }
                if variant == "core" and window == PRIMARY_WINDOW:
                    injected = series.obs.copy()
                    injected[start_indices[len(start_indices) // 2] :] *= 0.75
                    for kind, values in (
                        ("real", ratio),
                        ("drift_9", nine),
                        ("injected_25", series.ratio(window=window, obs=injected)),
                    ):
                        frames.append(
                            pl.DataFrame({"farm": farm, "day": days, "kind": kind, "ratio": values})
                        )
        report["farms"][farm] = farm_report
    pl.concat(frames).write_parquet(common.WORK_DIR / "ratio_series.parquet")
    (common.WORK_DIR / "detect.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
