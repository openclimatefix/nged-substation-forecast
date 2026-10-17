"""Print every number the experiment 72 page quotes, and the verdict the decision rule gives.

    uv run python ml-experiments/00072-*/e00072_report.py

The report is also written to `docs/ml-experiments/00072-*/report.txt`.
"""

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from typing import Final, cast

import numpy as np
import polars as pl
from e00072_detectors import robust_sd
from e00072_inject import METHODS
from e00072_score import OUTPUT_DIR
from e00072_signal import POOL_SITES, clean_baseline, load_record

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00072-p35-step-loss-monitor-for-solar-farms"
)
RESAMPLES: Final[int] = 10_000
BONFERRONI_INTERVAL: Final[tuple[float, float]] = (1.25, 98.75)
"""The 97.5% percentile interval, for two planned contrasts."""
SEED: Final[int] = 72
DETECTION_FLOOR: Final[float] = 0.80
CHANCE_SIZE: Final[float] = 0.0


def as_float(value: object) -> float:
    """Return a Polars aggregate as a float."""
    return float(cast("float", value))


def rate(*, events: pl.DataFrame, method: str, **filters: object) -> tuple[float, int]:
    """Return a monitor's pooled detection rate and the number of losses behind it."""
    selected = events.filter(pl.col("method") == method)
    for column, value in filters.items():
        selected = selected.filter(pl.col(column) == value)
    return (
        (as_float(selected["detected"].mean()), selected.height)
        if selected.height
        else (float("nan"), 0)
    )


def farm_bootstrap(
    *, events: pl.DataFrame, method_a: str, method_b: str, size: float
) -> tuple[float, tuple[float, float], dict[str, float]]:
    """Return a contrast of detection rates, its farm-resampling interval, and per-farm values.

    The contrast is `method_a` minus `method_b` on losses of `size` lasting 30 and 90 days that
    start from April to September. The interval resamples the five farms with replacement and keeps
    each farm's draws together.
    """
    base = events.filter(
        (pl.col("size") == size) & pl.col("days").is_in([30, 90]) & (pl.col("season") == "apr_sep")
    )
    per_farm = {}
    counts = {}
    for site in POOL_SITES:
        for method in (method_a, method_b):
            sub = base.filter((pl.col("site") == site) & (pl.col("method") == method))
            per_farm[(site, method)] = float(sub["detected"].sum())
            counts[site] = sub.height

    def pooled(sites: list[str]) -> float:
        hits_a = sum(per_farm[(s, method_a)] for s in sites)
        hits_b = sum(per_farm[(s, method_b)] for s in sites)
        total = sum(counts[s] for s in sites)
        return (hits_a - hits_b) / total

    rng = np.random.default_rng(SEED)
    draws = [
        pooled([POOL_SITES[i] for i in rng.integers(0, len(POOL_SITES), len(POOL_SITES))])
        for _ in range(RESAMPLES)
    ]
    low, high = np.percentile(draws, BONFERRONI_INTERVAL)
    farm_values = {
        site: (per_farm[(site, method_a)] - per_farm[(site, method_b)]) / counts[site]
        for site in POOL_SITES
    }
    return pooled(list(POOL_SITES)), (float(low), float(high)), farm_values


def report() -> str:  # noqa: PLR0912, PLR0915 - a flat sequence of printed tables
    """Return the report text."""
    events = pl.read_parquet(OUTPUT_DIR / "events.parquet")
    placebo = pl.read_parquet(OUTPUT_DIR / "placebo.parquet")
    unmodified = pl.read_parquet(OUTPUT_DIR / "unmodified.parquet")
    low_months = pl.read_parquet(OUTPUT_DIR / "low_months.parquet")
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        record = load_record()
        baseline = clean_baseline(record=record)
        print(
            "Signal: valid days, robust standard deviation of the signal, and valid days per "
            "week by half-year"
        )
        months = record.dates.astype("datetime64[M]").astype(int) % 12 + 1
        summer = (months >= 4) & (months <= 9)
        for farm, site in enumerate(record.sites):
            values = baseline.signal[farm]
            finite = np.isfinite(values)
            first, last = np.flatnonzero(finite)[[0, -1]]
            span = np.arange(len(values))
            inside = (span >= first) & (span <= last)
            per_week = {
                name: round(
                    7 * float(finite[inside & mask].sum() / max((inside & mask).sum(), 1)), 2
                )
                for name, mask in (("apr_sep", summer), ("oct_mar", ~summer))
            }
            print(
                f"farm {site}: {int(finite.sum())} valid days, robust SD "
                f"{robust_sd(values=values):.3f}, per week {per_week}"
            )
        print(
            "\nDetection rate by monitor, size, duration, and half-year (pooled over farms, draws)"
        )
        for season in ("apr_sep", "oct_mar"):
            for days in (30, 90, -1):
                for size in (0.0, 0.03, 0.10, 0.25):
                    cells = [
                        rate(events=events, method=m, season=season, days=days, size=size)
                        for m in METHODS
                    ]
                    if cells[0][1] == 0:
                        continue
                    text = "  ".join(
                        f"{m}={r:.2f}(n={n})" for m, (r, n) in zip(METHODS, cells, strict=True)
                    )
                    print(f"{season} days={days:>3} size={size:.2f}  {text}")
        print("\nSmallest loss found in at least 80% of April-September injections")
        for days in (30, 90):
            for method in METHODS:
                found = [
                    s
                    for s in (0.03, 0.10, 0.25)
                    if rate(events=events, method=method, season="apr_sep", days=days, size=s)[0]
                    >= DETECTION_FLOOR
                ]
                print(f"days={days} {method}: {min(found) if found else 'none of 3%, 10%, 25%'}")
        print("\nPer-farm detection rate of 10% losses lasting 90 days, April-September")
        for method in METHODS:
            print(
                method,
                {
                    s: round(
                        rate(
                            events=events,
                            method=method,
                            season="apr_sep",
                            days=90,
                            size=0.10,
                            site=s,
                        )[0],
                        2,
                    )
                    for s in POOL_SITES
                },
            )
        print("\nPer-farm detection rate of 25% losses lasting 30 days, April-September")
        for method in METHODS:
            print(
                method,
                {
                    s: round(
                        rate(
                            events=events,
                            method=method,
                            season="apr_sep",
                            days=30,
                            size=0.25,
                            site=s,
                        )[0],
                        2,
                    )
                    for s in POOL_SITES
                },
            )
        print(
            "\nMedian delay (days) and size error (points) of detected 25% losses, April-September"
        )
        for method in METHODS:
            sub = events.filter(
                (pl.col("method") == method)
                & (pl.col("size") == 0.25)
                & (pl.col("season") == "apr_sep")
                & pl.col("detected")
            )
            print(
                method,
                "delay",
                sub["delay_days"].median(),
                "size error",
                round(100 * as_float(sub["size_error"].median()), 1),
            )
        print("\nContrasts on 10% losses lasting 30 and 90 days starting April-September")
        verdict_inputs = {}
        for name, other in (("C1", "rolling_median"), ("C2", "hyst")):
            point, (low, high), per_farm = farm_bootstrap(
                events=events, method_a="changepoint", method_b=other, size=0.10
            )
            verdict_inputs[name] = (point, low, high)
            print(
                name,
                f"changepoint minus {other}: {point:+.3f} (97.5% interval {low:+.3f} to "
                f"{high:+.3f})",
                {s: round(v, 2) for s, v in per_farm.items()},
            )
        print("\nFalse alarms per farm-year")
        grouped = (
            placebo.group_by("method")
            .agg(alarms=pl.col("alarms").sum(), years=pl.col("farm_years").sum())
            .sort("method")
        )
        placebo_rate = {}
        for row in grouped.iter_rows(named=True):
            placebo_rate[row["method"]] = row["alarms"] / row["years"]
            print(
                f"scored placebo {row['method']}: {row['alarms']} alarms in "
                f"{row['years']:.1f} farm-years = {placebo_rate[row['method']]:.3f}"
            )
        unmodified_rate = {}
        for method in METHODS:
            sub = unmodified.filter((pl.col("method") == method) & (pl.col("site") != "E"))
            counted = sub.filter(~pl.col("near_farm_a_low_months"))
            print(
                f"unmodified record {method}: {counted.height} alarms away from farm A's low "
                f"months, {sub.height - counted.height} near them"
            )
        years_by_site = {}
        for site in POOL_SITES:
            years_by_site[site] = as_float(
                placebo.filter((pl.col("site") == site) & (pl.col("draw") == 0))["farm_years"].max()
            )
        total_years = sum(years_by_site.values())
        for method in METHODS:
            counted = unmodified.filter(
                (pl.col("method") == method)
                & (pl.col("site") != "E")
                & ~pl.col("near_farm_a_low_months")
            ).height
            unmodified_rate[method] = counted / total_years
            print(
                f"unmodified record {method}: {unmodified_rate[method]:.3f} per farm-year "
                f"over {total_years:.1f} farm-years"
            )
        print("\nUnmodified-record alarms of the change-point monitor")
        for row in (
            unmodified.filter(pl.col("method") == "changepoint")
            .sort("site", "date")
            .iter_rows(named=True)
        ):
            print(
                f"  farm {row['site']} {row['date']} size {row['size']:.2f} near farm A low "
                f"months: {row['near_farm_a_low_months']}"
            )
        print("\nFarm A's low months")
        print(low_months)
        found_changepoint = low_months.filter(pl.col("method") == "changepoint")["found"].sum()
        chance = {
            m: rate(events=events, method=m, size=CHANCE_SIZE, season="apr_sep")[0] for m in METHODS
        }
        print(
            "\nChance detection (0% control windows, April-September)",
            {m: round(v, 3) for m, v in chance.items()},
        )
        hit25 = rate(events=events, method="changepoint", season="apr_sep", days=30, size=0.25)[0]
        hit10_90 = rate(events=events, method="changepoint", season="apr_sep", days=90, size=0.10)[
            0
        ]
        stop = {
            "25% 30-day detection under 80%": hit25 < DETECTION_FLOOR,
            "upper limit of C1 or C2 at or below 0": verdict_inputs["C1"][2] <= 0
            or verdict_inputs["C2"][2] <= 0,
            "placebo false alarms above 1 per farm-year": placebo_rate["changepoint"] > 1.0,
            "none of farm A's low months found": found_changepoint == 0,
        }
        supported = {
            "10% 90-day detection at least 80%": hit10_90 >= DETECTION_FLOOR,
            "lower limit of C1 above 0": verdict_inputs["C1"][1] > 0,
            "lower limit of C2 above 0": verdict_inputs["C2"][1] > 0,
            "placebo false alarms at most 0.5": placebo_rate["changepoint"] <= 0.5,
            "all three low months found": found_changepoint == 3,
            "unmodified record at most 1 per farm-year": unmodified_rate["changepoint"] <= 1.0,
        }
        print("\nStop rules", stop)
        print("Supported conditions", supported)
        if any(stop.values()):
            verdict = "refuted"
        elif all(supported.values()):
            verdict = "supported"
        else:
            verdict = "partly supported"
        print("\nVERDICT by the decision rule:", verdict)
        print(
            "\nPOST HOC 1: detection window from 14 days before the loss to 28 days after "
            "(April-September)"
        )
        for days in (30, 90, -1):
            for size in (0.10, 0.25):
                cells = [
                    as_float(
                        events.filter(
                            (pl.col("method") == m)
                            & (pl.col("season") == "apr_sep")
                            & (pl.col("days") == days)
                            & (pl.col("size") == size)
                        )["detected_early_window"].mean()
                    )
                    for m in METHODS
                ]
                print(
                    f"days={days:>3} size={size:.2f}",
                    {m: round(c, 2) for m, c in zip(METHODS, cells, strict=True)},
                )
        print("\nPOST HOC 2: change-point monitor with a minimum segment of 7 valid days")
        short_events = pl.read_parquet(OUTPUT_DIR / "posthoc_events.parquet")
        short_placebo = pl.read_parquet(OUTPUT_DIR / "posthoc_placebo.parquet")
        print(
            "penalty", json.loads((OUTPUT_DIR / "posthoc_config.json").read_text())["penalty_scale"]
        )
        short_rate = as_float(short_placebo["alarms"].sum()) / as_float(
            short_placebo["farm_years"].sum()
        )
        print(f"scored placebo false alarms: {short_rate:.3f} per farm-year")
        for season in ("apr_sep", "oct_mar"):
            for days in (30, 90, -1):
                for size in (0.0, 0.10, 0.25):
                    sub = short_events.filter(
                        (pl.col("season") == season)
                        & (pl.col("days") == days)
                        & (pl.col("size") == size)
                    )
                    if sub.height:
                        print(
                            f"{season} days={days:>3} size={size:.2f}  detection "
                            f"{sub['detected'].mean():.2f} (n={sub.height})"
                        )
        base = events.filter(pl.col("method") == "changepoint")
        for name, other in (("C1", "rolling_median"), ("C2", "hyst")):
            both = pl.concat([short_events, events.filter(pl.col("method") == other)])
            point, (low, high), per_farm = farm_bootstrap(
                events=both, method_a="changepoint_short_segment", method_b=other, size=0.10
            )
            print(
                f"{name} (short segment minus {other}): {point:+.3f} (97.5% interval "
                f"{low:+.3f} to {high:+.3f})",
                {s: round(v, 2) for s, v in per_farm.items()},
            )
        del base
    return buffer.getvalue()


def main() -> None:
    """Print the report and save it."""
    text = report()
    print(text)
    (DOCS_DIR / "report.txt").write_text(text)


if __name__ == "__main__":
    main()
