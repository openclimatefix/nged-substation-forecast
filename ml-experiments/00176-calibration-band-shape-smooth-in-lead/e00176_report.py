"""Score the smooth-in-lead band arms against arm A and print every number the page quotes.

Reads `work/dp_preds/` and writes `work/report/report.json`: the median and mean q05 to q95 width by
lead day on two row sets (outside the switching events, and all rows), the pre-registered
non-decreasing test at every substation, the paired quantile-score contrast of every arm against
arm A with a month-resampled interval per substation, coverage by lead day, cold-night q95
coverage, tail pinball loss, and the checks.

    uv run --no-sync python ml-experiments/00176-*/e00176_report.py
"""

from typing import Final

import e00176_common as common
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.dp_calendar import BAND_NAMES
from studies.experiment_store import losses_for_bootstrap

KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
N_RESAMPLES: Final[int] = 20_000
LEVEL_PERCENT: Final[float] = 95.0
SITE_UPPER_LIMIT: Final[float] = 0.005
NON_DECREASING_TOLERANCE: Final[float] = 0.97
COLD_COVERAGE_FLOOR: Final[float] = 0.89
PRINTED_ARM_A_SCORE: Final[float] = 2.7519
COLD_DAY_C: Final[float] = 2.0
WINTER_MONTHS: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})
NIGHT_BANDS: Final[tuple[int, ...]] = (BAND_NAMES.index("night"), BAND_NAMES.index("evening"))
TAIL_QUANTILES: Final[tuple[tuple[str, float], ...]] = (
    ("q01", 0.01),
    ("q05", 0.05),
    ("q95", 0.95),
    ("q99", 0.99),
)


def daily_mean_temperature(*, features, end_times: np.ndarray) -> np.ndarray:  # noqa: ANN001
    """Return the ERA5 temperature's mean over each half-hour's UTC date, on the grid."""
    days = end_times.astype("datetime64[D]")
    codes = np.unique(days, return_inverse=True)[1]
    temperature = features.temperature_c.astype(np.float64)
    finite = np.isfinite(temperature)
    total = np.bincount(codes, weights=np.where(finite, temperature, 0.0))
    count = np.bincount(codes, weights=finite.astype(np.float64))
    return (total / np.maximum(count, 1))[codes]


def load_arm(*, arm: str, period: str, cache) -> pl.DataFrame:  # noqa: ANN001
    """Return an arm's rows for a period at every substation, with the cold-day flag."""
    pieces = []
    for site in common.SCORED_SITES:
        frame = pl.read_parquet(common.out_path(arm=arm, period=period, site=site)).with_columns(
            site=pl.lit(site)
        )
        features = cache.sites[site]
        target = np.searchsorted(
            cache.end_times, frame["valid_time"].dt.replace_time_zone(None).to_numpy()
        )
        daily = daily_mean_temperature(features=features, end_times=cache.end_times)
        pieces.append(
            frame.with_columns(
                cold2=pl.Series(np.nan_to_num(daily[target], nan=99.0) < COLD_DAY_C),
                valid_month=pl.col("valid_time").dt.month(),
            )
        )
    return pl.concat(pieces).sort([*KEYS])


def with_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the quantile score, the width, the coverage flags, and the pinball losses."""
    actual = frame["actual"].to_numpy().astype(np.float64)
    quantiles = frame.select(list(common.QUANTILES)).to_numpy().astype(np.float64)
    error = actual[:, None] - quantiles
    score = 2.0 * np.mean(
        np.maximum(common.LEVELS[None, :] * error, (common.LEVELS[None, :] - 1.0) * error), axis=1
    )
    extra = {
        "qs": score,
        "width": quantiles[:, 9] - quantiles[:, 1],
        "lead_day": common.lead_day(lead_hours=frame["lead_hours"].to_numpy()),
        "abs_mean": np.abs(actual - frame["mean"].to_numpy()),
        "median_minus_mean": quantiles[:, 5] - frame["mean"].to_numpy(),
    }
    for name, level in TAIL_QUANTILES:
        value = frame[name].to_numpy().astype(np.float64)
        diff = actual - value
        extra[f"pin_{name}"] = np.maximum(level * diff, (level - 1.0) * diff)
        extra[f"cov_{name}"] = (actual <= value).astype(np.float64)
    return frame.with_columns([pl.Series(k, v) for k, v in extra.items()])


def day_curve(*, frame: pl.DataFrame, column: str, statistic: str) -> list[float]:
    """Return the median or mean of a column on each lead day."""
    out = []
    for day in range(common.N_LEAD_DAYS):
        values = frame.filter(pl.col("lead_day") == day)[column].to_numpy()
        out.append(float(np.median(values) if statistic == "median" else np.mean(values)))
    return out


def smoothness(*, curve: list[float]) -> dict:
    """Return the lowest ratio of a day to the day before, and the mean absolute change."""
    values = np.array(curve)
    return {
        "minimum_ratio": float((values[1:] / values[:-1]).min()),
        "mean_absolute_change": float(np.abs(np.diff(values)).mean()),
        "non_decreasing_within_3_percent": bool(
            (values[1:] / values[:-1]).min() >= NON_DECREASING_TOLERANCE
        ),
    }


def score_period(*, period: str, cache) -> dict:  # noqa: ANN001
    """Return every number for one period."""
    frames = {
        arm: with_losses(frame=load_arm(arm=arm, period=period, cache=cache)) for arm in common.ARMS
    }
    out: dict = {"widths": {}, "smoothness": {}, "by_site": {}, "contrasts": {}, "checks": {}}
    for arm, frame in frames.items():
        all_rows = frame.filter(pl.col("lead_hours") <= common.HORIZON_HOURS)
        kept = all_rows.filter(~pl.col("switching_mask") & pl.col("actual").is_finite())
        for site in common.SCORED_SITES:
            for label, scope in (("kept", kept), ("all", all_rows)):
                part = scope.filter(pl.col("site") == site)
                for statistic in ("median", "mean"):
                    curve = day_curve(frame=part, column="width", statistic=statistic)
                    out["widths"][f"{arm}|{site}|{label}|{statistic}"] = curve
                    if statistic == "median":
                        out["smoothness"][f"{arm}|{site}|{label}"] = smoothness(curve=curve)
    scored = {
        arm: frame.filter(
            ~pl.col("switching_mask")
            & pl.col("actual").is_finite()
            & (pl.col("lead_hours") <= common.HORIZON_HOURS)
        )
        for arm, frame in frames.items()
    }
    decision = {
        arm: frame.filter(pl.col("site").is_in(list(common.DECISION_SITES)))
        for arm, frame in scored.items()
    }
    out["rows_decision"] = decision["a"].height
    for arm, frame in decision.items():
        night = frame.filter(pl.col("hour_band").is_in(list(NIGHT_BANDS)))
        cold = night.filter(pl.col("valid_month").is_in(list(WINTER_MONTHS)) & pl.col("cold2"))
        entry = {
            "score": float(frame["qs"].to_numpy().mean()),
            "cold_rows": cold.height,
            "cold_night_cov_q95": float(cold["cov_q95"].to_numpy().mean()),
            "cold_night_cov_q05": float(cold["cov_q05"].to_numpy().mean()),
            "cov_by_day": {
                name: [
                    float(frame.filter(pl.col("lead_day") == d)[f"cov_{name}"].to_numpy().mean())
                    for d in range(common.N_LEAD_DAYS)
                ]
                for name in ("q05", "q95", "q99")
            },
            "pinball": {
                name: float(frame[f"pin_{name}"].to_numpy().mean()) for name, _ in TAIL_QUANTILES
            },
            "cold_by_site": {
                site: float(cold.filter(pl.col("site") == site)["cov_q95"].to_numpy().mean())
                for site in common.DECISION_SITES
            },
            "abs_mean_error": float(frame["abs_mean"].to_numpy().mean()),
            "median_minus_mean": {
                site: float(
                    frame.filter(pl.col("site") == site)["median_minus_mean"].to_numpy().mean()
                )
                for site in common.SCORED_SITES
                if site in common.DECISION_SITES
            },
        }
        out["by_site"][arm] = entry
    keep = [*KEYS, "lead_hours", "seed", "qs", "site"]
    losses = losses_for_bootstrap(
        arm_losses={arm: scored[arm].select(keep).drop("site") for arm in scored}
    )
    label = {
        row["time_series_id"]: row["site"]
        for row in scored["a"].select("time_series_id", "site").unique().to_dicts()
    }
    for arm in common.ARMS[1:]:
        out["contrasts"][arm] = {}
        for series, site in label.items():
            part = losses.filter(pl.col("site") == series)
            t = part.filter(pl.col("arm") == arm)["qs"].to_numpy()
            r = part.filter(pl.col("arm") == "a")["qs"].to_numpy()
            lower, upper = bootstrap_difference_at_level(
                losses=part,
                treatment=arm,
                reference="a",
                metric="qs",
                level=LEVEL_PERCENT,
                n_resamples=N_RESAMPLES,
            )
            out["contrasts"][arm][site] = {
                "estimate": float(t.mean() - r.mean()),
                "lower": float(lower),
                "upper": float(upper),
            }
        five = losses.filter(
            pl.col("site").is_in([s for s, n in label.items() if n in common.DECISION_SITES])
        )
        lower, upper = bootstrap_difference_at_level(
            losses=five,
            treatment=arm,
            reference="a",
            metric="qs",
            level=LEVEL_PERCENT,
            n_resamples=N_RESAMPLES,
        )
        t = five.filter(pl.col("arm") == arm)["qs"].to_numpy()
        r = five.filter(pl.col("arm") == "a")["qs"].to_numpy()
        out["contrasts"][arm]["pooled"] = {
            "estimate": float(t.mean() - r.mean()),
            "lower": float(lower),
            "upper": float(upper),
        }
    return out


def decide(*, period: dict) -> dict:
    """Return which arms pass the three pre-registered criteria in one period."""
    verdict = {}
    for arm in common.ARMS:
        smooth_kept = period["smoothness"][f"{arm}|D5|kept"]["non_decreasing_within_3_percent"]
        smooth_all = period["smoothness"][f"{arm}|D5|all"]["non_decreasing_within_3_percent"]
        worst_upper = (
            max(period["contrasts"][arm][s]["upper"] for s in common.SCORED_SITES)
            if arm != "a"
            else 0.0
        )
        cold = period["by_site"][arm]["cold_night_cov_q95"]
        verdict[arm] = {
            "c1_d5_non_decreasing_kept": smooth_kept,
            "c1_d5_non_decreasing_all": smooth_all,
            "c2_worst_site_upper": worst_upper,
            "c2_pass": worst_upper <= SITE_UPPER_LIMIT,
            "c3_cold_night_cov_q95": cold,
            "c3_pass": cold >= COLD_COVERAGE_FLOOR,
            "passes": bool(
                smooth_kept
                and smooth_all
                and worst_upper <= SITE_UPPER_LIMIT
                and cold >= COLD_COVERAGE_FLOOR
            ),
        }
    return verdict


def main() -> None:
    """Score the replicate months and the scored year and write the report."""
    cache = common.load_cache()
    report = {}
    for period in ("scored", "replicate"):
        report[period] = score_period(period=period, cache=cache)
        report[period]["decision"] = decide(period=report[period])
        print(
            period,
            "arm A score at the five decision substations",
            report[period]["by_site"]["a"]["score"],
        )
    reproduction = abs(report["scored"]["by_site"]["a"]["score"] - PRINTED_ARM_A_SCORE)
    report["arm_a_reproduction_gap"] = reproduction
    common.write_json_atomic(path=common.WORK_DIR / "report" / "report.json", content=report)
    for period in ("scored", "replicate"):
        print("==", period)
        for arm in common.ARMS:
            d = report[period]["decision"][arm]
            s = report[period]["by_site"][arm]
            print(
                arm,
                f"score {s['score']:.4f}",
                {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()},
            )
        for arm in common.ARMS[1:]:
            print(
                arm,
                {
                    site: (round(c["estimate"], 4), round(c["lower"], 4), round(c["upper"], 4))
                    for site, c in report[period]["contrasts"][arm].items()
                },
            )
        for key in ("a", "p", "u", "s", "ps"):
            for label in ("kept", "all"):
                print(
                    f"D5 {key} {label} median width by day",
                    np.round(report[period]["widths"][f"{key}|D5|{label}|median"], 1),
                )
    print("arm A reproduction gap", reproduction)


if __name__ == "__main__":
    main()
