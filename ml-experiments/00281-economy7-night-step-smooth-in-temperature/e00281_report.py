"""Score experiment 281's arms and print every number the write-up quotes.

Reads `work/dp_preds/` and the stored predictions of experiment 87 (BG2, climatology, manual
heuristic), puts every arm on common rows (a reading, outside the hand-filtered switching events,
lead of at most 168 hours), and writes `work/report/report.json`: each arm's quantile score by lead
band and breakout, the planned contrasts with 99% month-resampled intervals, the tail tables, and
the checks.

    uv run --no-sync python ml-experiments/00281-*/e00281_report.py
"""

from datetime import UTC, datetime
from typing import Final

import e00281_c155 as common
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.dp_calendar import BAND_NAMES, HOLIDAY_PERIODS
from studies.dp_gam import QUANTILE_COLUMNS
from studies.experiment_store import losses_for_bootstrap
from studies.sources import ML_EXPERIMENTS_DATA_DIR

LEVELS: Final[np.ndarray] = common.LEVELS
LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0.5-3.5 h", 0.0, 3.5),
    ("3.5-9.5 h", 3.5, 9.5),
    ("9.5-24 h", 9.5, 24.0),
    ("under 24 h", 0.0, 24.0),
    ("24-72 h", 24.0, 72.0),
    ("72-168 h", 72.0, 168.0),
    ("D+1", 15.0, 39.0),
    ("12-36 h", 12.0, 36.0),
    ("all", 0.0, 168.0),
)
OURS: Final[tuple[str, ...]] = ("k0", "e1", "e2", "a0", "a1", "a2")
STORED_111: Final[dict[str, str]] = {}
STORED_87: Final[dict[str, str]] = {
    "bg2": "p_s6_bg2",
    "climatology": "ref_climatology",
    "manual_heuristic": "ref_manual_heuristic",
}
E87_PREDICTIONS: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00087-bg2-best-guess-assembled-with-drop-one-ablations"
    / "predictions"
)
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "C1": ("a1", "a0"),
    "C2": ("a2", "a1"),
    "C3": ("a2", "a0"),
    "C4": ("a0", "bg2"),
    "C5": ("a2", "bg2"),
    "R1": ("e1", "k0"),
    "R2": ("e2", "k0"),
    "R3": ("a0", "k0"),
}
LEVEL_PERCENT: Final[float] = 99.0
N_RESAMPLES: Final[int] = 20_000
COLD_TEMPERATURE_C: Final[float] = 3.0
COLD_DAY_C: Final[float] = 2.0
WINTER_MONTHS: Final[frozenset[int]] = common.WINTER_MONTHS
NIGHT_BANDS: Final[tuple[int, ...]] = (BAND_NAMES.index("night"), BAND_NAMES.index("evening"))
CHRISTMAS_CODES: Final[tuple[int, ...]] = tuple(
    HOLIDAY_PERIODS.index(name) for name in ("Christmas Day", "Boxing Day", "Christmas run")
)
STEP_KEY: Final[str] = "step hours (00:30 to 02:00 UTC, December to February)"
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
TAIL_QUANTILES: Final[tuple[tuple[str, float], ...]] = (
    ("q01", 0.01),
    ("q05", 0.05),
    ("q95", 0.95),
    ("q99", 0.99),
)
BASE_COLUMNS: Final[list[str]] = [
    "time_series_id", "power_fcst_init_time", "valid_time", "lead_hours", "seed",
    *QUANTILE_COLUMNS, "actual", "mean", "site",
]  # fmt: skip


def daily_mean_temperature(*, features, end_times: np.ndarray) -> np.ndarray:  # noqa: ANN001
    """Return the ERA5 temperature's mean over each half-hour's UTC date, on the grid."""
    days = end_times.astype("datetime64[D]")
    codes = np.unique(days, return_inverse=True)[1]
    temperature = features.temperature_c.astype(np.float64)
    finite = np.isfinite(temperature)
    total = np.bincount(codes, weights=np.where(finite, temperature, 0.0))
    count = np.bincount(codes, weights=finite.astype(np.float64))
    return (total / np.maximum(count, 1))[codes]


def annotate(*, frame: pl.DataFrame, cache) -> pl.DataFrame:  # noqa: ANN001
    """Keep scorable rows (a reading, not switching, lead of at most 168 hours); add breakouts."""
    last = cache.end_times[-1].astype("datetime64[us]").astype(datetime).replace(tzinfo=UTC)
    pieces = []
    for site in common.SCORED_SITES:
        part = frame.filter(pl.col("site") == site)
        if part.is_empty():
            continue
        features = cache.sites[site]
        target = np.searchsorted(
            cache.end_times, part["valid_time"].dt.replace_time_zone(None).to_numpy()
        )
        daily = daily_mean_temperature(features=features, end_times=cache.end_times)
        holiday_code = cache.calendar.holiday_period[target]
        part = part.with_columns(
            masked=pl.Series(features.switching_mask[target]),
            hour_band=pl.Series(cache.calendar.band[target].astype(np.int8)),
            holiday=pl.Series(holiday_code >= 0),
            christmas=pl.Series(np.isin(holiday_code, CHRISTMAS_CODES)),
            cold3=pl.Series(
                np.nan_to_num(features.temperature_c[target], nan=99.0) <= COLD_TEMPERATURE_C
            ),
            cold2=pl.Series(np.nan_to_num(daily[target], nan=99.0) < COLD_DAY_C),
            valid_month=pl.col("valid_time").dt.month(),
            utc_hh=pl.col("valid_time").dt.hour().cast(pl.Int32) * 2
            + pl.col("valid_time").dt.minute().cast(pl.Int32) // 30,
        )
        pieces.append(part)
    return pl.concat(pieces).filter(
        ~pl.col("masked")
        & pl.col("actual").is_finite()
        & (pl.col("lead_hours") <= 168.0)
        & (pl.col("valid_time") <= last)
    )


def in_period(*, frame: pl.DataFrame, period: str) -> pl.DataFrame:
    """Keep the rows whose run (by `power_fcst_init_time`) belongs to the period."""
    return frame.filter(pl.col("power_fcst_init_time").is_in(common.period_origins(period=period)))


def load(*, arm: str, period: str, cache) -> pl.DataFrame | None:  # noqa: ANN001
    """Return an arm's scorable rows for a period, or `None` if the arm has none."""
    ids = {s: cache.sites[s].table.time_series_id for s in common.SCORED_SITES}
    if arm in OURS or arm in STORED_111:
        pieces = []
        for site in common.SCORED_SITES:
            path = (
                common.out_path(arm=arm, period=period, site=site)
                if arm in OURS
                else common.E111_WORK_DIR
                / "dp_preds"
                / f"{STORED_111[arm]}_{period}_{site}.parquet"
            )
            if path.exists():
                pieces.append(pl.read_parquet(path).with_columns(site=pl.lit(site)))
        if not pieces:
            return None
        frame = pl.concat(pieces).select(BASE_COLUMNS)
    else:
        if period != "scored":
            return None
        label_of = {series: site for site, series in ids.items()}
        frame = pl.read_parquet(E87_PREDICTIONS / STORED_87[arm] / "predictions.parquet").filter(
            pl.col("time_series_id").is_in(list(label_of))
        )
        frame = frame.with_columns(
            site=pl.col("time_series_id").replace_strict(label_of, return_dtype=pl.String),
            actual=100.0 * pl.col("actual"),
            **{q: 100.0 * pl.col(q) for q in QUANTILE_COLUMNS},
        ).with_columns(mean=pl.col("q50"))
        frame = in_period(frame=frame.select(BASE_COLUMNS), period="scored")
    return annotate(frame=in_period(frame=frame, period=period), cache=cache)


def common_rows(*, frames: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Restrict every frame to the rows all frames hold."""
    shared = None
    for frame in frames.values():
        current = frame.select(KEYS).unique()
        shared = current if shared is None else shared.join(current, on=KEYS, how="inner")
    assert shared is not None
    schema = {
        "time_series_id": pl.Int32,
        "lead_hours": pl.Float64,
        "seed": pl.Int8,
        "actual": pl.Float32,
        "mean": pl.Float32,
        **dict.fromkeys(QUANTILE_COLUMNS, pl.Float32),
    }
    return {
        name: frame.join(shared, on=KEYS, how="inner")
        .sort(KEYS)
        .with_columns(pl.col(column).cast(dtype) for column, dtype in schema.items())
        for name, frame in frames.items()
    }


def with_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the quantile score, the mean and median absolute errors, and the coverage flags."""
    actual = frame["actual"].to_numpy().astype(np.float64)[:, None]
    quantiles = frame.select(list(QUANTILE_COLUMNS)).to_numpy().astype(np.float64)
    error = actual - quantiles
    score = 2.0 * np.mean(
        np.maximum(LEVELS[None, :] * error, (LEVELS[None, :] - 1.0) * error), axis=1
    )
    extra = {
        "qs": score,
        "abs_mean": np.abs(frame["actual"].to_numpy() - frame["mean"].to_numpy()).astype(
            np.float64
        ),
        "abs_median": np.abs(frame["actual"].to_numpy() - frame["q50"].to_numpy()).astype(
            np.float64
        ),
    }
    for name, level in TAIL_QUANTILES:
        value = frame[name].to_numpy().astype(np.float64)
        diff = frame["actual"].to_numpy() - value
        extra[f"pin_{name}"] = np.maximum(level * diff, (level - 1.0) * diff)
        extra[f"cov_{name}"] = (frame["actual"].to_numpy() <= value).astype(np.float64)
    return frame.with_columns([pl.Series(k, v) for k, v in extra.items()])


def lead_filter(*, frame: pl.DataFrame, low: float, high: float) -> pl.DataFrame:
    """Keep rows with a lead above `low` and at most `high` hours."""
    return frame.filter((pl.col("lead_hours") > low) & (pl.col("lead_hours") <= high))


def contrast(*, losses: pl.DataFrame, treatment: str, reference: str, metric: str = "qs") -> dict:
    """Return a paired contrast with a 99% month-resampled interval and its site and month signs."""
    t = losses.filter(pl.col("arm") == treatment)
    r = losses.filter(pl.col("arm") == reference)
    estimate = float(t[metric].to_numpy().mean() - r[metric].to_numpy().mean())
    lower, upper = bootstrap_difference_at_level(
        losses=losses,
        treatment=treatment,
        reference=reference,
        metric=metric,
        level=LEVEL_PERCENT,
        n_resamples=N_RESAMPLES,
    )
    by_site = (
        t.group_by("site")
        .agg(t=pl.col(metric).mean())
        .join(r.group_by("site").agg(r=pl.col(metric).mean()), on="site")
    )
    by_month = (
        t.group_by("month")
        .agg(t=pl.col(metric).mean())
        .join(r.group_by("month").agg(r=pl.col(metric).mean()), on="month")
    )
    return {
        "estimate": estimate,
        "lower": float(lower),
        "upper": float(upper),
        "sites_better": int((by_site["t"] < by_site["r"]).sum()),
        "n_sites": by_site.height,
        "months_negative": int((by_month["t"] < by_month["r"]).sum()),
        "n_months": by_month.height,
    }


def subset_filters() -> dict[str, pl.Expr]:
    """Return the breakout subsets, each as a row filter."""
    night = pl.col("hour_band").is_in(list(NIGHT_BANDS))
    winter = pl.col("valid_month").is_in(list(WINTER_MONTHS))
    return {
        STEP_KEY: pl.col("valid_month").is_in([12, 1, 2]) & pl.col("utc_hh").is_between(1, 4),
        "winter": winter,
        "winter nights": winter & night,
        "cold nights (day mean below 2 C)": winter & pl.col("cold2") & night,
        "cold nights (3 C or below, as experiment 136)": pl.col("cold3") & night,
        "holidays and Christmas": pl.col("holiday"),
        "Christmas only": pl.col("christmas"),
        "bank holidays without Christmas": pl.col("holiday") & ~pl.col("christmas"),
    }


COLD_KEY: Final[str] = "cold nights (day mean below 2 C)"


def arm_table(*, losses: pl.DataFrame, sites: tuple[str, ...]) -> dict:
    """Return each arm's mean score by lead band, by site, by breakout, and by hour band."""
    out: dict = {"by_lead": {}, "by_site": {}, "breakouts": {}, "by_hour_band": {}}
    scope = losses.filter(pl.col("site").is_in(list(sites)))
    for name, low, high in LEAD_BANDS:
        part = lead_filter(frame=scope, low=low, high=high)
        out["by_lead"][name] = part.group_by("arm").agg(pl.col("qs").mean()).sort("arm").to_dicts()
    out["by_site"] = (
        scope.group_by(["arm", "site"]).agg(pl.col("qs").mean()).sort(["arm", "site"]).to_dicts()
    )
    out["by_hour_band"] = (
        scope.group_by(["arm", "hour_band"])
        .agg(pl.col("qs").mean())
        .sort(["arm", "hour_band"])
        .to_dicts()
    )
    for label, expression in subset_filters().items():
        part = scope.filter(expression)
        if part.is_empty():
            continue
        out["breakouts"][label] = {
            "rows": part.filter(pl.col("arm") == part["arm"][0]).height,
            "scores": part.group_by("arm").agg(pl.col("qs").mean()).sort("arm").to_dicts(),
        }
    return out


def tail_table(*, losses: pl.DataFrame, sites: tuple[str, ...]) -> dict:
    """Return coverage and pinball loss at the four tail quantiles, overall and in subsets."""
    scope = losses.filter(pl.col("site").is_in(list(sites)))
    aggregations = [pl.col(f"cov_{n}").mean() for n, _ in TAIL_QUANTILES]
    aggregations += [pl.col(f"pin_{n}").mean() for n, _ in TAIL_QUANTILES]
    out: dict = {}
    parts = {
        "12-36 h": lead_filter(frame=scope, low=12.0, high=36.0),
        "all leads": scope,
        "cold nights (day mean below 2 C)": scope.filter(subset_filters()[COLD_KEY]),
        "cold nights (3 C or below, as experiment 136)": scope.filter(
            subset_filters()["cold nights (3 C or below, as experiment 136)"]
        ),
        "holidays and Christmas": scope.filter(pl.col("holiday")),
    }
    for label, part in parts.items():
        out[label] = part.group_by("arm").agg(aggregations).sort("arm").to_dicts()
    out["by hour band"] = (
        scope.group_by(["arm", "hour_band"]).agg(aggregations).sort(["arm", "hour_band"]).to_dicts()
    )
    return out


def winter_night_by_half_hour(*, losses: pl.DataFrame, sites: tuple[str, ...]) -> list[dict]:
    """Return the mean score by UTC half-hour from 21:00 to 09:00 on winter nights."""
    scope = losses.filter(
        pl.col("site").is_in(list(sites)) & pl.col("valid_month").is_in(list(WINTER_MONTHS))
    ).with_columns(
        minute=pl.col("valid_time").dt.hour().cast(pl.Int32) * 60
        + pl.col("valid_time").dt.minute().cast(pl.Int32)
    )
    night = scope.filter((pl.col("minute") >= 21 * 60) | (pl.col("minute") <= 9 * 60))
    return (
        night.group_by(["arm", "minute"])
        .agg(pl.col("qs").mean())
        .sort(["arm", "minute"])
        .to_dicts()
    )


def stack(*, frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return the bootstrap loss frame of arms already on common rows, with D-labels as `site`."""
    cache = common.load_cache()
    labels = {cache.sites[s].table.time_series_id: s for s in common.SCORED_SITES}
    keep = [*KEYS, "lead_hours", "seed", *QUANTILE_COLUMNS, "actual", "mean"]
    extra = ["hour_band", "holiday", "christmas", "cold2", "cold3", "valid_month", "utc_hh"]
    stacked = losses_for_bootstrap(
        arm_losses={
            name: with_losses(frame=f.select([*keep, *extra])) for name, f in frames.items()
        }
    )
    return stacked.with_columns(site=pl.col("site").replace_strict(labels, return_dtype=pl.String))


def site_gains(
    *, losses: pl.DataFrame, treatment: str, reference: str, rows: pl.Expr | None = None
) -> dict[str, float]:
    """Return each decision site's contrast, treatment minus reference, on all rows or `rows`."""
    scope = losses.filter(pl.col("site").is_in(list(common.DECISION_SITES)))
    if rows is not None:
        scope = scope.filter(rows)
    out = {}
    for site in common.DECISION_SITES:
        part = scope.filter(pl.col("site") == site)
        out[site] = float(
            part.filter(pl.col("arm") == treatment)["qs"].to_numpy().mean()
            - part.filter(pl.col("arm") == reference)["qs"].to_numpy().mean()
        )
    return out


def leave_one_out(*, losses: pl.DataFrame, treatment: str, reference: str) -> dict[str, float]:
    """Return the all-lead contrast with each decision site left out in turn."""
    scope = losses.filter(pl.col("site").is_in(list(common.DECISION_SITES)))
    out = {}
    for site in common.DECISION_SITES:
        part = scope.filter(pl.col("site") != site)
        t = part.filter(pl.col("arm") == treatment)["qs"].to_numpy().mean()
        r = part.filter(pl.col("arm") == reference)["qs"].to_numpy().mean()
        out[site] = float(t - r)
    return out


def checks(*, frames: dict[str, pl.DataFrame]) -> dict:
    """Return the median minus mean forecast per site and arm at 12 to 36 hours."""
    out = []
    for name, frame in frames.items():
        part = lead_filter(frame=frame, low=12.0, high=36.0)
        grouped = part.group_by("site").agg(
            (pl.col("q50") - pl.col("mean")).mean().alias("median_minus_mean")
        )
        out.extend({"arm": name, **row} for row in grouped.sort("site").to_dicts())
    return {"median_minus_mean": out}


def period_report(*, period: str) -> tuple[dict, dict[str, pl.DataFrame]]:
    """Return the period's tables and contrasts, and the common-row frames."""
    cache = common.load_cache()
    loaded = {
        arm: frame
        for arm in (*OURS, *STORED_111, *STORED_87)
        if (frame := load(arm=arm, period=period, cache=cache)) is not None
    }
    frames = common_rows(frames=loaded)
    losses = stack(frames=frames)
    decision = losses.filter(pl.col("site").is_in(list(common.DECISION_SITES)))
    report: dict = {
        "arms": list(frames),
        "rows": frames[next(iter(frames))].height,
        "tables": arm_table(losses=losses, sites=common.DECISION_SITES),
        "tail": tail_table(losses=losses, sites=common.DECISION_SITES),
        "winter_night_by_half_hour": winter_night_by_half_hour(
            losses=decision.filter(pl.col("arm").is_in(["k0", "a0", "a1", "a2"])),
            sites=common.DECISION_SITES,
        ),
        "contrasts": {},
        "site_gains": {},
        "leave_one_out": {},
    }
    for label, (treatment, reference) in CONTRASTS.items():
        if treatment not in frames or reference not in frames:
            continue
        report["site_gains"][label] = site_gains(
            losses=losses, treatment=treatment, reference=reference
        )
        report.setdefault("step_site_gains", {})[label] = site_gains(
            losses=losses,
            treatment=treatment,
            reference=reference,
            rows=subset_filters()[STEP_KEY],
        )
        report["leave_one_out"][label] = leave_one_out(
            losses=losses, treatment=treatment, reference=reference
        )
        entry: dict = {}
        full = period == "scored" and label.startswith("C")
        for scope_name, low, high in LEAD_BANDS:
            if not full and scope_name not in ("all", "D+1"):
                continue
            part = lead_filter(frame=decision, low=low, high=high)
            entry[scope_name] = {
                "qs": contrast(losses=part, treatment=treatment, reference=reference)
            }
            if scope_name in ("all", "D+1"):
                entry[scope_name]["abs_mean"] = contrast(
                    losses=part, treatment=treatment, reference=reference, metric="abs_mean"
                )
        for sub_name, expression in subset_filters().items():
            if not full and sub_name not in (COLD_KEY, "winter nights", STEP_KEY):
                continue
            part = decision.filter(expression)
            if not part.is_empty():
                entry[sub_name] = {
                    "qs": contrast(losses=part, treatment=treatment, reference=reference)
                }
                if sub_name == STEP_KEY:
                    entry[sub_name]["abs_mean"] = contrast(
                        losses=part, treatment=treatment, reference=reference, metric="abs_mean"
                    )
        cold = decision.filter(subset_filters()[COLD_KEY])
        entry["cold_night_q95_coverage"] = contrast(
            losses=cold, treatment=treatment, reference=reference, metric="cov_q95"
        )
        report["contrasts"][label] = entry
    return report, frames


def main() -> None:
    """Score the three periods and write the report."""
    report: dict = {}
    for period in ("scored", "replicate", "tuning"):
        period_data, frames = period_report(period=period)
        report[period] = period_data
        if period == "scored":
            report["checks"] = checks(frames=frames)
    common.write_json_atomic(path=common.WORK_DIR / "report" / "report.json", content=report)
    print("report written")


if __name__ == "__main__":
    main()
