"""Score experiment 168's arms and print every number the write-up quotes.

Reads `work/dp_preds/` (the production-cadence arms), the once-a-day arms A (experiment 136's A2)
and D (experiment 155's `d`), and the stored BG2, climatology, and manual-heuristic predictions of
experiment 87. Every scored row is a reading outside the hand-filtered switching events. The planned
contrasts pair rows in one of three ways, and the month-resampled interval is the 99% interval of
the paired differences:

- **Same lead.** A production-origin row is paired with the once-a-day row of the same run and the
  same lead. The difference is the cost of the older weather forecast.
- **Same valid time.** A production-origin row is paired with the once-a-day row of the same run
  and the same valid time. The difference is the gain of the later origin.
- **Same row.** A production-origin row with the 3.5-hour delay is paired with the same row with no
  delay. The difference is the cost of the delay.

    uv run --no-sync python ml-experiments/00168-*/e00168_report.py
"""

import json
from datetime import UTC, datetime, timedelta
from typing import Final

import e00168_common as common
import numpy as np
import polars as pl
from studies.dp_calendar import BAND_NAMES, HOLIDAY_PERIODS
from studies.dp_gam import QUANTILE_COLUMNS
from studies.experiment_store import losses_for_bootstrap
from studies.sources import ML_EXPERIMENTS_DATA_DIR

LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
PERIODS: Final[tuple[str, ...]] = ("scored", "replicate")
PRODUCTION_ARMS: Final[tuple[str, ...]] = ("ap", "dp", "ap0", "dp0")
ALL_LEAD_LIMIT_HOURS: Final[float] = 147.0
"""Beyond this lead only some of the four origins have a forecast (the stored window ends 168 hours
after the original origin, so the origin 21 hours later has 147 hours)."""
LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0.5-3.5 h", 0.0, 3.5),
    ("3.5-9.5 h", 3.5, 9.5),
    ("9.5-24 h", 9.5, 24.0),
    ("under 24 h", 0.0, 24.0),
    ("24-72 h", 24.0, 72.0),
    ("72-147 h", 72.0, ALL_LEAD_LIMIT_HOURS),
    ("D+1", 15.0, 39.0),
    ("all (to 147 h)", 0.0, ALL_LEAD_LIMIT_HOURS),
)
LEVEL_PERCENT: Final[float] = 99.0
N_RESAMPLES: Final[int] = 20_000
COLD_DAY_C: Final[float] = 2.0
WINTER_MONTHS: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})
NIGHT_BANDS: Final[tuple[int, ...]] = (BAND_NAMES.index("night"), BAND_NAMES.index("evening"))
CHRISTMAS_CODES: Final[tuple[int, ...]] = tuple(
    HOLIDAY_PERIODS.index(name) for name in ("Christmas Day", "Boxing Day", "Christmas run")
)
TAIL_QUANTILES: Final[tuple[tuple[str, float], ...]] = (
    ("q01", 0.01),
    ("q05", 0.05),
    ("q95", 0.95),
    ("q99", 0.99),
)
E87_PREDICTIONS: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00087-bg2-best-guess-assembled-with-drop-one-ablations"
    / "predictions"
)
STORED_87: Final[dict[str, str]] = {
    "bg2": "p_s6_bg2",
    "climatology": "ref_climatology",
    "manual_heuristic": "ref_manual_heuristic",
}
SLOT_LABELS: Final[tuple[str, ...]] = ("12:00", "18:00", "00:00", "06:00")
"""The production origin of each slot, in UTC; the 00:00 and 06:00 origins fall on the next day."""
SCORE_COLUMNS: Final[list[str]] = [
    "time_series_id", "run", "slot", "power_fcst_init_time", "valid_time", "lead_hours", "seed",
    *QUANTILE_COLUMNS, "actual", "mean", "hour_band", "quarter", "switching_mask", "site",
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
    """Keep scorable rows (a reading, not switching); add the breakout columns."""
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
        pieces.append(
            part.with_columns(
                masked=pl.Series(features.switching_mask[target]),
                hour_band=pl.Series(cache.calendar.band[target].astype(np.int8)),
                holiday=pl.Series(holiday_code >= 0),
                christmas=pl.Series(np.isin(holiday_code, CHRISTMAS_CODES)),
                cold2=pl.Series(np.nan_to_num(daily[target], nan=99.0) < COLD_DAY_C),
                valid_month=pl.col("valid_time").dt.month(),
            )
        )
    return pl.concat(pieces).filter(
        ~pl.col("masked") & pl.col("actual").is_finite() & (pl.col("valid_time") <= last)
    )


def common_origins(*, period: str) -> list[datetime]:
    """Return the `power_fcst_init_time` of every run of a period, as experiment 136 defines it."""
    return [
        run + common.PUBLICATION_DELAY
        for run in common.run_dates()
        if common.period_of_run(run=run) == period
    ]


def read_arm(*, arm: str, period: str, cache) -> pl.DataFrame:  # noqa: ANN001
    """Return an arm's scorable rows for a period, every scored site together.

    The once-a-day arms get `slot` -1 and `run_origin` equal to their origin.
    """
    pieces = []
    for site in common.SCORED_SITES:
        if arm in ("a", "d"):
            path = (
                common.E136_WORK_DIR / "dp_preds" / f"a2_{period}_{site}.parquet"
                if arm == "a"
                else common.E155_WORK_DIR / "dp_preds" / f"d_{period}_{site}.parquet"
            )
            part = pl.read_parquet(path).with_columns(slot=pl.lit(-1, dtype=pl.Int8))
        else:
            path = common.WORK_DIR / "dp_preds" / f"{arm}_{period}_{site}.parquet"
            part = pl.read_parquet(path)
            if arm in ("ai0", "di0"):
                part = part.with_columns(slot=pl.lit(-1, dtype=pl.Int8))
        pieces.append(part.with_columns(site=pl.lit(site)))
    frame = pl.concat(pieces).select(SCORE_COLUMNS)
    return annotate(frame=frame, cache=cache)


def read_stored_87(*, name: str, cache) -> pl.DataFrame:  # noqa: ANN001
    """Return a stored experiment 87 arm's scored-year rows, in points, once-a-day origins."""
    label_of = {cache.sites[s].table.time_series_id: s for s in common.SCORED_SITES}
    frame = pl.read_parquet(E87_PREDICTIONS / STORED_87[name] / "predictions.parquet").filter(
        pl.col("time_series_id").is_in(list(label_of))
    )
    frame = frame.with_columns(
        site=pl.col("time_series_id").replace_strict(label_of, return_dtype=pl.String),
        actual=100.0 * pl.col("actual"),
        slot=pl.lit(-1, dtype=pl.Int8),
        run=pl.lit(0, dtype=pl.Int32),
        hour_band=pl.lit(0, dtype=pl.Int8),
        quarter=pl.lit(0, dtype=pl.Int8),
        switching_mask=pl.lit(value=False),
        **{q: 100.0 * pl.col(q) for q in QUANTILE_COLUMNS},
    ).with_columns(mean=pl.col("q50"))
    frame = frame.select(SCORE_COLUMNS)
    origins = common_origins(period="scored")
    return annotate(frame=frame.filter(pl.col("power_fcst_init_time").is_in(origins)), cache=cache)


def with_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the quantile score, the absolute errors of the mean and median, and the tail columns."""
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


def subset_filters() -> dict[str, pl.Expr]:
    """Return the breakout subsets, each as a row filter on a paired or single-arm frame."""
    night = pl.col("hour_band").is_in(list(NIGHT_BANDS))
    winter = pl.col("valid_month").is_in(list(WINTER_MONTHS))
    return {
        "winter nights": winter & night,
        "cold nights (day mean below 2 C)": winter & pl.col("cold2") & night,
        "holidays and Christmas": pl.col("holiday"),
    }


SHARED_COLUMNS: Final[list[str]] = [
    "time_series_id", "power_fcst_init_time", "valid_time", "lead_hours", "seed", "slot",
    "hour_band", "holiday", "christmas", "cold2", "valid_month",
]  # fmt: skip
"""The columns a paired reference row takes from its treatment row, so that both rows fall in the
same lead band, origin, and subset."""
LOSS_INPUTS: Final[list[str]] = [*QUANTILE_COLUMNS, "actual", "mean"]


def pair(
    *, treatment: pl.DataFrame, reference: pl.DataFrame, on: list[str]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return both arms on the rows they share, the reference re-keyed to the treatment's rows.

    A reference row used by several treatment rows appears once per treatment row. The reference
    row keeps its own forecast and actual, and takes the treatment row's keys and breakout flags.
    """
    renamed = reference.select(*on, *[pl.col(c).alias(f"r_{c}") for c in LOSS_INPUTS])
    joined = treatment.join(renamed, on=on, how="inner", validate="m:1")
    t = joined.select([*SHARED_COLUMNS, *LOSS_INPUTS])
    r = joined.select(*SHARED_COLUMNS, *[pl.col(f"r_{c}").alias(c) for c in LOSS_INPUTS]).select(
        t.columns
    )
    return t, r


def stack(*, t: pl.DataFrame, r: pl.DataFrame, labels: dict[int, str]) -> pl.DataFrame:
    """Return the bootstrap loss frame of a treatment and a reference, D-labels as `site`."""
    keep = [*SHARED_COLUMNS, *LOSS_INPUTS]
    stacked = losses_for_bootstrap(
        arm_losses={
            "t": with_losses(frame=t.select(keep)),
            "r": with_losses(frame=r.select(keep)),
        }
    )
    return stacked.with_columns(site=pl.col("site").replace_strict(labels, return_dtype=pl.String))


def interval_of_months(
    *, difference: np.ndarray, count: np.ndarray, seed: int = 168
) -> tuple[float, float]:
    """Return the 99% interval of a mean from each month's summed difference and row count."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(count), size=(N_RESAMPLES, len(count)))
    resampled = difference[draws].sum(axis=1) / count[draws].sum(axis=1)
    tail = (100.0 - LEVEL_PERCENT) / 2.0
    return float(np.percentile(resampled, tail)), float(np.percentile(resampled, 100.0 - tail))


def month_interval(*, losses: pl.DataFrame, metric: str, seed: int = 168) -> tuple[float, float]:
    """Return the 99% interval of the mean paired difference, resampling whole calendar months.

    The design is that of `studies.bootstrap`: draw as many months as there are, with replacement,
    and take the mean over the rows of the drawn months. Because both arms hold the same rows, a
    month's total difference is the difference of the arms' month totals, so the draws resample
    12 numbers per arm instead of the rows. The draws are vectorised, which is why this does not
    call `bootstrap_difference_at_level`: its row-by-row loop takes minutes per contrast at this
    row count.
    """
    totals = (
        losses.group_by(["month", "arm"])
        .agg(total=pl.col(metric).sum(), n=pl.len())
        .pivot(on="arm", index="month", values=["total", "n"])
        .sort("month")
    )
    difference = (totals["total_t"] - totals["total_r"]).to_numpy()
    count = totals["n_t"].to_numpy().astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(count), size=(N_RESAMPLES, len(count)))
    resampled = difference[draws].sum(axis=1) / count[draws].sum(axis=1)
    tail = (100.0 - LEVEL_PERCENT) / 2.0
    return float(np.percentile(resampled, tail)), float(np.percentile(resampled, 100.0 - tail))


def held_contrast(
    *,
    treatment: pl.DataFrame,
    reference: pl.DataFrame,
    treatment_every_hours: int,
    labels: dict[int, str],
) -> dict:
    """Return the contrast between the newest forecasts two schedules hold at each request time.

    At every request time `t` (each half-hour from the first origin to the last), the once-a-day
    schedule holds the forecast issued at the newest 09:00 UTC origin at or before `t`, and the
    treatment schedule holds the forecast of the newest origin at or before `t` on a grid of
    `treatment_every_hours` hours. Both forecasts are scored at the same valid times, `t` plus
    0.5 to 24 hours, so the request time, the valid time, the actual, and the hour of the day are
    identical in both arms. A pair is kept when both schedules hold a scored forecast for that
    valid time. The interval resamples calendar months of the request time.
    """
    first = reference["power_fcst_init_time"].min()
    last = reference["power_fcst_init_time"].max() + timedelta(days=1)  # ty: ignore[unsupported-operator]
    grid = pl.datetime_range(first, last, interval="30m", eager=True, time_zone="UTC").alias("t")  # ty: ignore[no-matching-overload]
    request = (
        pl.DataFrame({"t": grid})
        .with_columns(
            origin_reference=(pl.col("t") - pl.duration(hours=9)).dt.truncate("1d")
            + pl.duration(hours=9),
            origin_treatment=pl.col("t").dt.truncate(f"{treatment_every_hours}h"),
        )
        .join(pl.DataFrame({"k": np.arange(1, 49)}), how="cross")
        .with_columns(valid_time=pl.col("t") + pl.duration(minutes=30) * pl.col("k"))
    )
    columns = [
        "time_series_id",
        "valid_time",
        "qs",
        "abs_mean",
        "hour_band",
        "valid_month",
        "cold2",
    ]

    def side(frame: pl.DataFrame, origin: str, suffix: str) -> pl.DataFrame:
        return frame.select(
            "time_series_id",
            pl.col("power_fcst_init_time").alias(origin),
            *[
                pl.col(c).alias(f"{c}_{suffix}") if c in ("qs", "abs_mean") else pl.col(c)
                for c in columns[1:]
            ],
        )

    sites = treatment["time_series_id"].unique().to_list()
    joined = (
        request.join(pl.DataFrame({"time_series_id": sites}), how="cross")
        .join(
            side(treatment, "origin_treatment", "t").select(
                "time_series_id", "origin_treatment", "valid_time", "qs_t", "abs_mean_t"
            ),
            on=["time_series_id", "origin_treatment", "valid_time"],
            how="inner",
        )
        .join(
            side(reference, "origin_reference", "r"),
            on=["time_series_id", "origin_reference", "valid_time"],
            how="inner",
        )
        .with_columns(
            horizon=pl.col("k") / 2.0,
            month=pl.col("t").dt.strftime("%Y-%m"),
            window=pl.col("t").dt.hour().is_between(9, 11),
            site=pl.col("time_series_id").replace_strict(labels, return_dtype=pl.String),
        )
    )

    def one(part: pl.DataFrame, metric: str = "qs") -> dict:
        diff = part[f"{metric}_t"] - part[f"{metric}_r"]
        by_month = (
            part.with_columns(diff=diff)
            .group_by("month")
            .agg(total=pl.col("diff").sum(), n=pl.len())
            .sort("month")
        )
        lower, upper = interval_of_months(
            difference=by_month["total"].to_numpy(), count=by_month["n"].to_numpy()
        )
        return {
            "estimate": float(diff.mean()),  # ty: ignore[invalid-argument-type]
            "lower": lower,
            "upper": upper,
            "rows": part.height,
            "score_treatment": float(part[f"{metric}_t"].mean()),  # ty: ignore[invalid-argument-type]
            "score_reference": float(part[f"{metric}_r"].mean()),  # ty: ignore[invalid-argument-type]
        }

    horizon = pl.col("horizon")
    winter = pl.col("valid_month").is_in(list(WINTER_MONTHS))
    night = pl.col("hour_band").is_in(list(NIGHT_BANDS))
    parts = {
        "0.5-3.5 h": joined.filter(horizon <= 3.5),
        "3.5-9.5 h": joined.filter((horizon > 3.5) & (horizon <= 9.5)),
        "9.5-24 h": joined.filter(horizon > 9.5),
        "request times 09:00-12:00 UTC": joined.filter(pl.col("window")),
        "other request times": joined.filter(~pl.col("window")),
        "winter nights": joined.filter(winter & night),
        "cold nights (day mean below 2 C)": joined.filter(winter & pl.col("cold2") & night),
    }
    out: dict = {
        "all": one(joined),
        "abs_mean": one(joined, metric="abs_mean"),
        "pairs": joined.height,
        "parts": {name: one(part) for name, part in parts.items() if not part.is_empty()},
    }
    by_site = joined.group_by("site").agg(
        (pl.col("qs_t") - pl.col("qs_r")).mean().alias("difference")
    )
    out["by_site"] = {row["site"]: row["difference"] for row in by_site.to_dicts()}
    return out


def contrast(*, losses: pl.DataFrame, metric: str = "qs") -> dict:
    """Return the treatment-minus-reference contrast, its 99% month interval, and its signs."""
    t = losses.filter(pl.col("arm") == "t")
    r = losses.filter(pl.col("arm") == "r")
    estimate = float(t[metric].to_numpy().mean() - r[metric].to_numpy().mean())
    lower, upper = month_interval(losses=losses, metric=metric)
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
        "rows": t.height,
        "sites_worse": int((by_site["t"] > by_site["r"]).sum()),
        "n_sites": by_site.height,
        "months_worse": int((by_month["t"] > by_month["r"]).sum()),
        "n_months": by_month.height,
    }


def contrast_table(*, losses: pl.DataFrame, full: bool, subsets: bool = True) -> dict:
    """Return one pairing's contrasts by lead band, by origin, and by subset."""
    out: dict = {"bands": {}, "by_slot_under_24": {}, "subsets": {}, "by_site": {}}
    for name, low, high in LEAD_BANDS:
        if not full and name not in ("under 24 h", "all (to 147 h)"):
            continue
        part = lead_filter(frame=losses, low=low, high=high)
        out["bands"][name] = {"qs": contrast(losses=part)}
        if name in ("under 24 h", "all (to 147 h)"):
            out["bands"][name]["abs_mean"] = contrast(losses=part, metric="abs_mean")
    if not full:
        return out
    under = lead_filter(frame=losses, low=0.0, high=24.0)
    for slot, label in enumerate(SLOT_LABELS):
        part = under.filter(pl.col("slot") == slot)
        if not part.is_empty():
            out["by_slot_under_24"][label] = contrast(losses=part)
    for name, expression in subset_filters().items():
        part = under.filter(expression)
        if subsets and not part.is_empty():
            out["subsets"][name] = contrast(losses=part)
    for site in common.DECISION_SITES:
        part = under.filter(pl.col("site") == site)
        t = part.filter(pl.col("arm") == "t")["qs"].to_numpy().mean()
        r = part.filter(pl.col("arm") == "r")["qs"].to_numpy().mean()
        out["by_site"][site] = float(t - r)
    return out


def growth_check(*, losses: pl.DataFrame, seed: int = 168) -> dict:
    """Return the lead check: growth from 0.5-3.5 h to 3.5-9.5 h, treatment minus reference.

    Months are resampled with replacement, as in experiment 155's lead check.
    """
    scope = losses.with_columns(
        band=pl.when((pl.col("lead_hours") > 0) & (pl.col("lead_hours") <= 3.5))
        .then(0)
        .when((pl.col("lead_hours") > 3.5) & (pl.col("lead_hours") <= 9.5))
        .then(1)
        .otherwise(-1)
    ).filter(pl.col("band") >= 0)
    grouped = scope.group_by(["arm", "month", "band"]).agg(total=pl.col("qs").sum(), n=pl.len())
    months = sorted(scope["month"].unique().to_list())
    table = {
        (row["arm"], row["month"], row["band"]): (row["total"], row["n"])
        for row in grouped.to_dicts()
    }
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(months), size=(N_RESAMPLES, len(months)))

    def growth(arm: str, picks: np.ndarray) -> np.ndarray:
        stat = []
        for band in (0, 1):
            total = np.array([table[(arm, months[m], band)][0] for m in range(len(months))])
            count = np.array([table[(arm, months[m], band)][1] for m in range(len(months))])
            stat.append(total[picks].sum(axis=1) / count[picks].sum(axis=1))
        return stat[1] - stat[0]

    everything = np.arange(len(months))[None, :]
    point_t, point_r = growth("t", everything)[0], growth("r", everything)[0]
    diff = growth("t", draws) - growth("r", draws)
    lower, upper = np.percentile(diff, [(100 - LEVEL_PERCENT) / 2, 100 - (100 - LEVEL_PERCENT) / 2])
    return {
        "growth_treatment": float(point_t),
        "growth_reference": float(point_r),
        "difference": float(point_t - point_r),
        "lower": float(lower),
        "upper": float(upper),
        "passes": bool(upper <= 0.02),
    }


def score_tables(*, losses: dict[str, pl.DataFrame]) -> dict:
    """Return each arm's mean score by lead band, by origin, by site, and by hour of lead."""
    out: dict = {"by_lead": {}, "by_slot_under_24": {}, "by_site": {}, "by_lead_hour": {}}
    decision = {
        arm: frame.filter(pl.col("site").is_in(list(common.DECISION_SITES)))
        for arm, frame in losses.items()
    }
    for arm, frame in decision.items():
        out["by_lead"][arm] = {
            name: float(lead_filter(frame=frame, low=low, high=high)["qs"].mean())  # ty: ignore[invalid-argument-type]
            for name, low, high in LEAD_BANDS
        }
        under = lead_filter(frame=frame, low=0.0, high=24.0)
        out["by_slot_under_24"][arm] = {
            label: float(under.filter(pl.col("slot") == slot)["qs"].mean())  # ty: ignore[invalid-argument-type]
            for slot, label in enumerate(SLOT_LABELS)
            if arm in PRODUCTION_ARMS
        }
        out["by_site"][arm] = {
            site: lead_filter(
                frame=losses[arm].filter(pl.col("site") == site), low=0.0, high=ALL_LEAD_LIMIT_HOURS
            )["qs"].mean()
            for site in common.SCORED_SITES
        }
        by_hour = (
            lead_filter(frame=frame, low=0.0, high=165.0)
            .with_columns(lead_bin=((pl.col("lead_hours") - 0.001) // 6.0 * 6.0 + 3.0))
            .group_by("lead_bin")
            .agg(qs=pl.col("qs").mean(), n=pl.len())
            .sort("lead_bin")
        )
        out["by_lead_hour"][arm] = by_hour.to_dicts()
        out.setdefault("by_valid_hour_under_24", {})[arm] = (
            lead_filter(frame=frame, low=0.0, high=24.0)
            .group_by(pl.col("valid_time").dt.hour().alias("hour"))
            .agg(qs=pl.col("qs").mean())
            .sort("hour")
            .to_dicts()
        )
    return out


def tail_tables(*, losses: dict[str, pl.DataFrame]) -> dict:
    """Return coverage and pinball loss at the four tail quantiles for the main arms."""
    aggregations = [pl.col(f"cov_{n}").mean() for n, _ in TAIL_QUANTILES]
    aggregations += [pl.col(f"pin_{n}").mean() for n, _ in TAIL_QUANTILES]
    out: dict = {}
    for scope_name, selector in (
        ("12-36 h", lambda f: lead_filter(frame=f, low=12.0, high=36.0)),
        (
            "all leads (to 147 h)",
            lambda f: lead_filter(frame=f, low=0.0, high=ALL_LEAD_LIMIT_HOURS),
        ),
        (
            "cold nights",
            lambda f: lead_filter(frame=f, low=0.0, high=ALL_LEAD_LIMIT_HOURS).filter(
                subset_filters()["cold nights (day mean below 2 C)"]
            ),
        ),
    ):
        rows = []
        for arm in ("a", "d", "ap", "dp", "bg2"):
            if arm not in losses:
                continue
            frame = selector(losses[arm].filter(pl.col("site").is_in(list(common.DECISION_SITES))))
            rows.append(
                {"arm": arm, "rows": frame.height, **frame.select(aggregations).to_dicts()[0]}
            )
        out[scope_name] = rows
    return out


def checks(*, frames: dict[str, pl.DataFrame]) -> dict:
    """Return the median minus mean forecast per site and arm at 12 to 36 hours."""
    out = []
    for name in ("a", "d", "ap", "dp"):
        part = lead_filter(frame=frames[name], low=12.0, high=36.0)
        grouped = part.group_by("site").agg(
            (pl.col("q50") - pl.col("mean")).mean().alias("median_minus_mean")
        )
        out.extend({"arm": name, **row} for row in grouped.sort("site").to_dicts())
    return {"median_minus_mean": out}


def period_report(*, period: str, cache) -> dict:  # noqa: ANN001
    """Return the period's tables and contrasts."""
    labels = {cache.sites[s].table.time_series_id: s for s in common.SCORED_SITES}
    frames = {
        arm: read_arm(arm=arm, period=period, cache=cache)
        for arm in ("a", "d", *PRODUCTION_ARMS, "ai0", "di0")
    }
    if period == "scored":
        for name in STORED_87:
            frames[name] = read_stored_87(name=name, cache=cache)
    losses_by_arm = {arm: with_losses(frame=frame) for arm, frame in frames.items()}
    decision = {
        arm: frame.filter(pl.col("site").is_in(list(common.DECISION_SITES)))
        for arm, frame in frames.items()
    }
    report: dict = {
        "rows": {arm: frame.height for arm, frame in decision.items()},
        "tables": score_tables(losses=losses_by_arm),
        "contrasts": {},
    }
    same_lead = ["time_series_id", "run", "lead_hours"]
    same_valid = ["time_series_id", "run", "valid_time"]
    same_row = ["time_series_id", "run", "slot", "valid_time"]
    pairings = {
        "C1": ("dp", "d", same_lead, "dp minus d, same lead (the headline)"),
        "C2": ("ap", "a", same_lead, "ap minus a, same lead"),
        "C3": ("dp", "dp0", same_row, "dp minus dp0, same row (the 3.5-hour delay)"),
        "C3A": ("ap", "ap0", same_row, "ap minus ap0, same row (the 3.5-hour delay)"),
        "C4": ("dp", "d", same_valid, "dp minus d, same valid time (the later origin)"),
        "C5": ("dp", "ap", same_row, "dp minus ap, same row (the stage 2 corrections)"),
        "C6": ("dp0", "d", same_lead, "dp0 minus d, same lead (older weather alone)"),
        "C7": (
            "d",
            "di0",
            same_valid,
            "d minus di0, same row (the 3.5-hour delay at the 09:00 origin)",
        ),
        "C7A": (
            "a",
            "ai0",
            same_valid,
            "a minus ai0, same row (the 3.5-hour delay at the 09:00 origin)",
        ),
    }
    for label, (treatment, reference, keys, description) in pairings.items():
        t, r = pair(treatment=decision[treatment], reference=decision[reference], on=keys)
        losses = stack(t=t, r=r, labels=labels)
        entry = contrast_table(losses=losses, full=period == "scored", subsets=keys != same_lead)
        entry["description"] = description
        entry["rows"] = t.height
        if label in ("C1", "C2") and period == "scored":
            entry["lead_check"] = growth_check(losses=losses)
        report["contrasts"][label] = entry
    report["held"] = {
        "dp_minus_d": held_contrast(
            treatment=losses_by_arm["dp"].filter(pl.col("site").is_in(list(common.DECISION_SITES))),
            reference=losses_by_arm["d"].filter(pl.col("site").is_in(list(common.DECISION_SITES))),
            treatment_every_hours=6,
            labels=labels,
        ),
        "ap_minus_a": held_contrast(
            treatment=losses_by_arm["ap"].filter(pl.col("site").is_in(list(common.DECISION_SITES))),
            reference=losses_by_arm["a"].filter(pl.col("site").is_in(list(common.DECISION_SITES))),
            treatment_every_hours=6,
            labels=labels,
        ),
        "dp0_minus_di0": held_contrast(
            treatment=losses_by_arm["dp0"].filter(
                pl.col("site").is_in(list(common.DECISION_SITES))
            ),
            reference=losses_by_arm["di0"].filter(
                pl.col("site").is_in(list(common.DECISION_SITES))
            ),
            treatment_every_hours=6,
            labels=labels,
        ),
    }
    if period == "scored":
        report["tail"] = tail_tables(losses=losses_by_arm)
        report["checks"] = checks(frames=decision)
    return report


def main() -> None:
    """Score the scored year and the replicate months and write the report."""
    cache = common.load_cache()
    report: dict = {period: period_report(period=period, cache=cache) for period in PERIODS}
    controls = [
        json.loads(path.read_text())
        for path in sorted((common.WORK_DIR / "control").glob("*.json"))
    ]
    report["controls"] = controls
    common.write_json_atomic(path=common.WORK_DIR / "report" / "report.json", content=report)
    print("report written")


if __name__ == "__main__":
    main()
