"""Score experiment 158's arms and print every number the write-up quotes.

Reads `work/dp_preds/`, experiment 136's stored arms A0 and A2, and experiment 87's stored BG2,
climatology, and manual heuristic. It puts every arm on common rows (a reading, outside the
hand-filtered switching events, lead of at most 168 hours), and writes `work/report/report.json`
and `work/report/report.txt`: each arm's quantile score by lead band and breakout, the nine planned
contrasts with 99% month-resampled intervals, the Christmas signs by substation, the tail table, and
the checks. The arms of the pooled fits are named in `work/tuned.json` by `e00158_tune.py`.

    uv run python ml-experiments/00158-*/e00158_report.py
"""

import json
from datetime import UTC, datetime
from typing import Final

import e00158_common as common
import e00158_run as run
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.dp_calendar import BAND_NAMES
from studies.dp_gam import QUANTILE_COLUMNS
from studies.experiment_store import losses_for_bootstrap
from studies.sources import ML_EXPERIMENTS_DATA_DIR

LEVELS: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)
LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("under 24 h", 0.0, 24.0),
    ("24-72 h", 24.0, 72.0),
    ("72-168 h", 72.0, 168.0),
    ("D+1 (15.5-39 h)", 15.5, 39.0),
    ("all", 0.0, 168.0),
)
STORED_136: Final[dict[str, str]] = {"e136_a0": "a0", "e136_a2": "a2"}
STORED_87: Final[dict[str, str]] = {
    "bg2": "p_s6_bg2",
    "climatology": "ref_climatology",
    "manual_heuristic": "ref_manual_heuristic",
}
E136_PREDICTIONS: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00136-dp-bg-stage1-assembly-on-the-ensemble-mean"
    / "work"
    / "dp_preds"
)
E87_PREDICTIONS: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00087-bg2-best-guess-assembled-with-drop-one-ablations"
    / "predictions"
)
LEVEL_PERCENT: Final[float] = 99.0
N_RESAMPLES: Final[int] = 20_000
COLD_TEMPERATURE_C: Final[float] = 3.0
WINTER_MONTHS: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})
SEASONS: Final[dict[str, frozenset[int]]] = {
    "winter (DJF)": frozenset({12, 1, 2}),
    "spring (MAM)": frozenset({3, 4, 5}),
    "summer (JJA)": frozenset({6, 7, 8}),
    "autumn (SON)": frozenset({9, 10, 11}),
}
NIGHT_BANDS: Final[tuple[int, ...]] = (BAND_NAMES.index("night"), 3)
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
TAIL_QUANTILES: Final[tuple[tuple[str, float], ...]] = (
    ("q01", 0.01),
    ("q05", 0.05),
    ("q95", 0.95),
    ("q99", 0.99),
)
BASE_COLUMNS: Final[list[str]] = [
    "time_series_id",
    "power_fcst_init_time",
    "valid_time",
    "lead_hours",
    "seed",
    *QUANTILE_COLUMNS,
    "actual",
    "mean",
    "site",
]
TUNED_PATH: Final = common.WORK_DIR / "tuned.json"


def tuned() -> dict:
    """Return the chosen pooled arms: `{"full": {"ph": ..., "ps": ...}, "3m": {...}}`."""
    return json.loads(TUNED_PATH.read_text())


def contrasts() -> dict[str, tuple[str, str, str]]:
    """Return the planned contrasts: label to (treatment, reference, row subset)."""
    chosen = tuned()
    ph, ps = f"full_{chosen['full']['ph']}", f"full_{chosen['full']['ps']}"
    ph3, ps3 = f"3m_{chosen['3m']['ph']}", f"3m_{chosen['3m']['ps']}"
    return {
        "C1": (ph, "full_k0", "all"),
        "C2": (ps, "full_k0", "all"),
        "C3": (ps, ph, "all"),
        "C4": (ps3, "3m_k0", "all"),
        "C5": (ph3, "3m_k0", "all"),
        "C6": (ps3, "full_k0", "all"),
        "C7": (ps, "full_k0", "holiday"),
        "C8": (ps, "full_k0", "ordinary"),
        "C9": (ps, "full_k0", "christmas"),
    }


def arm_names() -> list[str]:
    """Return the arms with stored predictions, in a stable order."""
    names = {p.name.split("_scored_")[0] for p in run.OUT_DIR.glob("*_scored_*.parquet")}
    return sorted(names)


def annotate(*, frame: pl.DataFrame, cache) -> pl.DataFrame:  # noqa: ANN001
    """Keep scorable rows (a reading, not switching, lead of at most 168 hours); add breakouts."""
    last = cache.end_times[-1].astype("datetime64[us]").astype(datetime).replace(tzinfo=UTC)
    pieces = []
    for site in common.FLEET:
        part = frame.filter(pl.col("site") == site)
        if part.is_empty():
            continue
        features = cache.sites[site]
        target = np.searchsorted(
            cache.end_times, part["valid_time"].dt.replace_time_zone(None).to_numpy()
        )
        holiday = cache.calendar.holiday_period[target] >= 0
        christmas = cache.calendar.tent[target] > 0
        part = part.with_columns(
            masked=pl.Series(features.switching_mask[target]),
            hour_band=pl.Series(cache.calendar.band[target].astype(np.int8)),
            holiday=pl.Series(holiday | christmas),
            christmas=pl.Series(christmas),
            cold=pl.Series(
                np.nan_to_num(features.temperature_c[target], nan=99.0) <= COLD_TEMPERATURE_C
            ),
            valid_month=pl.col("valid_time").dt.month(),
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
    origins = [
        common.origin_of(run=r) for r in common.run_dates() if common.period_of_run(run=r) == period
    ]
    return frame.filter(pl.col("power_fcst_init_time").is_in(origins))


def load(*, arm: str, period: str) -> pl.DataFrame | None:
    """Return an arm's scorable rows for a period, or `None` if the arm has none."""
    cache = common.load_cache()
    ids = {s: cache.sites[s].table.time_series_id for s in common.FLEET}
    if arm in STORED_87:
        if period != "scored":
            return None
        label_of = {series: site for site, series in ids.items()}
        frame = pl.read_parquet(E87_PREDICTIONS / STORED_87[arm] / "predictions.parquet").filter(
            pl.col("time_series_id").is_in(list(label_of))
        )
        levels = list(QUANTILE_COLUMNS)
        frame = frame.with_columns(
            site=pl.col("time_series_id").replace_strict(label_of, return_dtype=pl.String),
            actual=100.0 * pl.col("actual"),
            **{q: 100.0 * pl.col(q) for q in levels},
        ).with_columns(mean=pl.col("q50"))
        frame = in_period(frame=frame.select(BASE_COLUMNS), period="scored")
        return annotate(frame=frame, cache=cache)
    pieces = []
    for site in common.FLEET:
        if arm in STORED_136:
            path = E136_PREDICTIONS / f"{STORED_136[arm]}_{period}_{site}.parquet"
        else:
            path = run.out_path(arm=arm, period=period, site=site)
        if path.exists():
            pieces.append(pl.read_parquet(path).with_columns(site=pl.lit(site)))
    if not pieces:
        return None
    frame = pl.concat(pieces).select(BASE_COLUMNS)
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
        loss = np.maximum(
            level * (frame["actual"].to_numpy() - value),
            (level - 1.0) * (frame["actual"].to_numpy() - value),
        )
        extra[f"pin_{name}"] = loss
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
        "reference_score": float(r[metric].to_numpy().mean()),
        "sites_better": int((by_site["t"] < by_site["r"]).sum()),
        "n_sites": by_site.height,
        "months_negative": int((by_month["t"] < by_month["r"]).sum()),
        "n_months": by_month.height,
        "n_rows": t.height,
    }


def subset_filters() -> dict[str, pl.Expr]:
    """Return the breakout subsets, each as a row filter."""
    night = pl.col("hour_band").is_in(list(NIGHT_BANDS))
    winter = pl.col("valid_month").is_in(list(WINTER_MONTHS))
    return {
        "all": pl.lit(True),
        "winter": winter,
        "cold nights": pl.col("cold") & night,
        "holiday": pl.col("holiday"),
        "christmas": pl.col("christmas"),
        "ordinary": ~pl.col("holiday"),
        **{name: pl.col("valid_month").is_in(list(months)) for name, months in SEASONS.items()},
    }


def arm_table(*, losses: pl.DataFrame, sites: tuple[str, ...]) -> dict:
    """Return each arm's mean score by lead band, by site over all leads, and by breakout."""
    out: dict = {"by_lead": {}, "by_site": {}, "breakouts": {}}
    scope = losses.filter(pl.col("site").is_in(list(sites)))
    for name, low, high in LEAD_BANDS:
        part = lead_filter(frame=scope, low=low, high=high)
        out["by_lead"][name] = part.group_by("arm").agg(pl.col("qs").mean()).sort("arm").to_dicts()
    out["by_site"] = (
        scope.group_by(["arm", "site"]).agg(pl.col("qs").mean()).sort(["arm", "site"]).to_dicts()
    )
    for label, expression in subset_filters().items():
        part = scope.filter(expression)
        out["breakouts"][label] = {
            "rows": part.filter(pl.col("arm") == part["arm"][0]).height,
            "scores": part.group_by("arm").agg(pl.col("qs").mean()).sort("arm").to_dicts(),
        }
    return out


def tail_table(*, losses: pl.DataFrame, sites: tuple[str, ...]) -> dict:
    """Return coverage and pinball loss at the four tail quantiles, overall and on cold nights."""
    scope = losses.filter(pl.col("site").is_in(list(sites)))
    out: dict = {}
    for label, part in (
        ("all leads", scope),
        ("cold nights all leads", scope.filter(subset_filters()["cold nights"])),
        ("christmas all leads", scope.filter(subset_filters()["christmas"])),
    ):
        aggregations = [pl.col(f"cov_{n}").mean() for n, _ in TAIL_QUANTILES]
        aggregations += [pl.col(f"pin_{n}").mean() for n, _ in TAIL_QUANTILES]
        out[label] = part.group_by("arm").agg(aggregations).sort("arm").to_dicts()
    return out


def checks(*, frames: dict[str, pl.DataFrame]) -> dict:
    """Return the median minus mean forecast per site and arm, at 12 to 36 hours."""
    out = []
    for name, frame in frames.items():
        part = lead_filter(frame=frame, low=12.0, high=36.0)
        grouped = part.group_by("site").agg(
            (pl.col("q50") - pl.col("mean")).mean().alias("median_minus_mean")
        )
        out.extend({"arm": name, **row} for row in grouped.sort("site").to_dicts())
    return {"median_minus_mean": out}


def stack(*, frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return the bootstrap loss frame of arms already on common rows, with D-labels as `site`."""
    cache = common.load_cache()
    labels = {cache.sites[s].table.time_series_id: s for s in common.FLEET}
    keep = [*KEYS, "lead_hours", "seed", *QUANTILE_COLUMNS, "actual", "mean"]
    extra = ["hour_band", "holiday", "christmas", "cold", "valid_month"]
    stacked = losses_for_bootstrap(
        arm_losses={
            name: with_losses(frame=f.select([*keep, *extra])) for name, f in frames.items()
        }
    )
    return stacked.with_columns(site=pl.col("site").replace_strict(labels, return_dtype=pl.String))


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


def event_signs(*, losses: pl.DataFrame, treatment: str, reference: str) -> dict[str, float]:
    """Return each decision site's holiday and Christmas contrasts, treatment minus reference."""
    scope = losses.filter(pl.col("site").is_in(list(common.DECISION_SITES)))
    out = {}
    for label in ("holiday", "christmas"):
        part = scope.filter(subset_filters()[label])
        for site in common.DECISION_SITES:
            rows = part.filter(pl.col("site") == site)
            t = rows.filter(pl.col("arm") == treatment)["qs"].to_numpy()
            r = rows.filter(pl.col("arm") == reference)["qs"].to_numpy()
            out[f"{label}:{site}"] = float(t.mean() - r.mean()) if len(t) else float("nan")
    return out


def period_report(*, period: str) -> tuple[dict, dict[str, pl.DataFrame]]:
    """Return the period's tables and contrasts, and the common-row frames."""
    wanted = [*arm_names(), *STORED_136, *STORED_87]
    loaded = {arm: frame for arm in wanted if (frame := load(arm=arm, period=period)) is not None}
    frames = common_rows(frames=loaded)
    losses = stack(frames=frames)
    decision = losses.filter(pl.col("site").is_in(list(common.DECISION_SITES)))
    report: dict = {
        "arms": list(frames),
        "rows": frames[next(iter(frames))].height,
        "tables": arm_table(losses=losses, sites=common.DECISION_SITES),
        "tables_with_d9": arm_table(losses=losses, sites=common.FLEET),
        "tail": tail_table(losses=losses, sites=common.DECISION_SITES),
        "contrasts": {},
        "leave_one_out": {},
        "event_signs": {},
    }
    for label, (treatment, reference, subset) in contrasts().items():
        if treatment not in frames or reference not in frames:
            continue
        report["leave_one_out"][label] = leave_one_out(
            losses=losses, treatment=treatment, reference=reference
        )
        report["event_signs"][label] = event_signs(
            losses=losses, treatment=treatment, reference=reference
        )
        rows = decision.filter(subset_filters()[subset])
        entry: dict = {"arms": [treatment, reference], "subset": subset}
        for scope_name, low, high in LEAD_BANDS:
            part = lead_filter(frame=rows, low=low, high=high)
            entry[scope_name] = {
                "qs": contrast(losses=part, treatment=treatment, reference=reference),
                "abs_mean": contrast(
                    losses=part, treatment=treatment, reference=reference, metric="abs_mean"
                ),
            }
        for sub_name in ("winter", "cold nights", *SEASONS):
            part = rows.filter(subset_filters()[sub_name])
            if part.height and part["arm"].n_unique() == 2:
                entry[sub_name] = {
                    "qs": contrast(losses=part, treatment=treatment, reference=reference)
                }
        report["contrasts"][label] = entry
    return report, frames


def main() -> None:
    """Score the three periods and write the report."""
    report: dict = {"tuned": tuned()}
    for period in ("scored", "replicate"):
        period_data, frames = period_report(period=period)
        report[period] = period_data
        if period == "scored":
            report["checks"] = checks(frames=frames)
    target = common.WORK_DIR / "report" / "report.json"
    common.write_json_atomic(path=target, content=report)
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
