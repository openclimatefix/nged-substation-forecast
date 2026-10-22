"""Score the combined calibration fixes and the ensemble-spread model; print every number quoted.

Reads `work/dp_preds/` and the stored predictions of experiment 87 (BG2, climatology, manual
heuristic), puts every arm on common rows (a reading, outside the hand-filtered switching events,
lead of at most 168 hours), and writes `work/report/report.json`: the diagnosis by temperature bin
and hour band, the tail and coverage tables, the planned contrasts with month-resampled intervals,
the pre-registered decision, the checks, and the curtailment-cost proxy.

    uv run --no-sync python ml-experiments/00169-*/e00169_report.py
"""

import datetime as dt
from typing import Final

import e00169_common as common
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level, bootstrap_month_totals
from studies.dp_calendar import BAND_NAMES, bank_holiday_names
from studies.dp_gam import QUANTILE_COLUMNS
from studies.experiment_store import losses_for_bootstrap
from studies.node_curtailment import (
    NodeCurtailmentAssumptions,
    NodeTable,
    avoidable_cost,
    month_total_arrays,
    node_curtailment,
    scale_to_megawatts,
    scaled_limits,
    table_with_scaled_limits,
)
from studies.sources import ML_EXPERIMENTS_DATA_DIR

KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
OURS: Final[tuple[str, ...]] = common.ARMS
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
    "C1": ("a", "ref"),
    "C2": ("sp", "a"),
    "C3": ("sp", "ref"),
    "C4": ("at", "a"),
    "C5": ("sp", "sp0"),
    "C6": ("sp0", "a"),
}
LEVEL_PERCENT: Final[float] = 95.0
"""The 95% level; C1 and C2 are the two planned contrasts, so no correction."""
REFERENCE_SCORE: Final[float] = 2.7529
"""Experiment 160's printed arm I score over leads 0.5 to 168 hours in the scored year."""
ARM_A_SCORE: Final[float] = 2.7519
"""Experiment 164's printed fix O score over the same rows."""
CHRISTMAS_START: Final[tuple[int, int]] = (12, 24)
CHRISTMAS_END: Final[tuple[int, int]] = (1, 1)
PASS_COVERAGE: Final[float] = 0.90
TEMPERATURE_BIN_EDGES_C: Final[tuple[float, ...]] = (0.0, 2.0, 5.0, 8.0, 12.0)
N_RESAMPLES: Final[int] = 20_000
PASS_UPPER_BOUND: Final[float] = 0.01
COLD_DAY_C: Final[float] = 2.0
WINTER_MONTHS: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})
NIGHT_BANDS: Final[tuple[int, ...]] = (BAND_NAMES.index("night"), BAND_NAMES.index("evening"))
TAIL_QUANTILES: Final[tuple[tuple[str, float], ...]] = (
    ("q01", 0.01),
    ("q05", 0.05),
    ("q95", 0.95),
    ("q99", 0.99),
)
LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("under 24 h", 0.0, 24.0),
    ("24-72 h", 24.0, 72.0),
    ("72-168 h", 72.0, 168.0),
    ("12-36 h", 12.0, 36.0),
    ("all", 0.0, 168.0),
)
BASE_COLUMNS: Final[list[str]] = [
    "time_series_id", "power_fcst_init_time", "valid_time", "lead_hours", "seed",
    *QUANTILE_COLUMNS, "actual", "mean", "site", "ens_mean_c", "ens_std_c", "ens_sw_std",
]  # fmt: skip
CURTAILMENT_FRACTIONS: Final[tuple[float, ...]] = (0.9, 0.75, 0.5)
CURTAILMENT_PRICE: Final[float] = 2000.0
SCORED_START: Final[dt.datetime] = dt.datetime(2025, 7, 2, tzinfo=dt.UTC)


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
        pieces.append(
            part.with_columns(
                masked=pl.Series(features.switching_mask[target]),
                hour_band=pl.Series(cache.calendar.band[target].astype(np.int8)),
                cold2=pl.Series(np.nan_to_num(daily[target], nan=99.0) < COLD_DAY_C),
                valid_month=pl.col("valid_time").dt.month(),
            )
        )
    bank = list(bank_holiday_names())
    local_date = pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.date()
    return (
        pl.concat(pieces)
        .with_columns(
            bank_holiday=local_date.is_in(bank),
            christmas=(
                (
                    (local_date.dt.month() == CHRISTMAS_START[0])
                    & (local_date.dt.day() >= CHRISTMAS_START[1])
                )
                | (
                    (local_date.dt.month() == CHRISTMAS_END[0])
                    & (local_date.dt.day() <= CHRISTMAS_END[1])
                )
            ),
        )
        .filter(~pl.col("masked") & pl.col("actual").is_finite() & (pl.col("lead_hours") <= 168.0))
    )


def load_ours(*, arm: str, period: str, cache) -> pl.DataFrame:  # noqa: ANN001
    """Return an arm's scorable rows for a period."""
    pieces = [
        pl.read_parquet(common.out_path(arm=arm, period=period, site=site)).with_columns(
            site=pl.lit(site)
        )
        for site in common.SCORED_SITES
    ]
    return annotate(frame=pl.concat(pieces).select(BASE_COLUMNS), cache=cache)


def load_stored(*, name: str, origins: list, cache) -> pl.DataFrame:  # noqa: ANN001
    """Return a stored experiment 87 arm's scorable rows for the scored year."""
    ids = {s: cache.sites[s].table.time_series_id for s in common.SCORED_SITES}
    label_of = {series: site for site, series in ids.items()}
    frame = pl.read_parquet(E87_PREDICTIONS / STORED_87[name] / "predictions.parquet").filter(
        pl.col("time_series_id").is_in(list(label_of))
    )
    frame = frame.with_columns(
        site=pl.col("time_series_id").replace_strict(label_of, return_dtype=pl.String),
        actual=100.0 * pl.col("actual"),
        **{q: 100.0 * pl.col(q) for q in QUANTILE_COLUMNS},
    ).with_columns(
        mean=pl.col("q50"),
        ens_mean_c=pl.lit(None, dtype=pl.Float64),
        ens_std_c=pl.lit(None, dtype=pl.Float64),
        ens_sw_std=pl.lit(None, dtype=pl.Float64),
    )
    frame = frame.select(BASE_COLUMNS).filter(pl.col("power_fcst_init_time").is_in(origins))
    return annotate(frame=frame, cache=cache)


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
        "ens_mean_c": pl.Float64,
        "ens_std_c": pl.Float64,
        "ens_sw_std": pl.Float64,
    }
    return {
        name: frame.join(shared, on=KEYS, how="inner")
        .sort(KEYS)
        .with_columns(pl.col(column).cast(dtype) for column, dtype in schema.items())
        for name, frame in frames.items()
    }


def with_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the quantile score, the width, the mean absolute error, and the tail flags."""
    actual = frame["actual"].to_numpy().astype(np.float64)
    quantiles = frame.select(list(QUANTILE_COLUMNS)).to_numpy().astype(np.float64)
    error = actual[:, None] - quantiles
    score = 2.0 * np.mean(
        np.maximum(common.LEVELS[None, :] * error, (common.LEVELS[None, :] - 1.0) * error), axis=1
    )
    extra = {
        "qs": score,
        "width": quantiles[:, 9] - quantiles[:, 1],
        "inside": ((actual >= quantiles[:, 1]) & (actual <= quantiles[:, 9])).astype(np.float64),
        "abs_mean": np.abs(actual - frame["mean"].to_numpy()),
        "lead_day": np.clip(np.ceil(frame["lead_hours"].to_numpy() / 24.0) - 1, 0, 6).astype(
            np.int64
        ),
    }
    for name, level in TAIL_QUANTILES:
        value = frame[name].to_numpy().astype(np.float64)
        diff = actual - value
        extra[f"pin_{name}"] = np.maximum(level * diff, (level - 1.0) * diff)
        extra[f"cov_{name}"] = (actual <= value).astype(np.float64)
    return frame.with_columns([pl.Series(k, v) for k, v in extra.items()])


def stack(*, frames: dict[str, pl.DataFrame], cache) -> pl.DataFrame:  # noqa: ANN001
    """Return the bootstrap loss frame of arms already on common rows, with D-labels as `site`."""
    labels = {cache.sites[s].table.time_series_id: s for s in common.SCORED_SITES}
    keep = [
        *KEYS, "lead_hours", "seed", *QUANTILE_COLUMNS, "actual", "mean",
        "hour_band", "cold2", "valid_month", "ens_mean_c", "ens_std_c", "ens_sw_std",
        "bank_holiday", "christmas",
    ]  # fmt: skip
    stacked = losses_for_bootstrap(
        arm_losses={name: with_losses(frame=f.select(keep)) for name, f in frames.items()}
    )
    return stacked.with_columns(site=pl.col("site").replace_strict(labels, return_dtype=pl.String))


def lead_filter(*, frame: pl.DataFrame, low: float, high: float) -> pl.DataFrame:
    """Keep rows with a lead above `low` and at most `high` hours."""
    return frame.filter((pl.col("lead_hours") > low) & (pl.col("lead_hours") <= high))


def subsets() -> dict[str, pl.Expr]:
    """Return the breakout subsets, each as a row filter."""
    night = pl.col("hour_band").is_in(list(NIGHT_BANDS))
    winter = pl.col("valid_month").is_in(list(WINTER_MONTHS))
    return {
        "winter nights": winter & night,
        "cold nights (day mean below 2 C)": winter & pl.col("cold2") & night,
        "bank holidays": pl.col("bank_holiday"),
        "Christmas (24 Dec to 1 Jan)": pl.col("christmas"),
    }


COLD_KEY: Final[str] = "cold nights (day mean below 2 C)"


def contrast(*, losses: pl.DataFrame, treatment: str, reference: str, metric: str = "qs") -> dict:
    """Return a paired contrast with a 95% month-resampled interval."""
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
    by_month = (
        t.group_by("month")
        .agg(t=pl.col(metric).mean())
        .join(r.group_by("month").agg(r=pl.col(metric).mean()), on="month")
    )
    return {
        "estimate": estimate,
        "lower": float(lower),
        "upper": float(upper),
        "months_negative": int((by_month["t"] < by_month["r"]).sum()),
        "n_months": by_month.height,
    }


def width_by_lead_day(*, losses: pl.DataFrame) -> list[dict]:
    """Return, per arm, the decision substations' median q05 to q95 width and coverage by day."""
    scope = losses.filter(pl.col("site").is_in(list(common.DECISION_SITES)))
    return (
        scope.group_by(["arm", "lead_day"])
        .agg(
            median_width=pl.col("width").median(),
            cov_q95=pl.col("cov_q95").mean(),
            inside=pl.col("inside").mean(),
        )
        .sort(["arm", "lead_day"])
        .to_dicts()
    )


def score_tables(*, losses: pl.DataFrame) -> dict:
    """Return each arm's absolute score and tail tables."""
    scope = losses.filter(pl.col("site").is_in(list(common.DECISION_SITES)))
    aggregations = [pl.col(f"cov_{n}").mean() for n, _ in TAIL_QUANTILES]
    aggregations += [pl.col(f"pin_{n}").mean() for n, _ in TAIL_QUANTILES]
    out: dict = {"by_lead": {}, "tail": {}, "by_site": {}, "by_hour_band_tail": {}}
    for name, low, high in LEAD_BANDS:
        part = lead_filter(frame=scope, low=low, high=high)
        out["by_lead"][name] = part.group_by("arm").agg(pl.col("qs").mean()).sort("arm").to_dicts()
    out["by_site"] = (
        scope.group_by(["arm", "site"]).agg(pl.col("qs").mean()).sort(["arm", "site"]).to_dicts()
    )
    out["by_lead_day_qs"] = (
        scope.group_by(["arm", "lead_day"])
        .agg(pl.col("qs").mean())
        .sort(["arm", "lead_day"])
        .to_dicts()
    )
    parts = {
        "all leads": scope,
        "12-36 h": lead_filter(frame=scope, low=12.0, high=36.0),
        COLD_KEY: scope.filter(subsets()[COLD_KEY]),
        "winter nights": scope.filter(subsets()["winter nights"]),
        "bank holidays": scope.filter(subsets()["bank holidays"]),
        "Christmas (24 Dec to 1 Jan)": scope.filter(subsets()["Christmas (24 Dec to 1 Jan)"]),
    }
    for label, part in parts.items():
        out["tail"][label] = part.group_by("arm").agg(aggregations).sort("arm").to_dicts()
        out["tail"][label + " rows"] = int(part.filter(pl.col("arm") == part["arm"][0]).height)
    out["by_hour_band_tail"] = (
        scope.group_by(["arm", "hour_band"]).agg(aggregations).sort(["arm", "hour_band"]).to_dicts()
    )
    return out


def contrast_tables(*, losses: pl.DataFrame) -> dict:
    """Return the planned contrasts with intervals."""
    decision = losses.filter(pl.col("site").is_in(list(common.DECISION_SITES)))
    out: dict = {}
    for label, (treatment, reference) in CONTRASTS.items():
        entry: dict = {}
        for scope_name, low, high in LEAD_BANDS:
            part = lead_filter(frame=decision, low=low, high=high)
            entry[scope_name] = {
                "qs": contrast(losses=part, treatment=treatment, reference=reference)
            }
            if scope_name == "all":
                entry[scope_name]["abs_mean"] = contrast(
                    losses=part, treatment=treatment, reference=reference, metric="abs_mean"
                )
                for tail, _ in TAIL_QUANTILES:
                    entry[scope_name][f"pin_{tail}"] = contrast(
                        losses=part, treatment=treatment, reference=reference, metric=f"pin_{tail}"
                    )
        for sub_name, expression in subsets().items():
            part = decision.filter(expression)
            entry[sub_name] = {
                "qs": contrast(losses=part, treatment=treatment, reference=reference)
            }
        cold = decision.filter(subsets()[COLD_KEY])
        entry["cold_night_q95_coverage"] = contrast(
            losses=cold, treatment=treatment, reference=reference, metric="cov_q95"
        )
        out[label] = entry
    return out


def checks(*, frames: dict[str, pl.DataFrame]) -> list[dict]:
    """Return the median forecast minus the mean forecast at 12 to 36 hours, per site and arm."""
    out = []
    for name, frame in frames.items():
        part = lead_filter(frame=frame, low=12.0, high=36.0)
        grouped = part.group_by("site").agg(
            (pl.col("q50") - pl.col("mean")).mean().alias("median_minus_mean")
        )
        out.extend({"arm": name, **row} for row in grouped.sort("site").to_dicts())
    return out


def decision_label(*, report: dict) -> dict:
    """Return each fix's pre-registered pass flags for one period."""
    out = {}
    for label, (treatment, _) in CONTRASTS.items():
        upper = report["contrasts"][label]["all"]["qs"]["upper"]
        rows = [r for r in report["tables"]["tail"][COLD_KEY] if r["arm"] == treatment]
        coverage = float(rows[0]["cov_q95"])
        out[label] = {
            "arm": treatment,
            "cold_q95_coverage": coverage,
            "upper_bound": upper,
            "coverage_ok": coverage >= PASS_COVERAGE,
            "score_ok": upper <= PASS_UPPER_BOUND,
            "passes": bool(coverage >= PASS_COVERAGE and upper <= PASS_UPPER_BOUND),
        }
    return out


def read_flows(*, cache) -> pl.DataFrame:  # noqa: ANN001
    """Return the decision substations' demand in MW at valid times before the scored year.

    The readings are the reference's `actual` column in the tuning and replicate months
    (July 2024 to June 2025), scaled by each substation's effective capacity; a valid time that
    several runs forecast is kept once.
    """
    capacity = {
        cache.sites[s].table.time_series_id: float(cache.sites[s].table.effective_capacity_mw)
        for s in common.DECISION_SITES
    }
    pieces = [
        load_ours(arm="ref", period=period, cache=cache) for period in ("tuning", "replicate")
    ]
    readings = (
        pl.concat(pieces)
        .filter(
            pl.col("site").is_in(list(common.DECISION_SITES))
            & (pl.col("valid_time") < SCORED_START)
        )
        .unique(subset=["time_series_id", "valid_time"])
    )
    return readings.select(
        pl.col("time_series_id").cast(pl.Int64),
        "valid_time",
        flow_mw=pl.col("actual").cast(pl.Float64)
        / 100.0
        * pl.col("time_series_id").replace_strict(capacity, return_dtype=pl.Float64),
    )


def node_table(*, cache) -> tuple[NodeTable, dict[int, float]]:  # noqa: ANN001
    """Return a node table of the decision substations treated as nodes, and their capacities."""
    ids = {s: cache.sites[s].table.time_series_id for s in common.DECISION_SITES}
    capacity = {
        int(ids[s]): float(cache.sites[s].table.effective_capacity_mw)
        for s in common.DECISION_SITES
    }
    nodes = pl.DataFrame(
        {
            "node_id": list(ids),
            "time_series_id": [int(v) for v in ids.values()],
            "node_type": ["BSP"] * len(ids),
            "parent_node_id": [None] * len(ids),
            "export_sign": [1] * len(ids),
            "sign_source": ["assumed: a positive demand reading is the flow that can breach"]
            * len(ids),
        },
        schema_overrides={"parent_node_id": pl.String, "export_sign": pl.Int64},
    )
    limits = pl.DataFrame(
        schema={
            "node_id": pl.String,
            "season": pl.String,
            "export_limit_mw": pl.Float64,
            "import_limit_mw": pl.Float64,
            "limit_source": pl.String,
        }
    )
    generators = pl.DataFrame(
        schema={
            "generator_id": pl.Int64,
            "node_id": pl.String,
            "capacity_mw": pl.Float64,
            "connection_date": pl.Date,
            "is_forecast": pl.Boolean,
        }
    )
    return NodeTable(nodes=nodes, limits=limits, generators=generators), capacity


def curtailment_proxy(*, frames: dict[str, pl.DataFrame], cache) -> dict:  # noqa: ANN001
    """Score the arms with the node-only curtailment rule on the decision substations' demand."""
    table, capacity = node_table(cache=cache)
    flows = read_flows(cache=cache)
    keep = [
        "time_series_id",
        "power_fcst_init_time",
        "valid_time",
        "seed",
        "actual",
        *QUANTILE_COLUMNS,
    ]
    decision = [cache.sites[s].table.time_series_id for s in common.DECISION_SITES]
    arm_frames = {}
    for name, frame in frames.items():
        base = frame.filter(pl.col("time_series_id").is_in(decision)).select(keep)
        base = base.with_columns(
            pl.col("time_series_id").cast(pl.Int64),
            *[pl.col(c) / 100.0 for c in ["actual", *QUANTILE_COLUMNS]],
        )
        arm_frames[name] = scale_to_megawatts(frame=base, capacity_mw=capacity)
    perfect = arm_frames["ref"].with_columns(*[pl.col("actual").alias(q) for q in QUANTILE_COLUMNS])
    arm_frames["perfect"] = perfect
    assumptions = NodeCurtailmentAssumptions(use_generators=False)
    out: dict = {}
    for fraction in CURTAILMENT_FRACTIONS:
        scaled = scaled_limits(table=table, flows=flows, cutoff=SCORED_START, fraction=fraction)
        scaled_table = table_with_scaled_limits(table=table, scaled=scaled, fraction=fraction)
        result = node_curtailment(
            table=scaled_table,
            node_forecasts=arm_frames,
            generator_forecasts={},
            assumptions=assumptions,
        )
        avoid = avoidable_cost(per_arm=result.per_arm, perfect_arm="perfect")
        column = "energy_proxy_mwh_equivalent"
        perfect_sums, counts = month_total_arrays(
            month_totals=result.month_totals, arm="perfect", column=column
        )
        rows = []
        for arm in sorted(result.month_totals["arm"].unique()):
            sums, _ = month_total_arrays(month_totals=result.month_totals, arm=arm, column=column)
            diff = sums - perfect_sums
            low, high = bootstrap_month_totals(month_sums=diff, month_counts=counts)
            total = float(counts.sum()) * CURTAILMENT_PRICE
            rows.append(
                {
                    "arm": arm,
                    "avoidable_gbp": float(diff.sum(axis=1).mean()) * CURTAILMENT_PRICE,
                    "low_gbp": low * total,
                    "high_gbp": high * total,
                }
            )
        out[str(fraction)] = {
            "per_arm": result.per_arm.filter(pl.col("is_headline_price")).to_dicts(),
            "avoidable": avoid.to_dicts(),
            "intervals": rows,
            "excluded": result.excluded.to_dicts(),
        }
    return out


def diagnosis(*, losses: pl.DataFrame, arm: str) -> dict:
    """Return one arm's residual mean, residual spread, and coverage by temperature and hour band.

    The residual is the actual minus the mean forecast. Rows of the decision substations, by the
    ensemble-mean temperature bin and by hour band, and the ERA5-defined cold nights by hour band.
    """
    scope = losses.filter(
        pl.col("site").is_in(list(common.DECISION_SITES)) & (pl.col("arm") == arm)
    )
    scope = scope.with_columns(
        residual=pl.col("actual") - pl.col("mean"),
        temperature_bin=pl.col("ens_mean_c").cut(list(TEMPERATURE_BIN_EDGES_C)).cast(pl.String),  # ty: ignore[deprecated]
        half_width_95=pl.col("q95") - pl.col("mean"),
    )
    aggregations = [
        pl.col("residual").mean().alias("residual_mean"),
        pl.col("residual").std().alias("residual_std"),
        pl.col("half_width_95").mean().alias("mean_q95_offset"),
        pl.col("cov_q95").mean(),
        pl.col("cov_q05").mean(),
        pl.len().alias("rows"),
    ]
    night = scope.filter(pl.col("hour_band").is_in(list(NIGHT_BANDS)))
    return {
        "by_temperature_bin": scope.group_by("temperature_bin").agg(aggregations).to_dicts(),
        "night_by_temperature_bin": night.group_by("temperature_bin").agg(aggregations).to_dicts(),
        "cold_by_hour_band": scope.filter(pl.col("cold2"))
        .group_by("hour_band")
        .agg(aggregations)
        .sort("hour_band")
        .to_dicts(),
        "not_cold_by_hour_band": scope.filter(
            ~pl.col("cold2") & pl.col("valid_month").is_in(list(WINTER_MONTHS))
        )
        .group_by("hour_band")
        .agg(aggregations)
        .sort("hour_band")
        .to_dicts(),
        "cold_overall": scope.filter(subsets()[COLD_KEY]).select(aggregations).to_dicts(),
    }


def period_report(*, period: str, cache) -> tuple[dict, dict[str, pl.DataFrame]]:  # noqa: ANN001
    """Return the period's tables and contrasts, and the common-row frames."""
    loaded = {arm: load_ours(arm=arm, period=period, cache=cache) for arm in OURS}
    origins = loaded["ref"]["power_fcst_init_time"].unique().to_list()
    if period == "scored":
        for name in STORED_87:
            loaded[name] = load_stored(name=name, origins=origins, cache=cache)
    frames = common_rows(frames=loaded)
    losses = stack(frames=frames, cache=cache)
    ours = losses.filter(pl.col("arm").is_in(list(OURS)))
    widths = width_by_lead_day(losses=ours)
    report: dict = {
        "arms": list(frames),
        "rows": frames["ref"].height,
        "widths": widths,
        "diagnosis": diagnosis(losses=losses, arm="ref"),
        "tables": score_tables(losses=losses),
        "contrasts": contrast_tables(losses=ours),
        "hour_band_coverage": hour_band_coverage(losses=ours),
    }
    report["decision"] = decision_label(report=report)
    return report, frames


def hour_band_coverage(*, losses: pl.DataFrame) -> list[dict]:
    """Return the reference's q95 and q99 coverage by hour band at the decision substations."""
    scope = losses.filter(
        pl.col("site").is_in(list(common.DECISION_SITES)) & (pl.col("arm") == "ref")
    )
    return (
        scope.group_by("hour_band")
        .agg(cov_q95=pl.col("cov_q95").mean(), cov_q99=pl.col("cov_q99").mean(), rows=pl.len())
        .sort("hour_band")
        .to_dicts()
    )


def main() -> None:
    """Score the three periods and write the report."""
    cache = common.load_cache()
    report: dict = {}
    for period in ("scored", "replicate", "tuning"):
        period_data, frames = period_report(period=period, cache=cache)
        report[period] = period_data
        if period == "scored":
            report["checks"] = checks(frames=frames)
            by_arm = {r["arm"]: r["qs"] for r in period_data["tables"]["by_lead"]["all"]}
            report["reproduction"] = {
                "reference_score": by_arm["ref"],
                "experiment_160_score": REFERENCE_SCORE,
                "gap_ref": by_arm["ref"] - REFERENCE_SCORE,
                "arm_a_score": by_arm["a"],
                "experiment_164_score": ARM_A_SCORE,
                "gap_a": by_arm["a"] - ARM_A_SCORE,
            }
            report["curtailment"] = curtailment_proxy(frames=frames, cache=cache)
        print(period, "done", flush=True)
    common.write_json_atomic(path=common.WORK_DIR / "report" / "report.json", content=report)
    print("report written")


if __name__ == "__main__":
    main()
