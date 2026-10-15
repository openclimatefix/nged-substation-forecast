"""Build experiment 00031's arms from the fits, score them, and write the tables.

Every arm gets the 14-day error correction and the 1.15 spread factor that 00019 describes. B0 is
00013's N1 (stored) with both. Arm A5 is B0 plus a sunshine-shaped term: the slope `b` of the
14-day residuals on CAMS `p_unit`, fitted on a window that ends 48 hours before the origin, times
the ensemble-mean `p_unit` at the valid time minus the window's mean `p_unit`. Arm A4 holds the
offset refits for the directional Primaries and B0's rows for every other series. A7 is scored on
the 2 GSPs only. Intervals resample whole validation months and the three seeds; the six planned
contrasts are read at 99.17% over all leads and every other contrast at 95%.

Output under `work/results/`: `absolute.parquet`, `contrasts.parquet`, `per_site.parquet`,
`history.parquet`, and the prediction sets under `data/ml-experiments/00031-.../predictions/`.

    uv run python ml-experiments/00031-embedded-pv-features-primary-demand/e00031_score.py
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00031_common as common
import e00031_ecr as ecr
import e00031_pv as pv
import numpy as np
import polars as pl

from studies import bootstrap, experiment_store

ARMS_DIR: Final[Path] = common.WORK_DIR / "arms"
RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
HOLDOUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
STORED_ARMS: Final[tuple[str, ...]] = ("manual_heuristic", "climatology")
MEDIAN: Final[int] = 4
"""Position of the median (q50) among the nine levels."""
NEWEST_AGE: Final[int] = int(common.TELEMETRY_DELAY / common.STEP) + 1
"""Half-hours between the origin and the newest usable observation (00013's 3.5 hours)."""
WINDOW: Final[int] = common.CORRECTION_DAYS * common.STEPS_PER_DAY
MIN_COUNT: Final[int] = common.MIN_CORRECTION_DAYS * common.STEPS_PER_DAY
SHAPE_LAG: Final[int] = pv.FEATURE_DELAY_STEPS
"""The shaped term's window ends this many half-hours before the origin, where CAMS exists."""
FIT_ARMS: Final[tuple[str, ...]] = ("a1", "a2", "a3", "a4", "a6")
PLANNED: Final[dict[str, tuple[str, str]]] = {
    "C1: A1 minus B0": ("a1", "b0"),
    "C2: A2 minus B0": ("a2", "b0"),
    "C3: A3 minus B0": ("a3", "b0"),
    "C4: A4 minus B0": ("a4", "b0"),
    "C5: A5 minus B0": ("a5", "b0"),
    "C6: A3 minus A6": ("a3", "a6"),
}
EXPLORATORY: Final[dict[str, tuple[str, str]]] = {
    "E1: A3 minus A2": ("a3", "a2"),
    "E2: A4 minus A3": ("a4", "a3"),
    "E3: A6 minus B0": ("a6", "b0"),
    "E4: B0 minus manual_heuristic": ("b0", "manual_heuristic"),
    "E5: B0 minus climatology": ("b0", "climatology"),
}
PLANNED_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 6.0)
N_RESAMPLES: Final[int] = 20_000
DEFAULT_RESAMPLES: Final[int] = 2_000
SUBSETS: Final[dict[str, pl.Expr]] = {
    "all leads": pl.lit(True),
    "0-48 h": pl.col("lead_hours") < 48.0,
    "3-6 d": (pl.col("lead_hours") >= 72.0) & (pl.col("lead_hours") < 168.0),
    "7-15 d": pl.col("lead_hours") >= 168.0,
    "night": pl.col("sky") == "night",
    "daylight, index under 0.3": pl.col("sky") == "under 0.3",
    "daylight, index 0.3 to 0.7": pl.col("sky") == "0.3 to 0.7",
    "daylight, index 0.7 or more": pl.col("sky") == "0.7 or more",
    "April to September": pl.col("season") == "April to September",
    "October to March": pl.col("season") == "October to March",
    "sunny days": pl.col("day_class") == "sunny",
    "cloudy days": pl.col("day_class") == "cloudy",
    "April to September, index 0.7 or more": (pl.col("season") == "April to September")
    & (pl.col("sky") == "0.7 or more"),
    "directional meters": pl.col("directional"),
    "non-directional meters": ~pl.col("directional"),
}
LEAD_SUBSETS: Final[tuple[str, ...]] = ("all leads", "0-48 h", "3-6 d", "7-15 d")
OUTLIER_SITES: Final[tuple[str, ...]] = ("D8", "D12", "D22")


def _grids(*, series: pl.DataFrame) -> dict[str, pv.SiteGrid]:
    """Return each site's grid."""
    return {
        row["site"]: pv.site_grid(
            site=row["site"], latitude=row["latitude"], longitude=row["longitude"]
        )
        for row in series.iter_rows(named=True)
    }


def _actual_table(*, series: pl.DataFrame, power: dict[str, np.ndarray]) -> pl.DataFrame:
    """Return every validation case with the observed power over capacity (null if unobserved)."""
    pieces = []
    for row in series.iter_rows(named=True):
        keys = pl.concat(
            [
                pl.read_parquet(path, columns=list(KEYS))
                .filter(pl.col("time_series_id") == row["time_series_id"])
                .unique()
                for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet"))
            ]
        )
        index = np.asarray(common.grid_index(times=keys["valid_time"]))
        pieces.append(
            keys.with_columns(actual=pl.Series(common.take(values=power[row["site"]], index=index)))
        )
    return pl.concat(pieces)


def corrections_for(*, frame: pl.DataFrame, power: np.ndarray, grid: pv.SiteGrid) -> pl.DataFrame:
    """Return each run's flat correction and its sunshine-shaped term for one series and seed.

    The flat correction is 00019's: the mean of (observed minus median forecast) over the 14 days
    ending at the newest usable observation, each past half-hour using the forecast of the latest
    earlier run. The shaped term fits the same residuals on `p_unit` over the 14 days ending 48
    hours before the origin.

    Args:
        frame: One series' quantiles for one seed, keyed by run and valid time.
        power: The series' observed power over capacity on the dense grid.
        grid: The series' site grid, for CAMS `p_unit` and the daily clear-sky index.

    Returns:
        One row per run: `power_fcst_init_time`, `correction`, `history_count`, `slope`,
        `window_mean_unit_power`, and `sunny_days`.

    Raises:
        ValueError: If a leakage check fails.
    """
    runs = (
        frame.select("power_fcst_init_time")
        .unique()
        .sort("power_fcst_init_time")["power_fcst_init_time"]
    )
    origins = np.asarray(common.grid_index(times=runs))
    median = np.full(power.shape, np.nan, dtype=np.float64)
    source = np.full(power.shape, -1, dtype=np.int64)
    for position, origin in enumerate(runs):
        part = frame.filter(pl.col("power_fcst_init_time") == origin)
        index = np.asarray(common.grid_index(times=part["valid_time"]))
        end = origins[position + 1] if position + 1 < len(origins) else np.inf
        keep = (index >= origins[position]) & (index < end)
        median[index[keep]] = part[common.LEVEL_COLUMNS[MEDIAN]].to_numpy()[keep]
        source[index[keep]] = origins[position]
    rows = []
    for origin_time, origin in zip(runs, origins, strict=True):
        newest = origin - NEWEST_AGE
        window = np.arange(newest - WINDOW + 1, newest + 1)
        used = ~np.isnan(median[window]) & ~np.isnan(power[window])
        if (source[window][used] > window[used]).any():
            msg = "a past forecast comes from a run that began after its half-hour or at the origin"
            raise ValueError(msg)
        count = int(used.sum())
        residual = power[window][used].astype(np.float64) - median[window][used]
        slope, mean_unit, sunny = 0.0, 0.0, 0
        shaped_end = origin - SHAPE_LAG
        shaped = np.arange(shaped_end - WINDOW + 1, shaped_end + 1)
        x = grid.cams_unit_power[shaped]
        ok = ~np.isnan(median[shaped]) & ~np.isnan(power[shaped]) & ~np.isnan(x)
        if ok.sum() >= MIN_COUNT:
            day_index = grid.daily_clear_sky_index[shaped][ok]
            complete = shaped[ok] < (shaped_end // 48) * 48
            sunny = int(
                np.unique(np.floor(shaped[ok][complete & (day_index >= pv.SUNNY_INDEX)] / 48)).size
            )
            y = power[shaped][ok].astype(np.float64) - median[shaped][ok]
            mean_unit = float(x[ok].mean())
            spread = float(((x[ok] - mean_unit) ** 2).sum())
            if sunny >= pv.MIN_SUNNY_AND_CLOUDY_DAYS and spread > 0:
                slope = float(((x[ok] - mean_unit) * (y - y.mean())).sum() / spread)
        rows.append(
            {
                "power_fcst_init_time": origin_time,
                "correction": float(residual.mean()) if count >= MIN_COUNT else 0.0,
                "history_count": count,
                "slope": slope,
                "window_mean_unit_power": mean_unit,
                "sunny_days": sunny,
            }
        )
    return pl.DataFrame(rows)


def with_corrections(
    *,
    base: pl.DataFrame,
    series: pl.DataFrame,
    power: dict[str, np.ndarray],
    grids: dict[str, pv.SiteGrid],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Add the correction columns to a fit's quantiles, and return the per-run table too."""
    sites = series.select("time_series_id", "site")
    pieces, tables = [], []
    for (series_id, seed), part in base.group_by("time_series_id", "seed", maintain_order=True):
        site = sites.filter(pl.col("time_series_id") == series_id)["site"][0]
        table = corrections_for(frame=part, power=power[site], grid=grids[site])
        tables.append(table.with_columns(time_series_id=pl.lit(series_id), seed=pl.lit(seed)))
        pieces.append(part.join(table, on="power_fcst_init_time", how="left", validate="m:1"))
    return pl.concat(pieces), pl.concat(tables)


def stack(*, frame: pl.DataFrame, shape: pl.DataFrame | None = None) -> pl.DataFrame:
    """Apply the correction, an optional shaped term, and the spread factor to a fit's quantiles.

    Args:
        frame: A fit's quantiles with the correction columns.
        shape: When given, the ensemble-mean `p_unit` per case (`p_ens_mean`), which turns on the
            sunshine-shaped term.

    Returns:
        The corrected quantiles, keyed by run, valid time, and seed.
    """
    if shape is not None:
        frame = frame.join(shape, on=KEYS, how="left", validate="m:1").with_columns(
            shaped=pl.col("slope")
            * (pl.col("p_ens_mean") - pl.col("window_mean_unit_power")).fill_null(0.0)
        )
    quantiles = frame.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
    quantiles = quantiles + frame["correction"].to_numpy()[:, None]
    if shape is not None:
        quantiles = quantiles + frame["shaped"].to_numpy()[:, None]
    median = quantiles[:, [MEDIAN]]
    quantiles = median + common.SPREAD_FACTOR * (quantiles - median)
    return frame.select(*KEYS, "seed").with_columns(
        pl.DataFrame(
            np.sort(quantiles, axis=1).astype(np.float32),
            schema=list(common.LEVEL_COLUMNS),
            orient="row",
        )
    )


def load_fit(*, name: str) -> pl.DataFrame:
    """Return a fit's three seeds, keyed by series, run, valid time, and seed."""
    folder = common.ARMS_DIR_00013 if name == "n1" else ARMS_DIR
    paths = sorted(folder.glob(f"{name}_seed*.parquet"))
    if [p.name for p in paths] != [f"{name}_seed{s}.parquet" for s in common.SEEDS]:
        msg = f"fit {name} does not hold exactly seeds {common.SEEDS}"
        raise ValueError(msg)
    return pl.concat([pl.read_parquet(p) for p in paths]).sort(*KEYS, "seed")


def _sky_columns(*, series: pl.DataFrame, grids: dict[str, pv.SiteGrid]) -> pl.DataFrame:
    """Return, per series and valid time, the sky bin and the day class from CAMS."""
    cams = pl.read_parquet(pv.CAMS_PATH)
    pieces = []
    for row in series.iter_rows(named=True):
        hourly = cams.filter(pl.col("site") == row["site"]).select(
            hour_end="time",
            index=pl.when(pl.col("clear_sky_ghi_w_m2") > 0)
            .then(pl.col("ghi_w_m2") / pl.col("clear_sky_ghi_w_m2"))
            .otherwise(None),
            clear=pl.col("clear_sky_ghi_w_m2") > 0,
        )
        times = pl.DataFrame({"valid_time": common.grid_times()}).with_columns(
            hour_end=(pl.col("valid_time") - pl.duration(microseconds=1)).dt.truncate("1h")
            + pl.duration(hours=1),
            day_index=pl.Series(grids[row["site"]].daily_clear_sky_index),
        )
        joined = times.join(
            hourly, on="hour_end", how="left", validate="m:1", maintain_order="left"
        )
        pieces.append(
            joined.select(
                "valid_time",
                time_series_id=pl.lit(row["time_series_id"]),
                sky=pl.when(~pl.col("clear").fill_null(False))
                .then(pl.lit("night"))
                .when(pl.col("index") < pv.CLOUDY_INDEX)
                .then(pl.lit("under 0.3"))
                .when(pl.col("index") < pv.SUNNY_INDEX)
                .then(pl.lit("0.3 to 0.7"))
                .otherwise(pl.lit("0.7 or more")),
                day_class=pl.when(pl.col("day_index") >= pv.SUNNY_INDEX)
                .then(pl.lit("sunny"))
                .when(pl.col("day_index") <= pv.CLOUDY_INDEX)
                .then(pl.lit("cloudy"))
                .otherwise(pl.lit("middle")),
            )
        )
    return pl.concat(pieces)


def prepare(*, arms: dict[str, pl.DataFrame], context: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Restrict every arm to the shared cases and add the lead, losses, and context columns."""
    shared: pl.DataFrame | None = None
    for frame in arms.values():
        cases = (
            frame.filter(
                pl.all_horizontal(pl.col("actual", *common.LEVEL_COLUMNS).is_finite())
                & (pl.col("valid_time") < HOLDOUT)
            )
            .select(KEYS)
            .unique()
        )
        shared = cases if shared is None else shared.join(cases, on=KEYS, how="inner")
    assert shared is not None
    out = {}
    for name, frame in arms.items():
        kept = frame.join(shared, on=KEYS, how="semi").sort(*KEYS, "seed")
        quantiles = np.sort(kept.select(common.LEVEL_COLUMNS).to_numpy(), axis=1)
        actual = kept["actual"].to_numpy()
        out[name] = (
            kept.join(
                context, on=["time_series_id", "valid_time"], how="left", maintain_order="left"
            )
            .with_columns(
                lead_hours=(
                    pl.col("valid_time") - pl.col("power_fcst_init_time")
                ).dt.total_minutes()
                / 60.0,
                season=pl.when(pl.col("valid_time").dt.month().is_between(4, 9))
                .then(pl.lit("April to September"))
                .otherwise(pl.lit("October to March")),
                quantile_score=pl.Series(
                    common.pinball_quantile_score(quantiles=quantiles, actual=actual) * 100.0
                ),
                median_error=pl.Series(np.abs(quantiles[:, MEDIAN] - actual) * 100.0),
                signed_error=pl.Series((quantiles[:, MEDIAN] - actual) * 100.0),
                covered_90=pl.Series(
                    ((actual >= quantiles[:, 0]) & (actual <= quantiles[:, -1])).astype(float)
                ),
            )
            .with_columns(pl.col("quantile_score", "median_error", "signed_error").cast(pl.Float64))
            .drop(*common.LEVEL_COLUMNS, "actual")
        )
    return out


def _subset(*, frame: pl.DataFrame, where: pl.Expr, group: str) -> pl.DataFrame:
    return frame.filter((pl.col("group") == group) & where)


def contrast(
    *,
    arms: dict[str, pl.DataFrame],
    treatment: str,
    reference: str,
    subset: str,
    where: pl.Expr,
    group: str,
    metric: str,
    level: float,
    decisive: bool,
) -> dict[str, float | int | str]:
    """Return one paired contrast's estimate, interval, and month agreement."""
    losses = experiment_store.losses_for_bootstrap(
        arm_losses={
            name: _subset(frame=arms[name], where=where, group=group)
            for name in (treatment, reference)
        }
    )
    lower, upper = bootstrap.bootstrap_difference_at_level(
        losses=losses,
        treatment=treatment,
        reference=reference,
        metric=metric,
        level=level,
        n_resamples=N_RESAMPLES if decisive else DEFAULT_RESAMPLES,
    )
    paired = (
        losses.filter(pl.col("arm") == treatment)
        .select("site", "time", "seed", "month", t=pl.col(metric))
        .join(
            losses.filter(pl.col("arm") == reference).select(
                "site", "time", "seed", r=pl.col(metric)
            ),
            on=["site", "time", "seed"],
        )
        .with_columns(difference=pl.col("t") - pl.col("r"))
    )
    by_month = paired.group_by("month").agg(pl.col("difference").mean())
    return {
        "treatment": treatment,
        "reference": reference,
        "subset": subset,
        "group": group,
        "metric": metric,
        "level": level,
        "difference": float(paired["difference"].to_numpy().mean()),
        "lower": lower,
        "upper": upper,
        "months_negative": int((by_month["difference"] < 0).sum()),
        "months": by_month.height,
    }


def build_arms(
    *, series: pl.DataFrame, power: dict[str, np.ndarray], grids: dict[str, pv.SiteGrid]
) -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    """Return every scored arm's raw quantiles (before restriction) and the per-run history."""
    n1, history = with_corrections(
        base=load_fit(name="n1"), series=series, power=power, grids=grids
    )
    pens = pl.read_parquet(ARMS_DIR / "pens.parquet")
    arms = {"b0": stack(frame=n1), "a5": stack(frame=n1, shape=pens)}
    for name in FIT_ARMS:
        fit, _ = with_corrections(base=load_fit(name=name), series=series, power=power, grids=grids)
        arms[name] = stack(frame=fit)
    directional = series.filter(pl.col("site").is_in(ecr.directional_sites(series=series)))
    others = arms["b0"].join(directional.select("time_series_id"), on="time_series_id", how="anti")
    arms["a4"] = pl.concat([arms["a4"], others.select(arms["a4"].columns)]).sort(*KEYS, "seed")
    for name in STORED_ARMS:
        stored = pl.read_parquet(common.ARMS_DIR_00013 / f"{name}.parquet").select(
            *KEYS, *common.LEVEL_COLUMNS
        )
        arms[name] = pl.concat(
            [stored.with_columns(seed=pl.lit(seed).cast(pl.Int8)) for seed in common.SEEDS]
        )
    return arms, history


def main() -> None:
    """Build and score every arm and write the tables."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    series = common.series_table()
    power = common.load_power(series=series)
    grids = _grids(series=series)
    cases = _actual_table(series=series, power=power)
    arms, history = build_arms(series=series, power=power, grids=grids)
    history.write_parquet(RESULTS_DIR / "history.parquet")
    register = ecr.ecr_table(series=series)
    directional = ecr.directional_sites(series=series)
    context = _sky_columns(series=series, grids=grids).join(
        series.select(
            "time_series_id",
            "site",
            group=pl.col("substation_type"),
            directional=pl.col("site").is_in(directional),
        ),
        on="time_series_id",
        validate="m:1",
    )
    full = {
        name: frame.select(*KEYS, "seed", *common.LEVEL_COLUMNS).join(
            cases, on=KEYS, how="inner", validate="m:1"
        )
        for name, frame in arms.items()
    }
    scored = prepare(arms=full, context=context)
    rows = {name: frame.height // frame["seed"].n_unique() for name, frame in scored.items()}
    if len(set(rows.values())) != 1:
        msg = f"arms hold different numbers of shared cases per seed: {rows}"
        raise ValueError(msg)
    for name in (*FIT_ARMS, "a5", "b0"):
        try:
            experiment_store.write_predictions(
                experiment_id=common.EXPERIMENT_ID,
                name=name,
                frame=arms[name]
                .select(*KEYS, "seed", *common.LEVEL_COLUMNS)
                .filter(pl.col("valid_time") < HOLDOUT),
                kind="quantiles",
                row_set_id=common.ROW_SET_ID,
            )
        except FileExistsError:
            continue
    absolute = _absolute(scored=scored)
    absolute.write_parquet(RESULTS_DIR / "absolute.parquet")
    contrasts = _contrasts(scored=scored)
    contrasts.write_parquet(RESULTS_DIR / "contrasts.parquet")
    _per_site(scored=scored, register=register, directional=directional).write_parquet(
        RESULTS_DIR / "per_site.parquet"
    )
    print("rows per seed:", rows["b0"])
    with pl.Config(tbl_rows=100, tbl_cols=20, fmt_str_lengths=60):
        print(absolute.filter((pl.col("subset") == "all leads") & (pl.col("group") == "Primary")))
        print(
            contrasts.filter(pl.col("planned")).select(
                "label", "metric", "difference", "lower", "upper", "level", "months_negative"
            )
        )


def _absolute(*, scored: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return each arm's mean scores per group and subset, with month-resampled intervals."""
    rows = []
    for name, frame in scored.items():
        for group in ("Primary", "BSP", "GSP"):
            for subset, where in SUBSETS.items():
                part = _subset(frame=frame, where=where, group=group)
                if part.is_empty():
                    continue
                row: dict[str, float | str] = {"arm": name, "group": group, "subset": subset}
                for metric in ("quantile_score", "median_error"):
                    row[metric] = float(part[metric].mean())
                row["bias"] = float(part["signed_error"].mean())
                row["coverage_90"] = float(part["covered_90"].mean())
                row["rows"] = part.height
                rows.append(row)
    return pl.DataFrame(rows)


def _contrasts(*, scored: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return the planned contrasts, the exploratory ones, and the breakdowns."""
    rows = [
        {
            "label": label,
            "planned": label in PLANNED,
            **contrast(
                arms=scored,
                treatment=treatment,
                reference=reference,
                subset=subset,
                where=where,
                group="Primary",
                metric=metric,
                level=PLANNED_LEVEL if label in PLANNED and subset == "all leads" else 95.0,
                decisive=label in PLANNED and subset == "all leads",
            ),
        }
        for label, (treatment, reference) in {**PLANNED, **EXPLORATORY}.items()
        for subset, where in SUBSETS.items()
        if subset in LEAD_SUBSETS or (label not in PLANNED and subset == "all leads")
        if label not in PLANNED or subset in LEAD_SUBSETS
        for metric in ("quantile_score", "median_error")
    ]
    breakdown_arms = ("a1", "a2", "a3", "a4", "a5", "a6")
    for treatment in breakdown_arms:
        for subset, where in SUBSETS.items():
            if subset == "all leads":
                continue
            rows.append(
                {
                    "label": f"breakdown: {treatment} minus b0",
                    "planned": False,
                    **contrast(
                        arms=scored,
                        treatment=treatment,
                        reference="b0",
                        subset=subset,
                        where=where,
                        group="Primary",
                        metric="quantile_score",
                        level=95.0,
                        decisive=False,
                    ),
                }
            )
    rows.extend(_leave_outs(scored=scored))
    return pl.DataFrame(rows)


def _leave_outs(*, scored: dict[str, pl.DataFrame]) -> list[dict[str, object]]:
    """Return contrasts without the largest-B0-error site and without the three heavy PV sites."""
    b0 = scored["b0"].filter(pl.col("group") == "Primary")
    worst = (
        b0.group_by("site").agg(pl.col("quantile_score").mean()).sort("quantile_score")["site"][-1]
    )
    exclusions = {
        "without the worst B0 site": [worst],
        "without D8, D12, D22": list(OUTLIER_SITES),
    }
    return [
        {
            "label": f"leave-out ({tag}): {treatment} minus b0",
            "planned": False,
            **contrast(
                arms=scored,
                treatment=treatment,
                reference="b0",
                subset=tag,
                where=~pl.col("site").is_in(excluded),
                group="Primary",
                metric="quantile_score",
                level=95.0,
                decisive=False,
            ),
        }
        for tag, excluded in exclusions.items()
        for treatment in ("a1", "a2", "a3", "a4", "a5")
    ]


def _per_site(
    *, scored: dict[str, pl.DataFrame], register: pl.DataFrame, directional: frozenset[str]
) -> pl.DataFrame:
    """Return, per series, each arm's mean quantile score and the register share."""
    pieces = [
        frame.group_by("site", "group")
        .agg(score=pl.col("quantile_score").mean())
        .with_columns(arm=pl.lit(name))
        for name, frame in scored.items()
    ]
    return (
        pl.concat(pieces)
        .join(register.select("site", "role", "ecr_share"), on="site", validate="m:1")
        .with_columns(directional=pl.col("site").is_in(directional))
    )


if __name__ == "__main__":
    main()
