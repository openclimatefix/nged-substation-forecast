"""Build experiment 00024's stacked arms from the pooled fits, score them, and write the tables.

Each fit arm (`e00024_fit.py`) gets 00019's stack on its quantiles: a 14-day error correction (the
mean of observed minus median forecast over the 14 days ending at the newest usable observation,
00013's 3.5-hour delay, each past half-hour taking the forecast of the latest earlier validation
run)
and a spread factor of 1.15 about the corrected median. Scores are on the cases all arms hold, in
points of capacity; intervals resample whole validation months and the three seeds. The nine planned
contrasts are read at 99.44% over all leads; every other contrast is exploratory (95%).

Output under `work/results/`: `absolute.parquet`, `per_series.parquet`, `contrasts.parquet`,
`corrections.parquet`, and the prediction sets under `data/ml-experiments/00024-.../predictions/`.

    uv run python ml-experiments/00024-pooling-groupings-demand-xgboost/e00024_score.py
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00024_common as common
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
FIT_ARMS: Final[tuple[str, ...]] = tuple(common.ARMS)
STACK_SUFFIX: Final[str] = "_stack"
PLANNED_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 9.0)
N_RESAMPLES: Final[int] = 20_000
DEFAULT_RESAMPLES: Final[int] = 2_000
SCOPES: Final[tuple[str, ...]] = ("all leads", "up to 48 h", "3 d or more")
PRIMARY: Final[tuple[str, ...]] = ("Primary-MW", "Primary-MVA")
BSP_GSP: Final[tuple[str, ...]] = ("BSP", "GSP")
EVERY: Final[tuple[str, ...]] = (*PRIMARY, *BSP_GSP)
GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "Primary (16)": PRIMARY,
    "All (20)": EVERY,
    "BSP and GSP (4)": BSP_GSP,
    "Primary-MW (8)": ("Primary-MW",),
    "Primary-MVA (8)": ("Primary-MVA",),
    "BSP (2)": ("BSP",),
    "GSP (2)": ("GSP",),
}
PLANNED: Final[dict[str, tuple[str, str, str]]] = {
    "C1 (primary): by type with ID minus per series": ("type_id", "own", "Primary (16)"),
    "C2: all with ID minus per series": ("all_id", "own", "Primary (16)"),
    "C3: size bands with ID minus per series": ("size_id", "own", "Primary (16)"),
    "C4: by type with ID minus all with ID": ("type_id", "all_id", "All (20)"),
    "C5: by type without ID minus per series": ("type_noid", "own", "Primary (16)"),
    "C6: all without ID minus per series": ("all_noid", "own", "Primary (16)"),
    "C7: by type without ID minus by type with ID": ("type_noid", "type_id", "Primary (16)"),
    "C8: all without ID minus all with ID": ("all_noid", "all_id", "All (20)"),
    "C9: by type with ID minus random groups with ID": ("type_id", "random_id", "All (20)"),
}
EXPLORATORY_PAIRS: Final[dict[str, tuple[str, str]]] = {
    "by type with ID minus per series": ("type_id", "own"),
    "all with ID minus per series": ("all_id", "own"),
    "size bands with ID minus per series": ("size_id", "own"),
    "random groups with ID minus per series": ("random_id", "own"),
    "by type without ID minus per series": ("type_noid", "own"),
    "all without ID minus per series": ("all_noid", "own"),
}


def _actual_table(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return every validation case with the observed power over capacity (null if unobserved)."""
    arrays = common.load_series_arrays(series=series)
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
            keys.with_columns(
                actual=pl.Series(common.take(values=arrays[row["site"]].power, index=index))
            )
        )
    return pl.concat(pieces)


def corrections_for(*, frame: pl.DataFrame, power: np.ndarray) -> pl.DataFrame:
    """Return each run's correction for one series and seed, with its history count.

    Args:
        frame: One series' quantiles for one seed, keyed by run and valid time.
        power: The series' observed power over capacity on the dense grid.

    Returns:
        One row per run: `power_fcst_init_time`, `correction`, and `history_count`.

    Raises:
        ValueError: If a leakage check fails.
    """
    runs = (frame.select("power_fcst_init_time").unique().sort("power_fcst_init_time"))[
        "power_fcst_init_time"
    ]
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
        rows.append(
            {
                "power_fcst_init_time": origin_time,
                "correction": float(residual.mean()) if count >= MIN_COUNT else 0.0,
                "history_count": count,
            }
        )
    return pl.DataFrame(rows)


def with_corrections(
    *, base: pl.DataFrame, series: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Add a `correction` column to a fit's quantiles, and return the per-run corrections too."""
    arrays = common.load_series_arrays(series=series)
    sites = series.select("time_series_id", "site")
    pieces, tables = [], []
    for (series_id, seed), part in base.group_by("time_series_id", "seed", maintain_order=True):
        site = sites.filter(pl.col("time_series_id") == series_id)["site"][0]
        table = corrections_for(frame=part, power=arrays[site].power)
        tables.append(table.with_columns(time_series_id=pl.lit(series_id), seed=pl.lit(seed)))
        pieces.append(part.join(table, on="power_fcst_init_time", how="left", validate="m:1"))
    return pl.concat(pieces), pl.concat(tables)


def stack(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Apply the correction and the spread factor to a fit's quantiles."""
    quantiles = frame.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
    quantiles = quantiles + frame["correction"].to_numpy()[:, None]
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
    paths = sorted(ARMS_DIR.glob(f"{name}_seed*.parquet"))
    if [p.name for p in paths] != [f"{name}_seed{s}.parquet" for s in common.SEEDS]:
        msg = f"fit {name} does not hold exactly seeds {common.SEEDS}"
        raise ValueError(msg)
    return pl.concat([pl.read_parquet(p) for p in paths]).sort(*KEYS, "seed")


def pinball(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return 00013's quantile score per case, in fractions of capacity."""
    return common.pinball_quantile_score(quantiles=quantiles, actual=actual)


def prepare(*, arms: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Restrict every arm to the shared cases and add the lead, losses, coverage, and group."""
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
    groups = common.series_table().select(
        "time_series_id",
        group=pl.when(pl.col("substation_type") == "Primary")
        .then(pl.concat_str(pl.lit("Primary-"), pl.col("units")))
        .otherwise(pl.col("substation_type")),
        site_label=pl.col("site"),
    )
    out = {}
    for name, frame in arms.items():
        kept = frame.join(shared, on=KEYS, how="semi").sort(*KEYS, "seed")
        quantiles = np.sort(kept.select(common.LEVEL_COLUMNS).to_numpy(), axis=1)
        actual = kept["actual"].to_numpy()
        out[name] = (
            kept.join(groups, on="time_series_id", how="left", maintain_order="left")
            .with_columns(
                lead_hours=(
                    pl.col("valid_time") - pl.col("power_fcst_init_time")
                ).dt.total_minutes()
                / 60.0,
                quantile_score=pl.Series(pinball(quantiles=quantiles, actual=actual) * 100.0),
                median_error=pl.Series(np.abs(quantiles[:, MEDIAN] - actual) * 100.0),
                signed_error=pl.Series((quantiles[:, MEDIAN] - actual) * 100.0),
                covered_90=pl.Series(
                    ((actual >= quantiles[:, 0]) & (actual <= quantiles[:, -1])).astype(float)
                ),
                covered_80=pl.Series(
                    ((actual >= quantiles[:, 1]) & (actual <= quantiles[:, -2])).astype(float)
                ),
            )
            .with_columns(pl.col("quantile_score", "median_error", "signed_error").cast(pl.Float64))
            .drop(*common.LEVEL_COLUMNS, "actual")
        )
    return out


def _scope(*, frame: pl.DataFrame, scope: str, group: str) -> pl.DataFrame:
    frame = frame.filter(pl.col("group").is_in(GROUPS[group]))
    if scope == "all leads":
        return frame
    if scope == "up to 48 h":
        return frame.filter(pl.col("lead_hours") < 48.0)
    return frame.filter(pl.col("lead_hours") >= 72.0)


def contrast(
    *,
    arms: dict[str, pl.DataFrame],
    treatment: str,
    reference: str,
    group: str,
    scope: str,
    metric: str,
    level: float,
    decisive: bool,
) -> dict[str, float | int | str]:
    """Return one paired contrast's estimate, interval, and month agreement for a group."""
    losses = experiment_store.losses_for_bootstrap(
        arm_losses={
            name: _scope(frame=arms[name], scope=scope, group=group)
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
        "scope": scope,
        "metric": metric,
        "level": level,
        "difference": float(paired["difference"].to_numpy().mean()),
        "lower": lower,
        "upper": upper,
        "months_negative": int((by_month["difference"] < 0).sum()),
        "months": by_month.height,
    }


def main() -> None:
    """Build and score every arm and write the tables."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    series = common.series_table()
    cases = _actual_table(series=series)
    fits = {name: load_fit(name=name) for name in FIT_ARMS}
    corrected, tables = {}, []
    for name, fit in fits.items():
        corrected[name], table = with_corrections(base=fit, series=series)
        tables.append(table.with_columns(fit=pl.lit(name)))
    pl.concat(tables).write_parquet(RESULTS_DIR / "corrections.parquet")
    raw: dict[str, pl.DataFrame] = dict(fits)
    for name in FIT_ARMS:
        raw[f"{name}{STACK_SUFFIX}"] = stack(frame=corrected[name])
    for name in STORED_ARMS:
        stored = pl.read_parquet(common.ARMS_DIR_00013 / f"{name}.parquet").select(
            *KEYS, *common.LEVEL_COLUMNS
        )
        raw[name] = pl.concat(
            [stored.with_columns(seed=pl.lit(seed).cast(pl.Int8)) for seed in common.SEEDS]
        )
    full = {
        name: frame.select(*KEYS, "seed", *common.LEVEL_COLUMNS).join(
            cases, on=KEYS, how="inner", validate="m:1"
        )
        for name, frame in raw.items()
    }
    arms = prepare(arms=full)
    rows = {name: frame.height // frame["seed"].n_unique() for name, frame in arms.items()}
    if len(set(rows.values())) != 1:
        msg = f"arms hold different numbers of shared cases per seed: {rows}"
        raise ValueError(msg)
    for name in (*FIT_ARMS, *(f"{n}{STACK_SUFFIX}" for n in FIT_ARMS)):
        try:
            experiment_store.write_predictions(
                experiment_id=common.EXPERIMENT_ID,
                name=name,
                frame=raw[name]
                .select(*KEYS, "seed", *common.LEVEL_COLUMNS)
                .filter(pl.col("valid_time") < HOLDOUT),
                kind="quantiles",
                row_set_id=common.ROW_SET_ID,
            )
        except FileExistsError:
            continue
    absolute, per_series = [], []
    for name, frame in arms.items():
        for scope in SCOPES:
            for group in GROUPS:
                part = _scope(frame=frame, scope=scope, group=group)
                losses = experiment_store.losses_for_bootstrap(arm_losses={name: part})
                for metric in ("quantile_score", "median_error"):
                    interval = bootstrap.bootstrap_absolute(losses=losses, arm=name, metric=metric)
                    absolute.append(
                        {"arm": name, "scope": scope, "group": group, "metric": metric, **interval}
                    )
                absolute.append(
                    {
                        "arm": name,
                        "scope": scope,
                        "group": group,
                        "metric": "coverage",
                        "value": float(part["covered_90"].to_numpy().mean()),
                        "coverage_80": float(part["covered_80"].to_numpy().mean()),
                        "bias": float(part["signed_error"].to_numpy().mean()),
                    }
                )
        for site in frame["site_label"].unique().to_list():
            part = frame.filter(pl.col("site_label") == site)
            losses = experiment_store.losses_for_bootstrap(arm_losses={name: part})
            interval = bootstrap.bootstrap_absolute(
                losses=losses, arm=name, metric="quantile_score"
            )
            per_series.append({"arm": name, "site": site, "group": part["group"][0], **interval})
    pl.DataFrame(absolute, strict=False).write_parquet(RESULTS_DIR / "absolute.parquet")
    pl.DataFrame(per_series, strict=False).write_parquet(RESULTS_DIR / "per_series.parquet")
    suffixed = {
        label: (f"{t}{STACK_SUFFIX}", f"{r}{STACK_SUFFIX}", g)
        for label, (t, r, g) in PLANNED.items()
    }
    exploratory = {
        f"{label}, {group}": (f"{t}{STACK_SUFFIX}", f"{r}{STACK_SUFFIX}", group)
        for label, (t, r) in EXPLORATORY_PAIRS.items()
        for group in GROUPS
    }
    exploratory.update(
        {
            f"raw {label}, {group}": (t, r, group)
            for label, (t, r) in EXPLORATORY_PAIRS.items()
            for group in ("Primary (16)", "All (20)")
        }
    )
    exploratory.update(
        {
            f"{arm} stack minus manual_heuristic, {group}": (
                f"{arm}{STACK_SUFFIX}",
                "manual_heuristic",
                group,
            )
            for arm in FIT_ARMS
            for group in ("Primary (16)", "All (20)")
        }
    )
    contrasts = [
        {
            "label": label,
            "planned": label in suffixed,
            **contrast(
                arms=arms,
                treatment=treatment,
                reference=reference,
                group=group,
                scope=scope,
                metric=metric,
                level=PLANNED_LEVEL if label in suffixed and scope == "all leads" else 95.0,
                decisive=label in suffixed and scope == "all leads",
            ),
            "group": group,
        }
        for label, (treatment, reference, group) in {**suffixed, **exploratory}.items()
        for scope in SCOPES
        for metric in ("quantile_score", "median_error")
    ]
    pl.DataFrame(contrasts).write_parquet(RESULTS_DIR / "contrasts.parquet")
    runs = tables[0].filter(pl.col("seed") == 0)
    print("rows per seed:", rows[next(iter(rows))])
    print(
        "runs with no correction (seed 0, all series):",
        int((runs["history_count"] < MIN_COUNT).sum()),
        "of",
        runs.height,
    )


if __name__ == "__main__":
    main()
