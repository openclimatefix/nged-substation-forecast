"""Build experiment 00032's stacked arms from the fits, score them, and write the tables.

Each arm gets 00019's stack on its quantiles: a 14-day error correction (the mean of observed
minus median forecast over the 14 days ending at the newest usable observation, 00013's 3.5-hour
delay, each past half-hour taking the forecast of the latest earlier validation run) and a spread
factor of 1.15 about the corrected median. Scores are on the cases all arms of a table hold, in
points of capacity. The warm table holds the 7 arms on 20 series and three seeds; the cold-start
table holds the 4 cold-start arms, the warm arms with and without the identifier, and the two
stored baselines on the 8 held-out series and seed 0. The eight planned contrasts are read at
99.375% over all leads; every other contrast is exploratory (95%).

Output under `work/results/`: `absolute.parquet`, `per_series.parquet`, `contrasts.parquet`,
`corrections.parquet`, and the prediction sets under `data/ml-experiments/00032-.../predictions/`.

    uv run python ml-experiments/00032-substation-fingerprints-pooled-demand/e00032_score.py
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00032_common as common
import numpy as np
import polars as pl

from studies import bootstrap, experiment_store

ARMS_DIR: Final[Path] = common.WORK_DIR / "arms"
ARMS_DIR_00024: Final[Path] = common.ARMS_DIR_00024
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
WARM: Final[tuple[str, ...]] = tuple(common.WARM_ARMS)
COLD: Final[tuple[str, ...]] = tuple(common.COLD_ARMS)
FIT_ARMS: Final[tuple[str, ...]] = (*WARM, *COLD)
COLD_SEEDS: Final[tuple[int, ...]] = (0,)
STACK_SUFFIX: Final[str] = "_stack"
PLANNED_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 8.0)
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
    "Held-out (8)": PRIMARY,
}
PLANNED: Final[dict[str, tuple[str, str, str]]] = {
    "C1: ID and static minus ID": ("id_static", "type_id", "Primary (16)"),
    "C2: ID and training-window minus ID": ("id_train", "type_id", "Primary (16)"),
    "C3: ID and recent-window minus ID": ("id_recent", "type_id", "Primary (16)"),
    "C4: ID, training-window, and recent-window minus ID": (
        "id_train_recent",
        "type_id",
        "Primary (16)",
    ),
    "C5: training-window without ID minus ID": ("train_noid", "type_id", "Primary (16)"),
    "C6: training-window without ID minus no fingerprint, no ID": (
        "train_noid",
        "type_noid",
        "Primary (16)",
    ),
    "C7: held-out, training-window minus none": ("cold_train", "cold_none", "Held-out (8)"),
    "C8: held-out, training and recent minus none": (
        "cold_train_recent",
        "cold_none",
        "Held-out (8)",
    ),
}
EXPLORATORY_PAIRS: Final[dict[str, tuple[str, str]]] = {
    "held-out, static minus none (K1)": ("cold_static", "cold_none"),
    "held-out, none minus ID (cost of being new: the ID model saw the series)": (
        "cold_none",
        "type_id",
    ),
    "held-out, training and recent minus ID": ("cold_train_recent", "type_id"),
    "no ID minus ID": ("type_noid", "type_id"),
    "ID and recent minus ID and training": ("id_recent", "id_train"),
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


def load_fit(*, name: str, seeds: tuple[int, ...], directory: Path = ARMS_DIR) -> pl.DataFrame:
    """Return a fit's seeds, keyed by series, run, valid time, and seed."""
    paths = sorted(directory.glob(f"{name}_seed*.parquet"))
    if [p.name for p in paths] != [f"{name}_seed{s}.parquet" for s in seeds]:
        msg = f"fit {name} does not hold exactly seeds {seeds}"
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


def score_table(
    *,
    raw: dict[str, pl.DataFrame],
    cases: pl.DataFrame,
    planned: dict[str, tuple[str, str, str]],
    exploratory: dict[str, tuple[str, str, str]],
) -> tuple[list[dict], list[dict], list[dict], dict[str, int]]:
    """Score one table of arms and return its absolute, per-series, and contrast rows."""
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
    absolute, per_series = [], []
    for name, frame in arms.items():
        for scope in SCOPES:
            for group in GROUPS:
                if group == "Held-out (8)" and "cold_none" not in arms:
                    continue
                part = _scope(frame=frame, scope=scope, group=group)
                if part.is_empty():
                    continue
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
    contrasts = [
        {
            "label": label,
            "planned": label in planned,
            **contrast(
                arms=arms,
                treatment=treatment,
                reference=reference,
                group=group,
                scope=scope,
                metric=metric,
                level=PLANNED_LEVEL if label in planned and scope == "all leads" else 95.0,
                decisive=label in planned and scope == "all leads",
            ),
            "group": group,
        }
        for label, (treatment, reference, group) in {**planned, **exploratory}.items()
        for scope in SCOPES
        for metric in ("quantile_score", "median_error")
    ]
    return absolute, per_series, contrasts, rows


def main() -> None:
    """Build and score every arm and write the tables."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    series = common.series_table()
    cases = _actual_table(series=series)
    held_ids = series.filter(pl.col("site").is_in(common.HELD_OUT))["time_series_id"].to_list()
    fits = {
        name: load_fit(
            name=name,
            seeds=common.SEEDS,
            directory=ARMS_DIR_00024 if name in ("type_id", "type_noid") else ARMS_DIR,
        )
        for name in WARM
    }
    fits |= {name: load_fit(name=name, seeds=COLD_SEEDS) for name in COLD}
    reference_keys = fits["type_id"].select(KEYS).unique().sort(KEYS)
    for name in WARM:
        if not fits[name].select(KEYS).unique().sort(KEYS).equals(reference_keys):
            msg = f"arm {name} does not hold the same series, runs, and valid times as type_id"
            raise ValueError(msg)
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
    for name in (*common.NEW_WARM_ARMS, *COLD):
        for variant in (name, f"{name}{STACK_SUFFIX}"):
            try:
                experiment_store.write_predictions(
                    experiment_id=common.EXPERIMENT_ID,
                    name=variant,
                    frame=raw[variant]
                    .select(*KEYS, "seed", *common.LEVEL_COLUMNS)
                    .filter(pl.col("valid_time") < HOLDOUT),
                    kind="quantiles",
                    row_set_id=common.ROW_SET_ID,
                )
            except FileExistsError:
                continue
            except ValueError as error:
                print(f"predictions of {variant} not stored: {error}")
    warm_names = [*WARM, *(f"{n}{STACK_SUFFIX}" for n in WARM), *STORED_ARMS]
    cold_names = [
        *COLD,
        *(f"{n}{STACK_SUFFIX}" for n in COLD),
        "type_id",
        "type_noid",
        "type_id_stack",
        "type_noid_stack",
        *STORED_ARMS,
    ]
    warm_planned = {k: v for k, v in PLANNED.items() if v[2] != "Held-out (8)"}
    cold_planned = {k: v for k, v in PLANNED.items() if v[2] == "Held-out (8)"}
    suffix = lambda d: {  # noqa: E731
        k: (f"{t}{STACK_SUFFIX}", f"{r}{STACK_SUFFIX}", g) for k, (t, r, g) in d.items()
    }
    warm_exploratory = {
        f"{label}, {group}": (f"{t}{STACK_SUFFIX}", f"{r}{STACK_SUFFIX}", group)
        for label, (t, r) in EXPLORATORY_PAIRS.items()
        if not label.startswith("held-out")
        for group in ("Primary (16)", "All (20)", "BSP and GSP (4)")
    }
    warm_exploratory |= {
        f"raw {label}, {group}": (t, r, group)
        for label, (t, r) in {
            "C1": ("id_static", "type_id"),
            "C2": ("id_train", "type_id"),
            "C3": ("id_recent", "type_id"),
            "C4": ("id_train_recent", "type_id"),
            "C5": ("train_noid", "type_id"),
        }.items()
        for group in ("Primary (16)",)
    }
    warm_exploratory |= {
        f"{arm} stack minus manual_heuristic, {group}": (
            f"{arm}{STACK_SUFFIX}",
            "manual_heuristic",
            group,
        )
        for arm in WARM
        for group in ("Primary (16)",)
    }
    cold_exploratory = {
        f"{label}": (f"{t}{STACK_SUFFIX}", f"{r}{STACK_SUFFIX}", "Held-out (8)")
        for label, (t, r) in EXPLORATORY_PAIRS.items()
        if label.startswith("held-out")
    }
    cold_exploratory |= {
        f"raw {label}": (t, r, "Held-out (8)")
        for label, (t, r) in {
            "C7": ("cold_train", "cold_none"),
            "C8": ("cold_train_recent", "cold_none"),
        }.items()
    }
    warm_out = score_table(
        raw={n: raw[n] for n in warm_names},
        cases=cases,
        planned=suffix(warm_planned),
        exploratory=warm_exploratory,
    )
    held_cases = cases.filter(pl.col("time_series_id").is_in(held_ids))
    cold_raw = {
        n: raw[n].filter(pl.col("time_series_id").is_in(held_ids) & (pl.col("seed") == 0))
        for n in cold_names
    }
    cold_out = score_table(
        raw=cold_raw,
        cases=held_cases,
        planned=suffix(cold_planned),
        exploratory=cold_exploratory,
    )
    warm_absolute, warm_per_series, warm_contrasts, warm_rows = warm_out
    cold_absolute, cold_per_series, cold_contrasts, cold_rows = cold_out
    for file, warm_part, cold_part in (
        ("absolute", warm_absolute, cold_absolute),
        ("per_series", warm_per_series, cold_per_series),
        ("contrasts", warm_contrasts, cold_contrasts),
    ):
        pl.DataFrame(
            [
                *({**r, "table": "warm"} for r in warm_part),
                *({**r, "table": "cold"} for r in cold_part),
            ],
            strict=False,
            infer_schema_length=None,
        ).write_parquet(RESULTS_DIR / f"{file}.parquet")
    runs = tables[0].filter(pl.col("seed") == 0)
    print(
        "rows per seed (warm, cold):",
        next(iter(warm_rows.values())),
        next(iter(cold_rows.values())),
    )
    print(
        "runs with no correction (seed 0, first fit):",
        int((runs["history_count"] < MIN_COUNT).sum()),
        "of",
        runs.height,
    )


if __name__ == "__main__":
    main()
