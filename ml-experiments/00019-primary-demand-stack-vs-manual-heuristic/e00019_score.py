"""Build experiment 00019's stacked arms from N1 and N1H, score them, and write the tables.

The components are a 14-day error correction (C), a recency half-life of 0.5 years (H, the N1H fit)
and a spread factor of 1.15 (S). The correction adds, to every quantile of a run, the mean of
(observed minus median forecast) over the 14 days ending at the newest usable observation, using for
each past half-hour the forecast of the latest earlier validation run. Scores are on the cases all
arms hold, in points of capacity; intervals resample whole validation months and the three seeds.
The nine planned contrasts are read at 99.44% over all leads; every other contrast is exploratory.

Output under `work/results/`: `absolute.parquet`, `contrasts.parquet`, `bias.parquet`,
`corrections.parquet`, and the prediction sets under `data/ml-experiments/00019-.../predictions/`.

    uv run python ml-experiments/00019-primary-demand-stack-vs-manual-heuristic/e00019_score.py
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00019_common as common
import numpy as np
import polars as pl

from studies import bootstrap, experiment_store

ARMS_DIR: Final[Path] = common.WORK_DIR / "arms"
RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
HOLDOUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
STORED_ARMS: Final[tuple[str, ...]] = ("manual_heuristic", "climatology", "production")
MEDIAN: Final[int] = 4
"""Position of the median (q50) among the nine levels."""
NEWEST_AGE: Final[int] = int(common.TELEMETRY_DELAY / common.STEP) + 1
"""Half-hours between the origin and the newest usable observation (00013's 3.5 hours)."""
WINDOW: Final[int] = common.CORRECTION_DAYS * common.STEPS_PER_DAY
MIN_COUNT: Final[int] = common.MIN_CORRECTION_DAYS * common.STEPS_PER_DAY
STACK_ARMS: Final[dict[str, tuple[str, bool, bool]]] = {
    # name: (base fit, correction, spread)
    "n1_c": ("n1", True, False),
    "n1_s": ("n1", False, True),
    "stack": ("n1h", True, True),
    "stack_minus_c": ("n1h", False, True),
    "stack_minus_h": ("n1", True, True),
    "stack_minus_s": ("n1h", True, False),
}
FIT_ARMS: Final[tuple[str, ...]] = ("n1", "n1h")
PLANNED_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 9.0)
N_RESAMPLES: Final[int] = 20_000
DEFAULT_RESAMPLES: Final[int] = 2_000
SCOPES: Final[tuple[str, ...]] = ("all leads", "up to 48 h", "3 d or more")
PLANNED: Final[dict[str, tuple[str, str]]] = {
    "C1 (primary): stack minus manual_heuristic": ("stack", "manual_heuristic"),
    "C2: stack minus N1": ("stack", "n1"),
    "C3: stack minus production": ("stack", "production"),
    "C4: stack without correction minus stack": ("stack_minus_c", "stack"),
    "C5: stack without half-life minus stack": ("stack_minus_h", "stack"),
    "C6: stack without spread minus stack": ("stack_minus_s", "stack"),
    "C7: N1 plus correction minus N1": ("n1_c", "n1"),
    "C8: N1H minus N1": ("n1h", "n1"),
    "C9: N1 plus spread minus N1": ("n1_s", "n1"),
}
EXPLORATORY: Final[dict[str, tuple[str, str]]] = {
    "E1: N1 minus manual_heuristic": ("n1", "manual_heuristic"),
    "E2: stack minus climatology": ("stack", "climatology"),
    "E3: N1 plus correction minus manual_heuristic": ("n1_c", "manual_heuristic"),
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


def stack(*, frame: pl.DataFrame, correct: bool, spread: bool) -> pl.DataFrame:
    """Apply the correction and the spread factor to a fit's quantiles."""
    quantiles = frame.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
    if correct:
        quantiles = quantiles + frame["correction"].to_numpy()[:, None]
    if spread:
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
    folder = ARMS_DIR if name == "n1h" else common.ARMS_DIR_00013
    paths = sorted(folder.glob(f"{name}_seed*.parquet"))
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
        "time_series_id", group=pl.col("substation_type").cast(pl.String)
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
    frame = frame.filter(pl.col("group") == group)
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
    scope: str,
    metric: str,
    level: float,
    decisive: bool,
) -> dict[str, float | int | str]:
    """Return one paired contrast's estimate, interval, and month agreement, Primary only."""
    losses = experiment_store.losses_for_bootstrap(
        arm_losses={
            name: _scope(frame=arms[name], scope=scope, group="Primary")
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
    raw: dict[str, pl.DataFrame] = {}
    for name in FIT_ARMS:
        raw[name] = fits[name]
    for name, (base, correct, spread) in STACK_ARMS.items():
        raw[name] = stack(frame=corrected[base], correct=correct, spread=spread)
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
    for name in (*FIT_ARMS, *STACK_ARMS):
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
    absolute = []
    for name, frame in arms.items():
        for scope in SCOPES:
            part = _scope(frame=frame, scope=scope, group="Primary")
            losses = experiment_store.losses_for_bootstrap(arm_losses={name: part})
            for metric in ("quantile_score", "median_error"):
                interval = bootstrap.bootstrap_absolute(losses=losses, arm=name, metric=metric)
                absolute.append({"arm": name, "scope": scope, "metric": metric, **interval})
            absolute.append(
                {
                    "arm": name,
                    "scope": scope,
                    "metric": "coverage",
                    "value": float(part["covered_90"].to_numpy().mean()),
                    "coverage_80": float(part["covered_80"].to_numpy().mean()),
                    "bias": float(part["signed_error"].to_numpy().mean()),
                }
            )
    pl.DataFrame(absolute, strict=False).write_parquet(RESULTS_DIR / "absolute.parquet")
    contrasts = [
        {
            "label": label,
            "planned": label in PLANNED,
            **contrast(
                arms=arms,
                treatment=treatment,
                reference=reference,
                scope=scope,
                metric=metric,
                level=PLANNED_LEVEL if label in PLANNED and scope == "all leads" else 95.0,
                decisive=label in PLANNED and scope == "all leads",
            ),
        }
        for label, (treatment, reference) in {**PLANNED, **EXPLORATORY}.items()
        for scope in SCOPES
        for metric in ("quantile_score", "median_error")
    ]
    result = pl.DataFrame(contrasts)
    result.write_parquet(RESULTS_DIR / "contrasts.parquet")
    runs = tables[0].filter(pl.col("seed") == 0)
    print("rows per seed:", rows[next(iter(rows))])
    print(
        "runs with no correction (n1, seed 0, all series):",
        int((runs["history_count"] < MIN_COUNT).sum()),
        "of",
        runs.height,
    )
    with pl.Config(tbl_rows=100, tbl_cols=20, fmt_str_lengths=60):
        print(
            pl.DataFrame(absolute, strict=False)
            .filter(pl.col("scope") == "all leads")
            .select("arm", "metric", "value", "coverage_80", "bias", "lower_95", "upper_95")
        )
        print(
            result.filter(pl.col("scope") == "all leads").select(
                "label", "metric", "difference", "lower", "upper", "level", "months_negative"
            )
        )


if __name__ == "__main__":
    main()
