"""Build experiment 00030's lag-feature arms with and without the 14-day correction, and score them.

Arms A0 (00013's N1) to A5 are scored raw and with the correction (suffix `_c`); the 00019 stack
is N1H with the correction and a spread factor of 1.15. The correction adds, to every quantile of a
run, the mean of (observed minus median forecast) over the 14 days ending at the newest usable
observation, using for each past half-hour the forecast of the latest earlier validation run. Scores
are on the cases all arms hold, in points of capacity; intervals resample whole validation months
and the three seeds. The eight planned contrasts are read at 99.375% over all leads; every other
contrast is exploratory.

Output under `work/results/`: `absolute.parquet`, `contrasts.parquet`, `corrections.parquet`, and
the prediction sets under `data/ml-experiments/00030-.../predictions/`.

    uv run python ml-experiments/00030-lagged-power-feature-sets-primary-demand/e00030_score.py
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00030_common as common
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
FIT_ARMS: Final[tuple[str, ...]] = ("a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7")
"""A0 is 00013's N1; A6, the residual-lag arm, was not run; A7 is post hoc."""
STACK: Final[str] = "stack"
PLANNED_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 8.0)
N_RESAMPLES: Final[int] = 20_000
DEFAULT_RESAMPLES: Final[int] = 2_000
SCOPES: Final[tuple[str, ...]] = ("all leads", "up to 48 h", "3 d or more")
PLANNED: Final[dict[str, tuple[str, str]]] = {
    "C1: weekly analogues, A1 minus A0": ("a1", "a0"),
    "C2: annual analogues, A2 minus A1": ("a2", "a1"),
    "C3 (primary): all 13 lags, A2 minus A0": ("a2", "a0"),
    "C4: summaries instead of columns, A3 minus A2": ("a3", "a2"),
    "C5: recent values, A4 minus A2": ("a4", "a2"),
    "C6: same-day-type analogues, A5 minus A2": ("a5", "a2"),
    "C7: A2 minus manual_heuristic": ("a2", "manual_heuristic"),
    "C8: with the correction, A2 minus A0": ("a2_c", "a0_c"),
}
EXPLORATORY: Final[dict[str, tuple[str, str]]] = {
    "E1: A0 minus manual_heuristic": ("a0", "manual_heuristic"),
    "E2: A1 with the correction minus A0 with the correction": ("a1_c", "a0_c"),
    "E3: A3 with the correction minus A2 with the correction": ("a3_c", "a2_c"),
    "E4: A4 with the correction minus A2 with the correction": ("a4_c", "a2_c"),
    "E5: A5 with the correction minus A2 with the correction": ("a5_c", "a2_c"),
    "E6: A2 with the correction minus the 00019 stack": ("a2_c", STACK),
    "E7: A0 with the correction minus the 00019 stack": ("a0_c", STACK),
    "E8: A2 with the correction minus manual_heuristic": ("a2_c", "manual_heuristic"),
    "E9: A6 minus A2": ("a6", "a2"),
    "E10: A6 with the correction minus A2 with the correction": ("a6_c", "a2_c"),
    "E11: A0 minus climatology": ("a0", "climatology"),
    "E12: A3 with the correction minus A0 with the correction": ("a3_c", "a0_c"),
    "E13: A3 with the correction minus the 00019 stack": ("a3_c", STACK),
    "E14 (post hoc): A7 minus A0": ("a7", "a0"),
    "E15 (post hoc): A7 with the correction minus A0 with the correction": ("a7_c", "a0_c"),
    "E16 (post hoc): A7 with the correction minus the 00019 stack": ("a7_c", STACK),
    "E17 (post hoc): A7 minus A1": ("a7", "a1"),
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


def load_fit(*, name: str, seeds: tuple[int, ...]) -> pl.DataFrame:
    """Return a fit's three seeds, keyed by series, run, valid time, and seed."""
    folder, stem = {
        "a0": (common.ARMS_DIR_00013, "n1"),
        "n1h": (common.ARMS_DIR_00019, "n1h"),
    }.get(name, (ARMS_DIR, name))
    paths = sorted(folder.glob(f"{stem}_seed*.parquet"))
    if [p.name for p in paths if int(p.stem[-1]) in seeds] != [
        f"{stem}_seed{s}.parquet" for s in seeds
    ]:
        msg = f"fit {name} does not hold exactly seeds {seeds}"
        raise ValueError(msg)
    return pl.concat([pl.read_parquet(p) for p in paths if int(p.stem[-1]) in seeds]).sort(
        *KEYS, "seed"
    )


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(common.SEEDS))
    parser.add_argument("--arms", nargs="+", default=list(FIT_ARMS))
    parser.add_argument("--tag", default="results")
    return parser.parse_args()


def main() -> None:
    """Build and score the requested arms and seeds and write the tables."""
    args = _parse_args()
    seeds = tuple(args.seeds)
    results_dir = RESULTS_DIR.with_name(args.tag)
    results_dir.mkdir(parents=True, exist_ok=True)
    series = common.series_table()
    cases = _actual_table(series=series)
    names = [n for n in FIT_ARMS if n in args.arms]
    fits = {name: load_fit(name=name, seeds=seeds) for name in (*names, "n1h")}
    corrected, tables = {}, []
    for name, fit in fits.items():
        corrected[name], table = with_corrections(base=fit, series=series)
        tables.append(table.with_columns(fit=pl.lit(name)))
    pl.concat(tables).write_parquet(results_dir / "corrections.parquet")
    raw: dict[str, pl.DataFrame] = {}
    for name in names:
        raw[name] = fits[name]
        raw[f"{name}_c"] = stack(frame=corrected[name], correct=True, spread=False)
    raw[STACK] = stack(frame=corrected["n1h"], correct=True, spread=True)
    for name in STORED_ARMS:
        stored = pl.read_parquet(common.ARMS_DIR_00013 / f"{name}.parquet").select(
            *KEYS, *common.LEVEL_COLUMNS
        )
        raw[name] = pl.concat(
            [stored.with_columns(seed=pl.lit(seed).cast(pl.Int8)) for seed in seeds]
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
    for name in (*names, *(f"{n}_c" for n in names), STACK):
        if name == "a0" or len(seeds) != len(common.SEEDS):
            continue
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
    pl.DataFrame(absolute, strict=False).write_parquet(results_dir / "absolute.parquet")
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
        if treatment in arms and reference in arms
        for scope in SCOPES
        for metric in ("quantile_score", "median_error")
    ]
    result = pl.DataFrame(contrasts)
    result.write_parquet(results_dir / "contrasts.parquet")
    runs = tables[0].filter(pl.col("seed") == 0)
    print("rows per seed:", rows[next(iter(rows))])
    print(
        "runs with no correction (a0, seed 0, all series):",
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
