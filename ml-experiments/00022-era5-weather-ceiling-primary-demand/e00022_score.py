"""Score experiment 00022's ERA5-weather arms: stack each fit's quantiles and read the contrasts.

Every arm is experiment 00019's full stack (N1H, the 14-day error correction, the spread factor of
1.15) applied to the quantiles of `e00022_fit.py`; the correction history of an arm comes from that
arm's own past forecasts. Scores are on the cases all arms hold, Primary substations, in points of
capacity; intervals resample whole validation months and the three seeds. The eight planned
contrasts (each ERA5 arm minus `ens`, over all leads) are read at 99.38%; the lead-band contrasts
are exploratory, read at 95%.

The stored `manual_heuristic`, `climatology`, and `production` arms of 00013 join the arms, so
that the shared rows are 00019's.

Output under `work/results/`: `absolute.parquet`, `contrasts.parquet`, and the prediction sets
under `data/ml-experiments/00022-.../predictions/`.

    uv run python ml-experiments/00022-era5-weather-ceiling-primary-demand/e00022_score.py
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00022_common as common
import numpy as np
import polars as pl
from e00022_fit import ARM_NAMES

from studies import bootstrap, experiment_store

ARMS_DIR: Final[Path] = common.WORK_DIR / "arms"
RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
HOLDOUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
MEDIAN: Final[int] = 4
NEWEST_AGE: Final[int] = int(common.TELEMETRY_DELAY / common.STEP) + 1
WINDOW: Final[int] = common.CORRECTION_DAYS * common.STEPS_PER_DAY
MIN_COUNT: Final[int] = common.MIN_CORRECTION_DAYS * common.STEPS_PER_DAY
PLANNED_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 8.0)
N_RESAMPLES: Final[int] = 20_000
DEFAULT_RESAMPLES: Final[int] = 2_000
SCOPES: Final[tuple[str, ...]] = ("all leads", *(band[0] for band in common.LEAD_BANDS))
STORED_ARMS: Final[tuple[str, ...]] = ("manual_heuristic", "climatology", "production")
CHECK_ARM: Final[str] = "stack"
"""Experiment 00019's stored full stack, used as a check on the `ens` arm."""


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


def _scope(*, frame: pl.DataFrame, scope: str) -> pl.DataFrame:
    frame = frame.filter(pl.col("group") == "Primary")
    if scope == "all leads":
        return frame
    _, low, high = next(band for band in common.LEAD_BANDS if band[0] == scope)
    return frame.filter((pl.col("lead_hours") >= low) & (pl.col("lead_hours") < high))


def contrast(
    *, arms: dict[str, pl.DataFrame], treatment: str, reference: str, scope: str, decisive: bool
) -> dict[str, float | int | str]:
    """Return one paired contrast's estimate, interval, and month agreement, Primary only."""
    losses = experiment_store.losses_for_bootstrap(
        arm_losses={name: _scope(frame=arms[name], scope=scope) for name in (treatment, reference)}
    )
    level = PLANNED_LEVEL if decisive else 95.0
    lower, upper = bootstrap.bootstrap_difference_at_level(
        losses=losses,
        treatment=treatment,
        reference=reference,
        metric="quantile_score",
        level=level,
        n_resamples=N_RESAMPLES if decisive else DEFAULT_RESAMPLES,
    )
    paired = (
        losses.filter(pl.col("arm") == treatment)
        .select("site", "time", "seed", "month", t=pl.col("quantile_score"))
        .join(
            losses.filter(pl.col("arm") == reference).select(
                "site", "time", "seed", r=pl.col("quantile_score")
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
        "level": level,
        "difference": float(paired["difference"].to_numpy().mean()),
        "lower": lower,
        "upper": upper,
        "months_negative": int((by_month["difference"] < 0).sum()),
        "months": by_month.height,
    }


def load_arm(*, name: str) -> pl.DataFrame:
    """Return an arm's three seeds, keyed by series, run, valid time, and seed."""
    paths = [ARMS_DIR / f"{name}_seed{seed}.parquet" for seed in common.SEEDS]
    return pl.concat([pl.read_parquet(path) for path in paths]).sort(*KEYS, "seed")


def main() -> None:
    """Stack and score every arm and write the tables."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    series = common.series_table()
    cases = _actual_table(series=series)
    raw: dict[str, pl.DataFrame] = {}
    history = []
    for name in ARM_NAMES:
        corrected, table = with_corrections(base=load_arm(name=name), series=series)
        raw[name] = stack(frame=corrected, correct=True, spread=True)
        history.append(table.with_columns(arm=pl.lit(name)))
    pl.concat(history).write_parquet(RESULTS_DIR / "corrections.parquet")
    full = {
        name: frame.join(cases, on=KEYS, how="inner", validate="m:1") for name, frame in raw.items()
    }
    stored = experiment_store.read_predictions(
        experiment_id=common.EXPERIMENT_ID_00019, name=CHECK_ARM, row_set_id=common.ROW_SET_ID
    )
    full[CHECK_ARM] = stored.select(*KEYS, "seed", *common.LEVEL_COLUMNS).join(
        cases, on=KEYS, how="inner", validate="m:1"
    )
    for stored_name in STORED_ARMS:
        stored_arm = pl.read_parquet(common.ARMS_DIR_00013 / f"{stored_name}.parquet").select(
            *KEYS, *common.LEVEL_COLUMNS
        )
        full[stored_name] = pl.concat(
            [stored_arm.with_columns(seed=pl.lit(seed).cast(pl.Int8)) for seed in common.SEEDS]
        ).join(cases, on=KEYS, how="inner", validate="m:1")
    arms = prepare(arms=full)
    rows = {name: frame.height // frame["seed"].n_unique() for name, frame in arms.items()}
    if len(set(rows.values())) != 1:
        msg = f"arms hold different numbers of shared cases per seed: {rows}"
        raise ValueError(msg)
    for name in ARM_NAMES:
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
            part = _scope(frame=frame, scope=scope)
            losses = experiment_store.losses_for_bootstrap(arm_losses={name: part})
            interval = bootstrap.bootstrap_absolute(
                losses=losses, arm=name, metric="quantile_score"
            )
            absolute.append(
                {
                    "arm": name,
                    "scope": scope,
                    **interval,
                    "coverage_90": float(part["covered_90"].to_numpy().mean()),
                    "coverage_80": float(part["covered_80"].to_numpy().mean()),
                    "bias": float(part["signed_error"].to_numpy().mean()),
                }
            )
    pl.DataFrame(absolute, strict=False).write_parquet(RESULTS_DIR / "absolute.parquet")
    contrasts = [
        contrast(
            arms=arms,
            treatment=name,
            reference="ens",
            scope=scope,
            decisive=scope == "all leads",
        )
        for name in ARM_NAMES
        if name != "ens"
        for scope in SCOPES
    ]
    contrasts.append(
        contrast(arms=arms, treatment="ens", reference=CHECK_ARM, scope="all leads", decisive=False)
    )
    result = pl.DataFrame(contrasts)
    result.write_parquet(RESULTS_DIR / "contrasts.parquet")
    print("rows per seed:", rows[next(iter(rows))])
    with pl.Config(tbl_rows=100, tbl_cols=20, fmt_str_lengths=60):
        print(
            pl.DataFrame(absolute, strict=False)
            .filter(pl.col("scope") == "all leads")
            .select("arm", "value", "lower_95", "upper_95", "coverage_90", "bias")
        )
        print(result.filter(pl.col("scope") == "all leads"))


if __name__ == "__main__":
    main()
