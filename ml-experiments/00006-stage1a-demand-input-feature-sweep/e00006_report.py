"""Score the sweep: absolute errors, contrasts with intervals, and breakdowns, as tables.

Reads the prediction files `e00006_fit.py` wrote and writes tables under
`data/ml-experiments/00006-.../results/` that `e00006_charts.py` draws and the write-up
quotes. Every
number is the mean absolute error in points of capacity (a point is 1% of the series' effective
capacity), over a substation type's series, with 95% intervals that resample whole validation months
and the fitting seed.

    uv run python ml-experiments/00006-stage1a-demand-input-feature-sweep/e00006_report.py
"""

import sys
from pathlib import Path
from typing import Final

import e00006_fit as fit
import e00006_inputs as inputs
import e00006_sweep as sweep
import polars as pl
from baseline_forecasters.climatology import (
    CLIMATOLOGY_QUANTILE_COLUMNS,
    _local_calendar_cell_keys,
    _pooled_quantile_lookup,
)
from studies.bootstrap import bootstrap_absolute, bootstrap_difference_at_level
from studies.experiment_store import losses_for_bootstrap

RESULTS_DIR: Final[Path] = inputs.EXPERIMENT_DIR / "results"
TYPES: Final[tuple[str, ...]] = ("Primary", "BSP", "GSP")
N_PLANNED_CONTRASTS: Final[int] = 9
PLANNED_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / N_PLANNED_CONTRASTS)
"""Bonferroni-widened coverage, in percent, across the nine planned contrasts."""
N_RESAMPLES_PLANNED: Final[int] = 20_000
"""Resamples for the Bonferroni-widened planned intervals, whose tails hold 0.28% of the draws."""
N_RESAMPLES_95: Final[int] = 4_000
CLIMATOLOGY: Final[str] = "climatology"
MEDIAN_COLUMN: Final[str] = CLIMATOLOGY_QUANTILE_COLUMNS[25]
"""The climatology's equiprobable quantile at level 25.5 / 51 = 0.5."""


def series_types() -> pl.DataFrame:
    """Return `time_series_id` and `substation_type` for the 20 series."""
    return pl.read_parquet(inputs.INPUTS_DIR / "series.parquet").select(
        "time_series_id", "substation_type"
    )


def read_seeds(*, stage: str, fold_name: str, config: str = "series") -> pl.DataFrame:
    """Read every seed of a stage: rows `(time_series_id, valid_time, target, seed, <arms>)`."""
    folder = fit.FOLDS_DIR / stage / fold_name / config
    pieces = []
    for seed_dir in sorted(folder.glob("seed*")):
        seed = int(seed_dir.name.removeprefix("seed"))
        frame = sweep.read_stage(stage=stage, fold_name=fold_name, config=config, seed=seed)
        pieces.append(frame.with_columns(seed=pl.lit(seed)))
    return pl.concat(pieces, how="diagonal")


def of_type(*, rows: pl.DataFrame, substation_type: str) -> pl.DataFrame:
    """Keep the rows of one substation type's series."""
    keep = series_types().filter(pl.col("substation_type") == substation_type)["time_series_id"]
    return rows.filter(pl.col("time_series_id").is_in(keep.implode()))


def arm_columns(*, rows: pl.DataFrame) -> list[str]:
    """Return the prediction columns of a stage frame."""
    return [c for c in rows.columns if c not in ("time_series_id", "valid_time", "target", "seed")]


def loss_frame(*, rows: pl.DataFrame, arms: list[str]) -> pl.DataFrame:
    """Return the bootstrap's loss frame with `abs_error` in points of capacity."""
    keyed = rows.with_columns(power_fcst_init_time=pl.col("valid_time").dt.truncate("1mo"))
    return losses_for_bootstrap(
        arm_losses={
            arm: keyed.select(
                "time_series_id",
                "power_fcst_init_time",
                "valid_time",
                "seed",
                abs_error=(pl.col(arm) - pl.col("target")).abs() * 100.0,
            ).drop_nulls("abs_error")
            for arm in arms
        }
    )


def levels_table(*, rows: pl.DataFrame, arms: list[str], label: str) -> pl.DataFrame:
    """Return each arm's mean absolute error and 95% interval."""
    frame = loss_frame(rows=rows, arms=arms)
    out = []
    for arm in arms:
        result = bootstrap_absolute(losses=frame, arm=arm, metric="abs_error")
        out.append(
            {
                "scope": label,
                "arm": arm,
                "value": result["value"],
                "lower": result["lower_95"],
                "upper": result["upper_95"],
                "seed_spread": result["seed_spread"],
                "n_months": result["n_months"],
            }
        )
    return pl.DataFrame(out)


def contrast_table(
    *, rows: pl.DataFrame, pairs: list[tuple[str, str, str]], label: str, level: float = 95.0
) -> pl.DataFrame:
    """Return paired differences `(label, treatment, reference)` with an interval at `level`."""
    arms = sorted({a for _, t, r in pairs for a in (t, r)})
    frame = loss_frame(rows=rows, arms=arms)
    out = []
    for name, treatment, reference in pairs:
        low, high = bootstrap_difference_at_level(
            losses=frame,
            treatment=treatment,
            reference=reference,
            metric="abs_error",
            level=level,
            n_resamples=N_RESAMPLES_PLANNED if level > 99.0 else N_RESAMPLES_95,
        )
        difference = (
            frame.filter(pl.col("arm") == treatment)["abs_error"].to_numpy().mean()
            - frame.filter(pl.col("arm") == reference)["abs_error"].to_numpy().mean()
        )
        out.append(
            {
                "scope": label,
                "contrast": name,
                "treatment": treatment,
                "reference": reference,
                "difference": float(difference),
                "lower": low,
                "upper": high,
                "level": level,
            }
        )
    return pl.DataFrame(out)


def climatology_rows(*, fold_name: str, rows: pl.DataFrame) -> pl.DataFrame:
    """Add a `climatology` column: the month-by-half-hour median of the fold's training window.

    The cell is the series, the local month, the local half-hour of the day, and whether the day is
    a weekend, pooled over neighbouring months and half-hours, as the repo's
    `ClimatologyForecaster` does. A row whose cell holds no training sample gets null and is
    dropped from the climatology's score.
    """
    fold = fit.FOLDS[fold_name]
    features = pl.read_parquet(
        inputs.INPUTS_DIR / "features.parquet",
        columns=["time_series_id", "valid_time", "power_mw", "effective_capacity_mw"],
    )
    training = features.filter(
        pl.col("power_mw").is_not_null(),
        pl.col("valid_time") >= inputs.TRAIN_START,
        pl.col("valid_time") < fold.train_end,
    ).select("time_series_id", "valid_time", power=pl.col("power_mw"))
    lookup = _pooled_quantile_lookup(training).select(
        "time_series_id",
        "local_month",
        "local_half_hour_of_day",
        "local_is_weekend",
        median_mw=pl.col(MEDIAN_COLUMN),
    )
    keyed = rows.with_columns(
        **_local_calendar_cell_keys(pl.col("valid_time"), time_zone="Europe/London")
    ).join(
        lookup,
        on=["time_series_id", "local_month", "local_half_hour_of_day", "local_is_weekend"],
        how="left",
    )
    capacity = features.group_by("time_series_id").agg(c=pl.col("effective_capacity_mw").first())
    return (
        keyed.join(capacity, on="time_series_id")
        .with_columns(**{CLIMATOLOGY: pl.col("median_mw") / pl.col("c")})
        .drop("median_mw", "c", "local_month", "local_half_hour_of_day", "local_is_weekend")
    )


def write(*, table: pl.DataFrame, name: str) -> None:
    """Write a results table and print it."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.write_parquet(RESULTS_DIR / f"{name}.parquet")
    with pl.Config(tbl_rows=200, tbl_cols=12, fmt_str_lengths=80, tbl_width_chars=200):
        print(f"\n## {name}\n{table}")


def selection_fold() -> None:
    """Tables for the selection fold: add-one-in, planned contrasts P1 to P7, masking, finals."""
    fold_name = "select_2024"
    add_rows = read_seeds(stage="add_one_in", fold_name=fold_name)
    arms = arm_columns(rows=add_rows)
    base = "W+clock"
    pairs = [(a, a, base) for a in arms if a.startswith(("W+clock+", "W+clock|"))]
    base_w = "W"
    pairs_w = [(a, a, base_w) for a in arms if a.startswith(("W+", "W|")) and "clock" not in a]
    pairs_w.append(("W+clock", "W+clock", base_w))
    tables = []
    for substation_type in TYPES:
        sub = of_type(rows=add_rows, substation_type=substation_type)
        tables.append(contrast_table(rows=sub, pairs=pairs, label=substation_type))
        tables.append(contrast_table(rows=sub, pairs=pairs_w, label=f"{substation_type} vs W"))
    write(table=pl.concat(tables), name="add_one_in_selection_fold")

    planned_rows = read_seeds(stage="planned", fold_name=fold_name)
    planned_pairs = [
        ("P1", "W+clock", "W"),
        ("P2", "W+clock+doy1", "W+clock"),
        ("P3", "W+clock+hol_flags", "W+clock"),
        ("P4", "W+clock+t24", "W+clock"),
        ("P5", "W+windchill", "W"),
        ("P6", "W+clock+covid_flag", "W+clock"),
        ("P7", "W+clock|recency", "W+clock"),
    ]
    write(
        table=pl.concat(
            [
                contrast_table(
                    rows=of_type(rows=planned_rows, substation_type=t),
                    pairs=planned_pairs,
                    label=t,
                    level=PLANNED_LEVEL,
                )
                for t in TYPES
            ]
        ),
        name="planned_selection_fold",
    )

    masking = read_seeds(stage="masking", fold_name=fold_name)
    write(
        table=pl.concat(
            [
                levels_table(
                    rows=of_type(rows=masking, substation_type=t),
                    arms=arm_columns(rows=masking),
                    label=t,
                )
                for t in TYPES
            ]
        ),
        name="masking",
    )

    drops = [
        "",
        "-solar_zenith-solar_azimuth",
        "-temperature_2m-dew_point_temperature_2m",
        "-solar_zenith-solar_azimuth-temperature_2m-dew_point_temperature_2m",
    ]
    masking_pairs = [
        (f"W{drop}+clock{suffix} vs none", f"W{drop}+clock{suffix}", f"W{drop}+clock")
        for drop in drops
        for suffix in ("+doy1", "+doy2", "+month_cat")
    ]
    write(
        table=pl.concat(
            [
                contrast_table(
                    rows=of_type(rows=masking, substation_type=t), pairs=masking_pairs, label=t
                )
                for t in TYPES
            ]
        ),
        name="masking_contrasts",
    )

    graded = read_seeds(stage="graded_covid", fold_name=fold_name)
    write(
        table=pl.concat(
            [
                levels_table(
                    rows=of_type(rows=graded, substation_type=t),
                    arms=arm_columns(rows=graded),
                    label=t,
                )
                for t in TYPES
            ]
        ),
        name="graded_covid",
    )


def pooled_versus_series(*, fold_name: str) -> None:
    """Compare the pooled-by-type models with the per-series models on one seed."""
    pooled = sweep.read_stage(stage="pooled", fold_name=fold_name, config="type").with_columns(
        seed=pl.lit(0)
    )
    per_series = read_seeds(stage="finals", fold_name=fold_name).filter(pl.col("seed") == 0)
    tables = []
    for t in TYPES:
        sub_pooled = of_type(rows=pooled, substation_type=t)
        sub_series = of_type(rows=per_series, substation_type=t).rename(
            {c: f"series:{c}" for c in arm_columns(rows=per_series)}
        )
        both = sub_pooled.rename({c: f"pooled:{c}" for c in arm_columns(rows=sub_pooled)}).join(
            sub_series.drop("target"), on=["time_series_id", "valid_time", "seed"], how="inner"
        )
        tables.append(levels_table(rows=both, arms=arm_columns(rows=both), label=t))
    write(table=pl.concat(tables), name=f"pooled_vs_series_{fold_name}")


def finals_levels(*, fold_name: str) -> None:
    """Absolute scores of the finalist arms, with the climatology, by substation type."""
    rows = read_seeds(stage="finals", fold_name=fold_name)
    extra = (
        read_seeds(stage="type_sets", fold_name=fold_name) if fold_name == "confirm_2025" else None
    )
    if extra is not None:
        rows = rows.join(
            extra.drop("target"), on=["time_series_id", "valid_time", "seed"], how="left"
        )
    rows = climatology_rows(fold_name=fold_name, rows=rows)
    arms = [*arm_columns(rows=rows)]
    levels = []
    for t in TYPES:
        sub = of_type(rows=rows, substation_type=t)
        levels.append(levels_table(rows=sub, arms=arms, label=t))
    write(table=pl.concat(levels), name=f"finals_levels_{fold_name}")
    frozen = "W+clock+covid_flag+xmas"
    planned = [
        ("P1 clock", "W+clock", "W"),
        ("P8 frozen vs clock", frozen, "W+clock"),
        ("P9 from 2023-09", f"{frozen}|from2023-09", frozen),
    ]
    other = [
        ("P1 clock", "W+clock", "W"),
        ("P8 frozen vs clock", frozen, "W+clock"),
        ("P9 from 2023-09", f"{frozen}|from2023-09", frozen),
        ("frozen vs W", frozen, "W"),
        ("W vs climatology", "W", CLIMATOLOGY),
        ("W+clock vs climatology", "W+clock", CLIMATOLOGY),
        ("recency", f"{frozen}|recency", frozen),
        ("recency+from 2023-09", f"{frozen}|recency|from2023-09", f"{frozen}|recency"),
        ("frozen vs climatology", frozen, CLIMATOLOGY),
    ]
    contrasts = []
    for t in TYPES:
        sub = of_type(rows=rows, substation_type=t)
        contrasts.append(contrast_table(rows=sub, pairs=planned, label=t, level=PLANNED_LEVEL))
        contrasts.append(contrast_table(rows=sub, pairs=other, label=t))
    write(table=pl.concat(contrasts), name=f"finals_contrasts_{fold_name}")


def extra_selection_tables() -> None:
    """Tables behind the write-up's drop-one-out, replication, and windchill numbers."""
    drop = read_seeds(stage="drop_one", fold_name="select_2024")
    reference = "W+clock+covid_flag+xmas"
    drop_pairs = [(a, a, reference) for a in arm_columns(rows=drop) if a != reference]
    write(
        table=pl.concat(
            [
                contrast_table(
                    rows=of_type(rows=drop, substation_type=t), pairs=drop_pairs, label=t
                )
                for t in TYPES
            ]
        ),
        name="drop_one_selection_fold",
    )
    planned_pairs = [
        ("P1", "W+clock", "W"),
        ("P2", "W+clock+doy1", "W+clock"),
        ("P3", "W+clock+hol_flags", "W+clock"),
        ("P4", "W+clock+t24", "W+clock"),
        ("P5", "W+windchill", "W"),
        ("P6", "W+clock+covid_flag", "W+clock"),
        ("P7", "W+clock|recency", "W+clock"),
    ]
    second = read_seeds(stage="planned", fold_name="select_2023")
    write(
        table=contrast_table(
            rows=of_type(rows=second, substation_type="Primary"),
            pairs=planned_pairs,
            label="Primary",
            level=PLANNED_LEVEL,
        ),
        name="planned_second_selection_fold",
    )
    windchill = read_seeds(stage="production_windchill", fold_name="select_2024")
    pairs = [
        ("unguarded vs W", "W+windchill_unguarded", "W"),
        ("unguarded vs W+clock", "W+clock+windchill_unguarded", "W+clock"),
    ]
    write(
        table=pl.concat(
            [
                contrast_table(
                    rows=of_type(rows=windchill, substation_type=t), pairs=pairs, label=t
                )
                for t in TYPES
            ]
        ),
        name="production_windchill",
    )
    counts = []
    for fold_name in ("select_2023", "select_2024", "confirm_2025"):
        fitted = read_seeds(
            stage="planned" if fold_name != "confirm_2025" else "finals", fold_name=fold_name
        )
        types = series_types()
        ids = fitted["time_series_id"].unique()
        counts.append(
            types.filter(pl.col("time_series_id").is_in(ids.implode()))
            .group_by("substation_type")
            .len()
            .with_columns(fold=pl.lit(fold_name))
        )
    write(table=pl.concat(counts), name="series_counts")


def christmas_check_table() -> None:
    """Contrast each Christmas encoding with the clock plus COVID indicator, on every fold."""
    base = "W+clock+covid_flag"
    tables = []
    for fold_name in ("select_2024", "select_2023", "confirm_2025"):
        rows = of_type(
            rows=read_seeds(stage="christmas_check", fold_name=fold_name),
            substation_type="Primary",
        )
        pairs = [(a, a, base) for a in arm_columns(rows=rows) if a != base]
        tables.append(
            contrast_table(rows=rows, pairs=pairs, label=fold_name).with_columns(
                n_series=pl.lit(rows["time_series_id"].n_unique())
            )
        )
    write(table=pl.concat(tables), name="christmas_check")


def pinball_table(*, fold_name: str) -> None:
    """Mean pinball loss over the 9 levels, and the coverage of the 5-95 interval."""
    rows = read_seeds(stage="finals_quantiles", fold_name=fold_name)
    arms = sorted({c.split("@")[0] for c in rows.columns if "@q" in c})
    out = []
    for t in TYPES:
        sub = of_type(rows=rows, substation_type=t)
        for arm in arms:
            losses = [
                pl.max_horizontal(
                    level * (pl.col("target") - pl.col(f"{arm}@q{level}")),
                    (level - 1.0) * (pl.col("target") - pl.col(f"{arm}@q{level}")),
                )
                for level in fit.QUANTILE_LEVELS
            ]
            scored = sub.select(
                pinball=pl.mean_horizontal(*losses) * 100.0,
                median_error=(pl.col(f"{arm}@q0.5") - pl.col("target")).abs() * 100.0,
                inside=(
                    (pl.col("target") >= pl.col(f"{arm}@q0.05"))
                    & (pl.col("target") <= pl.col(f"{arm}@q0.95"))
                ).cast(pl.Float64),
                seed=pl.col("seed"),
            )
            out.append(
                {
                    "scope": t,
                    "arm": arm,
                    "pinball": float(scored["pinball"].mean()),  # ty: ignore[invalid-argument-type]
                    "median_mae": float(scored["median_error"].mean()),  # ty: ignore[invalid-argument-type]
                    "coverage_5_95": float(scored["inside"].mean()),  # ty: ignore[invalid-argument-type]
                }
            )
    write(table=pl.DataFrame(out), name=f"quantiles_{fold_name}")


def importance_table(*, fold_name: str) -> None:
    """Aggregate gain and permutation importance across series (median and quartiles)."""
    files = sorted(
        (fit.FOLDS_DIR / "importance_series" / fold_name / "series" / "seed0").glob(
            "*_importance.parquet"
        )
    )
    frame = pl.concat([pl.read_parquet(f) for f in files]).with_columns(
        time_series_id=pl.col("unit").cast(pl.Int32)
    )
    frame = frame.join(series_types(), on="time_series_id")
    summary = frame.group_by("substation_type", "arm", "column").agg(
        n_series=pl.len(),
        gain_median=pl.col("gain_share").median(),
        gain_q1=pl.col("gain_share").quantile(0.25),
        gain_q3=pl.col("gain_share").quantile(0.75),
        permutation_median=pl.col("permutation_points").median(),
        permutation_q1=pl.col("permutation_points").quantile(0.25),
        permutation_q3=pl.col("permutation_points").quantile(0.75),
    )
    write(
        table=summary.sort(
            "substation_type", "arm", "gain_median", descending=[False, False, True]
        ),
        name=f"importance_series_{fold_name}",
    )
    pooled_files = sorted(
        (fit.FOLDS_DIR / "importance_pooled" / fold_name / "type" / "seed0").glob(
            "*_importance.parquet"
        )
    )
    pooled = pl.concat([pl.read_parquet(f) for f in pooled_files]).rename(
        {"unit": "substation_type"}
    )
    write(
        table=pooled.sort("substation_type", "arm", "gain_share", descending=[False, False, True]),
        name=f"importance_pooled_{fold_name}",
    )


def month_error(*, fold_name: str, arms: list[str], stage: str) -> None:
    """Mean absolute error by calendar month and by local hour, for Primary and the other types."""
    rows = read_seeds(stage=stage, fold_name=fold_name)
    keyed = rows.with_columns(
        local=pl.col("valid_time").dt.convert_time_zone("Europe/London")
    ).with_columns(month=pl.col("local").dt.month(), hour=pl.col("local").dt.hour())
    out = []
    for t in TYPES:
        sub = of_type(rows=keyed, substation_type=t)
        for arm in arms:
            err = sub.with_columns(error=(pl.col(arm) - pl.col("target")).abs() * 100.0)
            signed = sub.with_columns(error=(pl.col(arm) - pl.col("target")) * 100.0)
            for key in ("month", "hour"):
                grouped = err.group_by(key).agg(mae=pl.col("error").mean())
                bias = signed.group_by(key).agg(bias=pl.col("error").mean())
                out.append(
                    grouped.join(bias, on=key)
                    .with_columns(scope=pl.lit(t), arm=pl.lit(arm), by=pl.lit(key))
                    .rename({key: "bucket"})
                )
    write(
        table=pl.concat(out).select("scope", "arm", "by", "bucket", "mae", "bias"),
        name=f"by_month_hour_{stage}_{fold_name}",
    )


def main() -> None:
    """Run the report named on the command line (default: everything for the selection fold)."""
    what = sys.argv[1] if len(sys.argv) > 1 else "selection"
    if what == "selection":
        selection_fold()
        finals_levels(fold_name="select_2024")
        pooled_versus_series(fold_name="select_2024")
        pinball_table(fold_name="select_2024")
        month_error(fold_name="select_2024", arms=["W+clock", "W+clock+doy1", "W"], stage="planned")
    elif what == "confirmation":
        fold_name = "confirm_2025"
        finals_levels(fold_name=fold_name)
        pooled_versus_series(fold_name=fold_name)
        pinball_table(fold_name=fold_name)
        importance_table(fold_name=fold_name)


if __name__ == "__main__":
    main()
