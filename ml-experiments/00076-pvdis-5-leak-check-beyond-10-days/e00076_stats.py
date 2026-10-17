"""Assemble `am0` and `am_train_only`, score them against experiment 00064, print every number.

Reads `work/arms_raw/` (see `e00076_fit.py`) and experiment 00064's `work/arms/` and
`work/arms_raw/`. Writes `work/arms/<arm>.parquet` (the 14-day correction and the 1.15 spread
factor applied, with `actual`), the prediction sets, and `work/results/*.parquet`.

The scored rows are experiment 00064's: daylight half-hours of April to September with a CAMS `u`
and an observed power, on the five verdict series. The score is the quantile score in points of
capacity. The decision rule is in `docs/ml-experiments/00076-*/result.md`; the three decision
contrasts at 10 days and beyond are read at 98.33% (95% split across three contrasts) from 50,000
resamples of whole ISO weeks of forecast start. Every other read is exploratory, at 95%.

    uv run python ml-experiments/00076-*/e00076_stats.py
"""

from typing import Final

import e00076_common as common
import numpy as np
import polars as pl

from studies import bootstrap, experiment_store

RAW_SOURCES: Final[dict[str, tuple[str, str]]] = {
    "am_raw": ("e64", "am"),
    "am0_raw": ("own", "am0"),
    "d1l_raw": ("e46", "d1l"),
}
"""Arms scored without the 14-day correction (the spread factor is kept), for the raw reads.

The correction adds the mean of an arm's own recent median errors, which come mostly from leads
under 10 days, so a scored contrast at 10 days and beyond mixes the models with their corrections.
"""
NEW_ARMS: Final[tuple[str, ...]] = ("am0", "am_train_only")
STORED_ARMS: Final[tuple[str, ...]] = ("d0", "d1l", "am")
LONG_BAND: Final[int] = 4
"""Index of the 10-days-and-beyond lead band."""
DECISION_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 3)
EXPLORATORY_LEVEL: Final[float] = 95.0
DECISION_RESAMPLES: Final[int] = 50_000
EXPLORATORY_RESAMPLES: Final[int] = 10_000
EQUALITY_TOLERANCE: Final[float] = 1e-6
DECISION: Final[tuple[tuple[str, str], ...]] = (
    ("am", "am0"),
    ("am0", "d1l"),
    ("am", "d1l"),
)
"""(treatment, reference) of the three decision contrasts at 10 days and beyond."""
RAW_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("am_raw", "am0_raw"),
    ("am0_raw", "d1l_raw"),
    ("am_raw", "d1l_raw"),
)
EXPLORATORY: Final[tuple[tuple[str, str], ...]] = (
    ("am", "d0"),
    ("d1l", "d0"),
    ("am0", "d0"),
    ("am_train_only", "am"),
)


def assemble() -> None:
    """Write the scored form of each new arm and its prediction set."""
    common.ARMS_DIR.mkdir(parents=True, exist_ok=True)
    series = common.series_table().filter(pl.col("site").is_in(list(common.SITES)))
    arrays = common.load_series_arrays(series=series)
    actual = common.actual_table(series=series, arrays=arrays)
    for arm in NEW_ARMS:
        target = common.ARMS_DIR / f"{arm}.parquet"
        if target.exists():
            continue
        raw = pl.concat(
            [
                pl.read_parquet(common.RAW_ARMS_DIR / f"{arm}_seed{seed}_{site}.parquet")
                .filter(pl.col("variant") == "base")
                .select(*common.KEYS, "seed", *common.LEVEL_COLUMNS)
                .with_columns(pl.col("seed").cast(pl.Int8))
                for seed in common.SEEDS
                for site in common.SITES
            ]
        )
        corrected = common.with_corrections(base=raw, series=series, arrays=arrays)
        scored = common.stack(frame=corrected, correct=True, spread=True).filter(
            pl.col("valid_time") < common.HOLDOUT
        )
        experiment_store.write_predictions(
            experiment_id=common.EXPERIMENT_ID,
            name=arm,
            frame=scored,
            kind="quantiles",
            row_set_id=common.ROW_SET_ID,
        )
        joined = scored.join(actual, on=list(common.KEYS), how="left", validate="m:1")
        common.write_atomically(frame=joined, target=target)
        print(f"{arm}: {joined.height:,} rows", flush=True)


def raw_equality() -> pl.DataFrame:
    """Return the largest raw-quantile difference between `am_train_only` and `am`, by lead band.

    Both are the same booster; `am_train_only` is predicted with a margin of zero at every lead, so
    the two must agree exactly at leads of 10 days and beyond, where `am`'s margin is zero.
    """
    pieces = []
    for seed in common.SEEDS:
        for site in common.SITES:
            own = pl.read_parquet(common.RAW_ARMS_DIR / f"am_train_only_seed{seed}_{site}.parquet")
            stored = pl.read_parquet(common.E64_RAW_ARMS_DIR / f"am_seed{seed}_{site}.parquet")
            both = own.join(
                stored.filter(pl.col("variant") == "base"),
                on=[*common.KEYS, "variant"],
                suffix="_am",
                validate="1:1",
            )
            difference = np.max(
                np.abs(
                    both.select(common.LEVEL_COLUMNS).to_numpy()
                    - both.select([f"{c}_am" for c in common.LEVEL_COLUMNS]).to_numpy()
                ),
                axis=1,
            )
            lead = (both["valid_time"] - both["power_fcst_init_time"]).dt.total_minutes() / 60.0
            pieces.append(
                pl.DataFrame(
                    {
                        "band": common.lead_band(lead_hours=lead.to_numpy()),
                        "difference": difference,
                    }
                )
            )
    return (
        pl.concat(pieces)
        .group_by("band")
        .agg(max_difference=pl.col("difference").max(), rows=pl.len())
        .sort("band")
    )


def with_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add `qs` in points of capacity to a frame of quantiles and `actual`."""
    quantiles = frame.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)
    return frame.with_columns(
        qs=pl.Series(common.pinball_quantile_score(quantiles=quantiles, actual=actual) * 100.0)
    )


def raw_arm(*, name: str, series: pl.DataFrame, actual: pl.DataFrame) -> pl.DataFrame:
    """Return an arm's quantiles with the spread factor but no 14-day correction, and `actual`."""
    source, arm = RAW_SOURCES[name]
    folder = {
        "e64": common.E64_RAW_ARMS_DIR,
        "own": common.RAW_ARMS_DIR,
        "e46": common.E46_RAW_ARMS_DIR,
    }[source]
    raw = pl.concat(
        [
            pl.read_parquet(folder / f"{arm}_seed{seed}_{site}.parquet")
            .filter(pl.col("variant") == "base")
            .select(*common.KEYS, "seed", *common.LEVEL_COLUMNS)
            .with_columns(pl.col("seed").cast(pl.Int8))
            for seed in common.SEEDS
            for site in common.SITES
        ]
    )
    stacked = common.stack(frame=raw, correct=False, spread=True).filter(
        pl.col("valid_time") < common.HOLDOUT
    )
    assert series.height == stacked["time_series_id"].n_unique()
    return stacked.join(actual, on=list(common.KEYS), how="left", validate="m:1")


def load_losses() -> pl.DataFrame:
    """Return every compared arm's per-row losses on experiment 00064's scored rows."""
    series = common.series_table().filter(pl.col("site").is_in(list(common.SITES)))
    arrays = common.load_series_arrays(series=series)
    context = (
        common.scored_rows(series=series, arrays=arrays)
        .filter(pl.col("scored"))
        .select(*common.KEYS, "band")
    )
    actual = common.actual_table(series=series, arrays=arrays)
    arm_losses = {}
    for name in RAW_SOURCES:
        arm_losses[name] = with_losses(
            frame=raw_arm(name=name, series=series, actual=actual).join(
                context, on=list(common.KEYS), how="inner", validate="m:1"
            )
        )
    for arm in (*STORED_ARMS, *NEW_ARMS):
        folder = common.E64_ARMS_DIR if arm in STORED_ARMS else common.ARMS_DIR
        arm_losses[arm] = with_losses(
            frame=pl.read_parquet(folder / f"{arm}.parquet").join(
                context, on=list(common.KEYS), how="inner", validate="m:1"
            )
        )
    losses = experiment_store.losses_for_bootstrap(arm_losses=arm_losses)
    return losses.with_columns(month=pl.col("power_fcst_init_time").dt.strftime("%G-W%V"))


def read_contrast(
    *,
    losses: pl.DataFrame,
    treatment: str,
    reference: str,
    scope_name: str,
    scope: pl.Expr,
    level: float,
    n_resamples: int,
) -> dict[str, str | float | int]:
    """Return one contrast's mean paired difference and its week-block interval."""
    scoped = losses.filter(scope)
    differences, weeks = bootstrap.paired_differences(
        losses=scoped, treatment=treatment, reference=reference, metric="qs"
    )
    lower, upper = bootstrap.bootstrap_difference_at_level(
        losses=scoped,
        treatment=treatment,
        reference=reference,
        metric="qs",
        level=level,
        n_resamples=n_resamples,
    )
    return {
        "treatment": treatment,
        "reference": reference,
        "scope": scope_name,
        "level": level,
        "difference": float(differences.mean()),
        "lower": lower,
        "upper": upper,
        "seed_spread": float(differences.mean(axis=1).std()),
        "rows_per_seed": differences.shape[1],
        "weeks": len(np.unique(weeks)),
    }


def absolute_scores(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Return each arm's mean quantile score at 10 days and beyond and at shorter leads."""
    return (
        losses.with_columns(long=pl.col("band") == LONG_BAND)
        .group_by("arm", "long")
        .agg(qs=pl.col("qs").mean())
        .sort("long", "arm")
    )


def print_section(*, title: str, table: pl.DataFrame) -> None:
    """Print a titled table without truncation."""
    print(f"\n{title}")
    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=220, float_precision=4):
        print(table)


def main() -> None:
    """Assemble the new arms, then print the equality check, scores, and contrasts."""
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    assemble()
    equality = raw_equality()
    print_section(
        title="am_train_only against am, largest raw-quantile difference by lead band",
        table=equality,
    )
    long_difference = equality.filter(pl.col("band") == LONG_BAND)["max_difference"][0]
    print(
        f"CHECK: at 10 days and beyond the largest difference is {long_difference:.2e} "
        f"({'PASS' if long_difference <= EQUALITY_TOLERANCE else 'FAIL'}; tolerance "
        f"{EQUALITY_TOLERANCE:.0e})"
    )
    losses = load_losses()
    first = losses.filter((pl.col("arm") == "d0") & (pl.col("seed") == 0))
    print(
        f"scored rows per seed: {first.height:,}; arms {sorted(losses['arm'].unique().to_list())}"
    )
    absolute = absolute_scores(losses=losses)
    print_section(
        title="Mean quantile score (points of capacity); long = 10 days and beyond", table=absolute
    )
    long_scope = pl.col("band") == LONG_BAND
    rows = [
        read_contrast(
            losses=losses,
            treatment=t,
            reference=r,
            scope_name="10+ d",
            scope=long_scope,
            level=DECISION_LEVEL,
            n_resamples=DECISION_RESAMPLES,
        )
        for t, r in DECISION
    ]
    decision = pl.DataFrame(rows)
    print_section(
        title=f"Decision contrasts at 10 days and beyond ({DECISION_LEVEL:.2f}%)", table=decision
    )
    explore_rows = []
    for t, r in EXPLORATORY:
        for name, scope in (
            ("10+ d", long_scope),
            ("under 10 d", pl.col("band") < LONG_BAND),
            ("all leads", pl.lit(value=True)),
        ):
            explore_rows.append(
                read_contrast(
                    losses=losses,
                    treatment=t,
                    reference=r,
                    scope_name=name,
                    scope=scope,
                    level=EXPLORATORY_LEVEL,
                    n_resamples=EXPLORATORY_RESAMPLES,
                )
            )
    for t, r in RAW_CONTRASTS:
        explore_rows.append(
            read_contrast(
                losses=losses,
                treatment=t,
                reference=r,
                scope_name="10+ d, no correction",
                scope=long_scope,
                level=EXPLORATORY_LEVEL,
                n_resamples=EXPLORATORY_RESAMPLES,
            )
        )
    exploratory = pl.DataFrame(explore_rows)
    print_section(title="Exploratory contrasts (95%)", table=exploratory)
    for name, table in (
        ("equality", equality),
        ("absolute", absolute),
        ("decision", decision),
        ("exploratory", exploratory),
    ):
        table.write_parquet(common.RESULTS_DIR / f"{name}.parquet")
    print(f"\nresults written to {common.RESULTS_DIR}")


if __name__ == "__main__":
    main()
