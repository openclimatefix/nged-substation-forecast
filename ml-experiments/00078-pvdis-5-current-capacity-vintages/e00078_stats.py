"""Assemble the new arms of experiment 00078, score them, and print every number.

Reads `work/arms_raw/` (see `e00078_fit.py`), `work/capacity_vintages.parquet`, experiment 00064's
`work/arms/` and `work/arms_raw/`, and experiment 00076's `work/arms/` and `work/arms_raw/`. Writes
`work/arms/<arm>.parquet` (the 14-day correction and the 1.15 spread factor applied, with `actual`),
the prediction sets, and `work/results/*.parquet`.

The scored rows are experiment 00064's: daylight half-hours of April to September with a CAMS `u`
and an observed power, on the five verdict series. The score is the quantile score in points of
capacity. The decision rule is in `docs/ml-experiments/00078-*/result.md`. The verdict family (C1 to
C4) is read at 98.75% and the starting-point family (D1 to D3) at 98.33%, from 50,000 resamples of
whole ISO weeks of forecast start. Every other read is exploratory, at 95%.

    uv run python ml-experiments/00078-*/e00078_stats.py
"""

import json
from pathlib import Path
from typing import Final

import e00078_common as common
import numpy as np
import polars as pl

from studies import bootstrap, experiment_store

NEW_ARMS: Final[tuple[str, ...]] = (
    "am_sy",
    "am_sy_twin",
    "am_wk",
    "d1l_r",
    "d0_r",
    "d0_m0",
)
"""The arms assembled here, each with the 14-day correction and the spread factor."""
STORED_E64: Final[tuple[str, ...]] = ("d0", "d1l", "am", "am_twin")
STORED_E76: Final[tuple[str, ...]] = ("am0",)
RAW_SOURCES: Final[dict[str, tuple[str, str]]] = {
    "am0_raw": ("e76", "am0"),
    "d1l_r_raw": ("own", "d1l_r"),
    "d0_r_raw": ("own", "d0_r"),
    "d0_m0_raw": ("own", "d0_m0"),
}
"""Arms scored without the 14-day correction (the spread factor is kept)."""
LONG_BAND: Final[int] = 4
"""Index of the 10-days-and-beyond lead band."""
SHORT_BANDS: Final[int] = 2
"""Lead bands below this index (0 to 24 hours and 1 to 2 days) are leads under 48 hours."""
THRESHOLD: Final[float] = 0.05
VERDICT_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 4)
START_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 3)
EXPLORATORY_LEVEL: Final[float] = 95.0
DECISION_RESAMPLES: Final[int] = 50_000
EXPLORATORY_RESAMPLES: Final[int] = 10_000
ALL_LEADS: Final[pl.Expr] = pl.lit(value=True)
UNDER_48H: Final[pl.Expr] = pl.col("band") < SHORT_BANDS
LONG: Final[pl.Expr] = pl.col("band") == LONG_BAND


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
    folder = common.E76_RAW_ARMS_DIR if source == "e76" else common.RAW_ARMS_DIR
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


def stored_arm_folder(*, arm: str) -> Path:
    """Return the folder holding a stored or new arm's scored form."""
    if arm in STORED_E64:
        return common.E64_ARMS_DIR
    return common.E76_ARMS_DIR if arm in STORED_E76 else common.ARMS_DIR


def load_losses(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return every compared arm's per-row losses on experiment 00064's scored rows."""
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
    for arm in (*STORED_E64, *STORED_E76, *NEW_ARMS):
        arm_losses[arm] = with_losses(
            frame=pl.read_parquet(stored_arm_folder(arm=arm) / f"{arm}.parquet").join(
                context, on=list(common.KEYS), how="inner", validate="m:1"
            )
        )
    losses = experiment_store.losses_for_bootstrap(arm_losses=arm_losses)
    sites = dict(zip(series["time_series_id"], series["site"], strict=True))
    vintages = pl.read_parquet(common.VINTAGES_FILE).select(
        "site", pl.col("run_date").alias("run_date"), "season_year_step"
    )
    return (
        losses.with_columns(
            week=pl.col("power_fcst_init_time").dt.strftime("%G-W%V"),
            valid_month=pl.col("valid_time").dt.strftime("%Y-%m"),
            site=pl.col("time_series_id").replace_strict(sites, return_dtype=pl.String),
            run_date=pl.col("power_fcst_init_time").dt.date(),
        )
        .join(vintages, on=["site", "run_date"], how="left", validate="m:1")
        .with_columns(step1=pl.col("season_year_step") == 1)
    )


def read_contrast(
    *,
    losses: pl.DataFrame,
    treatment: str,
    reference: str,
    scope_name: str,
    scope: pl.Expr,
    level: float,
    n_resamples: int,
    label: str = "",
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
        "id": label,
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


def verdict(*, decision: pl.DataFrame) -> str:
    """Apply the pre-registered decision rule to the four verdict contrasts."""
    upper = dict(zip(decision["id"], decision["upper"], strict=True))
    lower = dict(zip(decision["id"], decision["lower"], strict=True))
    solar_ok = upper["C3"] < 0 and upper["C4"] < 0
    if upper["C1"] < -THRESHOLD and solar_ok:
        return "supported"
    if upper["C1"] < 0 and solar_ok:
        return "small gain"
    if lower["C1"] > -THRESHOLD and upper["C1"] >= 0:
        return "refuted"
    return "inconclusive"


def absolute_scores(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Return each arm's mean quantile score over all leads, under 48 hours, and 10 days or more."""
    return (
        losses.group_by("arm")
        .agg(
            all_leads=pl.col("qs").mean(),
            under_48h=pl.col("qs").filter(UNDER_48H).mean(),
            ten_days_plus=pl.col("qs").filter(LONG).mean(),
        )
        .sort("all_leads")
    )


def print_section(*, title: str, table: pl.DataFrame) -> None:
    """Print a titled table without truncation."""
    print(f"\n{title}")
    with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=240, float_precision=4):
        print(table)


def base_scores() -> pl.DataFrame:
    """Return XGBoost's estimated base score and boosted rounds of each fitted arm."""
    rows = []
    for path in sorted((common.WORK_DIR / "base_scores").glob("*.json")):
        item = json.loads(path.read_text())
        rows.append({**item, "base_score": str(item["base_score"])})
    return pl.DataFrame(rows).sort("arm", "site", "seed")


def vintage_steps() -> pl.DataFrame:
    """Return the share of runs per series that each step of the fallback chain supplied."""
    return (
        pl.read_parquet(common.VINTAGES_FILE)
        .group_by("site", "season_year_step")
        .agg(runs=pl.len())
        .with_columns(share=pl.col("runs") / pl.col("runs").sum().over("site"))
        .sort("site", "season_year_step")
    )


def contrast_rows(*, losses: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return the verdict family, the starting-point family, and the exploratory reads."""
    verdict_rows = [
        read_contrast(
            losses=losses,
            treatment=t,
            reference=r,
            scope_name=name,
            scope=scope,
            level=VERDICT_LEVEL,
            n_resamples=DECISION_RESAMPLES,
            label=label,
        )
        for label, t, r, name, scope in (
            ("C1", "am_sy", "am", "all leads", ALL_LEADS),
            ("C2", "am_sy", "am", "under 48 h", UNDER_48H),
            ("C3", "am_sy", "am0", "all leads", ALL_LEADS),
            ("C4", "am_sy", "am_sy_twin", "all leads", ALL_LEADS),
        )
    ]
    start_rows = [
        read_contrast(
            losses=losses,
            treatment=t,
            reference=r,
            scope_name=name,
            scope=scope,
            level=START_LEVEL,
            n_resamples=DECISION_RESAMPLES,
            label=label,
        )
        for label, t, r, name, scope in (
            ("D1", "d1l_r_raw", "am0_raw", "10 d+, no correction", LONG),
            ("D2", "d0_r_raw", "d0_m0_raw", "10 d+, no correction", LONG),
            ("D3", "d1l_r", "am0", "all leads", ALL_LEADS),
        )
    ]
    plain = (
        ("am_wk", "am", ALL_LEADS, "all leads"),
        ("am_wk", "am", UNDER_48H, "under 48 h"),
        ("am_wk", "am_sy", ALL_LEADS, "all leads"),
        ("am_sy", "am", pl.col("step1"), "runs on step 1"),
        ("am", "am0", ALL_LEADS, "all leads"),
        ("am", "am0", UNDER_48H, "under 48 h"),
        ("am_sy", "am0", UNDER_48H, "under 48 h"),
        ("am_sy", "d0", ALL_LEADS, "all leads"),
        ("am_sy", "d0", UNDER_48H, "under 48 h"),
        ("am", "d0", ALL_LEADS, "all leads"),
        ("am", "d0", UNDER_48H, "under 48 h"),
        ("am_sy", "d0_r", ALL_LEADS, "all leads"),
        ("am_sy", "d0_r", UNDER_48H, "under 48 h"),
        ("am0", "d0_r", ALL_LEADS, "all leads"),
        ("d0_r", "d0", ALL_LEADS, "all leads"),
        ("d1l_r", "d1l", ALL_LEADS, "all leads"),
        ("d1l_r", "d1l", LONG, "10 d+"),
        ("d1l_r", "am0", LONG, "10 d+, corrected"),
        ("d0_m0", "d0_r", ALL_LEADS, "all leads"),
        ("am_sy_twin", "am_twin", ALL_LEADS, "all leads"),
    )
    splits = (
        *((f"band {b[0]}", pl.col("band") == i) for i, b in enumerate(common.LEAD_BANDS)),
        *(
            (f"valid month {m}", pl.col("valid_month") == m)
            for m in sorted(losses["valid_month"].unique().to_list())
        ),
        *((f"series {site}", pl.col("site") == site) for site in common.SITES),
    )
    reads = (
        *((t, r, scope, name) for t, r, scope, name in plain),
        *(("am_sy", "am", scope, name) for name, scope in splits),
    )
    exploratory = [
        read_contrast(
            losses=losses,
            treatment=t,
            reference=r,
            scope_name=name,
            scope=scope,
            level=EXPLORATORY_LEVEL,
            n_resamples=EXPLORATORY_RESAMPLES,
        )
        for t, r, scope, name in reads
    ]
    return pl.DataFrame(verdict_rows), pl.DataFrame(start_rows), pl.DataFrame(exploratory)


def main() -> None:
    """Assemble the new arms, then print the vintages, scores, contrasts, and the verdict."""
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    assemble()
    series = common.series_table().filter(pl.col("site").is_in(list(common.SITES)))
    print_section(title="Share of runs per step of the fallback chain", table=vintage_steps())
    print_section(
        title="XGBoost base score and boosted rounds of the fitted arms", table=base_scores()
    )
    losses = load_losses(series=series)
    first = losses.filter((pl.col("arm") == "d0") & (pl.col("seed") == 0))
    print(
        f"scored rows per seed: {first.height:,}; arms {sorted(losses['arm'].unique().to_list())}"
    )
    absolute = absolute_scores(losses=losses)
    print_section(title="Mean quantile score (points of capacity)", table=absolute)
    climatology = with_losses(frame=pl.read_parquet(common.E64_ARMS_DIR / "climatology.parquet"))
    print(f"climatology (experiment 00064's stored rows): {climatology['qs'].mean():.3f}")
    decision, start, exploratory = contrast_rows(losses=losses)
    print_section(title=f"Verdict family ({VERDICT_LEVEL:.2f}%)", table=decision)
    print_section(title=f"Starting-point family ({START_LEVEL:.2f}%)", table=start)
    print_section(title=f"Exploratory reads ({EXPLORATORY_LEVEL:.0f}%)", table=exploratory)
    print(f"\nVERDICT BY THE PRE-REGISTERED RULE: {verdict(decision=decision)}")
    for name, table in (
        ("absolute", absolute),
        ("decision", decision),
        ("start", start),
        ("exploratory", exploratory),
    ):
        table.write_parquet(common.RESULTS_DIR / f"{name}.parquet")
    print(f"\nresults written to {common.RESULTS_DIR}")


if __name__ == "__main__":
    main()
