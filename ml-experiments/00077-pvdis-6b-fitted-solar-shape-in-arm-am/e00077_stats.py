"""Score the arms of experiment 00077, read the pre-registered contrasts, and print every number.

Run it with `E77_ARM=blend` so that the scored rows come from the blend's tables, which share
experiment 00046's daylight mask with every other shape. It reads `work/arms/<arm>.parquet` (see
`e00077_assemble.py`) and writes under `work/results/`:

- `absolute.parquet`: each arm's mean quantile score and error of the median, with 95% intervals.
- `verdict.parquet`: contrasts C1, C2, and C3, at 98.33% (Bonferroni across the three contrasts)
  from 50,000 resamples of whole ISO weeks of forecast start.
- `exploratory.parquet`: every other contrast and split, at 95%.

Scored rows are the daylight half-hours of April to September with a CAMS `u` and an observed power
on the five verdict series, as in experiment 00064. The score is in points of capacity. The
decision rule is `verdict_rule()`, written before any arm was scored.

    E77_ARM=blend uv run python ml-experiments/00077-*/e00077_stats.py
"""

from typing import Final

import e00077_common as common
import e00077_model as model
import numpy as np
import polars as pl

from studies import bootstrap, experiment_store

ARMS_DIR: Final = common.BASE_WORK_DIR / "arms"
RESULTS_DIR: Final = common.BASE_WORK_DIR / "results"
THRESHOLD_FRACTION: Final[float] = 0.005
"""The smallest gain worth a table of three shares per Primary: 0.5% of the score of `am`."""
EXPECTED_ROWS_PER_SEED: Final[int] = 68_903
"""Experiment 00064's scored rows per seed."""
EXPECTED_AM_SCORE: Final[float] = 4.467
"""Experiment 00064's score of `am`, from which the threshold follows."""
N_VERDICT_CONTRASTS: Final[int] = 3
VERDICT_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / N_VERDICT_CONTRASTS)
"""98.33: a 5% familywise level split over the three contrasts."""
EXPLORATORY_LEVEL: Final[float] = 95.0
VERDICT_RESAMPLES: Final[int] = 50_000
EXPLORATORY_RESAMPLES: Final[int] = 10_000
SCORED_ARMS: Final[tuple[str, ...]] = (
    "d0",
    "d1l",
    "am",
    "am_twin",
    "am0",
    "am_sy",
    "am_blend",
    "am_blend_twin",
    "am_s25",
    "am_shuffled",
)
STORED_REFERENCE_ARMS: Final[tuple[str, ...]] = ("climatology", "manual_heuristic")
BAND_NAMES: Final[tuple[str, ...]] = tuple(name for name, _low, _high in common.LEAD_BANDS)
SHORT_BANDS: Final[tuple[str, ...]] = ("0-24 h", "1-2 d")
"""The lead bands that make up leads under 48 hours."""
LONG_BANDS: Final[tuple[str, ...]] = ("10+ d",)
"""The lead bands beyond 10 days, where the weight of the solar term is zero."""
ALL_ROWS: Final[pl.Expr] = pl.lit(value=True)


def context_table() -> pl.DataFrame:
    """Return the scored rows of the verdict series with the label, lead band, and valid month."""
    series = common.series_table().filter(pl.col("site").is_in(list(common.SITES)))
    arrays = common.load_series_arrays(series=series)
    rows = common.scored_rows(series=series, arrays=arrays).filter(pl.col("scored"))
    return rows.select(
        *common.KEYS,
        series_label=pl.col("site"),
        band=pl.col("band").map_elements(BAND_NAMES.__getitem__, return_dtype=pl.String),
        valid_month=pl.col("month"),
        lead_hours=pl.col("lead_hours"),
    )


def with_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add `qs` and `mae` in points of capacity to a frame of quantiles and `actual`."""
    quantiles = frame.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)
    return frame.with_columns(
        qs=pl.Series(common.pinball_quantile_score(quantiles=quantiles, actual=actual) * 100.0),
        mae=pl.Series(np.abs(quantiles[:, common.MEDIAN] - actual) * 100.0),
    )


def load_losses(*, context: pl.DataFrame) -> pl.DataFrame:
    """Return every scored arm's per-row losses on the context rows, resampled by ISO week.

    Args:
        context: `context_table()`.

    Returns:
        The frame `experiment_store.losses_for_bootstrap` builds, with the context columns, `qs`,
        `mae`, the nine quantiles, and `actual`; its `month` column holds the ISO week of the
        forecast start (`2025-W27`), so every bootstrap resamples whole weeks.
    """
    arm_losses = {
        arm: with_losses(
            frame=pl.read_parquet(ARMS_DIR / f"{arm}.parquet").join(
                context, on=list(common.KEYS), how="inner", validate="m:1"
            )
        )
        for arm in SCORED_ARMS
    }
    losses = experiment_store.losses_for_bootstrap(arm_losses=arm_losses)
    return losses.with_columns(month=pl.col("power_fcst_init_time").dt.strftime("%G-W%V"))


def read_contrast(
    *,
    losses: pl.DataFrame,
    contrast_id: str,
    treatment: str,
    reference: str,
    scope_name: str,
    scope: pl.Expr,
    level: float,
    n_resamples: int,
) -> dict[str, str | float | int]:
    """Return one contrast's mean paired difference and its week-block interval.

    Args:
        losses: `load_losses()`.
        contrast_id: The contrast's label.
        treatment: The arm whose score is compared.
        reference: The arm it is compared against.
        scope_name: What the scope is, for the printed table.
        scope: A Polars expression selecting the rows.
        level: The interval's coverage in percent.
        n_resamples: How many resamples.

    Returns:
        A row with the difference, bounds, half-width, rows per seed, blocks, and seed spread.
    """
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
        "contrast": contrast_id,
        "treatment": treatment,
        "reference": reference,
        "scope": scope_name,
        "level": level,
        "n_resamples": n_resamples,
        "difference": float(differences.mean()),
        "lower": lower,
        "upper": upper,
        "half_width": (upper - lower) / 2.0,
        "seed_spread": float(differences.mean(axis=1).std()),
        "rows_per_seed": differences.shape[1],
        "weeks": len(np.unique(weeks)),
    }


def verdict_family(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Read the three verdict contrasts at 98.33%."""
    return pl.DataFrame(
        [
            read_contrast(
                losses=losses,
                contrast_id=contrast_id,
                treatment="am_blend",
                reference=reference,
                scope_name="all leads",
                scope=ALL_ROWS,
                level=VERDICT_LEVEL,
                n_resamples=VERDICT_RESAMPLES,
            )
            for contrast_id, reference in (
                ("C1", "am"),
                ("C2", "am_s25"),
                ("C3", "am_blend_twin"),
            )
        ]
    )


def verdict_rule(*, family: pl.DataFrame, am_score: float) -> tuple[str, float]:
    """Return the verdict and the threshold of the pre-registered rule.

    The threshold is 0.5% of the score of `am`. `supported` when the upper end of C1 is below minus
    the threshold and the upper ends of C2 and C3 are below zero. `small gain` when the upper end of
    C1 is below zero and the rest of that rule is not met. `worse` when the lower end of C1 is above
    plus the threshold. `refuted` when the lower end of C1 is above minus the threshold and the
    upper end is at most plus the threshold. `inconclusive` otherwise.

    Args:
        family: `verdict_family()`.
        am_score: The mean quantile score of experiment 00064's `am` on the scored rows.

    Returns:
        The verdict word and the threshold in points of capacity.
    """
    threshold = THRESHOLD_FRACTION * am_score
    row = {r["contrast"]: r for r in family.iter_rows(named=True)}
    c1, c2, c3 = row["C1"], row["C2"], row["C3"]
    if c1["upper"] < -threshold and c2["upper"] < 0 and c3["upper"] < 0:
        return "supported", threshold
    if c1["upper"] < 0:
        return "small gain", threshold
    if c1["lower"] > threshold:
        return "worse", threshold
    if c1["lower"] > -threshold and c1["upper"] <= threshold:
        return "refuted", threshold
    return "inconclusive", threshold


def absolute_scores(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Return each arm's mean score and median error on the scored rows, with 95% intervals."""
    rows = []
    for arm in (a for a in SCORED_ARMS if a in set(losses["arm"].unique().to_list())):
        row: dict[str, str | float | int] = {"arm": arm}
        for metric in ("qs", "mae"):
            interval = bootstrap.bootstrap_absolute(losses=losses, arm=arm, metric=metric)
            row |= {
                metric: interval["value"],
                f"{metric}_lower": interval["lower_95"],
                f"{metric}_upper": interval["upper_95"],
            }
        rows.append(row)
    return pl.DataFrame(rows)


def stored_reference_scores(*, context: pl.DataFrame) -> pl.DataFrame:
    """Return the stored `climatology` and `manual_heuristic` arms' mean score on the scored rows.

    These arms hold a few rows fewer than the others, so they are scored on the rows they share with
    the context and not paired in the bootstrap.
    """
    rows = []
    for arm in STORED_REFERENCE_ARMS:
        frame = with_losses(
            frame=pl.read_parquet(ARMS_DIR / f"{arm}.parquet").join(
                context, on=list(common.KEYS), how="inner", validate="m:1"
            )
        )
        per_seed = frame.filter(pl.col("seed") == frame["seed"].min()).height
        rows.append(
            {
                "arm": arm,
                "qs": float(np.mean(frame["qs"].to_numpy())),
                "rows": per_seed,
                "missing_rows": context.height - per_seed,
            }
        )
    return pl.DataFrame(rows)


def exploratory_contrasts(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Return the exploratory contrasts at 95%: the other controls, splits, and per-series reads."""
    reads: list[tuple[str, str, str, str, pl.Expr]] = [
        ("E2", "am_blend", "am_shuffled", "all leads", ALL_ROWS),
        ("E3", "am_blend", "am0", "all leads", ALL_ROWS),
        ("E4", "am_blend", "am_sy", "all leads", ALL_ROWS),
        ("E5", "am_s25", "am", "all leads", ALL_ROWS),
        ("E6", "am_blend", "d0", "all leads", ALL_ROWS),
        ("E7", "am_blend", "am", "leads under 48 h", pl.col("band").is_in(list(SHORT_BANDS))),
        ("E8", "am_blend", "am", "leads beyond 10 d", pl.col("band").is_in(list(LONG_BANDS))),
        ("E9", "am_blend", "am", "leads 2 to 10 d", pl.col("band").is_in(["2-6 d", "6-10 d"])),
        ("E10", "am_blend_twin", "am_twin", "all leads", ALL_ROWS),
    ]
    reads += [
        (f"S {site}", "am_blend", "am", f"series {site}", pl.col("series_label") == site)
        for site in common.SITES
    ]
    reads += [
        (f"B {band}", "am_blend", "am", f"band {band}", pl.col("band") == band)
        for band in BAND_NAMES
    ]
    return pl.DataFrame(
        [
            read_contrast(
                losses=losses,
                contrast_id=contrast_id,
                treatment=treatment,
                reference=reference,
                scope_name=scope_name,
                scope=scope,
                level=EXPLORATORY_LEVEL,
                n_resamples=EXPLORATORY_RESAMPLES,
            )
            for contrast_id, treatment, reference, scope_name, scope in reads
        ]
    )


def by_month_table(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Return the mean paired difference of `am_blend` minus `am` in each local valid month."""
    paired = losses.filter(pl.col("arm") == "am_blend").join(
        losses.filter(pl.col("arm") == "am").select("site", "time", "seed", reference=pl.col("qs")),
        on=["site", "time", "seed"],
        validate="m:1",
    )
    return (
        paired.group_by("valid_month")
        .agg(difference=(pl.col("qs") - pl.col("reference")).mean(), rows=pl.len())
        .sort("valid_month")
    )


def capacity_table() -> pl.DataFrame:
    """Return each shape's capacity and mean fitted weight, to print beside the scores."""
    pieces = []
    for shape in common.SHAPE_ARMS:
        capacity = pl.read_parquet(common.BASE_WORK_DIR / "arm" / shape / "capacity.parquet")
        weights = (
            pl.read_parquet(common.BASE_WORK_DIR / "arm" / shape / "weights.parquet")
            .filter(pl.col("band") < model.NO_SKILL_BAND)
            .group_by("site")
            .agg(mean_weight=pl.col("slope_applied").mean())
        )
        pieces.append(
            capacity.join(weights, on="site").select(
                shape=pl.lit(shape),
                site="site",
                capacity_mw="capacity_mw",
                mean_weight="mean_weight",
            )
        )
    return pl.concat(pieces)


def print_section(*, title: str, table: pl.DataFrame) -> None:
    """Print a titled table with every row and column shown."""
    with pl.Config(tbl_rows=300, tbl_cols=30, fmt_float="mixed", tbl_width_chars=220):
        print(f"\n{title}")
        print(table)


def main() -> None:
    """Print the absolute scores, the verdict contrasts, and the exploratory reads."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    context = context_table()
    losses = load_losses(context=context)
    first = losses.filter((pl.col("arm") == "am") & (pl.col("seed") == 0))
    print(
        f"scored rows per seed on the five verdict series: {first.height:,}; "
        f"seeds {sorted(losses['seed'].unique().to_list())}; "
        f"forecast-start weeks {losses['month'].n_unique()}"
    )
    print(f"arms scored: {sorted(losses['arm'].unique().to_list())}")
    absolute = absolute_scores(losses=losses)
    print_section(title="Absolute scores (points of capacity; smaller is better)", table=absolute)
    reference = stored_reference_scores(context=context)
    print_section(title="Stored reference arms of experiment 00033", table=reference)
    family = verdict_family(losses=losses)
    print_section(
        title=f"Verdict contrasts ({VERDICT_LEVEL:.2f}% intervals, week blocks)",
        table=family.select(
            "contrast",
            "treatment",
            "reference",
            "difference",
            "lower",
            "upper",
            "half_width",
            "seed_spread",
            "rows_per_seed",
            "weeks",
        ),
    )
    am_score = float(absolute.filter(pl.col("arm") == "am")["qs"][0])
    assert first.height == EXPECTED_ROWS_PER_SEED, f"{first.height} scored rows per seed"
    assert abs(am_score - EXPECTED_AM_SCORE) < 0.005, f"am scores {am_score}, not experiment 64's"
    verdict, threshold = verdict_rule(family=family, am_score=am_score)
    print(f"\nThreshold: {THRESHOLD_FRACTION:.1%} of am's score ({am_score:.3f}) = {threshold:.4f}")
    print(f"VERDICT: {verdict}")
    exploratory = exploratory_contrasts(losses=losses)
    print_section(
        title="Exploratory contrasts (95%; week blocks; negative favours the treatment)",
        table=exploratory.select(
            "contrast",
            "treatment",
            "reference",
            "scope",
            "difference",
            "lower",
            "upper",
            "seed_spread",
            "rows_per_seed",
        ),
    )
    by_month = by_month_table(losses=losses)
    print_section(title="am_blend minus am by local valid month", table=by_month)
    capacities = capacity_table()
    print_section(
        title="Capacity (MW) and mean fitted weight by shape and series", table=capacities
    )
    for name, table in (
        ("absolute", absolute),
        ("verdict", family),
        ("exploratory", exploratory),
        ("by_month", by_month),
        ("capacity", capacities),
        ("stored_reference", reference),
    ):
        table.write_parquet(RESULTS_DIR / f"{name}.parquet")
    print(f"\nresults written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
