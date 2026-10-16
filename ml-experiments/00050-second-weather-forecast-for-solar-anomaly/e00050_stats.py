"""Score the demand arms of experiment 00050 and read the contrasts C7, C7s, C3b, and C6b.

Reads the arms under `work/arms/` (see `e00050_predict.py`) and experiment 00046's `d0` and `d1l`
arms, and writes under `work/results/`: `absolute.parquet` (each arm's mean quantile score),
`contrasts.parquet` (every contrast read), `by_month.parquet` (the sign check), and
`coverage.parquet` (the share of actuals below the 0.05 and 0.95 quantiles in sunny and cloudy
rows).

Scored rows are the daylight half-hours of April to September with a CAMS `u` and an observed
power, as in experiment 00046. The score is the quantile score in points of capacity (1 point is 1%
of the series' effective capacity). The two deciding contrasts are read at 97.5% (Bonferroni across
two); every other read is exploratory, labelled so, and read at 95%.

- C7: the selected arm's shift minus the ENS shift, on all scored rows of the verdict series.
- C7s: the same, on scored rows at leads of 0 to 48 hours.
- C3b: the selected arm's shift minus `d1l` (exploratory; experiment 00046's fitted C3 was -0.143).
- C6b: the selected arm's shift minus `d0`, on all leads and on 0 to 48 hours (exploratory).

The verdict is `supported` when either deciding interval lies wholly below -0.1, `small gain` when
either lies wholly below zero but neither lies wholly below -0.1, `refuted` when both lie wholly
above -0.1 and neither lies wholly below zero, and `inconclusive` otherwise.

    uv run python ml-experiments/00050-*/e00050_stats.py
"""

import json
from pathlib import Path
from typing import Final

import e00050_common as common
import numpy as np
import polars as pl

from studies import bootstrap, experiment_store

THRESHOLD_POINTS: Final[float] = -0.1
"""A gain counts as large enough when a deciding interval lies wholly below this."""
DECIDING_LEVEL: Final[float] = 97.5
DECIDING_RESAMPLES: Final[int] = 50_000
EXPLORATORY_LEVEL: Final[float] = 95.0
SPLIT_RESAMPLES: Final[int] = 10_000
SHORT_LEAD_HOURS: Final[float] = 48.0
D9: Final[str] = "D9"
BENCHMARK: Final[str] = "d1l_ens_shift_fitted"
BEST: Final[str] = "d1l_best_shift_fitted"
SOURCE_ARMS: Final[tuple[str, ...]] = ("d0", "d1l")
SHIFT_ARMS: Final[tuple[str, ...]] = (
    BENCHMARK,
    "d1l_gefs_shift_fitted",
    "d1l_blend_shift_fitted",
    "d1l_blend_fixed_shift_fitted",
    BEST,
)
CSI_BINS: Final[tuple[str, ...]] = ("cloudy", "middle", "sunny")
BAND_NAMES: Final[tuple[str, ...]] = tuple(name for name, _low, _high in common.LEAD_BANDS)
VERDICT_SCOPE: Final[pl.Expr] = pl.col("series_label").is_in(list(common.VERDICT_SITES))
SHORT_SCOPE: Final[pl.Expr] = pl.col("lead_hours") <= SHORT_LEAD_HOURS


def arm_path(*, arm: str) -> Path:
    """Return the file of an arm: experiment 00046's for `d0` and `d1l`, this experiment's else."""
    folder = common.SOURCE_ARMS_DIR if arm in SOURCE_ARMS else common.ARMS_DIR
    return folder / f"{arm}.parquet"


def _with_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add `qs` and `mae` in points of capacity to a frame of quantiles and `actual`."""
    quantiles = frame.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)
    return frame.with_columns(
        qs=pl.Series(common.pinball_quantile_score(quantiles=quantiles, actual=actual) * 100.0),
        mae=pl.Series(np.abs(quantiles[:, common.MEDIAN] - actual) * 100.0),
    )


def load_losses() -> pl.DataFrame:
    """Return every arm's per-row losses on the daylight, CAMS-present rows of all 12 months.

    Returns:
        The frame `experiment_store.losses_for_bootstrap` builds, with `qs` and `mae` (points of
        capacity), `q05`, `q50`, `q95`, `actual`, `series_label`, `band` (name of the lead band),
        `lead_hours`, `csi_bin`, `valid_month` (local calendar month), and `scored` (April to
        September).
    """
    series = common.series_table().filter(pl.col("site").is_in(list(common.SITES)))
    arrays = common.load_series_arrays(series=series)
    rows = common.scored_rows(series=series, arrays=arrays).filter(pl.col("scored_all_months"))
    context = rows.with_columns(
        series_label=pl.col("site"),
        band_name=pl.col("band").map_elements(BAND_NAMES.__getitem__, return_dtype=pl.String),
        csi_bin=pl.when(pl.col("clear_sky_index") >= common.SUNNY_INDEX)
        .then(pl.lit("sunny"))
        .when(pl.col("clear_sky_index") < common.CLOUDY_INDEX)
        .then(pl.lit("cloudy"))
        .otherwise(pl.lit("middle")),
    ).select(
        *common.KEYS,
        "series_label",
        band=pl.col("band_name"),
        valid_month=pl.col("month"),
        lead_hours=pl.col("lead_hours"),
        csi_bin=pl.col("csi_bin"),
        scored=pl.col("scored"),
    )
    arm_losses = {
        arm: _with_losses(
            frame=pl.read_parquet(arm_path(arm=arm)).join(
                context, on=list(common.KEYS), how="inner", validate="m:1"
            )
        )
        for arm in (*SOURCE_ARMS, *SHIFT_ARMS)
        if arm_path(arm=arm).exists()
    }
    return experiment_store.losses_for_bootstrap(arm_losses=arm_losses)


def scope_rows(*, losses: pl.DataFrame, scope: pl.Expr, scored_only: bool = True) -> pl.DataFrame:
    """Restrict the loss frame to a scope, on the April-to-September rows unless told otherwise."""
    selected = losses.filter(scope)
    return selected.filter(pl.col("scored")) if scored_only else selected


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
    scored_only: bool = True,
    block: str = "month",
) -> dict[str, str | float | int]:
    """Return one contrast's mean paired difference and its month-block (or week-block) interval.

    Args:
        losses: `load_losses()`.
        contrast_id: The contrast's label.
        treatment: The arm whose score is compared.
        reference: The arm it is compared against.
        scope_name: What the scope is, for the printed table.
        scope: A Polars expression selecting the rows.
        level: The interval's coverage in percent.
        n_resamples: How many resamples.
        scored_only: Restrict to April to September when true.
        block: `month` resamples whole months of the run's initialisation; `week` resamples whole
            ISO weeks of it.

    Returns:
        A row with the difference, bounds, half-width, rows per seed, blocks, and seed spread.
    """
    scoped = scope_rows(losses=losses, scope=scope, scored_only=scored_only)
    if block == "week":
        scoped = scoped.with_columns(month=pl.col("power_fcst_init_time").dt.strftime("%G-W%V"))
    differences, months = bootstrap.paired_differences(
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
        "block": block,
        "level": level,
        "n_resamples": n_resamples,
        "difference": float(differences.mean()),
        "lower": lower,
        "upper": upper,
        "half_width": (upper - lower) / 2.0,
        "seed_spread": float(differences.mean(axis=1).std()),
        "rows_per_seed": differences.shape[1],
        "blocks": len(np.unique(months)),
    }


def verdict(*, c7: dict[str, str | float | int], c7s: dict[str, str | float | int]) -> str:
    """Apply the pre-registered decision rule to the intervals of C7 and C7s.

    Args:
        c7: C7's row from `read_contrast`.
        c7s: C7s's row.

    Returns:
        `supported`, `small gain`, `refuted`, or `inconclusive`.
    """
    rows = (c7, c7s)
    if any(float(row["upper"]) < THRESHOLD_POINTS for row in rows):
        return "supported"
    if any(float(row["upper"]) < 0.0 for row in rows):
        return "small gain"
    if all(float(row["lower"]) > THRESHOLD_POINTS for row in rows):
        return "refuted"
    return "inconclusive"


def absolute_scores(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Return each arm's mean score and median error on the verdict series, with intervals."""
    scoped = scope_rows(losses=losses, scope=VERDICT_SCOPE)
    present = set(scoped["arm"].unique().to_list())
    rows = []
    for arm in (a for a in (*SOURCE_ARMS, *SHIFT_ARMS) if a in present):
        row: dict[str, str | float | int] = {"arm": arm}
        for metric in ("qs", "mae"):
            interval = bootstrap.bootstrap_absolute(losses=scoped, arm=arm, metric=metric)
            row |= {
                f"{metric}": interval["value"],
                f"{metric}_lower": interval["lower_95"],
                f"{metric}_upper": interval["upper_95"],
            }
        rows.append(row)
    return pl.DataFrame(rows)


def coverage_table(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Return the fraction of actuals below the 0.05 and 0.95 quantiles in sunny and cloudy rows."""
    scoped = scope_rows(losses=losses, scope=VERDICT_SCOPE)
    return (
        scoped.filter(pl.col("csi_bin").is_in(["sunny", "cloudy"]))
        .group_by("arm", "csi_bin")
        .agg(
            below_q05=(pl.col("actual") < pl.col("q05")).mean(),
            below_q95=(pl.col("actual") < pl.col("q95")).mean(),
            rows=pl.len(),
        )
        .sort("csi_bin", "arm")
    )


def by_month_table(*, losses: pl.DataFrame, treatment: str, reference: str) -> pl.DataFrame:
    """Return the mean paired difference of a contrast in each local valid month."""
    scoped = scope_rows(losses=losses, scope=VERDICT_SCOPE)
    paired = scoped.filter(pl.col("arm") == treatment).join(
        scoped.filter(pl.col("arm") == reference).select(
            "site", "time", "seed", reference=pl.col("qs")
        ),
        on=["site", "time", "seed"],
    )
    return (
        paired.group_by("valid_month")
        .agg(difference=(pl.col("qs") - pl.col("reference")).mean(), rows=pl.len())
        .sort("valid_month")
    )


def contrast_table(*, losses: pl.DataFrame, best: str) -> pl.DataFrame:
    """Read the deciding contrasts, then every exploratory contrast and split.

    Args:
        losses: `load_losses()`.
        best: The arm under test (`d1l_best_shift_fitted`).

    Returns:
        One row per contrast read.
    """
    deciding = [
        read_contrast(
            losses=losses,
            contrast_id=contrast_id,
            treatment=best,
            reference=BENCHMARK,
            scope_name=scope_name,
            scope=scope,
            level=DECIDING_LEVEL,
            n_resamples=DECIDING_RESAMPLES,
        )
        for contrast_id, scope_name, scope in (
            ("C7", "verdict series", VERDICT_SCOPE),
            ("C7s", "verdict series, leads 0 to 48 h", VERDICT_SCOPE & SHORT_SCOPE),
        )
    ]
    exploratory_scopes: list[tuple[str, pl.Expr]] = [
        ("verdict series", VERDICT_SCOPE),
        ("verdict series, leads 0 to 48 h", VERDICT_SCOPE & SHORT_SCOPE),
        ("without D9", VERDICT_SCOPE & (pl.col("series_label") != D9)),
        (
            "without D9, leads 0 to 48 h",
            VERDICT_SCOPE & SHORT_SCOPE & (pl.col("series_label") != D9),
        ),
        *((f"lead band {band}", VERDICT_SCOPE & (pl.col("band") == band)) for band in BAND_NAMES),
        *((f"series {site}", pl.col("series_label") == site) for site in common.SITES),
        *((f"clear-sky index {c}", VERDICT_SCOPE & (pl.col("csi_bin") == c)) for c in CSI_BINS),
    ]
    contrasts: dict[str, tuple[str, str]] = {
        "C7": (best, BENCHMARK),
        "C3b": (best, "d1l"),
        "C6b": (best, "d0"),
    }
    present = set(losses["arm"].unique().to_list())
    for other in SHIFT_ARMS:
        if other not in (BENCHMARK, best) and other in present:
            contrasts[f"{other} minus ENS shift"] = (other, BENCHMARK)
    rows = []
    for contrast_id, (treatment, reference) in contrasts.items():
        for scope_name, scope in exploratory_scopes:
            if contrast_id != "C7" and "lead band" in scope_name:
                continue
            rows.append(
                read_contrast(
                    losses=losses,
                    contrast_id=contrast_id,
                    treatment=treatment,
                    reference=reference,
                    scope_name=scope_name,
                    scope=scope,
                    level=EXPLORATORY_LEVEL,
                    n_resamples=SPLIT_RESAMPLES,
                )
            )
        rows.append(
            read_contrast(
                losses=losses,
                contrast_id=contrast_id,
                treatment=treatment,
                reference=reference,
                scope_name="verdict series, week blocks",
                scope=VERDICT_SCOPE,
                level=EXPLORATORY_LEVEL,
                n_resamples=SPLIT_RESAMPLES,
                block="week",
            )
        )
    return pl.DataFrame([*deciding, *rows])


def main() -> None:
    """Print the absolute scores, the contrasts, the sign check, and the verdict."""
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    selection = json.loads(common.SELECTION_PATH.read_text())
    best = BEST if BEST in {a for a in SHIFT_ARMS if arm_path(arm=a).exists()} else BENCHMARK
    if selection["selected"] == "ens":
        print(
            "stage 1 selected ENS: the demand contrasts are not read and the verdict is `refuted`"
        )
        return
    losses = load_losses()
    scored = scope_rows(losses=losses, scope=VERDICT_SCOPE).filter(pl.col("arm") == "d1l")
    print(
        f"stage 1 selected {selection['selected']}; scored rows per seed on the verdict series: "
        f"{scored.filter(pl.col('seed') == 0).height:,}; seeds {sorted(scored['seed'].unique())}"
    )
    with pl.Config(tbl_rows=200, tbl_cols=30, fmt_float="mixed", tbl_width_chars=200):
        absolute = absolute_scores(losses=losses)
        print("\nAbsolute scores on the verdict series (points of capacity; smaller is better)")
        print(absolute)
        coverage = coverage_table(losses=losses)
        print("\nCoverage of the 0.05 and 0.95 quantiles (fraction of actuals below each)")
        print(coverage)
        contrasts = contrast_table(losses=losses, best=best)
        print("\nContrasts (points of capacity; negative favours the treatment)")
        print(
            contrasts.select(
                "contrast",
                "scope",
                "block",
                "level",
                "difference",
                "lower",
                "upper",
                "half_width",
                "seed_spread",
                "rows_per_seed",
                "blocks",
            )
        )
        by_month = by_month_table(losses=losses, treatment=best, reference=BENCHMARK)
        print("\nC7 by local valid month (the sign check: the sign must match in every month)")
        print(by_month)
    deciding = contrasts.filter(pl.col("level") == DECIDING_LEVEL)
    c7 = deciding.filter(pl.col("contrast") == "C7").row(0, named=True)
    c7s = deciding.filter(pl.col("contrast") == "C7s").row(0, named=True)
    print(f"\nVerdict by the pre-registered rule, verdict series: {verdict(c7=c7, c7s=c7s)}")
    without_d9 = {}
    for contrast_id, scope in (
        ("C7", VERDICT_SCOPE & (pl.col("series_label") != D9)),
        ("C7s", VERDICT_SCOPE & SHORT_SCOPE & (pl.col("series_label") != D9)),
    ):
        without_d9[contrast_id] = read_contrast(
            losses=losses,
            contrast_id=contrast_id,
            treatment=best,
            reference=BENCHMARK,
            scope_name="without D9",
            scope=scope,
            level=DECIDING_LEVEL,
            n_resamples=DECIDING_RESAMPLES,
        )
    print(f"Verdict without D9: {verdict(c7=without_d9['C7'], c7s=without_d9['C7s'])}")
    print(
        f"Realised half-width at 97.5%: C7 {float(c7['half_width']):.3f}, C7s "
        f"{float(c7s['half_width']):.3f} points (the plan expected 0.03 to 0.05 and 0.05 to 0.10)"
    )
    c6 = contrasts.filter(
        (pl.col("contrast") == "C6b") & (pl.col("scope") == "verdict series")
    ).row(0, named=True)
    c6_short = contrasts.filter(
        (pl.col("contrast") == "C6b") & (pl.col("scope") == "verdict series, leads 0 to 48 h")
    ).row(0, named=True)
    print(
        f"C6b (best minus d0): all leads {float(c6['difference']):.3f} "
        f"[{float(c6['lower']):.3f}, {float(c6['upper']):.3f}]; leads 0 to 48 h "
        f"{float(c6_short['difference']):.3f} [{float(c6_short['lower']):.3f}, "
        f"{float(c6_short['upper']):.3f}] at 95%: PVDIS-5 is a gain test at 0 to 48 h only if "
        "the second interval lies wholly below zero"
    )
    absolute.write_parquet(common.RESULTS_DIR / "absolute.parquet")
    coverage.write_parquet(common.RESULTS_DIR / "coverage.parquet")
    contrasts.write_parquet(common.RESULTS_DIR / "contrasts.parquet")
    by_month.write_parquet(common.RESULTS_DIR / "by_month.parquet")
    print(f"results written to {common.RESULTS_DIR}")


if __name__ == "__main__":
    main()
