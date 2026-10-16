"""Score the arms of experiment 00046, print the coefficient diagnostic, and read the contrasts.

Reads `work/arms/<arm>.parquet` (see `e00046_assemble.py`) and writes under `work/results/`:

- `absolute.parquet`: each arm's mean quantile score and error of the median, with 95% intervals.
- `diagnostic.parquet`: the slope of the residual of the `d1l` median on the oracle shift.
- `coverage.parquet`: the fraction of actuals below the 0.05 and 0.95 quantiles, sunny and cloudy.
- `contrasts.parquet`: every contrast read, with its interval, level, and scope.
- `by_month.parquet`: the mean paired difference of C3 to C6 in each valid month.
- `binned.parquet`: the `d1l` residual binned by oracle shift, for the first chart.

Scored rows are the daylight half-hours of April to September with a CAMS `u` and an observed
power. The deciding contrasts C3 and C4 are read at 97.5% (Bonferroni across two). Every other read
is exploratory and read at 95%. The score is in points of capacity (1 point is 1% of the series'
effective capacity).

    uv run python ml-experiments/00046-*/e00046_stats.py
"""

from typing import Final

import e00046_common as common
import numpy as np
import polars as pl

from studies import bootstrap, experiment_store

THRESHOLD_POINTS: Final[float] = -0.35
"""A gain counts as large enough when the interval lies wholly below this (points of capacity)."""
DECIDING_LEVEL: Final[float] = 97.5
DECIDING_RESAMPLES: Final[int] = 50_000
EXPLORATORY_LEVEL: Final[float] = 95.0
SPLIT_RESAMPLES: Final[int] = 10_000
TRIGGER_SLOPE: Final[float] = -0.8
"""A pooled slope of the residual on the shift above this in any lead band triggers extra reads."""
D9: Final[str] = "D9"
ARMS: Final[tuple[str, ...]] = (
    "d0",
    "d1l",
    "d1l_ens_shift",
    "d1l_cams_shift",
    "d1l_ens_shift_headline",
    "d1l_cams_shift_headline",
    "d1l_ens_mean_shift",
    "d1l_ens_shift_fitted",
    "d1l_cams_shift_fitted",
)
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "C3": ("d1l_ens_shift", "d1l"),
    "C4": ("d1l_cams_shift", "d1l"),
    "C5": ("d1l_cams_shift", "d1l_ens_shift"),
    "C6": ("d1l_ens_shift", "d0"),
}
"""The planned contrasts, each as (treatment, reference); a negative value favours the treatment."""
EXTRA_CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "d0_minus_d1l": ("d0", "d1l"),
    "C3_headline": ("d1l_ens_shift_headline", "d1l"),
    "C4_headline": ("d1l_cams_shift_headline", "d1l"),
    "C3_ens_mean": ("d1l_ens_mean_shift", "d1l"),
    "C3_fitted": ("d1l_ens_shift_fitted", "d1l"),
    "C4_fitted": ("d1l_cams_shift_fitted", "d1l"),
}
CSI_BINS: Final[tuple[str, ...]] = ("cloudy", "middle", "sunny")
BAND_NAMES: Final[tuple[str, ...]] = tuple(name for name, _low, _high in common.LEAD_BANDS)
VERDICT_SCOPE: Final[pl.Expr] = pl.col("series_label").is_in(list(common.VERDICT_SITES))


def load_losses() -> pl.DataFrame:
    """Return every arm's per-row losses on the daylight, CAMS-present rows of all 12 months.

    Returns:
        The frame `experiment_store.losses_for_bootstrap` builds, with `qs` and `mae` (points of
        capacity), `q05`, `q50`, `q95`, `actual`, `series_label`, `band` (name of the lead band),
        `csi_bin`, `valid_month` (local calendar month), `scored` (April to September), and
        `oracle_shift` (capacity over the series' effective capacity times the CAMS anomaly).
    """
    series = common.series_table().filter(pl.col("site").is_in(list(common.SITES)))
    arrays = common.load_series_arrays(series=series)
    rows = common.scored_rows(series=series, arrays=arrays).filter(pl.col("scored_all_months"))
    capacity = (
        common.capacities()
        .select("site", "C")
        .join(series.select("site", "effective_capacity_mw"), on="site")
    )
    context = (
        rows.join(capacity, on="site")
        .with_columns(
            series_label=pl.col("site"),
            band_name=pl.col("band").map_elements(BAND_NAMES.__getitem__, return_dtype=pl.String),
            csi_bin=pl.when(pl.col("clear_sky_index") >= common.SUNNY_INDEX)
            .then(pl.lit("sunny"))
            .when(pl.col("clear_sky_index") < common.CLOUDY_INDEX)
            .then(pl.lit("cloudy"))
            .otherwise(pl.lit("middle")),
            oracle_shift=pl.col("C") * pl.col("anomaly_cams") / pl.col("effective_capacity_mw"),
        )
        .select(
            *common.KEYS,
            "series_label",
            band=pl.col("band_name"),
            valid_month=pl.col("month"),
            lead_hours=pl.col("lead_hours"),
            csi_bin=pl.col("csi_bin"),
            scored=pl.col("scored"),
            oracle_shift=pl.col("oracle_shift"),
        )
    )
    arm_losses = {
        arm: _with_losses(
            frame=pl.read_parquet(common.ARMS_DIR / f"{arm}.parquet").join(
                context, on=list(common.KEYS), how="inner", validate="m:1"
            )
        )
        for arm in ARMS
        if (common.ARMS_DIR / f"{arm}.parquet").exists()
    }
    return experiment_store.losses_for_bootstrap(arm_losses=arm_losses)


def _with_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add `qs` and `mae` in points of capacity to a frame of quantiles and `actual`."""
    quantiles = frame.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)
    return frame.with_columns(
        qs=pl.Series(common.pinball_quantile_score(quantiles=quantiles, actual=actual) * 100.0),
        mae=pl.Series(np.abs(quantiles[:, common.MEDIAN] - actual) * 100.0),
    )


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


def verdict(*, c3: dict[str, str | float | int], c4: dict[str, str | float | int]) -> str:
    """Apply the pre-registered decision rule to the intervals of C4 (the gate) and C3.

    Args:
        c3: C3's row from `read_contrast`.
        c4: C4's row.

    Returns:
        `inconclusive` when C4 does not lie wholly below the threshold. Otherwise `supported`,
        `small gain`, `refuted`, or `inconclusive`, by C3's interval.
    """
    if float(c4["upper"]) >= THRESHOLD_POINTS:
        return "inconclusive (the gate failed: C4 does not lie wholly below -0.35)"
    upper, lower = float(c3["upper"]), float(c3["lower"])
    if upper < THRESHOLD_POINTS:
        return "supported"
    if upper < 0.0:
        return "small gain"
    if lower > THRESHOLD_POINTS:
        return "refuted"
    return "inconclusive"


def absolute_scores(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Return each arm's mean score and median error on the verdict series, with intervals."""
    scoped = scope_rows(losses=losses, scope=VERDICT_SCOPE)
    rows = []
    for arm in (a for a in ARMS if a in set(scoped["arm"].unique().to_list())):
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


def mean_residual_frame(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Return, per scored verdict-series case, the mean over seeds of the `d1l` median's residual.

    Args:
        losses: `load_losses()`.

    Returns:
        Columns `series_label`, `band`, `csi_bin`, `valid_month`, `oracle_shift`, and `residual`
        (actual minus the corrected median of `d1l`, averaged over seeds), in power over capacity.
    """
    return (
        scope_rows(losses=losses, scope=VERDICT_SCOPE)
        .filter(pl.col("arm") == "d1l")
        .group_by("site", "time", maintain_order=True)
        .agg(
            pl.col("series_label").first(),
            pl.col("band").first(),
            pl.col("csi_bin").first(),
            pl.col("valid_month").first(),
            pl.col("oracle_shift").first(),
            residual=(pl.col("actual") - pl.col("q50")).mean(),
        )
    )


def slope_through_origin(*, frame: pl.DataFrame) -> float:
    """Return the least-squares slope through the origin of `residual` on `oracle_shift`."""
    shift = frame["oracle_shift"].to_numpy()
    return float((frame["residual"].to_numpy() * shift).sum() / (shift**2).sum())


def diagnostic_table(*, residuals: pl.DataFrame) -> pl.DataFrame:
    """Return the slope of the `d1l` residual on the oracle shift, by lead band and other splits.

    The ideal slope is -1: the forecast holds no solar information beyond the climatology, and the
    shift's capacity is right. A slope near zero says the forecast already carries the anomaly, or
    the capacity is too large. Slopes are also cross-fitted by month: a slope fitted on odd local
    months is applied to the even months and the reverse, and `variance_ratio` is the residual's
    sum of squares after that adjustment over its sum of squares before it.

    Args:
        residuals: `mean_residual_frame()`.

    Returns:
        One row per split (`pooled`, a lead band, a series, or a clear-sky-index bin) and group.
    """
    rows = []
    groupings: list[tuple[str, str | None, list[str]]] = [
        ("pooled", None, ["all"]),
        ("lead band", "band", list(BAND_NAMES)),
        ("series", "series_label", list(common.VERDICT_SITES)),
        ("clear-sky index", "csi_bin", list(CSI_BINS)),
    ]
    for split, column, groups in groupings:
        for group in groups:
            part = residuals if column is None else residuals.filter(pl.col(column) == group)
            odd = part.filter(pl.col("valid_month") % 2 == 1)
            even = part.filter(pl.col("valid_month") % 2 == 0)
            slope_odd, slope_even = (
                slope_through_origin(frame=odd),
                slope_through_origin(frame=even),
            )
            adjusted = np.concatenate(
                [
                    even["residual"].to_numpy() - slope_odd * even["oracle_shift"].to_numpy(),
                    odd["residual"].to_numpy() - slope_even * odd["oracle_shift"].to_numpy(),
                ]
            )
            rows.append(
                {
                    "split": split,
                    "group": group,
                    "slope": slope_through_origin(frame=part),
                    "slope_odd_months": slope_odd,
                    "slope_even_months": slope_even,
                    "variance_ratio": float((adjusted**2).sum() / (part["residual"] ** 2).sum()),
                    "rows": part.height,
                }
            )
    return pl.DataFrame(rows)


def coverage_table(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Return the fraction of actuals below the 0.05 and 0.95 quantiles in sunny and cloudy rows."""
    scoped = scope_rows(losses=losses, scope=VERDICT_SCOPE).filter(
        pl.col("arm").is_in(["d1l", "d1l_ens_shift", "d1l_ens_mean_shift", "d0"])
    )
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


def by_month_table(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Return the mean paired difference of C3 to C6 in each local valid month."""
    scoped = scope_rows(losses=losses, scope=VERDICT_SCOPE)
    pieces = []
    for contrast_id, (treatment, reference) in CONTRASTS.items():
        paired = scoped.filter(pl.col("arm") == treatment).join(
            scoped.filter(pl.col("arm") == reference).select(
                "site", "time", "seed", reference=pl.col("qs")
            ),
            on=["site", "time", "seed"],
        )
        pieces.append(
            paired.group_by("valid_month")
            .agg(difference=(pl.col("qs") - pl.col("reference")).mean(), rows=pl.len())
            .with_columns(contrast=pl.lit(contrast_id))
            .sort("valid_month")
        )
    return pl.concat(pieces)


def binned_table(*, residuals: pl.DataFrame) -> pl.DataFrame:
    """Return the `d1l` residual binned by oracle shift (2-point bins), per lead band, in points."""
    return (
        residuals.with_columns(
            shift_points=pl.col("oracle_shift") * 100.0,
            residual_points=pl.col("residual") * 100.0,
        )
        .with_columns(bin_centre=(pl.col("shift_points") / 2.0).floor() * 2.0 + 1.0)
        .filter(pl.col("bin_centre").is_between(-21.0, 31.0))
        .group_by("band", "bin_centre")
        .agg(
            residual_points=pl.col("residual_points").mean(),
            shift_points=pl.col("shift_points").mean(),
            rows=pl.len(),
        )
        .filter(pl.col("rows") >= 50)
        .sort("band", "bin_centre")
    )


def split_scopes() -> list[tuple[str, pl.Expr]]:
    """Return the exploratory splits of the scored rows, each as (name, row selector)."""
    return [
        *((f"lead band {band}", VERDICT_SCOPE & (pl.col("band") == band)) for band in BAND_NAMES),
        *(
            (f"clear-sky index {csi_bin}", VERDICT_SCOPE & (pl.col("csi_bin") == csi_bin))
            for csi_bin in CSI_BINS
        ),
        *((f"series {site}", pl.col("series_label") == site) for site in common.SITES),
    ]


def contrast_rows(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Read the deciding contrasts, then every exploratory contrast and split."""
    rows = [
        read_contrast(
            losses=losses,
            contrast_id=contrast_id,
            treatment=treatment,
            reference=reference,
            scope_name="verdict series",
            scope=VERDICT_SCOPE,
            level=DECIDING_LEVEL if contrast_id in ("C3", "C4") else EXPLORATORY_LEVEL,
            n_resamples=DECIDING_RESAMPLES,
        )
        for contrast_id, (treatment, reference) in CONTRASTS.items()
    ]
    scopes = [
        ("without D9", VERDICT_SCOPE & (pl.col("series_label") != D9)),
        ("verdict series and D5", pl.col("series_label").is_in(list(common.SITES))),
    ]
    for contrast_id, (treatment, reference) in {**CONTRASTS, **EXTRA_CONTRASTS}.items():
        planned = contrast_id in CONTRASTS
        if treatment not in set(losses["arm"].unique().to_list()):
            continue
        read = [
            (name, scope, "month", True)
            for name, scope in [*scopes, *(split_scopes() if planned else [])]
        ]
        if contrast_id.endswith("_fitted"):
            read.append(("verdict series", VERDICT_SCOPE, "month", True))
        read.append(("verdict series, all 12 months", VERDICT_SCOPE, "month", False))
        if planned:
            read.append(("verdict series", VERDICT_SCOPE, "week", True))
        rows.extend(
            read_contrast(
                losses=losses,
                contrast_id=contrast_id,
                treatment=treatment,
                reference=reference,
                scope_name=name,
                scope=scope,
                level=DECIDING_LEVEL
                if (block == "week" and contrast_id in ("C3", "C4"))
                else EXPLORATORY_LEVEL,
                n_resamples=SPLIT_RESAMPLES,
                scored_only=scored_only,
                block=block,
            )
            for name, scope, block, scored_only in read
        )
    return pl.DataFrame(rows)


def main() -> None:
    """Print the absolute scores, the coefficient diagnostic, the contrasts, and the verdict."""
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    losses = load_losses()
    scored = scope_rows(losses=losses, scope=VERDICT_SCOPE).filter(pl.col("arm") == "d1l")
    print(
        f"scored rows per seed on the verdict series: "
        f"{scored.filter(pl.col('seed') == 0).height:,}; "
        f"seeds {sorted(scored['seed'].unique().to_list())}; "
        f"valid months {sorted(scored['valid_month'].unique().to_list())}"
    )
    with pl.Config(tbl_rows=200, tbl_cols=30, fmt_float="mixed", tbl_width_chars=200):
        absolute = absolute_scores(losses=losses)
        print("\nAbsolute scores on the verdict series (points of capacity; smaller is better)")
        print(absolute)
        residuals = mean_residual_frame(losses=losses)
        diagnostic = diagnostic_table(residuals=residuals)
        print("\nCoefficient diagnostic (ideal slope -1; read before the verdict)")
        print(diagnostic)
        bands = diagnostic.filter(pl.col("split") == "lead band")
        triggered = bands.filter(pl.col("slope") > TRIGGER_SLOPE)
        print(
            f"pooled slope above {TRIGGER_SLOPE} in lead bands: "
            f"{triggered['group'].to_list() or 'none'} "
            f"-> extra exploratory reads {'TRIGGERED' if triggered.height else 'not triggered'}"
        )
        coverage = coverage_table(losses=losses)
        print("\nCoverage of the 0.05 and 0.95 quantiles (fraction of actuals below each)")
        print(coverage)
        contrasts = contrast_rows(losses=losses)
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
        by_month = by_month_table(losses=losses)
        print("\nMean paired difference by valid month (the sign check)")
        print(by_month.pivot(on="valid_month", index="contrast", values="difference"))
    main_rows = contrasts.filter(
        (pl.col("scope") == "verdict series") & (pl.col("block") == "month")
    )
    c3 = main_rows.filter(pl.col("contrast") == "C3").row(0, named=True)
    c4 = main_rows.filter(pl.col("contrast") == "C4").row(0, named=True)
    print(f"\nVerdict by the pre-registered rule, verdict series: {verdict(c3=c3, c4=c4)}")
    for label, scope_name in (("without D9", "without D9"),):
        pair = {}
        for contrast_id in ("C3", "C4"):
            pair[contrast_id] = read_contrast(
                losses=losses,
                contrast_id=contrast_id,
                treatment=CONTRASTS[contrast_id][0],
                reference=CONTRASTS[contrast_id][1],
                scope_name=scope_name,
                scope=VERDICT_SCOPE & (pl.col("series_label") != D9),
                level=DECIDING_LEVEL,
                n_resamples=DECIDING_RESAMPLES,
            )
        print(
            f"Verdict {label} (97.5% intervals: C3 {pair['C3']['difference']:.3f} "
            f"[{pair['C3']['lower']:.3f}, {pair['C3']['upper']:.3f}], C4 "
            f"{pair['C4']['difference']:.3f} [{pair['C4']['lower']:.3f}, "
            f"{pair['C4']['upper']:.3f}]): "
            f"{verdict(c3=pair['C3'], c4=pair['C4'])}"
        )
    print(
        f"Realised half-width at 97.5%: C3 {float(c3['half_width']):.3f}, C4 "
        f"{float(c4['half_width']):.3f} points (the plan expected about 0.4 for C3)"
    )
    c6 = main_rows.filter(pl.col("contrast") == "C6").row(0, named=True)
    print(
        f"C6 (ENS shift minus d0) {float(c6['difference']):.3f} [{float(c6['lower']):.3f}, "
        f"{float(c6['upper']):.3f}] at 95%: PVDIS-5 is a "
        f"{'gain test' if float(c6['upper']) < 0.0 else 'non-inferiority and explanation test'} "
        "(also needs C3 below -0.35)"
    )
    absolute.write_parquet(common.RESULTS_DIR / "absolute.parquet")
    diagnostic.write_parquet(common.RESULTS_DIR / "diagnostic.parquet")
    coverage.write_parquet(common.RESULTS_DIR / "coverage.parquet")
    contrasts.write_parquet(common.RESULTS_DIR / "contrasts.parquet")
    by_month.write_parquet(common.RESULTS_DIR / "by_month.parquet")
    binned_table(residuals=residuals).write_parquet(common.RESULTS_DIR / "binned.parquet")
    print(f"results written to {common.RESULTS_DIR}")


if __name__ == "__main__":
    main()
