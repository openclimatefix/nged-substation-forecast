"""Print the one report that holds every number the write-up of experiment 00045 quotes.

The report collects what the earlier scripts already wrote, and computes nothing a second way:

- the recommendation and the column notes, read from `capacity_weekly.parquet`'s file metadata;
- the tuning tables of both attempts, read from `frozen_config.json`;
- `scores/score_report.txt` verbatim: conditions (i) to (vi) for the frozen pair, the
  comparison-fit table, the detection curve, falls, transfers, seasonal load, false steps by
  Primary, vintages on injections, coverage, and the post hoc block (conditions and coverage of
  `alt_post_hoc`, `a = 2`, `r = 10`);
- the post hoc pair's breakdown tables, from `e00045_score._add_breakdowns` run on the post hoc
  unit scores and placebo counts, so the post hoc pair is read in the same terms as the frozen
  pair (tables in `scores/post_hoc_tables/`; in them the fit called `asymmetric` is the post hoc
  pair);
- the path error by case family with 95% cluster-bootstrap intervals for every fit, the frozen
  pair and the post hoc pair included (`scores/family_intervals.parquet`), computed with
  `e00045_score`'s own `Draws`, `boot`, and `median_of_medians`; the frozen fits' points are
  asserted equal to `scores/by_family.parquet`;
- three tables the charts need and the frozen scoring does not print for every fit, computed from
  the same unit scores and placebo counts: the share of steps detected by Primary, season, step
  size, and fit (`detection_by_primary.parquet`); the false-step rate and ratchet drift
  distribution by Primary, fit, and sky (`false_step_distribution.parquet`); and the path error of
  each unit under both baselines, for the additivity check (`additivity.parquet`);
- `real_report.txt` verbatim: the real-series summary per Primary, recovery ratios, false steps on
  the placebo sky, the gap diagnostic, held falls, vintages on the real series, and the notes on
  the intervals and the recommendation;
- `outside_report.txt` verbatim: the outside checks;
- the polygon check tables and `validation_report.txt` verbatim (the U1 validation facts).

Run it after `e00045_outputs.py`:

    POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
        uv run python ml-experiments/00045-pvdis-4-weekly-capacity-asymmetric-l1/e00045_report.py
"""

import json
from pathlib import Path
from typing import Final

import e00045_common as common
import e00045_score as score
import numpy as np
import polars as pl
import pyarrow.parquet as pq

REPORT_PATH: Final[Path] = common.DATA_DIR / "report.txt"
SCORES: Final[Path] = score.SCORES_DIR
POST_HOC_FITS_DIR: Final[Path] = SCORES / score.POST_HOC_DIR_NAME
POST_HOC_TABLES_DIR: Final[Path] = SCORES / "post_hoc_tables"
FAMILY_INTERVALS_PATH: Final[Path] = SCORES / "family_intervals.parquet"
POST_HOC_FIT: Final[str] = "post_hoc"
"""The name the combined frames give the post hoc pair; the frozen pair is `asymmetric`."""
POST_HOC_LABEL: Final[str] = "post hoc (a=2, r=10)"
FIT_ORDER: Final[tuple[str, ...]] = (*score.FIT_NAMES, POST_HOC_FIT)
LABELS: Final[dict[str, str]] = {**score.FIT_LABELS, POST_HOC_FIT: POST_HOC_LABEL}
POINT_TOLERANCE: Final[float] = 1e-9
BANNER: Final[str] = "=" * 100


def banner(*, title: str) -> str:
    """Return a part heading that stands out from the `== section ==` lines of the parts."""
    return f"\n{BANNER}\nPART: {title}\n{BANNER}\n"


def tuning_text() -> str:
    """Return the tuning tables of both attempts, with the notes the frozen file carries."""
    frozen = json.loads(score.FROZEN_PATH.read_text())
    second = pl.DataFrame(frozen["tuning_table"])
    first = pl.DataFrame(frozen["first_attempt_real_baseline"]["tuning_table"])
    limits = frozen["limits"]
    return "\n\n".join(
        [
            (
                f"Frozen at {frozen['frozen_at']} by commit "
                f"{frozen['tuning_script_commit']}; status "
                f"{frozen['status']}; chosen pair a={frozen['a']:g}, r={frozen['r']:g}. "
                f"Tuning units {frozen['tuning_units']} (farms {frozen['tuning_farms']}, "
                f"path seeds "
                f"{frozen['tuning_seeds']}, placebo seeds {frozen['placebo_seeds']}). Limits: "
                f"{limits}. 'violation' is the largest excess over a limit; no pair is feasible."
            ),
            "Second attempt (the baseline subtracts the season-year constant fit; the constant "
            "path is on the placebo sky):\n" + score.table_text(second),
            "First attempt (superseded; the baseline subtracts the real series):\n"
            + score.table_text(first),
            "What the second attempt replaced: " + frozen["supersedes"],
            "Redefinition of the constant-path unit after the first attempt's results were seen: "
            + frozen["constant_path_redefinition"],
            "Recovery ratio by Primary (frozen pair):\n"
            + score.table_text(
                pl.DataFrame(
                    {
                        "primary": list(frozen["recovery_ratio_by_primary"]),
                        "ratio": list(frozen["recovery_ratio_by_primary"].values()),
                    }
                )
            ),
        ]
    )


def recommendation_text() -> str:
    """Return the recommendation and the column notes from the capacity file's metadata."""
    metadata = pq.read_metadata(common.DATA_DIR / "capacity_weekly.parquet").metadata
    keys = (
        "recommended_variant",
        "recommended_reason",
        "uncertainty",
        "interval_and_recommendation_notes",
    )
    return "\n".join(f"{key}: {metadata[key.encode()].decode()}" for key in keys)


def combined_scores() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return the frozen and the post hoc unit scores, placebo counts, and post hoc alone.

    The post hoc rows carry the fit name `post_hoc`.

    Returns:
        The frozen scores with the post hoc rows added, the same for the placebo counts (each
        with the false-step rates per year), and the post hoc scores under the fit `asymmetric`.
    """
    frozen_scores = pl.read_parquet(SCORES / "unit_scores.parquet")
    post_hoc_scores = pl.read_parquet(SCORES / "unit_scores_post_hoc.parquet")
    placebo_frames = []
    for directory, rename in ((SCORES, False), (POST_HOC_FITS_DIR, True)):
        frame = pl.concat(
            [pl.read_parquet(f) for f in sorted(directory.glob("scoring_placebo_*.parquet"))]
        )
        if rename:
            frame = frame.with_columns(fit=pl.lit(POST_HOC_FIT))
        placebo_frames.append(frame)
    placebo = pl.concat(placebo_frames).with_columns(
        rate_10=pl.col("steps_10") / (pl.col("n_weeks") / score.metrics.WEEKS_PER_YEAR),
        rate_05=pl.col("steps_05") / (pl.col("n_weeks") / score.metrics.WEEKS_PER_YEAR),
    )
    combined = pl.concat(
        [frozen_scores, post_hoc_scores.with_columns(fit=pl.lit(POST_HOC_FIT))],
        how="vertical_relaxed",
    )
    return combined, placebo, post_hoc_scores


def family_intervals(*, scores: pl.DataFrame) -> pl.DataFrame:
    """Return the path error by case family and fit, with 95% cluster-bootstrap intervals.

    Args:
        scores: The combined unit scores, post hoc rows included.

    Returns:
        One row per case family, fit, and variant: the point, the 2.5th and the 97.5th percentile.
    """
    draws = score.Draws(n_primaries=scores["primary"].n_unique())
    rows = []
    for family, cases in score.CASE_FAMILIES.items():
        for fit in FIT_ORDER:
            for variant in score.VARIANTS:
                values = score.tensor_of(
                    frame=scores, fit=fit, cases=cases, column="path_error", variant=variant
                )
                if np.isnan(values).all():
                    continue
                point = score.median_of_medians(values)
                result = score.interval(
                    point=point,
                    samples=score.boot(
                        values=values, statistic=score.median_of_medians, draws=draws
                    ),
                )
                rows.append(
                    {
                        "family": family,
                        "fit": LABELS[fit],
                        "variant": variant,
                        "path_error": point,
                        "low": result.low,
                        "high": result.high,
                    }
                )
    return pl.DataFrame(rows)


def check_against_by_family(*, intervals: pl.DataFrame) -> None:
    """Raise when a frozen fit's point differs from `scores/by_family.parquet`."""
    stored = pl.read_parquet(SCORES / "by_family.parquet")
    joined = stored.join(intervals, on=["family", "fit", "variant"], suffix="_new")
    assert joined.height == stored.height, "a stored by_family row has no interval row"
    worst = (joined["path_error"] - joined["path_error_new"]).abs().max()
    assert worst is not None, "no stored by_family rows"
    assert float(worst) < POINT_TOLERANCE, f"by_family differs by {worst}"  # ty: ignore[invalid-argument-type]


def post_hoc_breakdowns(*, post_hoc_scores: pl.DataFrame, placebo: pl.DataFrame) -> str:
    """Return the post hoc pair's breakdown tables, written by the frozen scoring's own code.

    `_add_breakdowns` loops over `score.FIT_NAMES`; the post hoc scores hold one fit, so the loop
    is restricted to that fit for the call.

    Args:
        post_hoc_scores: `unit_scores_post_hoc.parquet` (the fit is named `asymmetric`).
        placebo: The combined placebo counts with rates.

    Returns:
        The report text of the breakdowns.
    """
    POST_HOC_TABLES_DIR.mkdir(exist_ok=True)
    report = score.Report(out_dir=POST_HOC_TABLES_DIR)
    mine = placebo.filter(pl.col("fit") == POST_HOC_FIT).with_columns(fit=pl.lit("asymmetric"))
    original = score.FIT_NAMES
    score.FIT_NAMES = ("asymmetric",)
    try:
        score._add_breakdowns(
            report=report,
            scores=post_hoc_scores,
            placebo=mine,
            detection=score.detection_limits(scores=post_hoc_scores, fit="asymmetric"),
        )
    finally:
        score.FIT_NAMES = original
    return "\n".join(report.lines)


def detection_by_primary(*, scores: pl.DataFrame) -> pl.DataFrame:
    """Return the share of steps detected, by fit, Primary, season of the step, and step size.

    Same rows and definition as `score.detection_limits` (semi-synthetic steps with a detection
    verdict; a step placed from April to September is a summer step).

    Args:
        scores: The combined unit scores.

    Returns:
        One row per fit, Primary, season, and step case: the share detected and the unit count.
    """
    return (
        scores.filter(
            (pl.col("variant") == "semi_synthetic")
            & pl.col("case").is_in(list(score.STEPS))
            & pl.col("detected").is_not_null()
        )
        .with_columns(
            season=pl.when(pl.col("step_summer"))
            .then(pl.lit("Apr-Sep"))
            .otherwise(pl.lit("Oct-Mar")),
            fit=pl.col("fit").replace(LABELS),
        )
        .group_by("fit", "primary", "season", "case")
        .agg(share_detected=pl.col("detected").cast(pl.Float64).mean(), n=pl.len())
        .sort("fit", "primary", "season", "case")
    )


def false_step_distribution(*, scores: pl.DataFrame, placebo: pl.DataFrame) -> pl.DataFrame:
    """Return the spread of false steps and ratchet drift by fit, Primary, and sky.

    A false step is a rise of 10% of the real series' mean capacity (`steps_10`, per year). The
    constant-path sky has one value per farm-seed unit and the placebo sky one per permutation.

    Args:
        scores: The combined unit scores.
        placebo: The combined placebo counts with their rates.

    Returns:
        One row per fit, Primary, and sky: the mean, the 10th and 90th percentiles of the steps
        per Primary-year, the share of replicates with any step, the mean ratchet drift per 5
        years, and the replicate count.
    """
    constant = (
        scores.filter((pl.col("case") == "constant_path") & (pl.col("variant") == "semi_synthetic"))
        .select("fit", "primary", rate=pl.col("steps_10"), drift=pl.col("drift_per_5y"))
        .with_columns(sky=pl.lit("constant path"))
    )
    sky = placebo.select(
        "fit", "primary", rate=pl.col("rate_10"), drift=pl.col("drift_per_5y")
    ).with_columns(sky=pl.lit("placebo sky"))
    return (
        pl.concat([constant, sky])
        .with_columns(fit=pl.col("fit").replace(LABELS))
        .group_by("fit", "primary", "sky")
        .agg(
            mean_steps=pl.col("rate").mean(),
            p10_steps=pl.col("rate").quantile(0.1),
            p90_steps=pl.col("rate").quantile(0.9),
            share_with_a_step=(pl.col("rate") > 0).cast(pl.Float64).mean(),
            mean_drift=pl.col("drift").mean(),
            n=pl.len(),
        )
        .sort("fit", "sky", "primary")
    )


def additivity(*, scores: pl.DataFrame) -> pl.DataFrame:
    """Return each unit's path error under the semi-synthetic and the real-plus-injection baseline.

    Args:
        scores: The combined unit scores.

    Returns:
        One row per fit and unit scored under both baselines, with both path errors.
    """
    key = ["fit", "primary", "farm", "seed", "case"]
    parts = {
        variant: scores.filter(pl.col("variant") == variant).select(
            *key, pl.col("path_error").alias(variant)
        )
        for variant in score.VARIANTS
    }
    return (
        parts["semi_synthetic"]
        .join(parts["real_plus_injection"], on=key)
        .with_columns(fit=pl.col("fit").replace(LABELS))
        .sort(key)
    )


def polygon_text() -> str:
    """Return the polygon check tables."""
    return "\n\n".join(
        f"[{name}]\n" + score.table_text(pl.read_parquet(common.DATA_DIR / f"{name}.parquet"))
        for name in ("polygon_check", "polygon_overlaps")
    )


def main() -> None:
    """Write `report.txt` and `scores/family_intervals.parquet`, and print the report."""
    scores, placebo, post_hoc_scores = combined_scores()
    intervals = family_intervals(scores=scores)
    check_against_by_family(intervals=intervals)
    score.atomic_write(frame=intervals, path=FAMILY_INTERVALS_PATH)
    extra = {
        "detection_by_primary": detection_by_primary(scores=scores),
        "false_step_distribution": false_step_distribution(scores=scores, placebo=placebo),
        "additivity": additivity(scores=scores),
    }
    for name, frame in extra.items():
        score.atomic_write(frame=frame, path=SCORES / f"{name}.parquet")
    parts = [
        (
            "Experiment 45 (PVDIS-4) report: every number the write-up quotes. "
            "In the frozen scoring, the fit called `asymmetric` is the frozen pair (a=4, r=30), "
            "which equals the non-decreasing fit. The post hoc pair (a=2, r=10) was chosen after "
            "seeing the frozen pair fail, so every post hoc number is post hoc."
        ),
        banner(title="the recommended series and the column notes") + recommendation_text(),
        banner(title="tuning, both attempts") + tuning_text(),
        banner(
            title="frozen scoring (conditions, comparison fits, breakdowns, vintages on "
            "injections, coverage) and the post hoc block: scores/score_report.txt"
        )
        + (SCORES / "score_report.txt").read_text().rstrip("\n"),
        banner(
            title="path error by case family with 95% intervals, every fit, post hoc included "
            "(family_intervals.parquet)"
        )
        + score.table_text(intervals),
        banner(title="detection, false-step distribution, and additivity tables for the charts")
        + "\n\n".join(
            f"[{name}]\n"
            + (
                score.table_text(frame)
                if name != "additivity"
                else score.table_text(
                    frame.group_by("fit")
                    .agg(
                        units=pl.len(),
                        median_gap=(
                            pl.col("real_plus_injection") - pl.col("semi_synthetic")
                        ).median(),
                        correlation=pl.corr("semi_synthetic", "real_plus_injection"),
                    )
                    .sort("fit")
                )
                + "\n(one row per unit in additivity.parquet; summarised by fit here)"
            )
            for name, frame in extra.items()
        ),
        banner(
            title="post hoc pair: breakdown tables from the frozen scoring's own code "
            "(the fit called asymmetric here is a=2, r=10)"
        )
        + post_hoc_breakdowns(post_hoc_scores=post_hoc_scores, placebo=placebo),
        banner(title="post hoc coverage: scores/coverage_report_post_hoc.txt")
        + (SCORES / "coverage_report_post_hoc.txt").read_text().rstrip("\n"),
        banner(title="real series: real_report.txt")
        + (common.DATA_DIR / "real_report.txt").read_text().rstrip("\n"),
        banner(title="outside checks: outside_report.txt")
        + (common.DATA_DIR / "outside_report.txt").read_text().rstrip("\n"),
        banner(title="polygon check") + polygon_text(),
        banner(title="U1 validation facts: validation_report.txt")
        + common.VALIDATION_REPORT_PATH.read_text().rstrip("\n"),
    ]
    text = "\n".join(parts) + "\n"
    REPORT_PATH.write_text(text)
    print(f"wrote {REPORT_PATH} ({text.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
