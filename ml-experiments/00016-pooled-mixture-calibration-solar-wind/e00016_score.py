"""Score experiment 00016's arms: CRPS contrasts, coverage by lead band, and the verdict.

The CRPS is on every case, in points of capacity (100 times the mass-weighted pinball estimate).
Coverage is the share of non-trivial cases (uncorrected 95% quantile above 1% of capacity) whose
observed power lies in the 5% to 95% band inclusive, in percent. Intervals resample whole
initialisation months. The 8 planned CRPS contrasts are read at 99.4% (a Bonferroni correction of
5% across 8); the coverage intervals are 95%.

Output: `work/results/{report.md, absolute.parquet, contrasts.parquet, coverage.parquet}`.

    uv run python ml-experiments/00016-*/e00016_score.py
"""

from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00016_common import (
    ARMS,
    EXPERIMENT_ID,
    KEY_COLUMNS,
    LEAD_BANDS,
    LEVEL_COLUMNS,
    POST_HOC_ARM,
    ROW_SET_ID,
    SCORES_DIR,
    SEEDS,
    TECHNOLOGIES,
    WORK_DIR,
    ArmType,
    TechnologyType,
    covered,
    crps_estimate,
)

from studies import bootstrap, experiment_store

RESULTS_DIR: Final[Path] = WORK_DIR / "results"
PLANNED_LEVEL: Final[float] = 99.4
N_RESAMPLES: Final[int] = 10_000
COVERAGE_RESAMPLES: Final[int] = 2_000
COVERAGE_SEED: Final[int] = 20261015
TOLERANCE: Final[float] = 2.5
"""A band reaches 90% when its coverage lies within this many points of 90."""
MAX_HARM: Final[float] = 0.05
"""The largest upper bound of a CRPS interval, in points of capacity, for a supported verdict."""
REPRODUCTION_TOLERANCE: Final[float] = 0.05
PUBLISHED_CRPS: Final[dict[TechnologyType, float]] = {"wind": 11.11, "solar": 5.05}
"""Experiment 00002's `b_mixture_full` mean CRPS per technology (points of capacity)."""
REFERENCE_ARMS: Final[dict[str, str]] = {
    "mixture_blend": "b_mixture_full_blp",
    "stage2": "c_stage2_full",
    "climatology": "climatology_q",
}
PLANNED: Final[tuple[tuple[str, TechnologyType, ArmType], ...]] = (
    ("C1", "wind", "spread"),
    ("C2", "solar", "spread"),
    ("C3", "wind", "spread_band"),
    ("C4", "solar", "spread_band"),
    ("C5", "wind", "locscale"),
    ("C6", "solar", "locscale"),
    ("C7", "wind", "free_quantile"),
    ("C8", "solar", "free_quantile"),
)


def arm_rows(*, arm: ArmType) -> pl.DataFrame:
    """Return one arm's per-case CRPS (points) and coverage indicator (0 or 1), all seeds."""
    frame = experiment_store.read_predictions(
        experiment_id=EXPERIMENT_ID, name=arm.replace("_", "-"), row_set_id=ROW_SET_ID
    )
    quantiles = frame.select(LEVEL_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual_norm"].to_numpy().astype(np.float64)
    return frame.select(
        *KEY_COLUMNS, "seed", "lead_hours", "technology", "band", "trivial"
    ).with_columns(
        crps=pl.Series(100.0 * crps_estimate(quantiles=quantiles, actual=actual)),
        covered=pl.Series(covered(quantiles=quantiles, actual=actual)).cast(pl.Float64),
    )


def reference_rows(*, none: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return the per-case CRPS and coverage of the 00002 reference arms, on `none`'s cases."""
    scaffold = none.select(*KEY_COLUMNS, "seed", "lead_hours", "technology", "band", "trivial")
    out = {}
    for name, arm in REFERENCE_ARMS.items():
        stored = pl.concat(
            [
                pl.read_parquet(SCORES_DIR / f"rows_seed{seed}.parquet").filter(
                    pl.col("arm") == arm
                )
                for seed in SEEDS
            ]
        ).select(*KEY_COLUMNS, "seed", "crps", covered=pl.col("cover_90") / 100.0)
        out[name] = scaffold.join(stored, on=[*KEY_COLUMNS, "seed"], how="left", validate="1:1")
    return out


def _month(*, frame: pl.DataFrame) -> np.ndarray:
    return frame["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()


def coverage_interval(*, frame: pl.DataFrame) -> dict[str, float]:
    """Return the coverage (percent) of non-trivial cases with a month-resampled 95% interval."""
    scope = frame.filter(~pl.col("trivial"))
    months = _month(frame=scope)
    labels = np.unique(months)
    hits = np.array([scope["covered"].to_numpy()[months == m].sum() for m in labels])
    counts = np.array([float((months == m).sum()) for m in labels])
    generator = np.random.default_rng(COVERAGE_SEED)
    draws = generator.integers(0, labels.size, size=(COVERAGE_RESAMPLES, labels.size))
    resampled = 100.0 * hits[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "coverage": 100.0 * float(hits.sum() / counts.sum()),
        "lower": float(np.percentile(resampled, 2.5)),
        "upper": float(np.percentile(resampled, 97.5)),
        "n_cases": int(counts.sum() / len(SEEDS)),
    }


def coverage_table(*, rows: dict[str, pl.DataFrame], corrected: pl.DataFrame) -> pl.DataFrame:
    """Return coverage per arm, technology, and lead band, for all months and corrected months.

    Args:
        rows: Each arm's per-case frame.
        corrected: Columns `technology`, `band`, `month`: the months in which the per-band fit
            existed, so that the corrected-months scope is the same for every arm.

    Returns:
        One row per arm, technology, lead band, and month scope, with the coverage and its interval.
    """
    out = []
    for arm, frame in rows.items():
        keyed = frame.with_columns(month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"))
        for technology in TECHNOLOGIES:
            tech = keyed.filter(pl.col("technology") == technology)
            for band, _, _ in LEAD_BANDS:
                scope = tech.filter(pl.col("band") == band)
                fitted = scope.join(corrected, on=["technology", "band", "month"], how="semi")
                for label, part in (("all", scope), ("corrected", fitted)):
                    if part.filter(~pl.col("trivial")).is_empty():
                        continue
                    out.append(
                        {
                            "arm": arm,
                            "technology": technology,
                            "band": band,
                            "months": label,
                            **coverage_interval(frame=part),
                        }
                    )
    return pl.DataFrame(out)


def corrected_months() -> pl.DataFrame:
    """Return the technology, band, and month combinations where the per-band fit existed."""
    parameters = pl.read_parquet(WORK_DIR / "parameters.parquet")
    return (
        parameters.filter((pl.col("arm") == "spread_band") & pl.col("fitted"))
        .select("technology", "band", "month")
        .unique()
    )


def absolute_table(*, rows: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return each arm's mean CRPS per technology with a 95% interval."""
    out = []
    for arm, frame in rows.items():
        for technology in TECHNOLOGIES:
            losses = experiment_store.losses_for_bootstrap(
                arm_losses={
                    arm: frame.filter(pl.col("technology") == technology).select(
                        *KEY_COLUMNS, "seed", "crps"
                    )
                }
            )
            interval = bootstrap.bootstrap_absolute(losses=losses, arm=arm, metric="crps")
            out.append({"technology": technology, "arm": arm, **interval})
    return pl.DataFrame(out)


POST_HOC: Final[tuple[tuple[str, TechnologyType, ArmType], ...]] = (
    ("H1", "wind", POST_HOC_ARM),
    ("H2", "solar", POST_HOC_ARM),
)


def planned_contrasts(
    *,
    rows: dict[str, pl.DataFrame],
    contrasts: tuple[tuple[str, TechnologyType, ArmType], ...] = PLANNED,
) -> pl.DataFrame:
    """Score CRPS contrasts against `none` at `PLANNED_LEVEL` (the 8 planned ones by default)."""
    out = []
    for contrast, technology, arm in contrasts:
        losses = experiment_store.losses_for_bootstrap(
            arm_losses={
                name: rows[name]
                .filter(pl.col("technology") == technology)
                .select(*KEY_COLUMNS, "seed", "crps")
                for name in (arm, "none")
            }
        )
        differences, _ = bootstrap.paired_differences(
            losses=losses, treatment=arm, reference="none", metric="crps"
        )
        lower, upper = bootstrap.bootstrap_difference_at_level(
            losses=losses,
            treatment=arm,
            reference="none",
            metric="crps",
            level=PLANNED_LEVEL,
            n_resamples=N_RESAMPLES,
        )
        out.append(
            {
                "contrast": contrast,
                "technology": technology,
                "arm": arm,
                "difference": float(differences.mean()),
                "lower": lower,
                "upper": upper,
            }
        )
    return pl.DataFrame(out)


def reaches_ninety(*, coverage: pl.DataFrame, technology: TechnologyType, arm: ArmType) -> bool:
    """Apply the decision rule's coverage condition to one arm over all months."""
    part = coverage.filter((pl.col("technology") == technology) & (pl.col("months") == "all"))
    mine = part.filter(pl.col("arm") == arm).sort("band")
    base = part.filter(pl.col("arm") == "none").sort("band")
    gap = (mine["coverage"] - 90.0).abs()
    base_gap = (base["coverage"] - 90.0).abs()
    within = (gap <= TOLERANCE).all()
    better = ((base_gap <= TOLERANCE) | (gap < base_gap)).all()
    return bool(within and better)


def verdicts(*, coverage: pl.DataFrame, contrasts: pl.DataFrame) -> dict[TechnologyType, str]:
    """Apply the decision rule per technology: supported, refuted, or inconclusive."""
    out: dict[TechnologyType, str] = {}
    for technology in TECHNOLOGIES:
        part = coverage.filter(
            (pl.col("technology") == technology)
            & (pl.col("months") == "all")
            & (pl.col("arm") == "none")
        )
        if ((part["coverage"] - 90.0).abs() <= TOLERANCE).all():
            out[technology] = "refuted"
            continue
        reached = [
            row["arm"]
            for row in contrasts.filter(pl.col("technology") == technology).iter_rows(named=True)
            if reaches_ninety(coverage=coverage, technology=technology, arm=row["arm"])
        ]
        if not reached:
            out[technology] = "refuted"
        elif any(
            row["upper"] <= MAX_HARM
            for row in contrasts.filter(
                (pl.col("technology") == technology) & pl.col("arm").is_in(reached)
            ).iter_rows(named=True)
        ):
            out[technology] = "supported"
        else:
            out[technology] = "inconclusive"
    return out


def reproduction(*, absolute: pl.DataFrame) -> list[str]:
    """Check `none` against experiment 00002's mean CRPS and return a table of both.

    Args:
        absolute: `absolute_table(...)`.

    Returns:
        Markdown lines of a table of the two scores per technology.

    Raises:
        ValueError: If a technology differs by more than `REPRODUCTION_TOLERANCE`.
    """
    lines = ["## Reproduction of experiment 00002", "", "| technology | here | 00002 |"]
    lines.append("| --- | --- | --- |")
    for technology in TECHNOLOGIES:
        here = absolute.filter((pl.col("technology") == technology) & (pl.col("arm") == "none"))[
            "value"
        ].item()
        lines.append(f"| {technology} | {here:.3f} | {PUBLISHED_CRPS[technology]:.2f} |")
        if abs(here - PUBLISHED_CRPS[technology]) > REPRODUCTION_TOLERANCE:
            msg = f"{technology}: none scores {here:.3f}, experiment 00002 scored it as published"
            raise ValueError(msg)
    return lines


def write_report(
    *,
    absolute: pl.DataFrame,
    contrasts: pl.DataFrame,
    post_hoc: pl.DataFrame,
    coverage: pl.DataFrame,
    verdict: dict[TechnologyType, str],
    reproduction_lines: list[str],
) -> str:
    """Return the report as markdown."""
    lines = [f"# Experiment 00016 report (verdicts: {verdict})", ""]
    lines += ["## Mean CRPS (points of capacity, 95% interval)", ""]
    lines += [
        "| technology | arm | CRPS | interval | seed spread |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r['technology']} | {r['arm']} | {r['value']:.2f} | {r['lower_95']:.2f} to "
        f"{r['upper_95']:.2f} | {r['seed_spread']:.3f} |"
        for r in absolute.sort("technology", "value").iter_rows(named=True)
    ]
    lines += ["", f"## Planned CRPS contrasts ({PLANNED_LEVEL}% intervals)", ""]
    lines += [
        "| id | technology | arm minus none | difference | interval |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r['contrast']} | {r['technology']} | {r['arm']} | {r['difference']:+.3f} | "
        f"{r['lower']:+.3f} to {r['upper']:+.3f} |"
        for r in pl.concat([contrasts, post_hoc]).iter_rows(named=True)
    ]
    bands = [name for name, _, _ in LEAD_BANDS]
    for months in ("all", "corrected"):
        for technology in TECHNOLOGIES:
            lines += [
                "",
                f"## Coverage of the 5% to 95% band, {technology}, {months} months (%)",
                "",
            ]
            lines += ["| arm | " + " | ".join(bands) + " |", "| --- |" + " --- |" * len(bands)]
            part = coverage.filter(
                (pl.col("technology") == technology) & (pl.col("months") == months)
            )
            for arm in part["arm"].unique(maintain_order=True):
                cells = []
                for band in bands:
                    cell = part.filter((pl.col("arm") == arm) & (pl.col("band") == band))
                    cells.append(
                        f"{cell['coverage'].item():.1f} ({cell['lower'].item():.1f} to "
                        f"{cell['upper'].item():.1f})"
                        if cell.height
                        else "n/a"
                    )
                lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    return "\n".join([*lines, "", *reproduction_lines]) + "\n"


def main() -> None:
    """Score every arm, write the report, and save the tables."""
    rows: dict[str, pl.DataFrame] = {arm: arm_rows(arm=arm) for arm in (*ARMS, POST_HOC_ARM)}
    for frame in rows.values():
        if frame.height != rows["none"].height:
            msg = "arms hold different numbers of cases"
            raise ValueError(msg)
    rows |= reference_rows(none=rows["none"])
    absolute = absolute_table(rows=rows)
    reproduction_lines = reproduction(absolute=absolute)
    contrasts = planned_contrasts(rows=rows)
    post_hoc = planned_contrasts(rows=rows, contrasts=POST_HOC)
    coverage = coverage_table(rows=rows, corrected=corrected_months())
    verdict = verdicts(coverage=coverage, contrasts=contrasts)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    absolute.write_parquet(RESULTS_DIR / "absolute.parquet")
    contrasts.write_parquet(RESULTS_DIR / "contrasts.parquet")
    post_hoc.write_parquet(RESULTS_DIR / "post_hoc.parquet")
    coverage.write_parquet(RESULTS_DIR / "coverage.parquet")
    report = write_report(
        absolute=absolute,
        contrasts=contrasts,
        post_hoc=post_hoc,
        coverage=coverage,
        verdict=verdict,
        reproduction_lines=reproduction_lines,
    )
    (RESULTS_DIR / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
