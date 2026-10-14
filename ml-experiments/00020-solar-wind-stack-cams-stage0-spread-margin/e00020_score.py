"""Score experiment 00020's arms: CRPS contrasts, coverage by lead band, and the verdict.

The CRPS is on every case, in points of capacity (100 times the mass-weighted pinball estimate).
Coverage is the share of non-trivial cases whose observed power lies in the 5% to 95% band
inclusive, in percent. Intervals resample whole initialisation months. The 12 planned contrasts
are read at 99.58% (a Bonferroni correction of 5% across 12); coverage intervals are 95%.

Output: `work/results/{report.md, absolute.parquet, contrasts.parquet, coverage.parquet}`.

    uv run python ml-experiments/00020-*/e00020_score.py
"""

from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00020_common import (
    ARM_SOURCE,
    ARMS,
    EXPERIMENT_ID,
    KEY_COLUMNS,
    LEAD_BANDS,
    LEVEL_COLUMNS,
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
PLANNED_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 12)
N_RESAMPLES: Final[int] = 20_000
COVERAGE_RESAMPLES: Final[int] = 2_000
COVERAGE_SEED: Final[int] = 20261015
GAIN_THRESHOLD: Final[float] = -0.05
"""The full stack's gain must be at most this many points of capacity (see the plan)."""
KEEP_MARGIN: Final[dict[str, float]] = {"s0": 0.02, "spread": 0.02, "cams": 0.0, "margin": 0.0}
REFERENCE_ARMS: Final[dict[str, str]] = {"climatology": "climatology_q"}
PLANNED: Final[tuple[tuple[str, TechnologyType, ArmType, ArmType], ...]] = (
    ("S1", "solar", "full", "base"),
    ("S2", "solar", "cams", "base"),
    ("S3", "solar", "cams_s0", "cams"),
    ("S4", "solar", "base_spread", "base"),
    ("S5", "solar", "full_no_cams", "full"),
    ("S6", "solar", "full_no_s0", "full"),
    ("S7", "solar", "cams_s0", "full"),
    ("W1", "wind", "full", "base"),
    ("W2", "wind", "margin", "base"),
    ("W3", "wind", "base_spread", "base"),
    ("W4", "wind", "base_spread", "full"),
    ("W5", "wind", "margin", "full"),
)
"""Each contrast is the arm named first minus the arm named second."""
KEEP_COMPONENT: Final[dict[str, str]] = {
    "S5": "cams",
    "S6": "s0",
    "S7": "spread",
    "W4": "margin",
    "W5": "spread",
}
"""The component each drop-one contrast tests, which sets its keep threshold."""


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


def reference_rows(*, base: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return the per-case CRPS and coverage of the 00002 reference arms, on `base`'s cases."""
    scaffold = base.select(*KEY_COLUMNS, "seed", "lead_hours", "technology", "band", "trivial")
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


def coverage_interval(*, frame: pl.DataFrame) -> dict[str, float]:
    """Return the coverage (percent) of non-trivial cases with a month-resampled 95% interval."""
    scope = frame.filter(~pl.col("trivial"))
    months = scope["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    labels = np.unique(months)
    covered_flags = scope["covered"].to_numpy()
    hits = np.array([covered_flags[months == m].sum() for m in labels])
    counts = np.array([float((months == m).sum()) for m in labels])
    generator = np.random.default_rng(COVERAGE_SEED)
    draws = generator.integers(0, labels.size, size=(COVERAGE_RESAMPLES, labels.size))
    resampled = 100.0 * hits[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "coverage": 100.0 * float(hits.sum() / counts.sum()),
        "lower": float(np.percentile(resampled, 2.5)),
        "upper": float(np.percentile(resampled, 97.5)),
    }


def coverage_table(*, rows: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return coverage per arm, technology, and lead band."""
    out = []
    for arm, frame in rows.items():
        for technology in TECHNOLOGIES:
            tech = frame.filter(pl.col("technology") == technology)
            if tech.is_empty():
                continue
            for band, _, _ in LEAD_BANDS:
                scope = tech.filter(pl.col("band") == band)
                if scope.filter(~pl.col("trivial")).is_empty():
                    continue
                out.append(
                    {"arm": arm, "technology": technology, "band": band}
                    | coverage_interval(frame=scope)
                )
    return pl.DataFrame(out)


def _losses(*, rows: dict[str, pl.DataFrame], technology: TechnologyType, arms: list[str]):  # noqa: ANN202
    return experiment_store.losses_for_bootstrap(
        arm_losses={
            name: rows[name]
            .filter(pl.col("technology") == technology)
            .select(*KEY_COLUMNS, "seed", "crps")
            for name in arms
        }
    )


def absolute_table(*, rows: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return each arm's mean CRPS per technology with a 95% interval."""
    out = []
    for arm, frame in rows.items():
        for technology in TECHNOLOGIES:
            if frame.filter(pl.col("technology") == technology).is_empty():
                continue
            losses = _losses(rows=rows, technology=technology, arms=[arm])
            out.append(
                {"technology": technology, "arm": arm}
                | bootstrap.bootstrap_absolute(losses=losses, arm=arm, metric="crps")
            )
    return pl.DataFrame(out)


def planned_contrasts(*, rows: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Score the 12 planned contrasts at `PLANNED_LEVEL`."""
    out = []
    for contrast, technology, arm, reference in PLANNED:
        losses = _losses(rows=rows, technology=technology, arms=[arm, reference])
        differences, _ = bootstrap.paired_differences(
            losses=losses, treatment=arm, reference=reference, metric="crps"
        )
        lower, upper = bootstrap.bootstrap_difference_at_level(
            losses=losses,
            treatment=arm,
            reference=reference,
            metric="crps",
            level=PLANNED_LEVEL,
            n_resamples=N_RESAMPLES,
        )
        out.append(
            {
                "contrast": contrast,
                "technology": technology,
                "arm": arm,
                "reference": reference,
                "difference": float(differences.mean()),
                "lower": lower,
                "upper": upper,
            }
        )
    return pl.DataFrame(out)


def technology_verdict(*, row: dict[str, float]) -> str:
    """Classify a headline contrast (S1 or W1): gain, small gain, refuted, or inconclusive."""
    if row["upper"] < 0.0:
        return "gain" if row["difference"] <= GAIN_THRESHOLD else "small gain"
    if row["lower"] > GAIN_THRESHOLD:
        return "refuted"
    return "inconclusive"


def overall_verdict(*, per_technology: dict[str, str]) -> str:
    """Combine the two technologies' classifications as the decision rule does."""
    values = list(per_technology.values())
    if values.count("gain") == len(values):
        return "supported"
    if "gain" in values:
        return "partly supported"
    if "small gain" in values:
        return "small gain"
    if values.count("refuted") == len(values):
        return "refuted"
    return "inconclusive"


def keep_decisions(*, contrasts: pl.DataFrame) -> pl.DataFrame:
    """Return whether each component stays in the stack: its drop-one interval is above zero."""
    out = []
    for contrast, component in KEEP_COMPONENT.items():
        row = contrasts.filter(pl.col("contrast") == contrast).row(0, named=True)
        keep = row["lower"] > 0.0 and row["difference"] >= KEEP_MARGIN[component]
        out.append({"contrast": contrast, "component": component, "keep": keep})
    return pl.DataFrame(out)


def write_report(
    *,
    absolute: pl.DataFrame,
    contrasts: pl.DataFrame,
    coverage: pl.DataFrame,
    keep: pl.DataFrame,
    per_technology: dict[str, str],
    verdict: str,
) -> str:
    """Return the report as markdown."""
    lines = [f"# Experiment 00020 report (verdict: {verdict}; {per_technology})", ""]
    lines += ["## Mean CRPS (points of capacity, 95% interval)", ""]
    lines += [
        "| technology | arm | CRPS | interval | seed spread |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r['technology']} | {r['arm']} | {r['value']:.3f} | {r['lower_95']:.3f} to "
        f"{r['upper_95']:.3f} | {r['seed_spread']:.3f} |"
        for r in absolute.sort("technology", "value").iter_rows(named=True)
    ]
    lines += ["", f"## Planned CRPS contrasts ({PLANNED_LEVEL:.2f}% intervals)", ""]
    lines += [
        "| id | technology | arm minus reference | difference | interval |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r['contrast']} | {r['technology']} | {r['arm']} - {r['reference']} | "
        f"{r['difference']:+.3f} | {r['lower']:+.3f} to {r['upper']:+.3f} |"
        for r in contrasts.iter_rows(named=True)
    ]
    lines += ["", "## Components kept (drop-one contrast above zero by the margin)", ""]
    lines += ["| contrast | component | keep |", "| --- | --- | --- |"]
    lines += [
        f"| {r['contrast']} | {r['component']} | {r['keep']} |" for r in keep.iter_rows(named=True)
    ]
    bands = [name for name, _, _ in LEAD_BANDS]
    for technology in TECHNOLOGIES:
        lines += ["", f"## Coverage of the 5% to 95% band, {technology} (%)", ""]
        lines += ["| arm | " + " | ".join(bands) + " |", "| --- |" + " --- |" * len(bands)]
        part = coverage.filter(pl.col("technology") == technology)
        for arm in part["arm"].unique(maintain_order=True):
            cells = []
            for band in bands:
                cell = part.filter((pl.col("arm") == arm) & (pl.col("band") == band))
                cells.append(f"{cell['coverage'].item():.1f}" if cell.height else "n/a")
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Score every arm, write the report, and save the tables."""
    rows: dict[str, pl.DataFrame] = {arm: arm_rows(arm=arm) for arm in ARMS if _exists(arm)}
    rows |= reference_rows(base=rows["base"])
    absolute = absolute_table(rows=rows)
    contrasts = planned_contrasts(rows=rows)
    per_technology = {
        technology: technology_verdict(
            row=contrasts.filter(pl.col("contrast") == head).row(0, named=True)
        )
        for technology, head in (("solar", "S1"), ("wind", "W1"))
    }
    verdict = overall_verdict(per_technology=per_technology)
    keep = keep_decisions(contrasts=contrasts)
    coverage = coverage_table(rows=rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    absolute.write_parquet(RESULTS_DIR / "absolute.parquet")
    contrasts.write_parquet(RESULTS_DIR / "contrasts.parquet")
    coverage.write_parquet(RESULTS_DIR / "coverage.parquet")
    report = write_report(
        absolute=absolute,
        contrasts=contrasts,
        coverage=coverage,
        keep=keep,
        per_technology=per_technology,
        verdict=verdict,
    )
    (RESULTS_DIR / "report.md").write_text(report)
    print(report)


def _exists(arm: ArmType) -> bool:
    """Return whether any technology builds the arm."""
    return any(arm in ARM_SOURCE[technology] for technology in TECHNOLOGIES)


if __name__ == "__main__":
    main()
