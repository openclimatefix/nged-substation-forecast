"""Score experiment 00026's arms: CRPS contrasts, coverage by lead band, per-farm gains, verdict.

The CRPS is on every case, in points of capacity (100 times the mass-weighted pinball estimate).
Coverage is the share of non-trivial cases whose observed power lies in the 5% to 95% band
inclusive, in percent. Intervals resample whole initialisation months. The decisive contrasts (C1
and C2 for each technology, 4 in all) are read at 98.75% (a Bonferroni correction of 5% across 4);
the other contrasts are descriptive and read at the same level. Coverage intervals are 95%.

Output: `work/results/{report.md, absolute.parquet, contrasts.parquet, coverage.parquet,
per_farm.parquet}`.

    uv run python ml-experiments/00026-*/e00026_score.py
"""

from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00026_common import (
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
    site_table,
)

from studies import bootstrap, experiment_store

RESULTS_DIR: Final[Path] = WORK_DIR / "results"
PLANNED_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 4)
N_RESAMPLES: Final[int] = 20_000
COVERAGE_RESAMPLES: Final[int] = 2_000
COVERAGE_SEED: Final[int] = 20261015
GAIN_THRESHOLD: Final[float] = -0.03
"""C1's gain must be at most this many points of capacity (see the plan)."""
REFERENCE_ARMS: Final[dict[str, str]] = {"climatology": "climatology_q"}
PLANNED: Final[tuple[tuple[str, ArmType, ArmType], ...]] = (
    ("C1", "a2", "a0"),
    ("C2", "a2", "a7"),
    ("C3", "a3", "a6"),
    ("C4", "a3", "a2"),
    ("C5", "a4", "a2"),
    ("C6", "a5", "a2"),
)
"""Each contrast is the arm named first minus the arm named second, for each technology."""


def arm_rows(*, arm: ArmType) -> pl.DataFrame:
    """Return one arm's per-case CRPS (points) and coverage indicator (0 or 1), all seeds."""
    frame = experiment_store.read_predictions(
        experiment_id=EXPERIMENT_ID, name=arm, row_set_id=ROW_SET_ID
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
    """Score the planned contrasts, for each technology, at `PLANNED_LEVEL`."""
    out = []
    for technology in TECHNOLOGIES:
        for contrast, arm, reference in PLANNED:
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


def per_farm(*, rows: dict[str, pl.DataFrame], sites: pl.DataFrame) -> pl.DataFrame:
    """Return each farm's mean CRPS difference (points of capacity) of A2, A3 and A7 against A0."""
    labels = sites.select("time_series_id", "site")
    out = []
    for arm in ("a2", "a3", "a7"):
        joined = (
            rows[arm]
            .join(
                rows["a0"].select(*KEY_COLUMNS, "seed", reference="crps"), on=[*KEY_COLUMNS, "seed"]
            )
            .join(labels, on="time_series_id")
            .group_by("site")
            .agg(difference=(pl.col("crps") - pl.col("reference")).mean())
            .with_columns(arm=pl.lit(arm))
        )
        out.append(joined)
    return pl.concat(out).sort("arm", "site")


def technology_verdict(*, c1: dict[str, float], c2: dict[str, float]) -> str:
    """Classify C1 as the decision rule does, and demand C2 wholly below zero for a real gain."""
    if c1["upper"] < 0.0:
        if c2["upper"] >= 0.0:
            return "inconclusive"
        return "gain" if c1["difference"] <= GAIN_THRESHOLD else "small gain"
    if c1["lower"] > GAIN_THRESHOLD:
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


def write_report(
    *,
    absolute: pl.DataFrame,
    contrasts: pl.DataFrame,
    coverage: pl.DataFrame,
    per_technology: dict[str, str],
    verdict: str,
) -> str:
    """Return the report as markdown."""
    lines = [f"# Experiment 00026 report (verdict: {verdict}; {per_technology})", ""]
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
    lines += ["", f"## CRPS contrasts ({PLANNED_LEVEL:.2f}% intervals)", ""]
    lines += [
        "| id | technology | arm minus reference | difference | interval |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r['contrast']} | {r['technology']} | {r['arm']} - {r['reference']} | "
        f"{r['difference']:+.3f} | {r['lower']:+.3f} to {r['upper']:+.3f} |"
        for r in contrasts.iter_rows(named=True)
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
    rows: dict[str, pl.DataFrame] = {arm: arm_rows(arm=arm) for arm in ARMS}
    rows |= reference_rows(base=rows["a0"])
    absolute = absolute_table(rows=rows)
    contrasts = planned_contrasts(rows=rows)
    farms = per_farm(rows=rows, sites=site_table())
    per_technology: dict[str, str] = {
        technology: technology_verdict(
            c1=contrasts.filter(
                (pl.col("contrast") == "C1") & (pl.col("technology") == technology)
            ).row(0, named=True),
            c2=contrasts.filter(
                (pl.col("contrast") == "C2") & (pl.col("technology") == technology)
            ).row(0, named=True),
        )
        for technology in TECHNOLOGIES
    }
    verdict = overall_verdict(per_technology=per_technology)
    coverage = coverage_table(rows=rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    absolute.write_parquet(RESULTS_DIR / "absolute.parquet")
    contrasts.write_parquet(RESULTS_DIR / "contrasts.parquet")
    coverage.write_parquet(RESULTS_DIR / "coverage.parquet")
    farms.write_parquet(RESULTS_DIR / "per_farm.parquet")
    report = write_report(
        absolute=absolute,
        contrasts=contrasts,
        coverage=coverage,
        per_technology=per_technology,
        verdict=verdict,
    )
    farm_lines = ["", "## Mean CRPS difference against A0 by farm (points of capacity)", ""]
    farm_lines += ["| arm | farm | difference |", "| --- | --- | --- |"]
    farm_lines += [
        f"| {r['arm']} | {r['site']} | {r['difference']:+.3f} |"
        for r in farms.iter_rows(named=True)
    ]
    report += "\n".join(farm_lines) + "\n"
    (RESULTS_DIR / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
