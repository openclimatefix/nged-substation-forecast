"""Score experiment 00023's arms: CRPS contrasts, coverage by lead band, and the verdict.

The CRPS is on every case, in points of capacity (100 times the mass-weighted pinball estimate).
Coverage is the share of non-trivial cases whose observed power lies in the 5% to 95% band
inclusive, in percent. Intervals resample whole initialisation months. The 12 planned contrasts
(6 per technology) are read at 99.58% (a Bonferroni correction of 5% across 12); coverage intervals
are 95%.

Output: `work/results/{report.md, absolute.parquet, contrasts.parquet, coverage.parquet,
by_band.parquet}`.

    uv run python ml-experiments/00023-*/e00023_score.py
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00023_common import (
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
"""A gain must be at most this many points of capacity (see the plan)."""
REFERENCE_ARMS: Final[dict[str, str]] = {"climatology": "climatology_q"}
PLANNED: Final[tuple[tuple[str, str, str], ...]] = (
    ("C1", "cal_lags", "cal0"),
    ("C2", "cal_sums", "cal0"),
    ("C3", "cal_both", "cal_lags"),
    ("C4", "cal_both", "mixture"),
    ("C5", "mix_c14", "mixture"),
    ("C6", "cal_both_c14", "cal_both"),
)
"""Each contrast is the arm named first minus the arm named second, per technology."""
POST_HOC: Final[tuple[tuple[str, str, str], ...]] = (("P1", "mix_c14_median", "mixture"),)
"""Post hoc contrasts, read at 95% and outside the Bonferroni family."""
DECIDING: Final[tuple[str, ...]] = ("C1", "C5")


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
    for technology in TECHNOLOGIES:
        for contrast, arm, reference in (*PLANNED, *POST_HOC):
            level = PLANNED_LEVEL if (contrast, arm, reference) in PLANNED else 95.0
            losses = _losses(rows=rows, technology=technology, arms=[arm, reference])
            differences, _ = bootstrap.paired_differences(
                losses=losses, treatment=arm, reference=reference, metric="crps"
            )
            lower, upper = bootstrap.bootstrap_difference_at_level(
                losses=losses,
                treatment=arm,
                reference=reference,
                metric="crps",
                level=level,
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


def by_band(*, rows: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return each planned contrast's mean difference (points) by lead band, without intervals."""
    out = []
    for technology in TECHNOLOGIES:
        for contrast, arm, reference in (*PLANNED, *POST_HOC):
            joined = (
                rows[arm]
                .filter(pl.col("technology") == technology)
                .select(*KEY_COLUMNS, "seed", "band", "crps")
                .join(
                    rows[reference].select(*KEY_COLUMNS, "seed", pl.col("crps").alias("other")),
                    on=[*KEY_COLUMNS, "seed"],
                    validate="1:1",
                )
            )
            for band, mean in (
                joined.group_by("band").agg((pl.col("crps") - pl.col("other")).mean()).iter_rows()
            ):
                out.append(
                    {"contrast": contrast, "technology": technology, "band": band, "diff": mean}
                )
    return pl.DataFrame(out)


def technology_verdict(*, row: dict[str, float]) -> str:
    """Classify a deciding contrast: gain, small gain, refuted, or inconclusive."""
    if row["upper"] < 0.0:
        return "gain" if row["difference"] <= GAIN_THRESHOLD else "small gain"
    if row["lower"] > GAIN_THRESHOLD:
        return "refuted"
    return "inconclusive"


def overall_verdict(*, per_technology: Mapping[str, str]) -> str:
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


def technology_class(*, classes: list[str]) -> str:
    """Combine the deciding contrasts of one technology: the best class, or refuted if all are."""
    for best in ("gain", "small gain"):
        if best in classes:
            return best
    return "refuted" if all(c == "refuted" for c in classes) else "inconclusive"


def write_report(
    *,
    absolute: pl.DataFrame,
    contrasts: pl.DataFrame,
    coverage: pl.DataFrame,
    per_technology: Mapping[str, str],
    verdict: str,
) -> str:
    """Return the report as markdown."""
    lines = [f"# Experiment 00023 report (verdict: {verdict}; {per_technology})", ""]
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
    rows |= reference_rows(base=rows["mixture"])
    absolute = absolute_table(rows=rows)
    contrasts = planned_contrasts(rows=rows)
    per_technology: dict[str, str] = {
        technology: technology_class(
            classes=[
                technology_verdict(
                    row=contrasts.filter(
                        (pl.col("contrast") == head) & (pl.col("technology") == technology)
                    ).row(0, named=True)
                )
                for head in DECIDING
            ]
        )
        for technology in TECHNOLOGIES
    }
    verdict = overall_verdict(per_technology=per_technology)
    coverage = coverage_table(rows=rows)
    bands = by_band(rows=rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    absolute.write_parquet(RESULTS_DIR / "absolute.parquet")
    contrasts.write_parquet(RESULTS_DIR / "contrasts.parquet")
    coverage.write_parquet(RESULTS_DIR / "coverage.parquet")
    bands.write_parquet(RESULTS_DIR / "by_band.parquet")
    report = write_report(
        absolute=absolute,
        contrasts=contrasts,
        coverage=coverage,
        per_technology=per_technology,
        verdict=verdict,
    )
    band_names_in_order = [name for name, _, _ in LEAD_BANDS]
    report += "\n## Mean difference by lead band (points)\n\n"
    report += "| technology | contrast | " + " | ".join(band_names_in_order) + " |\n"
    report += "| --- | --- |" + " --- |" * len(band_names_in_order) + "\n"
    for technology in TECHNOLOGIES:
        for contrast, _, _ in (*PLANNED, *POST_HOC):
            part = bands.filter(
                (pl.col("technology") == technology) & (pl.col("contrast") == contrast)
            )
            cells = [
                f"{part.filter(pl.col('band') == b)['diff'].item():+.3f}"
                for b in band_names_in_order
            ]
            report += f"| {technology} | {contrast} | " + " | ".join(cells) + " |\n"
    (RESULTS_DIR / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
