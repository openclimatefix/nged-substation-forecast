"""Score experiment 00012's arms, save their predictions, and write the report.

Every arm is scored on the cases of fold A (`fold_a_tuesdays_v1`): the 53 validation runs' rows of
the nine series with an observed power. The error is the absolute difference between the member
average and the observed power, in points of capacity. Intervals resample whole validation months
and the three fitting seeds. The ten planned contrasts are read at 99.5% (a Bonferroni correction
of 5% across ten); every other interval is 95%.

Output: the prediction sets under `data/ml-experiments/00012-.../predictions/`, and
`work/results/{report.md, absolute.parquet, contrasts.parquet, by_lead.parquet}`.

    uv run python ml-experiments/00012-*/e00012_score.py
"""

from pathlib import Path
from typing import Final

import e00012_common as common
import numpy as np
import polars as pl

from studies import bootstrap, experiment_store

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
PLANNED_LEVEL: Final[float] = 99.5
N_RESAMPLES: Final[int] = 10_000
"""Resamples for the planned contrasts, so each tail of a 99.5% interval holds 25 draws."""
MIN_GAIN: Final[float] = 0.15
"""The smallest point estimate, in points of capacity, that counts as a gain."""
REPRODUCTION_TOLERANCE: Final[float] = 0.05
"""Arm `w` must match experiment 00001's `s1_memberavg` to within this, in points of capacity."""
LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0 to 24 h", 0.0, 24.0),
    ("1 to 2 days", 24.0, 72.0),
    ("3 to 6 days", 72.0, 168.0),
    ("7 to 10 days", 168.0, 264.0),
    ("11 to 15 days", 264.0, 10_000.0),
)
ContrastRow = tuple[str, common.TechnologyType, str, str, float, float]
PLANNED: Final[tuple[ContrastRow, ...]] = (
    ("C1", "solar", "cams", "w", 0.15, 0.15),
    ("C2", "solar", "cams_stage0", "cams", 0.15, 0.15),
    ("C3", "solar", "csr", "w", 0.15, 0.30),
    ("C4", "wind", "base_margin", "w", 0.15, 0.30),
    ("C5", "solar", "start_2023", "w", 0.15, 0.30),
    ("C6", "wind", "start_2023", "w", 0.15, 0.30),
    ("C7", "solar", "start_2024", "w", 0.15, 0.30),
    ("C8", "wind", "start_2024", "w", 0.15, 0.30),
    ("C9", "solar", "small_block", "w", 0.15, 0.15),
    ("C10", "wind", "speed_block", "w", 0.15, 0.15),
)
"""Planned contrasts as (id, technology, treatment, reference, gain threshold, equivalence band).

The first number is the smallest point estimate (as a negative) that counts as a gain; the second
is the half-width of the interval that counts as an equivalence."""


def validation_cases(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return one row per case with an observed power: keys, lead hours, and actual power."""
    capacity = sites.select("time_series_id", "effective_capacity_mw")
    ids = sites["time_series_id"].to_list()
    pieces = [
        pl.read_parquet(path, columns=[*common.KEY_COLUMNS, "ensemble_member", "power"])
        .filter(
            pl.col("time_series_id").is_in(ids)
            & (pl.col("ensemble_member") == 0)
            & pl.col("power").is_finite()
        )
        .select(
            *common.KEY_COLUMNS,
            "power",
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0,
        )
        for path in sorted(common.ENS_VALIDATION_DIR.glob("init=*.parquet"))
    ]
    return (
        pl.concat(pieces)
        .join(capacity, on="time_series_id", how="left", validate="m:1")
        .select(
            *common.KEY_COLUMNS,
            "lead_hours",
            actual_norm=pl.col("power") / pl.col("effective_capacity_mw"),
        )
    )


def climatology_points(*, cases: pl.DataFrame, sites: pl.DataFrame) -> pl.DataFrame:
    """Return the month-by-half-hour climatology as a point forecast of each case.

    Built from the cleaned observations before the first validation run, in local time.
    """
    key = [
        pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.month().alias("month"),
        (
            pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.hour() * 2
            + pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.minute() // 30
        ).alias("slot"),
    ]
    mean = (
        common.observed_training_power(sites=sites)
        .with_columns(*key)
        .group_by("time_series_id", "month", "slot")
        .agg(point_norm=pl.col("target").mean())
    )
    return (
        cases.select(*common.KEY_COLUMNS)
        .with_columns(*key)
        .join(mean, on=["time_series_id", "month", "slot"], how="left", validate="m:1")
        .select(*common.KEY_COLUMNS, "point_norm")
    )


def arm_rows(*, points: pl.DataFrame, cases: pl.DataFrame, arm: str, seed: int) -> pl.DataFrame:
    """Return one arm's per-case absolute error for one seed.

    Raises:
        ValueError: If a case of the arm's series has no forecast.
    """
    covered = cases.join(points.select("time_series_id").unique(), on="time_series_id", how="semi")
    joined = covered.join(points, on=list(common.KEY_COLUMNS), how="left", validate="1:1")
    if joined["point_norm"].null_count():
        msg = f"arm {arm}: {joined['point_norm'].null_count()} cases have no forecast"
        raise ValueError(msg)
    return joined.select(
        *common.KEY_COLUMNS,
        "lead_hours",
        "actual_norm",
        "point_norm",
        seed=pl.lit(seed),
        abs_error=100.0 * (pl.col("point_norm") - pl.col("actual_norm")).abs(),
    )


def all_arm_rows(*, cases: pl.DataFrame, sites: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return every arm's per-case errors over all seeds, keyed by arm name."""
    climatology = climatology_points(cases=cases, sites=sites)
    rows: dict[str, list[pl.DataFrame]] = {}
    for seed in common.SEEDS:
        member = pl.read_parquet(common.WORK_DIR / "member_average" / f"seed{seed}.parquet")
        for arm in member["arm"].unique().to_list():
            points = member.filter(pl.col("arm") == arm)
            rows.setdefault(arm, []).append(
                arm_rows(points=points, cases=cases, arm=arm, seed=seed)
            )
            if "control_norm" in points.columns and arm in ("cams", "cams_stage0"):
                control = points.select(*common.KEY_COLUMNS, point_norm=pl.col("control_norm"))
                rows.setdefault(f"{arm}_control", []).append(
                    arm_rows(points=control, cases=cases, arm=f"{arm}_control", seed=seed)
                )
        ceiling = pl.read_parquet(common.WORK_DIR / "ceiling" / f"seed{seed}.parquet")
        ceiling_points = cases.select(*common.KEY_COLUMNS).join(
            ceiling, on=["time_series_id", "valid_time"], how="left", validate="m:1"
        )
        rows.setdefault("era5_ceiling", []).append(
            arm_rows(points=ceiling_points, cases=cases, arm="era5_ceiling", seed=seed)
        )
        rows.setdefault("climatology", []).append(
            arm_rows(points=climatology, cases=cases, arm="climatology", seed=seed)
        )
    return {arm: pl.concat(frames) for arm, frames in rows.items()}


def reproduction(*, rows: dict[str, pl.DataFrame], sites: pl.DataFrame) -> list[str]:
    """Compare arms `w` and `cams` with experiment 00001's `s1_memberavg` and `s1_cams`.

    Raises:
        ValueError: If arm `w` differs from `s1_memberavg` by more than `REPRODUCTION_TOLERANCE`
            for a technology.
    """
    pilot = pl.concat(
        [
            pl.read_parquet(common.PILOT_SCORES_DIR / f"rows_seed{seed}.parquet").filter(
                pl.col("arm").is_in(["s1_memberavg", "s1_cams"])
            )
            for seed in common.SEEDS
        ]
    )
    lines = ["## Reproduction of experiment 00001", ""]
    lines += ["| technology | arm | here | 00001 |", "| --- | --- | --- | --- |"]
    for technology in ("solar", "wind"):
        ids = sites.filter(pl.col("technology") == technology)["time_series_id"].to_list()
        checks = [("w", "s1_memberavg"), *([("cams", "s1_cams")] if technology == "solar" else [])]
        for ours, theirs in checks:
            here = _mean_error(frame=rows[ours], ids=ids)
            there = _mean_error(frame=pilot.filter(pl.col("arm") == theirs), ids=ids)
            lines.append(f"| {technology} | {ours} | {here:.3f} | {there:.3f} |")
            if ours == "w" and abs(here - there) > REPRODUCTION_TOLERANCE:
                msg = f"{technology}: arm w scores {here:.3f}, experiment 00001 scored {there:.3f}"
                raise ValueError(msg)
    return lines


def _mean_error(*, frame: pl.DataFrame, ids: list[int]) -> float:
    """Return the mean absolute error (points of capacity) of the rows of the given series."""
    return float(frame.filter(pl.col("time_series_id").is_in(ids))["abs_error"].mean())  # ty: ignore[invalid-argument-type]


def technology_losses(
    *, rows: dict[str, pl.DataFrame], arms: list[str], ids: list[int]
) -> pl.DataFrame:
    """Return the bootstrap loss frame of the named arms, restricted to one technology's series."""
    return experiment_store.losses_for_bootstrap(
        arm_losses={
            arm: rows[arm]
            .filter(pl.col("time_series_id").is_in(ids))
            .select(*common.KEY_COLUMNS, "seed", "abs_error", "lead_hours")
            for arm in arms
        }
    )


def verdict_for(*, difference: float, lower: float, upper: float, gain: float, band: float) -> str:
    """Return `gain`, `harm`, `equivalent`, or `unresolved` for one contrast."""
    if upper < 0.0 and difference <= -gain:
        return "gain"
    if lower > 0.0 and difference >= gain:
        return "harm"
    if -band <= lower and upper <= band:
        return "equivalent"
    return "unresolved"


def planned_contrasts(*, rows: dict[str, pl.DataFrame], sites: pl.DataFrame) -> pl.DataFrame:
    """Score the ten planned contrasts at the corrected level."""
    out = []
    for contrast, technology, treatment, reference, gain, band in PLANNED:
        ids = sites.filter(pl.col("technology") == technology)["time_series_id"].to_list()
        losses = technology_losses(rows=rows, arms=[treatment, reference], ids=ids)
        differences, _ = bootstrap.paired_differences(
            losses=losses, treatment=treatment, reference=reference, metric="abs_error"
        )
        lower, upper = bootstrap.bootstrap_difference_at_level(
            losses=losses,
            treatment=treatment,
            reference=reference,
            metric="abs_error",
            level=PLANNED_LEVEL,
            n_resamples=N_RESAMPLES,
        )
        difference = float(differences.mean())
        out.append(
            {
                "contrast": contrast,
                "technology": technology,
                "treatment": treatment,
                "reference": reference,
                "difference": difference,
                "lower": lower,
                "upper": upper,
                "seed_spread": float(differences.mean(axis=1).std()),
                "verdict": verdict_for(
                    difference=difference, lower=lower, upper=upper, gain=gain, band=band
                ),
            }
        )
    return pl.DataFrame(out)


def overall_verdict(*, contrasts: pl.DataFrame) -> str:
    """Apply the decision rule to C1 to C4: supported, refuted, or inconclusive."""
    first_four = contrasts.filter(pl.col("contrast").is_in(["C1", "C2", "C3", "C4"]))
    if (first_four["verdict"] == "gain").any():
        return "supported"
    if (first_four["lower"] > -MIN_GAIN).all():
        return "refuted"
    return "inconclusive"


def absolute_table(*, rows: dict[str, pl.DataFrame], sites: pl.DataFrame) -> pl.DataFrame:
    """Return each arm's absolute error per technology with a 95% interval."""
    out = []
    for technology in ("solar", "wind"):
        ids = sites.filter(pl.col("technology") == technology)["time_series_id"].to_list()
        for arm, frame in rows.items():
            if not frame["time_series_id"].is_in(ids).any():
                continue
            losses = technology_losses(rows={arm: frame}, arms=[arm], ids=ids)
            interval = bootstrap.bootstrap_absolute(losses=losses, arm=arm, metric="abs_error")
            out.append({"technology": technology, "arm": arm, **interval})
    return pl.DataFrame(out)


def contrasts_by_lead(*, rows: dict[str, pl.DataFrame], sites: pl.DataFrame) -> pl.DataFrame:
    """Return each planned contrast's mean difference within each lead band (descriptive)."""
    out = []
    for contrast, technology, treatment, reference, _, _ in PLANNED:
        ids = sites.filter(pl.col("technology") == technology)["time_series_id"].to_list()
        losses = technology_losses(rows=rows, arms=[treatment, reference], ids=ids)
        for band, low, high in LEAD_BANDS:
            scoped = losses.filter((pl.col("lead_hours") >= low) & (pl.col("lead_hours") < high))
            differences, _ = bootstrap.paired_differences(
                losses=scoped, treatment=treatment, reference=reference, metric="abs_error"
            )
            out.append(
                {"contrast": contrast, "band": band, "difference": float(np.mean(differences))}
            )
    return pl.DataFrame(out)


def write_report(
    *,
    absolute: pl.DataFrame,
    contrasts: pl.DataFrame,
    by_lead: pl.DataFrame,
    reproduction_lines: list[str],
    verdict: str,
) -> str:
    """Return the report as markdown."""
    lines = [f"# Experiment 00012 report (decision rule on C1 to C4: {verdict})", ""]
    lines += ["## Absolute error, all leads (points of capacity, 95% interval)", ""]
    lines += [
        "| technology | arm | error | interval | seed spread |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r['technology']} | {r['arm']} | {r['value']:.2f} | {r['lower_95']:.2f} to "
        f"{r['upper_95']:.2f} | {r['seed_spread']:.3f} |"
        for r in absolute.sort("technology", "value").iter_rows(named=True)
    ]
    lines += ["", f"## Planned contrasts ({PLANNED_LEVEL}% intervals)", ""]
    lines += [
        "| id | technology | treatment minus reference | difference | interval | verdict |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r['contrast']} | {r['technology']} | {r['treatment']} minus {r['reference']} | "
        f"{r['difference']:+.3f} | {r['lower']:+.3f} to {r['upper']:+.3f} | {r['verdict']} |"
        for r in contrasts.iter_rows(named=True)
    ]
    lines += ["", "## Mean difference by lead band (descriptive)", ""]
    bands = [name for name, _, _ in LEAD_BANDS]
    lines += ["| id | " + " | ".join(bands) + " |", "| --- |" + " --- |" * len(bands)]
    for contrast in contrasts["contrast"]:
        part = by_lead.filter(pl.col("contrast") == contrast)
        cells = [f"{part.filter(pl.col('band') == b)['difference'].item():+.2f}" for b in bands]
        lines.append(f"| {contrast} | " + " | ".join(cells) + " |")
    return "\n".join([*lines, "", *reproduction_lines]) + "\n"


def save_predictions(*, rows: dict[str, pl.DataFrame]) -> None:
    """Save every arm's member-average forecasts as a prediction set."""
    for arm, frame in rows.items():
        name = arm.replace("_", "-")
        experiment_store.write_predictions(
            experiment_id=common.EXPERIMENT_ID,
            name=name,
            frame=frame.select(*common.KEY_COLUMNS, "seed", "point_norm"),
            kind="stage1",
            row_set_id=common.ROW_SET_ID,
        )


def main() -> None:
    """Score every arm, write the report, and save the predictions."""
    sites = common.site_table()
    cases = validation_cases(sites=sites)
    rows = all_arm_rows(cases=cases, sites=sites)
    reproduction_lines = reproduction(rows=rows, sites=sites)
    absolute = absolute_table(rows=rows, sites=sites)
    contrasts = planned_contrasts(rows=rows, sites=sites)
    by_lead = contrasts_by_lead(rows=rows, sites=sites)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    absolute.write_parquet(RESULTS_DIR / "absolute.parquet")
    contrasts.write_parquet(RESULTS_DIR / "contrasts.parquet")
    by_lead.write_parquet(RESULTS_DIR / "by_lead.parquet")
    report = write_report(
        absolute=absolute,
        contrasts=contrasts,
        by_lead=by_lead,
        reproduction_lines=reproduction_lines,
        verdict=overall_verdict(contrasts=contrasts),
    )
    (RESULTS_DIR / "report.md").write_text(report)
    print(report)
    save_predictions(rows=rows)


if __name__ == "__main__":
    main()
