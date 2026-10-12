"""Print every table the quick-wins page quotes into one `summary.md`.

Reads the parquet files that `report_quick_wins.py` and `calibrate.py` wrote, scores the stored
baselines on the same sampled runs, and prints: the fold A leaderboard with the baselines, the
paired contrasts by series type and lead band, the per-seed spread, fold B, calibration, the round 0
cleaning tables, and every arm's feature columns.

    uv run python studies/quick_wins/report_summary.py
"""

from pathlib import Path
from typing import Final

import arms as arm_definitions
import month_bootstrap
import polars as pl
import quick_wins_inputs as inputs
from contracts.settings import Settings
from per_run_scores import normalised_by_capacity
from report_tables import markdown_table

REPORT_DIR: Final[Path] = inputs.OUTPUT_ROOT / "report"
CALIBRATION_DIR: Final[Path] = inputs.OUTPUT_ROOT / "calibration"
BASELINE_DIR: Final[Path] = inputs.OUTPUT_ROOT / "production_arm" / "colsample0.8_seed0" / "report"
BASELINES: Final[dict[str, str]] = {
    "stored production (xgboost_baseline)": "scores_stored_xgboost_baseline.parquet",
    "manual_heuristic": "scores_stored_manual_heuristic.parquet",
    "climatology": "scores_stored_climatology.parquet",
}
"""Stored experiments scored on the same sampled runs, by display name and scores file."""


def _read(*, fold: str, kind: str, tag: str) -> pl.DataFrame | None:
    path = REPORT_DIR / fold / f"{kind}_{tag}.parquet"
    return pl.read_parquet(path) if path.exists() else None


def _round(frame: pl.DataFrame, columns: list[str], digits: int = 4) -> pl.DataFrame:
    return frame.with_columns(pl.col(c).round(digits) for c in columns)


def _leaderboard(*, lines: list[str]) -> None:
    fold = inputs.FOLD_A
    levels = _read(fold=fold, kind="levels", tag="seeds012")
    if levels is None:
        return
    lines += [
        "## Fold A leaderboard: mean over three seeds, all series types, all leads",
        "",
        (
            "Normalised CRPS and MAE (fraction of effective capacity; smaller is better), with "
            "the 95% interval from resampling whole months."
        ),
        "",
    ]
    wide = (
        levels.filter((pl.col("group") == "all types") & (pl.col("lead_band") == "all"))
        .select("arm", "metric", "value", "lower_95", "upper_95")
        .pivot(on="metric", index="arm", values=["value", "lower_95", "upper_95"])
        .sort("value_crps")
    )
    lines.append(markdown_table(frame=_round(wide, [c for c in wide.columns if c != "arm"])))
    settings = Settings()
    capacity = pl.DataFrame(inputs.effective_capacity(settings=settings))
    rows = []
    for name, file in BASELINES.items():
        scores = normalised_by_capacity(
            scores=pl.read_parquet(BASELINE_DIR / file), capacity=capacity
        )
        row: dict[str, object] = {"baseline": name}
        for metric, column in (("crps", "crps_norm"), ("mae", "abs_error_norm")):
            level = month_bootstrap.level_interval(
                table=month_bootstrap.monthly_table(scores=scores, metric=column)
            )
            row |= {
                f"{metric}": level.estimate,
                f"{metric}_lower_95": level.lower_95,
                f"{metric}_upper_95": level.upper_95,
            }
        rows.append(row)
    lines += [
        "",
        (
            "Stored baselines on the same 53 runs (single fit; manual_heuristic and climatology "
            "have no seeds)."
        ),
        "",
        markdown_table(frame=_round(pl.DataFrame(rows), [c for c in rows[0] if c != "baseline"])),
        "",
    ]


def _contrast_tables(*, lines: list[str], fold: str, tag: str, reference: str) -> None:
    contrasts = _read(fold=fold, kind="contrasts", tag=tag)
    if contrasts is None:
        return
    for group_by, heading in (("group", "by series type"), ("lead_band", "by lead band")):
        for metric in ("crps", "mae"):
            frame = contrasts.filter(
                (pl.col("reference") == reference)
                & (pl.col("metric") == metric)
                & (
                    (pl.col("lead_band") == "all")
                    if group_by == "group"
                    else (pl.col("group") == "all types")
                )
            )
            if frame.is_empty():
                continue
            lines += [
                f"### {fold}, {tag}: {metric} difference from `{reference}`, {heading}",
                "",
                (
                    "Negative favours the arm. Interval: 95% month-resampled; months better of "
                    "months scored."
                ),
                "",
            ]
            frame = frame.with_columns(
                interval=pl.format(
                    "[{}, {}]", pl.col("lower_95").round(4), pl.col("upper_95").round(4)
                ),
                months=pl.format("{} of {}", pl.col("months_better"), pl.col("n_months")),
                difference=pl.col("difference").round(4),
            )
            wide = frame.pivot(
                on=group_by, index="arm", values=["difference", "interval", "months"]
            ).sort("arm")
            lines += [markdown_table(frame=wide), ""]


def _seed_spread(*, lines: list[str]) -> None:
    rows = []
    for seed in (0, 1, 2):
        levels = _read(fold=inputs.FOLD_A, kind="levels", tag=f"seeds{seed}")
        if levels is None:
            continue
        rows.append(
            levels.filter(
                (pl.col("group") == "all types")
                & (pl.col("lead_band") == "all")
                & (pl.col("metric") == "crps")
            ).select("arm", pl.col("value").alias(f"crps_seed{seed}"))
        )
    if not rows:
        return
    wide = rows[0]
    for other in rows[1:]:
        wide = wide.join(other, on="arm", how="inner")
    columns = [c for c in wide.columns if c != "arm"]
    wide = wide.with_columns(spread=pl.max_horizontal(columns) - pl.min_horizontal(columns)).sort(
        "crps_seed0"
    )
    lines += [
        "## Spread across the three fitting seeds, fold A, normalised CRPS, all types",
        "",
        markdown_table(frame=_round(wide, [c for c in wide.columns if c != "arm"], 5)),
        "",
    ]


def _calibration(*, lines: list[str]) -> None:
    for fold in (inputs.FOLD_A, inputs.FOLD_B):
        for path in sorted((CALIBRATION_DIR / fold).glob("calibration_seed*.parquet")):
            frame = pl.read_parquet(path).filter(pl.col("group").is_in(["all types", "PV", "Wind"]))
            lines += [
                f"## Spread calibration (post hoc), {fold}, {path.stem}",
                "",
                (
                    "Runs after the first eight only. Difference is calibrated minus "
                    "uncalibrated normalised CRPS. Coverage is the share of outcomes inside the "
                    "members' 10% to 90% band; nominal 0.8."
                ),
                "",
                markdown_table(
                    frame=_round(frame, [c for c in frame.columns if frame[c].dtype == pl.Float64])
                ),
                "",
            ]


def _arm_scores_with_types(*, fold: str, arm: str, seed: int, types: pl.DataFrame) -> pl.DataFrame:
    """Read one arm's per-row scores with each series' type."""
    path = inputs.OUTPUT_ROOT / "arms" / fold / f"{arm}_seed{seed}" / "scores.parquet"
    return pl.read_parquet(path).join(types, on="time_series_id")


def _paired_rows(
    *, arm: pl.DataFrame, reference: pl.DataFrame, label: str
) -> list[dict[str, object]]:
    """Month-resampled difference `arm - reference` by series type, for CRPS and MAE."""
    rows: list[dict[str, object]] = []
    groups: list[str | None] = [None, *sorted(arm["time_series_type"].unique().to_list())]
    for series_type in groups:
        for metric, column in (("crps", "crps_norm"), ("mae", "abs_error_norm")):
            tables = []
            for frame in (arm, reference):
                part = (
                    frame
                    if series_type is None
                    else frame.filter(pl.col("time_series_type") == series_type)
                )
                tables.append(month_bootstrap.monthly_table(scores=part, metric=column))
            diff = month_bootstrap.difference_interval(arm=tables[0], reference=tables[1])
            rows.append(
                {
                    "comparison": label,
                    "group": series_type or "all types",
                    "metric": metric,
                    "reference_value": month_bootstrap.level_interval(table=tables[1]).estimate,
                    "difference": diff.estimate,
                    "interval_95": f"[{diff.lower_95:.4f}, {diff.upper_95:.4f}]",
                    "t_interval_95": f"[{diff.t_lower_95:.4f}, {diff.t_upper_95:.4f}]",
                    "months_better": f"{diff.months_better} of {diff.n_months}",
                }
            )
    return rows


def _types() -> pl.DataFrame:
    settings = Settings()
    series_ids = inputs.eligible_series_ids(settings=settings, fold_id=inputs.FOLD_A)
    return pl.DataFrame(
        inputs.series_metadata(settings=settings, time_series_ids=series_ids)
    ).select("time_series_id", pl.col("time_series_type").cast(pl.String))


def _g1_against_quantile(*, lines: list[str]) -> None:
    """Contrast the quantile combination with the heuristic's inputs against the same without."""
    types = _types()
    rows: list[dict[str, object]] = []
    for fold in (inputs.FOLD_A, inputs.FOLD_B):
        paths = [
            inputs.OUTPUT_ROOT / "arms" / fold / f"{name}_seed0" / "scores.parquet"
            for name in ("r2_e1_c1_a3w_d3_g1", "r2_e1_c1_a3w_d3")
        ]
        if not all(path.exists() for path in paths):
            continue
        rows += _paired_rows(
            arm=_arm_scores_with_types(fold=fold, arm="r2_e1_c1_a3w_d3_g1", seed=0, types=types),
            reference=_arm_scores_with_types(fold=fold, arm="r2_e1_c1_a3w_d3", seed=0, types=types),
            label=fold,
        )
    if not rows:
        return
    lines += [
        "## The heuristic's inputs added to the quantile combination (seed 0)",
        "",
        (
            "Difference is `r2_e1_c1_a3w_d3_g1` minus `r2_e1_c1_a3w_d3` in normalised CRPS "
            "and MAE; "
            "negative favours the arm with the heuristic's inputs. Interval: 95% "
            "month-resampled; t-interval over the monthly differences; months in which "
            "the arm with the heuristic's inputs was better."
        ),
        "",
        markdown_table(frame=pl.DataFrame(rows)),
        "",
    ]


CALIBRATED_ARMS: Final[tuple[str, ...]] = (
    "production_cleaned",
    "r2_c4_c2_c1_a3w_d3",
    "r2_c4_c1_a3w_d3",
)
"""Arms with a calibration run on fold A seed 0, from which the best point arm is chosen."""

QUANTILE_ARMS: Final[tuple[str, ...]] = (
    "e1_quantile",
    "r2_e1_c1_a3w_d3",
    "r2_e1_c1_a3w_d3_g1",
)
"""The quantile arms fitted on fold A seed 0."""


def _calibrated_against_quantile(*, lines: list[str]) -> None:
    """Compare calibrated point arms with the quantile arms on the runs after the warm-up."""
    types = _types()
    base = CALIBRATION_DIR / inputs.FOLD_A
    calibrated: dict[str, pl.DataFrame] = {}
    for name in CALIBRATED_ARMS:
        path = base / f"{name}_seed0" / "calibrated.parquet"
        if path.exists():
            calibrated[name] = pl.read_parquet(path)
    if "production_cleaned" not in calibrated:
        return
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    scored_rows = calibrated["production_cleaned"].select(keys)
    n_runs = scored_rows["power_fcst_init_time"].n_unique()

    def on_those_rows(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.join(scored_rows, on=keys, how="semi").join(types, on="time_series_id")

    models = {f"{name} (calibrated)": on_those_rows(frame) for name, frame in calibrated.items()}
    for name in QUANTILE_ARMS:
        path = inputs.OUTPUT_ROOT / "arms" / inputs.FOLD_A / f"{name}_seed0" / "scores.parquet"
        if path.exists():
            models[name] = on_those_rows(pl.read_parquet(path))
    best_point = min(
        (name for name in calibrated if name != "production_cleaned"),
        key=lambda name: float(models[f"{name} (calibrated)"]["crps_norm"].to_numpy().mean()),
        default=None,
    )
    best_quantile = min(
        (name for name in QUANTILE_ARMS if name in models),
        key=lambda name: float(models[name]["crps_norm"].to_numpy().mean()),
        default=None,
    )
    rows: list[dict[str, object]] = []
    for label, frame in models.items():
        per_series = frame.group_by("time_series_id").agg(
            pl.col("crps_norm").mean(), pl.col("abs_error_norm").mean()
        )
        rows.append(
            {
                "model": label,
                "rows": frame.height,
                "crps": float(per_series["crps_norm"].to_numpy().mean()),
                "mae": float(per_series["abs_error_norm"].to_numpy().mean()),
            }
        )
    contrasts: list[dict[str, object]] = []
    comparisons = [
        (f"{best_point} (calibrated)", "production_cleaned (calibrated)"),
        (best_quantile, "production_cleaned (calibrated)"),
        (best_quantile, f"{best_point} (calibrated)"),
        ("e1_quantile", "production_cleaned (calibrated)"),
    ]
    for arm_label, reference_label in comparisons:
        if arm_label not in models or reference_label not in models:
            continue
        contrasts += _paired_rows(
            arm=models[arm_label],
            reference=models[reference_label],
            label=f"{arm_label} minus {reference_label}",
        )
    lines += [
        "## Calibrated point arms against quantile arms, same runs (fold A, seed 0)",
        "",
        (
            f"All rows are the {n_runs} runs after the calibration warm-up, on the same series "
            "and valid times. Calibrated arms use only outcomes known at each run's "
            "initialisation time. Calibration leaves the MAE unchanged. Best calibrated point "
            f"arm: `{best_point}`. Best quantile arm: `{best_quantile}`."
        ),
        "",
        markdown_table(frame=pl.DataFrame(rows)),
        "",
        (
            "Paired differences (first minus second; negative favours the first). Interval: 95% "
            "month-resampled."
        ),
        "",
        markdown_table(frame=pl.DataFrame(contrasts)),
        "",
    ]


def _cleaning(*, lines: list[str]) -> None:
    contrasts = _read(fold=inputs.FOLD_A, kind="contrasts", tag="seeds0")
    if contrasts is None:
        return
    cleaning_arms = [a.name for a in arm_definitions.CLEANING_ARMS]
    frame = contrasts.filter(
        pl.col("arm").is_in(cleaning_arms)
        & (pl.col("reference") == "production")
        & (pl.col("group") == "all types")
        & (pl.col("lead_band") == "all")
    ).with_columns(
        interval=pl.format("[{}, {}]", pl.col("lower_95").round(4), pl.col("upper_95").round(4)),
        difference=pl.col("difference").round(4),
        months=pl.format("{} of {}", pl.col("months_better"), pl.col("n_months")),
    )
    wide = frame.pivot(on="metric", index="arm", values=["difference", "interval", "months"])
    lines += [
        "## Round 0: cleaning arms against `production`, fold A, seed 0, all types",
        "",
        markdown_table(frame=wide.sort("arm")),
        "",
    ]
    settings = Settings()
    series_ids = inputs.eligible_series_ids(settings=settings, fold_id=inputs.FOLD_A)
    types = pl.DataFrame(
        inputs.series_metadata(settings=settings, time_series_ids=series_ids)
    ).select("time_series_id", pl.col("time_series_type").cast(pl.String))
    rows = []
    for path in sorted((inputs.OUTPUT_ROOT / "cleaning" / inputs.FOLD_A).glob("*.parquet")):
        dropped = pl.read_parquet(path).join(types, on="time_series_id")
        counts = dropped.group_by("time_series_type").agg(rows=pl.len())
        rows.append(
            counts.with_columns(rule=pl.lit(path.stem)).select("rule", "time_series_type", "rows")
        )
    if rows:
        lines += [
            "### Training half-hours each rule drops, by series type (fold A)",
            "",
            markdown_table(frame=pl.concat(rows).sort("rule", "time_series_type")),
            "",
        ]


def _lead_band_baselines(*, lines: list[str]) -> None:
    """Print each model's normalised MAE and CRPS by series type and lead band (seed 0, fold A)."""
    settings = Settings()
    capacity = pl.DataFrame(inputs.effective_capacity(settings=settings))
    series_ids = inputs.eligible_series_ids(settings=settings, fold_id=inputs.FOLD_A)
    types = pl.DataFrame(
        inputs.series_metadata(settings=settings, time_series_ids=series_ids)
    ).select("time_series_id", pl.col("time_series_type").cast(pl.String))
    sources = {
        "manual_heuristic": BASELINE_DIR / BASELINES["manual_heuristic"],
        "climatology": BASELINE_DIR / BASELINES["climatology"],
    }
    models: dict[str, pl.DataFrame] = {
        name: normalised_by_capacity(scores=pl.read_parquet(path), capacity=capacity)
        for name, path in sources.items()
    }
    for arm in ("production_cleaned", "r2_c4_c2_c1_a3w_d3", "r2_e1_c1_a3w_d3"):
        path = inputs.OUTPUT_ROOT / "arms" / inputs.FOLD_A / f"{arm}_seed0" / "scores.parquet"
        if path.exists():
            models[arm] = pl.read_parquet(path)
    bands = {
        "days 0 to 2": pl.col("lead_hours") < 72,
        "days 3 to 10": (pl.col("lead_hours") >= 72) & (pl.col("lead_hours") < 264),
        "days 11 and later": pl.col("lead_hours") >= 264,
    }
    rows = []
    for model, scores in models.items():
        typed = scores.join(types, on="time_series_id")
        for band, condition in bands.items():
            in_band = typed.filter(condition)
            for group in ("all types", "PV", "Wind", "Disaggregated Demand", "Raw Flow"):
                part = (
                    in_band
                    if group == "all types"
                    else in_band.filter(pl.col("time_series_type") == group)
                )
                per_series = part.group_by("time_series_id").agg(
                    mae=pl.col("abs_error_norm").mean(), crps=pl.col("crps_norm").mean()
                )
                rows.append(
                    per_series.select(
                        pl.lit(group).alias("group"),
                        pl.lit(band).alias("lead_band"),
                        pl.lit(model).alias("model"),
                        (pl.col("mae").mean() * 100).alias("mae_percent"),
                        (pl.col("crps").mean() * 100).alias("crps_percent"),
                    )
                )
    table = pl.concat(rows).with_columns(pl.col("mae_percent", "crps_percent").round(2))
    lines += [
        "## Normalised MAE and CRPS by lead band: where XGBoost beats the heuristic (fold A)",
        "",
        "Percent of capacity, mean over series; baselines are single fits on the same runs.",
        "",
        markdown_table(frame=table.sort("group", "lead_band", "model")),
        "",
    ]


def _blend(*, lines: list[str]) -> None:
    for path in sorted((inputs.OUTPUT_ROOT / "blend" / inputs.FOLD_A).glob("blend_*.parquet")):
        frame = pl.read_parquet(path)
        lines += [
            f"## Blend of XGBoost members with the heuristic's members (post hoc): {path.stem}",
            "",
            markdown_table(
                frame=frame.with_columns(
                    pl.col(c).round(4) for c in frame.columns if frame[c].dtype == pl.Float64
                )
            ),
            "",
        ]


def _columns(*, lines: list[str]) -> None:
    lines += ["## Feature columns of every arm", ""]
    for name, arm in arm_definitions.ALL_ARMS.items():
        shown = {
            series_type: arm.columns(series_type)
            for series_type in ("PV", "Wind", "Disaggregated Demand", "Raw Flow")
        }
        extra = " (plus `site_id`)" if arm.pooled_site_id else ""
        lines.append(
            f"- `{name}`{extra}: "
            + "; ".join(f"{t} {len(cols)} columns" for t, cols in shown.items())
            + f". Cleaning: {arm.cleaning}"
            + (f" + {', '.join(arm.extra_cleaning)}" if arm.extra_cleaning else "")
            + f". Objective: {arm.objective}. colsample_bytree {arm.colsample_bytree}."
        )
    base = set(arm_definitions.PRODUCTION_COLUMNS)
    lines += ["", "Columns by arm that differ from production's 24:", ""]
    for name, arm in arm_definitions.ALL_ARMS.items():
        for series_type in ("PV", "Wind", "Disaggregated Demand"):
            cols = set(arm.columns(series_type))
            if cols != base:
                lines.append(
                    f"- `{name}`, {series_type}: removes {sorted(base - cols)}, adds "
                    f"{sorted(cols - base)}"
                )
    lines.append("")


def main() -> None:
    """Write `summary.md` under the report directory."""
    lines = ["# Quick-wins study: tables quoted on the page", ""]
    _leaderboard(lines=lines)
    lines += ["## Paired contrasts, fold A, three seeds, against `production_cleaned`", ""]
    _contrast_tables(
        lines=lines, fold=inputs.FOLD_A, tag="seeds012", reference="production_cleaned"
    )
    lines += ["## Against `production` (as the study's control without cleaning)", ""]
    _contrast_tables(lines=lines, fold=inputs.FOLD_A, tag="seeds012", reference="production")
    lines += ["## Fold A, seed 0 only (arms fitted on one seed), against `production_cleaned`", ""]
    _contrast_tables(lines=lines, fold=inputs.FOLD_A, tag="seeds0", reference="production_cleaned")
    _seed_spread(lines=lines)
    lines += ["## Fold B (robustness), against `production_cleaned`", ""]
    for tag in ("seeds01", "seeds0"):
        _contrast_tables(lines=lines, fold=inputs.FOLD_B, tag=tag, reference="production_cleaned")
    _calibration(lines=lines)
    _calibrated_against_quantile(lines=lines)
    _g1_against_quantile(lines=lines)
    _lead_band_baselines(lines=lines)
    _blend(lines=lines)
    _cleaning(lines=lines)
    _columns(lines=lines)
    path = REPORT_DIR / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
