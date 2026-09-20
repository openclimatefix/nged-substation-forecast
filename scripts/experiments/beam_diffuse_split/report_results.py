"""Turn the experiment's parquet output into the markdown tables the write-up quotes.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>.

**Every number that reaches a human is printed by this script rather than read off a frame by
hand.** Transcription is where a result quietly acquires a digit it never had, and where an
inconvenient row goes missing.

No site identifier can reach the output: `build_dataset.py` relabelled the sites before writing
anything, so the results files hold only the shuffled letters.

Run it with `uv run --no-project --with polars python
scripts/experiments/beam_diffuse_split/report_results.py`.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Final, Literal

import polars as pl

REPO_DATA_DIR: Final[Path] = Path("/home/jack/dev/nged-substation-forecast/data")

InstrumentType = Literal["xgboost", "physics"]
"""Which of the two instruments' results to report.

`xgboost` is `run_experiment.py`'s gradient-boosted tree and `physics` is
`run_physics_experiment.py`'s fitted five-parameter PV model. The two write the same files with the
same columns, and differ only in their arm names and in which extra sections they have to report.
"""

SHORT_ARM_LABELS: Final[dict[str, str]] = {
    "A_global_only": "A",
    "B_erbs": "B",
    "C_era5_split": "C",
    "D_direct_fraction": "D",
    "B_disc": "B-DISC",
    "B_learned": "B-LEARNED",
    "P_A_global_only": "P-A",
    "P_B_erbs": "P-B",
    "P_B_disc": "P-B-DISC",
    "P_C_source_split": "P-C",
    "P_E_blended": "P-E",
}
"""Arm keys to the short label a contrast row uses."""

ARM_LABELS: Final[dict[str, str]] = {
    "A_global_only": "A — global irradiance only",
    "B_erbs": "B — Erbs separation model",
    "C_era5_split": "C — the source's own split",
    "D_direct_fraction": "D — the source's direct fraction",
    "B_disc": "B-DISC — DISC separation model",
    "B_learned": "B-LEARNED — the best separation model derivable from arm A's features",
    "P_A_global_only": "P-A — global irradiance only, no transposition",
    "P_B_erbs": "P-B — transposed with the Erbs split",
    "P_B_disc": "P-B-DISC — transposed with the DISC split",
    "P_C_source_split": "P-C — transposed with the source's own split",
    "P_E_blended": "P-E — transposed with all three splits, weights fitted",
}

HEADLINE_CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "xgboost": ("C_era5_split", "B_erbs"),
    "physics": ("P_C_source_split", "P_B_erbs"),
}
"""The contrast each instrument's per-site table is drawn for."""

PERCENTAGE_POINTS: Final[float] = 100.0


def _pooled_mean(*, summary: pl.DataFrame, setting: str, arm: str, column: str) -> float:
    """Return one arm's row-weighted mean of `column` across every site.

    Args:
        summary: The per-site summary.
        setting: Which hyperparameter setting to read.
        arm: Which arm to read.
        column: The metric column.

    Returns:
        The pooled figure.
    """
    rows = summary.filter((pl.col("setting") == setting) & (pl.col("arm") == arm))
    return float((rows[column] * rows["n_rows"]).sum()) / float(rows["n_rows"].sum())


def _arm_table(*, summary: pl.DataFrame, setting: str) -> list[str]:
    """Render the per-arm pooled metrics as a markdown table."""
    lines = [
        "| Arm | MAE (% of P99 output) | MAE (MW) | CRPS (MW) |",
        "|---|---|---|---|",
    ]
    arms = [
        arm
        for arm in ARM_LABELS
        if not summary.filter((pl.col("setting") == setting) & (pl.col("arm") == arm)).is_empty()
    ]
    for arm in arms:
        normalised = _pooled_mean(
            summary=summary, setting=setting, arm=arm, column="mae_fraction_of_capacity"
        )
        absolute = _pooled_mean(summary=summary, setting=setting, arm=arm, column="mae_mw")
        crps_rows = summary.filter((pl.col("setting") == setting) & (pl.col("arm") == arm))
        scores_probabilistically = (
            "crps_mw" in summary.columns and crps_rows["crps_mw"].null_count() == 0
        )
        crps = (
            f"{_pooled_mean(summary=summary, setting=setting, arm=arm, column='crps_mw'):.4f}"
            if scores_probabilistically
            else "—"
        )
        lines.append(
            f"| {ARM_LABELS[arm]} | {normalised * PERCENTAGE_POINTS:.3f}"
            f" | {absolute:.4f} | {crps} |"
        )
    return lines


def _contrast_table(*, intervals: pl.DataFrame, summary: pl.DataFrame, setting: str) -> list[str]:
    """Render the pooled arm-to-arm contrasts, in percentage points of P99 output."""
    lines = [
        (
            "| Contrast | ΔMAE (pp of P99 output) | 95% interval | Relative | Excludes zero? |"
            " Folds with the same sign |"
        ),
        "|---|---|---|---|---|---|",
    ]
    pooled = intervals.filter(
        (pl.col("setting") == setting)
        & (pl.col("scope") == "all_sites")
        & (pl.col("metric") == "absolute_error_fraction_of_capacity")
    )
    for row in pooled.iter_rows(named=True):
        reference_mae = _pooled_mean(
            summary=summary,
            setting=setting,
            arm=row["reference"],
            column="mae_fraction_of_capacity",
        )
        relative = row["difference"] / reference_mae * PERCENTAGE_POINTS
        excludes_zero = "**yes**" if row["lower_95"] * row["upper_95"] > 0 else "no"
        folds = row["per_fold_differences"]
        agreeing = (
            f"{sum(1 for value in folds if value * row['difference'] > 0)} of {len(folds)}"
            if len(folds) > 0
            else "—"
        )
        marker = " **(headline)**" if row["is_headline"] else ""
        lines.append(
            f"| {SHORT_ARM_LABELS[row['treatment']]} − {SHORT_ARM_LABELS[row['reference']]}"
            f"{marker} | "
            f"{row['difference'] * PERCENTAGE_POINTS:+.4f} | "
            f"[{row['lower_95'] * PERCENTAGE_POINTS:+.4f}, "
            f"{row['upper_95'] * PERCENTAGE_POINTS:+.4f}] | "
            f"{relative:+.2f}% | {excludes_zero} | {agreeing} |"
        )
    return lines


def _crps_contrast_table(*, intervals: pl.DataFrame, setting: str) -> list[str]:
    """Render the same contrasts scored by CRPS rather than by mean absolute error.

    The continuous ranked probability score is the run's only probabilistic evidence, and an arm
    that sharpened its central estimate while widening its distribution would show up here and
    nowhere else.

    Args:
        intervals: The bootstrap intervals frame.
        setting: Which hyperparameter setting to report.

    Returns:
        The table's lines, or an empty list where the instrument scores no distribution.
    """
    pooled = intervals.filter(
        (pl.col("setting") == setting)
        & (pl.col("scope") == "all_sites")
        & (pl.col("metric") == "crps_mw")
    )
    if pooled.is_empty():
        return []
    lines = [
        "| Contrast | ΔCRPS (MW) | 95% interval | Excludes zero? |",
        "|---|---|---|---|",
    ]
    for row in pooled.iter_rows(named=True):
        excludes_zero = "**yes**" if row["lower_95"] * row["upper_95"] > 0 else "no"
        marker = " **(headline)**" if row["is_headline"] else ""
        lines.append(
            f"| {SHORT_ARM_LABELS[row['treatment']]} − {SHORT_ARM_LABELS[row['reference']]}"
            f"{marker} | {row['difference']:+.6f} | "
            f"[{row['lower_95']:+.6f}, {row['upper_95']:+.6f}] | {excludes_zero} |"
        )
    return lines


def _per_site_table(
    *, intervals: pl.DataFrame, setting: str, contrast: tuple[str, str]
) -> list[str]:
    """Render one contrast's per-site intervals."""
    treatment, reference = contrast
    rows = intervals.filter(
        (pl.col("setting") == setting)
        & (pl.col("metric") == "absolute_error_fraction_of_capacity")
        & (pl.col("treatment") == treatment)
        & (pl.col("reference") == reference)
        & (pl.col("scope") != "all_sites")
    ).sort("scope")
    lines = [
        f"{SHORT_ARM_LABELS[treatment]} − {SHORT_ARM_LABELS[reference]}, per site:",
        "",
        "| Site | ΔMAE (pp of P99 output) | 95% interval | Hours |",
        "|---|---|---|---|",
    ]
    lines.extend(
        f"| {row['scope']} | {row['difference'] * PERCENTAGE_POINTS:+.4f} | "
        f"[{row['lower_95'] * PERCENTAGE_POINTS:+.4f}, "
        f"{row['upper_95'] * PERCENTAGE_POINTS:+.4f}] | "
        f"{row['n_rows']:,} |"
        for row in rows.iter_rows(named=True)
    )
    return lines


def _fitted_parameter_table(*, results_dir: Path) -> list[str]:
    """Render what geometry the physical model settles on when fitted to each site's whole span."""
    path = results_dir / "fitted_parameters.parquet"
    if not path.exists():
        return []
    rows = pl.read_parquet(path).filter(pl.col("arm") == "P_C_source_split").sort("site")
    lines = [
        "Fitted on each site's whole span, for the arm given the source's own split:",
        "",
        "| Site | Tilt (degrees) | Azimuth (degrees) | Capacity / P99 output | Clip / P99 output |",
        "|---|---|---|---|---|",
    ]
    for row in rows.iter_rows(named=True):
        clip = row["clip_fraction_of_p99"]
        clip_cell = "—" if clip is None else f"{clip:.3f}"
        lines.append(
            f"| {row['site']} | {row['tilt_degrees']:.1f} | {row['azimuth_degrees']:.1f} | "
            f"{row['capacity_fraction_of_p99']:.3f} | {clip_cell} |"
        )
    lines.extend(["", "An em dash means the clip never binds, so the data does not identify it."])
    return lines


def _blend_weight_table(*, results_dir: Path) -> list[str]:
    """Render the convex weights the all-three-splits arm settles on."""
    path = results_dir / "fitted_parameters.parquet"
    if not path.exists():
        return []
    rows = pl.read_parquet(path).filter(pl.col("arm") == "P_E_blended").sort("site")
    lines = [
        "Weights arm P-E puts on each beam estimate, fitted on each site's whole span:",
        "",
        "| Site | The source's own split | Erbs | DISC |",
        "|---|---|---|---|",
    ]
    lines.extend(
        f"| {row['site']} | {row['weights'][0]:.3f} | {row['weights'][1]:.3f} | "
        f"{row['weights'][2]:.3f} |"
        for row in rows.iter_rows(named=True)
    )
    return lines


def main() -> int:
    """Write every table to `report.md` and to standard output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("cds", "open-meteo", "cams"), default="open-meteo")
    parser.add_argument("--alignment", choices=("as-labelled", "shifted"), default="as-labelled")
    parser.add_argument("--instrument", choices=("xgboost", "physics"), default="xgboost")
    arguments = parser.parse_args()
    instrument: InstrumentType = arguments.instrument
    stem = "results" if instrument == "xgboost" else "physics"
    results_dir = (
        REPO_DATA_DIR / "ERA5" / f"beam_diffuse_{stem}_{arguments.source}_{arguments.alignment}"
    )

    intervals = pl.read_parquet(results_dir / "bootstrap_intervals.parquet")
    summary = pl.read_parquet(results_dir / "per_site_summary.parquet")

    lines: list[str] = [
        (
            "**P99 output** is each site's 99th-percentile absolute metered output over its whole"
            " history, which every percentage below is a percentage of. It is not the site's"
            " registered capacity, which this experiment never reads."
        ),
        "",
        "## Arms, primary setting",
        "",
    ]
    lines += _arm_table(summary=summary, setting="primary")
    lines += ["", "## Contrasts, primary setting", ""]
    lines += _contrast_table(intervals=intervals, summary=summary, setting="primary")
    crps_lines = _crps_contrast_table(intervals=intervals, setting="primary")
    if crps_lines:
        lines += ["", "## Contrasts by CRPS, primary setting", ""]
        lines += crps_lines
    if instrument == "xgboost":
        lines += ["", "## Contrasts, sensitivity hyperparameters", ""]
        lines += _contrast_table(intervals=intervals, summary=summary, setting="sensitivity")
    lines += ["", "## Positive control: synthetic transposed-plane target", ""]
    lines += _arm_table(summary=summary, setting="positive_control")
    lines += [""]
    lines += _contrast_table(intervals=intervals, summary=summary, setting="positive_control")
    lines += ["", "## Per-site, headline contrast", ""]
    lines += _per_site_table(
        intervals=intervals, setting="primary", contrast=HEADLINE_CONTRASTS[instrument]
    )
    if instrument == "physics":
        lines += ["", "## What the fit settles on", ""]
        lines += _fitted_parameter_table(results_dir=results_dir)
        lines += [""]
        lines += _blend_weight_table(results_dir=results_dir)
    else:
        diagnostic = json.loads((results_dir / "diagnostic.json").read_text())
        lines += ["", "## Predictability diagnostic", ""]
        lines += [
            "Out-of-fold, predicting the source's direct fraction from arm A's own feature set:",
            "",
            (
                "- unexplained variance fraction: "
                f"**{diagnostic['unexplained_variance_fraction']:.3f}**"
            ),
            f"- residual standard deviation: {diagnostic['residual_standard_deviation']:.4f}",
            (
                "- direct fraction standard deviation: "
                f"{diagnostic['direct_fraction_standard_deviation']:.4f}"
            ),
            f"- rows: {diagnostic['n_rows']:,}",
        ]

    report = "\n".join(lines) + "\n"
    (results_dir / "report.md").write_text(report)
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
