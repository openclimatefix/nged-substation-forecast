"""Tabulate every fitted arm: its error with an interval, and its paired difference from references.

Reads each arm's per-row `scores.parquet` (written by `fit_arms.py`), and writes `report.md` plus
parquet tables to `data/studies/per_study/quick_wins/report/<fold>/`. Two metrics are reported:
the continuous ranked probability score (CRPS) and the mean absolute error (MAE) of the ensemble
mean, each as a fraction of the series' effective capacity (so the MAE is the NMAE) and in MW.
Intervals resample whole validation months, paired across arms.

    uv run python studies/quick_wins/report_quick_wins.py [--fold ...] [--seeds 0 1 2]
"""

import argparse
from pathlib import Path
from typing import Final

import arms as arm_definitions
import fit_arms
import month_bootstrap
import polars as pl
import quick_wins_inputs as inputs
from contracts.settings import Settings
from per_run_scores import LEAD_BAND_ORDER
from report_tables import markdown_table

METRICS: Final[dict[str, str]] = {"crps": "crps_norm", "mae": "abs_error_norm"}
"""The two reported metrics, by name, and the per-row column each is read from."""

REFERENCES: Final[tuple[str, ...]] = ("production_cleaned", "production", "production_colsample0.8")
"""The arms every other arm is contrasted with: production at colsample 1.0, and as stored."""


def _filtered(
    *, scores: pl.DataFrame, series_type: str | None, band: str, types: pl.DataFrame
) -> pl.DataFrame:
    """Keep the rows of one series type (or all) within one lead band."""
    rows = scores.join(types, on="time_series_id")
    if series_type is not None:
        rows = rows.filter(pl.col("time_series_type") == series_type)
    if band == "days 0 to 2":
        rows = rows.filter(pl.col("lead_hours") < 72)
    elif band == "days 3 to 10":
        rows = rows.filter((pl.col("lead_hours") >= 72) & (pl.col("lead_hours") < 264))
    return rows


def _arm_scores(*, fold_id: str, seeds: list[int]) -> dict[str, pl.DataFrame]:
    """Read each arm's scores, averaging the per-row scores over seeds.

    An arm is kept only if every seed in `seeds` has scored it. The scores, not the forecasts, are
    averaged, so each seed's own spread stays in the number.
    """
    found = {}
    for name, arm in arm_definitions.ALL_ARMS.items():
        paths = [
            fit_arms.arm_dir(fold_id=fold_id, arm=arm, seed=seed) / "scores.parquet"
            for seed in seeds
        ]
        if not all(path.exists() for path in paths):
            continue
        frames = [pl.read_parquet(path) for path in paths]
        if len(frames) == 1:
            found[name] = frames[0]
            continue
        keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
        found[name] = (
            pl.concat(frames)
            .group_by(keys)
            .agg(
                pl.col("lead_hours").first(),
                pl.col("effective_capacity_mw").first(),
                pl.col("horizon_slice").first(),
                pl.col("crps", "abs_error", "crps_norm", "abs_error_norm").mean(),
                n_seeds=pl.len(),
            )
            .filter(pl.col("n_seeds") == len(frames))
            .drop("n_seeds")
        )
    return found


def main() -> None:
    """Write the level and contrast tables for every fitted arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", default=inputs.FOLD_A)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()

    settings = Settings()
    series_ids = inputs.eligible_series_ids(settings=settings, fold_id=inputs.FOLD_A)
    types = pl.DataFrame(
        inputs.series_metadata(settings=settings, time_series_ids=series_ids)
    ).select("time_series_id", pl.col("time_series_type").cast(pl.String))
    scores = _arm_scores(fold_id=args.fold, seeds=args.seeds)
    tag = "seeds" + "".join(str(seed) for seed in args.seeds)
    if not scores:
        msg = "no arm has scores yet"
        raise SystemExit(msg)
    out_dir = inputs.OUTPUT_ROOT / "report" / args.fold
    out_dir.mkdir(parents=True, exist_ok=True)

    level_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    groups: list[tuple[str, str | None]] = [("all types", None)] + [
        (t, t) for t in sorted(types["time_series_type"].unique().to_list())
    ]
    for label, series_type in groups:
        for band in LEAD_BAND_ORDER:
            for metric, column in METRICS.items():
                tables = {
                    name: month_bootstrap.monthly_table(
                        scores=_filtered(
                            scores=frame, series_type=series_type, band=band, types=types
                        ),
                        metric=column,
                    )
                    for name, frame in scores.items()
                }
                for name, table in tables.items():
                    level = month_bootstrap.level_interval(table=table)
                    level_rows.append(
                        {
                            "arm": name,
                            "group": label,
                            "lead_band": band,
                            "metric": metric,
                            "value": level.estimate,
                            "lower_95": level.lower_95,
                            "upper_95": level.upper_95,
                        }
                    )
                    for reference in REFERENCES:
                        if reference == name or reference not in tables:
                            continue
                        diff = month_bootstrap.difference_interval(
                            arm=table, reference=tables[reference]
                        )
                        contrast_rows.append(
                            {
                                "arm": name,
                                "reference": reference,
                                "group": label,
                                "lead_band": band,
                                "metric": metric,
                                "difference": diff.estimate,
                                "lower_95": diff.lower_95,
                                "upper_95": diff.upper_95,
                                "t_lower_95": diff.t_lower_95,
                                "t_upper_95": diff.t_upper_95,
                                "months_better": diff.months_better,
                                "n_months": diff.n_months,
                            }
                        )
    levels = pl.DataFrame(level_rows)
    contrasts = pl.DataFrame(contrast_rows)
    levels.write_parquet(out_dir / f"levels_{tag}.parquet")
    contrasts.write_parquet(out_dir / f"contrasts_{tag}.parquet")
    _write_report(levels=levels, contrasts=contrasts, path=out_dir / f"report_{tag}.md")


def _write_report(*, levels: pl.DataFrame, contrasts: pl.DataFrame, path: Path) -> None:
    lines = ["# Quick-wins arms", ""]
    for metric in METRICS:
        lines += [
            f"## {metric.upper()}, fraction of capacity, all types, all leads (smaller is better)",
            "",
            markdown_table(
                frame=levels.filter(
                    (pl.col("metric") == metric)
                    & (pl.col("group") == "all types")
                    & (pl.col("lead_band") == "all")
                ).sort("value")
            ),
            "",
        ]
        for reference in REFERENCES:
            lines += [
                (
                    f"## {metric.upper()}: difference from `{reference}`, all types, all leads "
                    "(negative favours the arm)"
                ),
                "",
                markdown_table(
                    frame=contrasts.filter(
                        (pl.col("metric") == metric)
                        & (pl.col("reference") == reference)
                        & (pl.col("group") == "all types")
                        & (pl.col("lead_band") == "all")
                    ).sort("difference")
                ),
                "",
            ]
    path.write_text("\n".join(lines))
    print(path.read_text())


if __name__ == "__main__":
    main()
