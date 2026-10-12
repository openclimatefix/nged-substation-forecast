"""Paired contrasts of named arms against one reference, by series type and four lead bands.

`report_quick_wins.py` splits leads into two bands. This script splits them into four, under 24
hours, 24 to 48, 48 to 72, and 72 and over, for a chosen list of arms. Scores are averaged over
the fitting seeds that exist for every named arm, and intervals resample whole validation months,
paired between each arm and the reference.

    uv run python studies/quick_wins/lead_band_contrasts.py --reference production_cleaned \
        --arms a1b_lags_removed_only a1_generators_without_lags --seeds 0 1 2 --name lag_only
"""

import argparse
from typing import Final

import month_bootstrap
import polars as pl
import quick_wins_inputs as inputs
from contracts.settings import Settings
from report_quick_wins import METRICS, _arm_scores
from report_tables import markdown_table

LEAD_BANDS_4: Final[dict[str, pl.Expr]] = {
    "all": pl.lit(value=True),
    "under 24 h": pl.col("lead_hours") < 24,
    "24 to 48 h": (pl.col("lead_hours") >= 24) & (pl.col("lead_hours") < 48),
    "48 to 72 h": (pl.col("lead_hours") >= 48) & (pl.col("lead_hours") < 72),
    "72 h and over": pl.col("lead_hours") >= 72,
}
"""The four lead bands, plus `all`, as filters on the `lead_hours` column."""


def contrast_table(
    *, scores: dict[str, pl.DataFrame], reference: str, types: pl.DataFrame
) -> pl.DataFrame:
    """Contrast every arm in `scores` (except `reference`) with `reference`.

    Args:
        scores: Per-row scores by arm name, including the reference. All arms must hold the same
            rows.
        reference: The name of the arm every other arm is subtracted from.
        types: Columns `time_series_id` and `time_series_type`.

    Returns:
        One row per arm, series type (and `all types`), lead band, and metric, with the
        difference (arm minus reference), its 95% month-resampled interval, and the months better.
    """
    typed = {name: frame.join(types, on="time_series_id") for name, frame in scores.items()}
    groups: list[str | None] = [None, *sorted(types["time_series_type"].unique().to_list())]
    rows: list[dict[str, object]] = []
    for series_type in groups:
        for band, condition in LEAD_BANDS_4.items():
            for metric, column in METRICS.items():
                tables = {}
                for name, frame in typed.items():
                    part = (
                        frame
                        if series_type is None
                        else frame.filter(pl.col("time_series_type") == series_type)
                    )
                    tables[name] = month_bootstrap.monthly_table(
                        scores=part.filter(condition), metric=column
                    )
                level = month_bootstrap.level_interval(table=tables[reference])
                for name, table in tables.items():
                    if name == reference:
                        continue
                    diff = month_bootstrap.difference_interval(
                        arm=table, reference=tables[reference]
                    )
                    rows.append(
                        {
                            "arm": name,
                            "group": series_type or "all types",
                            "lead_band": band,
                            "metric": metric,
                            "reference_value": level.estimate,
                            "difference": diff.estimate,
                            "lower_95": diff.lower_95,
                            "upper_95": diff.upper_95,
                            "months_better": diff.months_better,
                            "n_months": diff.n_months,
                        }
                    )
    return pl.DataFrame(rows)


def main() -> None:
    """Write the contrast table of the named arms to `report/<fold>/lead4_<name>.md`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--fold", default=inputs.FOLD_A)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    settings = Settings()
    series_ids = inputs.eligible_series_ids(settings=settings, fold_id=inputs.FOLD_A)
    types = pl.DataFrame(
        inputs.series_metadata(settings=settings, time_series_ids=series_ids)
    ).select("time_series_id", pl.col("time_series_type").cast(pl.String))
    found = _arm_scores(fold_id=args.fold, seeds=args.seeds)
    wanted = [args.reference, *args.arms]
    missing = [name for name in wanted if name not in found]
    if missing:
        msg = f"no scores for {missing} on seeds {args.seeds}"
        raise SystemExit(msg)
    table = contrast_table(
        scores={name: found[name] for name in wanted}, reference=args.reference, types=types
    )
    out_dir = inputs.OUTPUT_ROOT / "report" / args.fold
    table.write_parquet(out_dir / f"lead4_{args.name}.parquet")
    seeds = ", ".join(str(seed) for seed in args.seeds)
    lines = [
        f"# Contrasts against `{args.reference}` by four lead bands: {args.name}",
        "",
        (
            f"Fold {args.fold}, seeds {seeds} averaged. Difference is arm minus reference in "
            "normalised CRPS or MAE (fraction of capacity); negative favours the arm. Interval: "
            "95% month-resampled."
        ),
        "",
    ]
    for metric in METRICS:
        for arm in args.arms:
            part = table.filter((pl.col("metric") == metric) & (pl.col("arm") == arm)).with_columns(
                interval=pl.format(
                    "[{}, {}]", pl.col("lower_95").round(4), pl.col("upper_95").round(4)
                ),
                difference=pl.col("difference").round(4),
                months=pl.format("{} of {}", pl.col("months_better"), pl.col("n_months")),
            )
            wide = part.pivot(
                on="lead_band", index="group", values=["difference", "interval", "months"]
            ).sort("group")
            lines += [f"## {metric.upper()}: `{arm}`", "", markdown_table(frame=wide), ""]
    path = out_dir / f"lead4_{args.name}.md"
    path.write_text("\n".join(lines) + "\n")
    print(path.read_text())


if __name__ == "__main__":
    main()
