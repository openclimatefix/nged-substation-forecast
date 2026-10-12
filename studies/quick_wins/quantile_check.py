"""Check the quantile arm's predictive distribution: its coverage, and what crossing adds to CRPS.

The `e1_quantile` arm forecasts nine quantile levels under each of 51 ensemble members, a sample of
459 values per row. This script reads that saved sample (written by `fit_arms.py
--save-quantile-samples`) and reports two things.

1. The share of outcomes inside the sample's 10% to 90% band and its 5% to 95% band, by series type
   and lead band. A calibrated forecast holds 80% and 90%.
2. The fair CRPS of the nine control-member quantiles alone (the control member is ensemble member 0
   and takes codes 0 to 8) against the fair CRPS of the full 459-value sample, on the same rows.
   The first is a quantile forecast of the control weather only. The second also crosses in the 50
   perturbed members, so the gap is what the ensemble's weather spread adds.

    uv run python studies/quick_wins/quantile_check.py [--arm e1_quantile] [--fold ...] [--seed 0]
"""

import argparse
from typing import Final

import arms as arm_definitions
import fit_arms
import month_bootstrap
import polars as pl
import quick_wins_inputs as inputs
from contracts.settings import Settings
from lead_band_contrasts import LEAD_BANDS_4
from per_run_scores import SCORE_KEYS, per_run_scores
from report_tables import markdown_table
from studies.power import scan_power

N_LEVELS: Final[int] = len(arm_definitions.QUANTILE_LEVELS)
"""Quantile levels per ensemble member; the control member's codes are `0` to `N_LEVELS - 1`."""

COVERAGE_BANDS: Final[dict[str, tuple[float, float]]] = {
    "10-90": (0.1, 0.9),
    "5-95": (0.05, 0.95),
}
"""The central bands whose coverage is reported, by label, as lower and upper quantile levels."""


def _run_rows(*, forecasts: pl.DataFrame, actuals: pl.DataFrame) -> pl.DataFrame:
    """Score one run's sample, its control-member quantiles alone, and the band coverage."""
    full = per_run_scores(forecasts=forecasts, actuals=actuals).select(
        *SCORE_KEYS, "lead_hours", "actual", crps_sample="crps"
    )
    control = per_run_scores(
        forecasts=forecasts.filter(pl.col("ensemble_member") < N_LEVELS), actuals=actuals
    ).select(*SCORE_KEYS, crps_control="crps", abs_error_control="abs_error")
    band = forecasts.group_by(*SCORE_KEYS).agg(
        *[
            expr
            for label, (low, high) in COVERAGE_BANDS.items()
            for expr in (
                pl.col("power_fcst").quantile(low).alias(f"low_{label}"),
                pl.col("power_fcst").quantile(high).alias(f"high_{label}"),
            )
        ]
    )
    return (
        full.join(control, on=list(SCORE_KEYS))
        .join(band, on=list(SCORE_KEYS))
        .with_columns(
            [
                (
                    (pl.col("actual") >= pl.col(f"low_{label}"))
                    & (pl.col("actual") <= pl.col(f"high_{label}"))
                ).alias(f"inside_{label}")
                for label in COVERAGE_BANDS
            ]
        )
        .select(
            *SCORE_KEYS,
            "lead_hours",
            "crps_sample",
            "crps_control",
            *[f"inside_{label}" for label in COVERAGE_BANDS],
        )
    )


def main() -> None:
    """Write `report/quantile_check.md`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="e1_quantile")
    parser.add_argument("--fold", default=inputs.FOLD_A)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    settings = Settings()
    window = inputs.window_for(fold_id=args.fold)
    series_ids = inputs.eligible_series_ids(settings=settings, fold_id=inputs.FOLD_A)
    types = pl.DataFrame(
        inputs.series_metadata(settings=settings, time_series_ids=series_ids)
    ).select("time_series_id", pl.col("time_series_type").cast(pl.String))
    capacity = pl.DataFrame(inputs.effective_capacity(settings=settings))
    actuals = (
        scan_power()
        .filter(pl.col("time") >= window.val_start, pl.col("time") <= window.val_end)
        .collect()
    )
    arm = arm_definitions.ALL_ARMS[args.arm]
    out_dir = fit_arms.arm_dir(fold_id=args.fold, arm=arm, seed=args.seed)
    files = sorted((out_dir / "forecasts").glob("init=*.parquet"))
    if not files:
        msg = f"{args.arm} has no saved samples; run fit_arms.py with --save-quantile-samples"
        raise SystemExit(msg)
    rows = pl.concat(
        _run_rows(forecasts=pl.read_parquet(path), actuals=actuals) for path in files
    ).join(types, on="time_series_id")
    scale = capacity.select("time_series_id", "effective_capacity_mw")
    rows = rows.join(scale, on="time_series_id").with_columns(
        crps_sample_norm=pl.col("crps_sample") / pl.col("effective_capacity_mw"),
        crps_control_norm=pl.col("crps_control") / pl.col("effective_capacity_mw"),
    )

    stored = pl.read_parquet(out_dir / "scores.parquet")
    check = rows.join(stored.select(*SCORE_KEYS, "crps_norm"), on=list(SCORE_KEYS))
    worst = float((check["crps_sample_norm"] - check["crps_norm"]).abs().to_numpy().max())

    groups: list[str | None] = [None, *sorted(types["time_series_type"].unique().to_list())]
    coverage_rows, crps_rows = [], []
    for series_type in groups:
        part = (
            rows if series_type is None else rows.filter(pl.col("time_series_type") == series_type)
        )
        label = series_type or "all types"
        for band, condition in LEAD_BANDS_4.items():
            in_band = part.filter(condition)
            per_series = in_band.group_by("time_series_id").agg(
                pl.col(f"inside_{b}").mean() for b in COVERAGE_BANDS
            )
            coverage_rows.append(
                {
                    "group": label,
                    "lead_band": band,
                    **{
                        f"coverage_{b}": float(per_series[f"inside_{b}"].to_numpy().mean())
                        for b in COVERAGE_BANDS
                    },
                }
            )
            sample = month_bootstrap.monthly_table(scores=in_band, metric="crps_sample_norm")
            control = month_bootstrap.monthly_table(scores=in_band, metric="crps_control_norm")
            diff = month_bootstrap.difference_interval(arm=control, reference=sample)
            crps_rows.append(
                {
                    "group": label,
                    "lead_band": band,
                    "crps_459_sample": month_bootstrap.level_interval(table=sample).estimate,
                    "crps_9_control_quantiles": month_bootstrap.level_interval(
                        table=control
                    ).estimate,
                    "control_minus_sample": diff.estimate,
                    "lower_95": diff.lower_95,
                    "upper_95": diff.upper_95,
                    "months_control_better": diff.months_better,
                    "n_months": diff.n_months,
                }
            )
    lines = [
        f"# Quantile arm check: `{args.arm}`, fold {args.fold}, seed {args.seed}",
        "",
        (
            "The sample is the 51 ensemble members each crossed with 9 quantile levels, 459 values "
            f"per row. Rows scored: {rows.height:,}. The CRPS recomputed from the saved sample "
            f"matches the stored scores to within {worst:.2e}."
        ),
        "",
        "## Coverage of the 459-value sample",
        "",
        (
            "Share of outcomes inside the band, mean over series. Nominal: 0.80 for 10-90 and 0.90 "
            "for 5-95."
        ),
        "",
        markdown_table(frame=pl.DataFrame(coverage_rows).sort("group", "lead_band")),
        "",
        "## CRPS of the 9 control-member quantiles against the 459-value sample",
        "",
        (
            "Fair CRPS as a fraction of capacity, on the same rows. The difference is the control "
            "quantiles' CRPS minus the sample's, so a negative difference means the control "
            "quantiles alone score better than the crossed sample. Interval: 95% month-resampled."
        ),
        "",
        markdown_table(frame=pl.DataFrame(crps_rows).sort("group", "lead_band")),
        "",
    ]
    path = inputs.OUTPUT_ROOT / "report" / "quantile_check.md"
    path.write_text("\n".join(lines) + "\n")
    print(path.read_text())


if __name__ == "__main__":
    main()
