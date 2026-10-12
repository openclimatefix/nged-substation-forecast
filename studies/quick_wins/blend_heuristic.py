"""Diagnostic (post hoc): does adding the manual heuristic's members to an XGBoost ensemble help?

For each sampled run, the 51 XGBoost members are combined with the stored `manual_heuristic`
experiment's 13 members into one equal-weight sample, either as they are (64 members) or with each
heuristic member repeated 4 times (103 members, about half heuristic). The fair CRPS of each sample
is compared with the XGBoost ensemble alone and the heuristic alone, on exactly the rows where the
heuristic has a forecast. The heuristic has no weather input, so a gain from the blend shows that
its analogue observations carry information the XGBoost forecast lacks.

    uv run python studies/quick_wins/blend_heuristic.py --arms production_cleaned [--seed 0]
"""

import argparse
from typing import Final

import arms as arm_definitions
import fit_arms
import month_bootstrap
import polars as pl
import quick_wins_inputs as inputs
from contracts.settings import Settings
from per_run_scores import normalised_by_capacity, per_run_scores
from report_tables import markdown_table
from studies.power import scan_power

HEURISTIC_MEMBER_OFFSET: Final[int] = 1000
"""Added to a heuristic member's number so it cannot collide with an XGBoost member's."""

HEURISTIC_REPEATS: Final[int] = 4
"""How often each heuristic member is repeated in the heavier blend."""


def _heuristic_run(*, init_time: object, fold_id: str, copies: int) -> pl.DataFrame:
    """Read the stored heuristic forecast of one run, with each member repeated `copies` times."""
    settings = Settings()
    stored = (
        pl.scan_delta(settings.power_forecasts_data_path)
        .filter(
            (pl.col("experiment_name") == "manual_heuristic")
            & (pl.col("fold_id") == fold_id)
            & (pl.col("power_fcst_init_time") == init_time)
        )
        .select(
            "time_series_id",
            "power_fcst_init_time",
            "valid_time",
            "ensemble_member",
            "power_fcst",
        )
        .collect(engine="streaming")
    )
    return pl.concat(
        stored.with_columns(
            ensemble_member=pl.col("ensemble_member").cast(pl.Int32)
            + HEURISTIC_MEMBER_OFFSET * (copy + 1)
        )
        for copy in range(copies)
    )


def main() -> None:
    """Score the blends for each named arm and print the contrasts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--fold", default=inputs.FOLD_A)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    settings = Settings()
    window = inputs.window_for(fold_id=args.fold)
    capacity = pl.DataFrame(inputs.effective_capacity(settings=settings))
    actuals = (
        scan_power()
        .filter(pl.col("time") >= window.val_start, pl.col("time") <= window.val_end)
        .collect()
    )
    series_ids = inputs.eligible_series_ids(settings=settings, fold_id=inputs.FOLD_A)
    types = pl.DataFrame(
        inputs.series_metadata(settings=settings, time_series_ids=series_ids)
    ).select("time_series_id", pl.col("time_series_type").cast(pl.String))

    out_dir = inputs.OUTPUT_ROOT / "blend" / args.fold
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Blend of XGBoost members with the manual heuristic's members (post hoc)", ""]
    for name in args.arms:
        arm = arm_definitions.ALL_ARMS[name]
        files = sorted(
            (fit_arms.arm_dir(fold_id=args.fold, arm=arm, seed=args.seed) / "forecasts").glob(
                "init=*.parquet"
            )
        )
        parts: dict[str, list[pl.DataFrame]] = {
            "xgboost": [],
            "heuristic": [],
            "blend_64_members": [],
            "blend_equal_weight": [],
        }
        for path in files:
            xgb_run = pl.read_parquet(path).select(
                "time_series_id",
                "power_fcst_init_time",
                "valid_time",
                "ensemble_member",
                "power_fcst",
            )
            init_time = xgb_run["power_fcst_init_time"][0]
            heuristic_once = _heuristic_run(init_time=init_time, fold_id=args.fold, copies=1)
            heuristic_heavy = _heuristic_run(
                init_time=init_time, fold_id=args.fold, copies=HEURISTIC_REPEATS
            )
            keys = heuristic_once.select("time_series_id", "valid_time").unique()
            xgb_run = xgb_run.join(keys, on=["time_series_id", "valid_time"]).with_columns(
                pl.col("ensemble_member").cast(pl.Int32)
            )
            samples = {
                "xgboost": xgb_run,
                "heuristic": heuristic_once,
                "blend_64_members": pl.concat([xgb_run, heuristic_once]),
                "blend_equal_weight": pl.concat([xgb_run, heuristic_heavy]),
            }
            for label, sample in samples.items():
                parts[label].append(per_run_scores(forecasts=sample, actuals=actuals))
        scores = {
            label: normalised_by_capacity(scores=pl.concat(frames), capacity=capacity)
            for label, frames in parts.items()
        }
        rows = []
        for group in ["all types", *sorted(types["time_series_type"].unique().to_list())]:
            tables = {}
            for label, frame in scores.items():
                rows_of_group = frame.join(types, on="time_series_id")
                if group != "all types":
                    rows_of_group = rows_of_group.filter(pl.col("time_series_type") == group)
                tables[label] = month_bootstrap.monthly_table(
                    scores=rows_of_group, metric="crps_norm"
                )
            row: dict[str, object] = {"group": group}
            for label, table in tables.items():
                row[label] = month_bootstrap.level_interval(table=table).estimate
            for label in ("blend_64_members", "blend_equal_weight"):
                diff = month_bootstrap.difference_interval(
                    arm=tables[label], reference=tables["xgboost"]
                )
                row[f"{label}_minus_xgboost"] = diff.estimate
                row[f"{label}_interval"] = f"[{diff.lower_95:.4f}, {diff.upper_95:.4f}]"
            rows.append(row)
        table_frame = pl.DataFrame(rows)
        table_frame.write_parquet(out_dir / f"blend_{name}_seed{args.seed}.parquet")
        lines += [
            f"## {name}, seed {args.seed}",
            "",
            (
                "Normalised CRPS (fraction of capacity) on the rows where the heuristic has a "
                "forecast. Differences are blend minus XGBoost alone."
            ),
            "",
            markdown_table(
                frame=table_frame.with_columns(
                    pl.col(c).round(4)
                    for c in table_frame.columns
                    if table_frame[c].dtype == pl.Float64
                )
            ),
            "",
        ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
