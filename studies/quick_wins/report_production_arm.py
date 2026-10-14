"""Score the refitted production forecast against the stored experiments and print the gate.

Reads the forecasts `fit_production_arm.py` wrote, scores them with the leaderboard's own CRPS and
mean absolute error on the sampled ECMWF runs, scores the stored `xgboost_baseline`, its two
retrains, `manual_heuristic`, and `climatology` rows on the same runs, and writes `report.md`,
`scores_<experiment>.parquet`, and the per-series tables. The refit passes the reproduction gate
when its scores sit within the range of the three stored production runs on the same rows.

    uv run python studies/quick_wins/report_production_arm.py [--arm colsample0.8_seed0]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Final

import mlflow
import polars as pl
import quick_wins_inputs as inputs
from contracts.settings import Settings
from per_run_scores import (
    LEAD_BAND_ORDER,
    lead_band_means,
    normalised_by_capacity,
    per_run_scores,
)
from report_tables import markdown_table
from studies.guards import refuse_to_overwrite
from studies.power import scan_power

STORED_EXPERIMENTS: Final[tuple[str, ...]] = (
    "xgboost_baseline",
    "xgboost_baseline_retrain_638",
    "xgboost_baseline_retrain_790",
    "manual_heuristic",
    "climatology",
)
"""The stored experiments scored on the sample beside the refit."""


def _score_refit(*, arm_dir: Path, actuals: pl.DataFrame) -> pl.DataFrame:
    """Score each sampled run's file on its own, so a 51-member frame never sits in memory whole."""
    parts = [
        per_run_scores(forecasts=pl.read_parquet(path), actuals=actuals)
        for path in sorted((arm_dir / "forecasts").glob("init=*.parquet"))
    ]
    return pl.concat(parts)


def _score_stored(
    *, experiment: str, init_times: list[datetime], fold_id: str, actuals: pl.DataFrame
) -> pl.DataFrame:
    """Score a stored experiment on the sampled runs, one initialisation time at a time."""
    settings = Settings()
    scan = pl.scan_delta(settings.power_forecasts_data_path).filter(
        (pl.col("experiment_name") == experiment) & (pl.col("fold_id") == fold_id)
    )
    parts = [
        per_run_scores(
            forecasts=scan.filter(pl.col("power_fcst_init_time") == init_time).collect(
                engine="streaming"
            ),
            actuals=actuals,
        )
        for init_time in init_times
    ]
    return pl.concat(parts)


def _stored_config(*, experiment: str) -> str:
    """Describe whether the stored experiment's registered configuration equals the yaml's.

    The comparison drops the two fields that name the experiment, and `model_release`, which
    is not a model setting. The stored rows' `power_fcst` was
    rounded to a reduced significand on write, which the refit's unrounded values are not.
    """
    mlflow.set_tracking_uri(Settings().mlflow_tracking_uri)
    found = mlflow.get_experiment_by_name(experiment)
    if found is None or "config" not in found.tags:
        return f"No registered configuration was found for {experiment}, so it was not compared."
    stored = json.loads(found.tags["config"])
    ignored = {"experiment_name", "ml_flow_experiment_id", "model_release"}
    current = inputs.production_config().model_dump(mode="json")
    differing = sorted(
        key for key in (set(stored) | set(current)) - ignored if stored.get(key) != current.get(key)
    )
    versions = (
        pl.scan_delta(Settings().power_forecasts_data_path)
        .filter(pl.col("experiment_name") == experiment)
        .select("power_fcst_model_version")
        .unique()
        .collect()["power_fcst_model_version"]
        .to_list()
    )
    return (
        f"Fields differing between the registered configuration of {experiment} and "
        f"`conf/model/xgboost.yaml`: {differing or 'none'}. Stored rows carry model version "
        f"{versions}."
    )


def _summary(*, scores: pl.DataFrame, capacity: pl.DataFrame, types: pl.DataFrame) -> pl.DataFrame:
    """Mean over series of each series' mean CRPS (MW), MAE (MW), and NMAE, per type and band."""
    per_series = lead_band_means(scores=normalised_by_capacity(scores=scores, capacity=capacity))
    typed = per_series.join(types, on="time_series_id")
    overall = typed.with_columns(time_series_type=pl.lit("all types"))
    return (
        pl.concat([typed, overall])
        .group_by("time_series_type", "lead_band")
        .agg(
            n_series=pl.len(),
            crps_mw=pl.col("crps").mean(),
            mae_mw=pl.col("mae").mean(),
            nmae=pl.col("nmae").mean(),
        )
    )


def main() -> None:
    """Score every arm on the sample, write the tables, and print the reproduction gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="colsample0.8_seed0")
    args = parser.parse_args()

    settings = Settings()
    window = inputs.fold_window()
    arm_dir = inputs.OUTPUT_ROOT / "production_arm" / args.arm
    out_dir = arm_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.md"
    refuse_to_overwrite(paths=[report_path])

    series_ids = inputs.eligible_series_ids(settings=settings, fold_id=window.fold_id)
    metadata = inputs.series_metadata(settings=settings, time_series_ids=series_ids)
    types = pl.DataFrame(metadata).select(
        "time_series_id", pl.col("time_series_type").cast(pl.String)
    )
    capacity = pl.DataFrame(inputs.effective_capacity(settings=settings))
    actuals = (
        scan_power()
        .filter(pl.col("time") >= window.val_start, pl.col("time") <= window.val_end)
        .collect()
    )

    refit_scores = _score_refit(arm_dir=arm_dir, actuals=actuals)
    init_times = sorted(refit_scores["power_fcst_init_time"].unique().to_list())
    all_scores = {"refit (this study)": refit_scores}
    for experiment in STORED_EXPERIMENTS:
        all_scores[f"stored {experiment}"] = _score_stored(
            experiment=experiment, init_times=init_times, fold_id=window.fold_id, actuals=actuals
        )

    stored_series = set(all_scores["stored xgboost_baseline"]["time_series_id"].unique().to_list())
    refit_series = set(refit_scores["time_series_id"].unique().to_list())
    if stored_series != refit_series:
        msg = (
            f"the refit scores {sorted(refit_series - stored_series)} that the stored experiment "
            f"does not, and lacks {sorted(stored_series - refit_series)} that it has"
        )
        raise ValueError(msg)
    stored_config = _stored_config(experiment="xgboost_baseline")

    lines = [
        f"# Production arm on the sampled runs ({args.arm})",
        "",
        (
            f"Runs scored: {len(init_times)} initialisation times, every "
            f"{inputs.SAMPLE_STRIDE_DAYS} days from {window.val_start:%Y-%m-%d}."
        ),
        "",
        "## Configuration",
        "",
        stored_config,
        "",
        "## Row keys",
        "",
        "| arm | scored rows | rows also in the refit | rows not in the refit |",
        "|---|---|---|---|",
    ]
    refit_keys = refit_scores.select("time_series_id", "power_fcst_init_time", "valid_time")
    for name, scores in all_scores.items():
        keys = scores.select("time_series_id", "power_fcst_init_time", "valid_time")
        shared = keys.join(refit_keys, on=keys.columns, how="inner").height
        lines.append(f"| {name} | {scores.height:,} | {shared:,} | {scores.height - shared:,} |")

    summaries = {
        name: _summary(scores=scores, capacity=capacity, types=types)
        for name, scores in all_scores.items()
    }
    for name, scores in all_scores.items():
        slug = name.replace(" ", "_").replace("(", "").replace(")", "")
        scores.write_parquet(out_dir / f"scores_{slug}.parquet")
        summaries[name].write_parquet(out_dir / f"summary_{slug}.parquet")

    for band in LEAD_BAND_ORDER:
        for metric in ("crps_mw", "mae_mw", "nmae"):
            lines += ["", f"## {metric}, lead band: {band}", ""]
            wide = pl.concat(
                [
                    summaries[name]
                    .filter(pl.col("lead_band") == band)
                    .select("time_series_type", pl.col(metric).alias(name))
                    for name in all_scores
                ],
                how="align",
            ).sort("time_series_type")
            lines.append(markdown_table(frame=wide))

    gate = pl.concat(
        [
            summaries[name]
            .filter((pl.col("time_series_type") == "all types") & (pl.col("lead_band") == "all"))
            .select(pl.lit(name).alias("arm"), "crps_mw", "nmae")
            for name in all_scores
            if name.startswith(("refit", "stored xgboost"))
        ]
    )
    stored = gate.filter(pl.col("arm").str.starts_with("stored"))
    lines += [
        "",
        "## Reproduction gate",
        "",
        markdown_table(frame=gate),
    ]
    refit_row = gate.filter(pl.col("arm").str.starts_with("refit")).row(0, named=True)
    for metric in ("crps_mw", "nmae"):
        low, high = stored[metric].min(), stored[metric].max()
        inside = low <= refit_row[metric] <= high
        lines.append(
            f"- {metric}: refit {refit_row[metric]:.4f}; stored production runs span "
            f"[{low:.4f}, {high:.4f}]; the refit is {'inside' if inside else 'OUTSIDE'} that range "
            f"({(refit_row[metric] / stored[metric].mean() - 1) * 100:+.2f}% of the stored mean)."
        )
    report_path.write_text("\n".join(lines) + "\n")
    print(report_path.read_text())


if __name__ == "__main__":
    main()
