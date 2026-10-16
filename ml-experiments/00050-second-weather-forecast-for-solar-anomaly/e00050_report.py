"""Print every number the write-up of experiment 00050 quotes, and save the text beside the results.

Reads `work/results/*.parquet` and `selection.json` (see `e00050_anomaly.py`), and writes
`work/results/report.txt`. The write-up quotes only numbers printed here.

    uv run python ml-experiments/00050-*/e00050_report.py
"""

import contextlib
import io
import json

import e00050_common as common
import polars as pl


def cv_summary() -> None:
    """Print the cross-validation errors, pooled and by fold, as ratios to ENS."""
    errors = pl.read_parquet(common.RESULTS_DIR / "cv_errors.parquet")
    selection = json.loads(common.SELECTION_PATH.read_text())
    print("Stage 1 selection (training window only)")
    print(json.dumps(selection, indent=2))
    scored_folds = errors.filter((pl.col("arm") == "ens") & (pl.col("n") > 0))["fold"].to_list()
    print(f"\nFolds holding scored April-to-September rows: {len(scored_folds)}: {scored_folds}")
    wide = errors.pivot(on="arm", index="fold", values="sse").sort("fold")
    wide = wide.filter(pl.col("ens") > 0).with_columns(
        gefs_ratio=pl.col("gefs") / pl.col("ens"), blend_ratio=pl.col("blend") / pl.col("ens")
    )
    print("\nSquared error by fold (sum over rows) and the ratio to ENS")
    print(wide)
    print("\nFolds in which each arm beats ENS")
    print(
        wide.select(gefs=(pl.col("gefs_ratio") < 1).sum(), blend=(pl.col("blend_ratio") < 1).sum())
    )


def validation_summary() -> None:
    """Print the validation-year anomaly scores and their intervals."""
    results = common.RESULTS_DIR
    scores = pl.read_parquet(results / "anomaly_scores.parquet")
    bands = ["all", *(name for name, _low, _high in common.LEAD_BANDS)]
    for metric in ("corr", "slope_calibrated", "slope_raw", "mse_skill"):
        print(f"\n{metric}, pooled over the five verdict series")
        table = scores.filter((pl.col("scope") == "pooled") & (pl.col("metric") == metric))
        print(
            table.pivot(on="arm", index="band", values="value").sort(
                pl.col("band").replace_strict({b: i for i, b in enumerate(bands)})
            )
        )
    print("\nCorrelation by series, all leads")
    print(
        scores.filter((pl.col("band") == "all") & (pl.col("metric") == "corr"))
        .pivot(on="arm", index="scope", values="value")
        .sort("scope")
    )
    print("\nCorrelation by series, 0-24 h band")
    print(
        scores.filter((pl.col("band") == "0-24 h") & (pl.col("metric") == "corr"))
        .pivot(on="arm", index="scope", values="value")
        .sort("scope")
    )
    print("\nRows behind each pooled band")
    print(
        scores.filter(
            (pl.col("scope") == "pooled") & (pl.col("metric") == "corr") & (pl.col("arm") == "ens")
        ).select("band", "n")
    )
    print("\nDifference in correlation from ENS, 95% interval from resampling init months")
    print(pl.read_parquet(results / "anomaly_intervals.parquet"))
    print("\nCorrelation of 3-hour mean anomalies (exploratory)")
    print(
        pl.read_parquet(results / "anomaly_three_hour.parquet").pivot(
            on="arm", index="band", values="corr"
        )
    )
    weights = pl.read_parquet(results / "blend_weights.parquet")
    print("\nBlend weights, mean over the five verdict series, by lead band")
    print(
        weights.filter(pl.col("site").is_in(list(common.VERDICT_SITES)))
        .group_by("band")
        .agg(
            w_ens=pl.col("w_ens").mean(),
            w_gefs=pl.col("w_gefs").mean(),
            slope_ens=pl.col("slope_ens").mean(),
            slope_gefs=pl.col("slope_gefs").mean(),
            negative=pl.col("negative_weight").sum(),
        )
        .sort("band")
    )


def latency_summary() -> None:
    """Print the latency decision."""
    decision = json.loads(common.LATENCY_PATH.read_text())
    print("\nGEFS run used")
    print(f"run_offset_days = {decision['run_offset_days']}: {decision['reason']}")
    print(f"observations recorded: {len(decision['observations'])}")


def main() -> None:
    """Print the report and save a copy."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        cv_summary()
        validation_summary()
        latency_summary()
    text = buffer.getvalue()
    print(text)
    (common.RESULTS_DIR / "report.txt").write_text(text)


if __name__ == "__main__":
    main()
