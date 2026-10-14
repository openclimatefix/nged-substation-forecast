"""Check whether the long ERA5 history hurts demand because of bad early data or because of drift.

Exploratory follow-up to experiment 00003. The ERA5-trained model with the 2019 onward history has
a larger error than the same model trained from 2024-04 only. Two explanations are tested, each by
refitting the ERA5-trained model with a different training window, in the weather-only (V0) and
weather-plus-calendar (V1) variants:

- **Bad early data:** `drop_first6m` drops the first six months of each series' history, for the
  series with more than three years of history.
- **Drift:** `start_<year>-09` trains from September of 2020, 2021, 2022, and 2023. If the error
  falls smoothly as the start moves later, the old data differ from the validation year's, whatever
  their quality.

`full` (2019 onward, every row) is refitted so every contrast is paired. Rows, folds, seeds, and
scoring follow `demand_score.py`. Output: `data/studies/per_study/stage1_pilot/demand/results/
history_check.md`, with per-series errors.

    uv run python studies/stage1_pilot/demand_history_check.py
"""

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import demand_fit as fit
import demand_inputs as demand
import demand_score as score
import numpy as np
import polars as pl
import stage1_inputs as inputs
import stage1_stats as stats
import xgboost as xgb

RESULT_PATH: Final[Path] = demand.demand_subfolder(name="results") / "history_check.md"
VARIANTS: Final[tuple[str, ...]] = ("V0", "V1")
START_YEARS: Final[tuple[int, ...]] = (2020, 2021, 2022, 2023)
LONG_HISTORY_DAYS: Final[int] = 3 * 365
DROPPED_DAYS: Final[int] = 183
ARMS: Final[tuple[str, ...]] = (
    "full",
    "drop_first6m",
    *[f"start_{year}-09" for year in START_YEARS],
    "short",
)


def window_rows(*, arm: str, era5: pl.DataFrame) -> pl.DataFrame:
    """Return the training rows of an arm: every series' rows within the arm's window."""
    if arm == "full":
        return era5
    if arm == "short":
        return era5.filter(pl.col("valid_time") >= fit.SHORT_START)
    if arm == "drop_first6m":
        span = era5.group_by("time_series_id").agg(
            first=pl.col("valid_time").min(), last=pl.col("valid_time").max()
        )
        return (
            era5.join(span, on="time_series_id")
            .filter(
                ((pl.col("last") - pl.col("first")).dt.total_days() <= LONG_HISTORY_DAYS)
                | (pl.col("valid_time") >= pl.col("first") + pl.duration(days=DROPPED_DAYS))
            )
            .drop("first", "last")
        )
    year = int(arm.removeprefix("start_")[:4])
    return era5.filter(pl.col("valid_time") >= datetime(year, 9, 1, tzinfo=UTC))


def predicted_errors(*, seed: int, series: pl.DataFrame) -> pl.DataFrame:
    """Fit every arm and variant for one seed and return the summed error by series and month."""
    era5 = fit.era5_training_rows(series=series, seed=seed)
    ids = [int(i) for i in series["time_series_id"]]
    capacity = {
        int(r["time_series_id"]): float(r["effective_capacity_mw"])
        for r in series.iter_rows(named=True)
    }
    boosters = {}
    started = time.monotonic()
    for arm in ARMS:
        rows = window_rows(arm=arm, era5=era5)
        for variant in VARIANTS:
            for sid in ids:
                part = rows.filter(pl.col("time_series_id") == sid)
                boosters[arm, variant, sid] = fit.xgb.train(
                    {**fit.PARAMETERS, "seed": seed},
                    xgb.DMatrix(
                        fit._matrix(frame=part, columns=demand.VARIANTS[variant]),
                        label=part["target"].to_numpy(),
                    ),
                    num_boost_round=fit.N_ROUNDS,
                )
    print(f"seed {seed}: fitted in {time.monotonic() - started:.0f} s", flush=True)
    pieces = []
    for path in sorted(fit.ENS_FEATURES_DIR.joinpath("val").glob("init=*.parquet")):
        frame = pl.read_parquet(path).sort("time_series_id", "valid_time", "ensemble_member")
        columns = {f"{a}__{v}": np.empty(frame.height) for a in ARMS for v in VARIANTS}
        offset = 0
        for (key,), part in frame.partition_by(
            "time_series_id", as_dict=True, maintain_order=True
        ).items():
            sid = int(key)
            for variant in VARIANTS:
                matrix = xgb.DMatrix(fit._matrix(frame=part, columns=demand.VARIANTS[variant]))
                for arm in ARMS:
                    columns[f"{arm}__{variant}"][offset : offset + part.height] = (
                        boosters[arm, variant, sid].predict(matrix) * capacity[sid]
                    )
            offset += part.height
        cases = (
            frame.with_columns(*[pl.Series(k, v) for k, v in columns.items()])
            .group_by("time_series_id", "valid_time", maintain_order=True)
            .agg(actual=pl.col("power").first(), **{k: pl.col(k).mean() for k in columns})
            .filter(pl.col("actual").is_finite() & (pl.col("valid_time") < score.VAL_END))
            .join(series.select("time_series_id", "effective_capacity_mw"), on="time_series_id")
            .with_columns(
                month=pl.col("valid_time").dt.strftime("%Y-%m"),
                **{
                    k: 100.0
                    * (pl.col(k) - pl.col("actual")).abs()
                    / pl.col("effective_capacity_mw")
                    for k in columns
                },
            )
            .group_by("time_series_id", "month")
            .agg(n=pl.len(), **{k: pl.col(k).sum() for k in columns})
        )
        pieces.append(cases)
    return (
        pl.concat(pieces)
        .group_by("time_series_id", "month")
        .agg(pl.all().exclude("time_series_id", "month").sum())
        .with_columns(seed=pl.lit(seed))
    )


def main() -> None:
    """Fit the window arms, score them, and write the report."""
    series = demand.series_table()
    ids = sorted(int(i) for i in series["time_series_id"])
    agg = pl.concat([predicted_errors(seed=seed, series=series) for seed in inputs.SEEDS]).join(
        series.select("time_series_id", "group"), on="time_series_id"
    )
    agg = agg.with_columns(band=pl.lit("all"), top=pl.lit(value=False))
    lines = [
        "# History check for experiment 00003 (exploratory)",
        "",
        (
            "Mean absolute error of the ensemble-mean forecast, points of capacity, 22 series, "
            "all leads, with the difference from `full` (2019 onward) and its 95% "
            "month-resampled interval."
        ),
        "",
        "| variant | training window | error | difference from full | months better |",
        "| --- | --- | --- | --- | --- |",
    ]
    for variant in VARIANTS:
        reference = score.monthly_for(
            agg=agg, arm=f"full__{variant}", band="all", group="all", tail="all cases", series=ids
        )
        for arm in ARMS:
            table = score.monthly_for(
                agg=agg,
                arm=f"{arm}__{variant}",
                band="all",
                group="all",
                tail="all cases",
                series=ids,
            )
            level = stats.level(table=table)
            if arm == "full":
                lines.append(f"| {variant} | {arm} | {level.value:.2f} | | |")
                continue
            result = stats.contrast(treatment=table, reference=reference)
            lines.append(
                f"| {variant} | {arm} | {level.value:.2f} | {result.difference:+.2f} "
                f"({result.lower:+.2f} to {result.upper:+.2f}) | {result.months_better} of 12 |"
            )
    lines += [
        "",
        "## Error by series, V0, all leads",
        "",
        "| series | group | " + " | ".join(ARMS) + " |",
        "| --- | --- |" + " --- |" * len(ARMS),
    ]
    per = agg.group_by("time_series_id", "group").agg(
        n=pl.col("n").sum(), **{f"{a}__V0": pl.col(f"{a}__V0").sum() for a in ARMS}
    )
    labels = series.select("time_series_id", "site")
    for row in (
        per.join(labels, on="time_series_id")
        .sort(pl.col("site").str.slice(1).cast(pl.Int32))
        .iter_rows(named=True)
    ):
        cells = [f"{row[f'{a}__V0'] / row['n']:.2f}" for a in ARMS]
        lines.append(f"| {row['site']} | {row['group']} | " + " | ".join(cells) + " |")
    RESULT_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
