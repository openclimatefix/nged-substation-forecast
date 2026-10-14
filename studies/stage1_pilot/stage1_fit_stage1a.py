"""Fit stage 1a on ERA5 and CAMS and apply it to ENS weather, to ERA5 itself, and out of fold.

Stage 1a is one XGBoost model per series (squared error, production hyperparameters, GPU) from
weather and calendar columns to power divided by effective capacity. It has no power lag and no
lead input. Three kinds of model are fitted, each for seeds 0, 1, and 2:

- `pre`: trained on valid times before 2025-06-17 (14 days before the first validation run), used
  on every validation row. For the solar farms a second `pre` model is trained with CAMS irradiance
  in place of ERA5's.
- `short`: the same, trained only on valid times from 2024-04, the months the ENS-trained arms
  train on, so that it separates the training weather from the training length.
- `month`: for each month from 2024-04 to 2025-06, trained on valid times up to 14 days before the
  month starts, used on that month's rows of the ENS training frame. These are the forward-chained
  out-of-fold values the control with the extra column trains on.

Outputs under `data/studies/per_study/stage1_pilot/stage1a/` (each skipped if it exists):

- `val/seed<s>/init=<date>.parquet`: keys and the prediction (a fraction of capacity) for every
  validation row and member, from the ERA5-trained and (solar) CAMS-trained models.
- `train_oof/seed<s>.parquet`: keys and the forward-chained prediction for the training frame.
- `ceiling/seed<s>.parquet`: the `pre` models applied to ERA5 (and CAMS) at the validation times.
- `sanity.md`: with `--stage sanity`, a seed-0 fit and the diurnal and power-curve checks.

    uv run python studies/stage1_pilot/stage1_fit_stage1a.py --stage sanity
    uv run python studies/stage1_pilot/stage1_fit_stage1a.py --stage all
"""

import argparse
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import stage1_inputs as inputs
import xgboost as xgb
from studies.power import scan_power

STAGE_DIR: Final[Path] = inputs.OUTPUT_ROOT / "stage1a"
INPUT_DIR: Final[Path] = inputs.INPUTS_DIR

PARAMETERS: Final[dict[str, float | int | str]] = {
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "device": "cuda",
    "nthread": 4,
}
"""Production's hyperparameters (`conf/model/xgboost.yaml`), every column offered to each tree."""

N_ROUNDS: Final[int] = 500
"""Production's `n_estimators`."""

SETTLING_LEVEL_FRACTION: Final[float] = 0.05
SETTLING_RUN_HALF_HOURS: Final[int] = 48
SETTLING_WEEKS: Final[int] = 6
"""The quick-wins `new_meter_settling` rule: a series is live from the first run of 48 consecutive
half-hours above 5% of capacity, and the six weeks after that are dropped, as are all earlier rows.
This study applies it to the whole ERA5 era, with the effective capacity as the level."""

OOF_MONTHS: Final[tuple[datetime, ...]] = tuple(
    datetime(2024 + (3 + i) // 12, (3 + i) % 12 + 1, 1, tzinfo=UTC) for i in range(15)
)
"""First days of 2024-04 to 2025-06, the months of the ENS training frame."""


def month_label(*, month: datetime) -> str:
    """Return `YYYY-MM`."""
    return f"{month:%Y-%m}"


def observed_training_power(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return cleaned power as a fraction of capacity, from the ERA5 era to the validation start.

    Rows before each series' settling cutoff are dropped. Power is read through `scan_power`, the
    same cleaned observations the leaderboard scorer uses.
    """
    ids = sites["time_series_id"].to_list()
    power = (
        scan_power()
        .filter(
            pl.col("time_series_id").is_in(ids),
            pl.col("time") >= inputs.STAGE1_TRAIN_START,
            pl.col("time") < inputs.VALIDATION_START,
        )
        .select("time_series_id", pl.col("time").alias("valid_time"), "power")
        .collect()
        .join(sites.select("time_series_id", "effective_capacity_mw"), on="time_series_id")
        .sort("time_series_id", "valid_time")
    )
    live = (
        pl.col("power").abs() > SETTLING_LEVEL_FRACTION * pl.col("effective_capacity_mw")
    ).fill_null(value=False)
    run_end = (
        live.cast(pl.Int32).rolling_sum(window_size=SETTLING_RUN_HALF_HOURS).over("time_series_id")
        == SETTLING_RUN_HALF_HOURS
    )
    cutoffs = (
        power.filter(run_end)
        .group_by("time_series_id")
        .agg(
            cutoff=pl.col("valid_time")
            .min()
            .dt.offset_by(f"-{SETTLING_RUN_HALF_HOURS * 30 - 30}m")
            .dt.offset_by(f"{SETTLING_WEEKS}w")
        )
    )
    return (
        power.join(cutoffs, on="time_series_id", how="left")
        .filter(pl.col("cutoff").is_null() | (pl.col("valid_time") >= pl.col("cutoff")))
        .filter(pl.col("power").is_finite())
        .select(
            "time_series_id", "valid_time", target=pl.col("power") / pl.col("effective_capacity_mw")
        )
    )


def _matrix(*, frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    return frame.select([pl.col(c).cast(pl.Float32) for c in columns]).to_numpy()


def fit(*, rows: pl.DataFrame, columns: list[str], seed: int) -> xgb.Booster:
    """Fit one stage 1a model on rows that carry `target` and the columns."""
    if rows.height == 0:
        msg = "no training rows"
        raise ValueError(msg)
    return xgb.train(
        {**PARAMETERS, "seed": seed},
        xgb.DMatrix(_matrix(frame=rows, columns=columns), label=rows["target"].to_numpy()),
        num_boost_round=N_ROUNDS,
    )


def predict(*, booster: xgb.Booster, frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    """Return the prediction (a fraction of capacity) for every row of the frame."""
    return booster.predict(xgb.DMatrix(_matrix(frame=frame, columns=columns)))


def series_columns(*, sites: pl.DataFrame) -> dict[int, list[str]]:
    """Return each series' stage 1a column list from its technology."""
    return {
        int(row["time_series_id"]): list(inputs.FEATURES[row["technology"]])
        for row in sites.iter_rows(named=True)
    }


def training_rows(
    *,
    series: int,
    analysis: pl.DataFrame,
    observed: pl.DataFrame,
    before: datetime,
    arm: str,
) -> pl.DataFrame:
    """Return one series' ERA5 (or CAMS-irradiance) rows with their target, before a cut-off."""
    rows = analysis.filter(pl.col("time_series_id") == series).join(
        observed.filter(pl.col("time_series_id") == series),
        on=["time_series_id", "valid_time"],
        how="inner",
        validate="1:1",
    )
    if arm == "cams":
        rows = rows.with_columns(shortwave=pl.col("shortwave_cams")).drop_nulls("shortwave")
    if arm == "short":
        rows = rows.filter(pl.col("valid_time") >= OOF_MONTHS[0])
    return rows.filter(pl.col("valid_time") < before)


def sanity(*, sites: pl.DataFrame, analysis: pl.DataFrame, observed: pl.DataFrame) -> list[str]:
    """Fit seed 0's pre-validation models and print diurnal and power-curve checks on ERA5."""
    columns = series_columns(sites=sites)
    lines = ["# Stage 1a sanity checks (seed 0, pre-validation models applied to ERA5)", ""]
    solar_curve, wind_curve = [], []
    for row in sites.iter_rows(named=True):
        series = int(row["time_series_id"])
        rows = training_rows(
            series=series,
            analysis=analysis,
            observed=observed,
            before=inputs.STAGE1_TRAIN_END,
            arm="era5",
        )
        booster = fit(rows=rows, columns=columns[series], seed=0)
        validation = analysis.filter(
            (pl.col("time_series_id") == series)
            & (pl.col("valid_time") >= inputs.VALIDATION_START)
            & (pl.col("valid_time") < datetime(2026, 7, 1, tzinfo=UTC))
        )
        prediction = predict(booster=booster, frame=validation, columns=columns[series])
        frame = validation.with_columns(prediction=pl.Series(prediction)).join(
            observed_validation(series=series, sites=sites),
            on=["time_series_id", "valid_time"],
            how="inner",
        )
        lines.append(
            f"- {row['site']}: {rows.height:,} training half-hours; on {frame.height:,} validation "
            f"half-hours the ERA5-driven prediction has MAE "
            f"{100 * float((frame['prediction'] - frame['target']).abs().mean()):.2f} points of "  # ty: ignore[invalid-argument-type]
            f"capacity and bias {100 * float((frame['prediction'] - frame['target']).mean()):+.2f}."  # ty: ignore[invalid-argument-type]
        )
        if row["technology"] == "solar":
            solar_curve.append(
                frame.group_by(hour=pl.col("valid_time").dt.hour())
                .agg(
                    predicted=pl.col("prediction").mean(),
                    observed=pl.col("target").mean(),
                )
                .with_columns(site=pl.lit(row["site"]))
            )
        else:
            wind_curve.append(
                frame.group_by(speed_bin=(pl.col("wind_speed_100m").floor()).cast(pl.Int32))
                .agg(
                    predicted=pl.col("prediction").mean(),
                    observed=pl.col("target").mean(),
                    n=pl.len(),
                )
                .with_columns(site=pl.lit(row["site"]))
            )
    lines += [
        "",
        "## Diurnal profile at the solar farms (mean fraction of capacity by UTC hour)",
        "",
    ]
    lines += ["| hour | predicted | observed |", "| --- | --- | --- |"]
    curve = (
        pl.concat(solar_curve)
        .group_by("hour")
        .agg(pl.col("predicted").mean(), pl.col("observed").mean())
        .sort("hour")
    )
    lines += [
        f"| {r['hour']} | {r['predicted']:.3f} | {r['observed']:.3f} |"
        for r in curve.iter_rows(named=True)
    ]
    lines += [
        "",
        "## Power curve at the wind farms (mean fraction of capacity by 100 m speed bin, m/s)",
        "",
    ]
    lines += ["| speed bin | predicted | observed | half-hours |", "| --- | --- | --- | --- |"]
    wind = (
        pl.concat(wind_curve)
        .group_by("speed_bin")
        .agg(pl.col("predicted").mean(), pl.col("observed").mean(), pl.col("n").sum())
        .sort("speed_bin")
    )
    lines += [
        f"| {r['speed_bin']} | {r['predicted']:.3f} | {r['observed']:.3f} | {r['n']:,} |"
        for r in wind.iter_rows(named=True)
    ]
    return lines


def observed_validation(*, series: int, sites: pl.DataFrame) -> pl.DataFrame:
    """Return cleaned power as a fraction of capacity across the validation year, for one series."""
    capacity = sites.filter(pl.col("time_series_id") == series)["effective_capacity_mw"].item()
    return (
        scan_power()
        .filter(
            pl.col("time_series_id") == series,
            pl.col("time") >= inputs.VALIDATION_START,
            pl.col("time") < datetime(2026, 7, 1, tzinfo=UTC),
        )
        .select(
            "time_series_id", pl.col("time").alias("valid_time"), target=pl.col("power") / capacity
        )
        .collect()
        .filter(pl.col("target").is_finite())
    )


def fit_all(  # noqa: PLR0912
    *, sites: pl.DataFrame, analysis: pl.DataFrame, observed: pl.DataFrame
) -> None:
    """Fit the `pre` and `month` models for every seed and write all three prediction kinds."""
    columns = series_columns(sites=sites)
    technology = {int(r["time_series_id"]): r["technology"] for r in sites.iter_rows(named=True)}
    val_files = sorted((INPUT_DIR / "ens_features" / "val").glob("init=*.parquet"))
    train = pl.read_parquet(INPUT_DIR / "ens_features" / "train.parquet")
    for seed in inputs.SEEDS:
        todo = [
            STAGE_DIR / "train_oof" / f"seed{seed}.parquet",
            STAGE_DIR / "ceiling" / f"seed{seed}.parquet",
            *[STAGE_DIR / "val" / f"seed{seed}" / p.name for p in val_files],
        ]
        if all(p.exists() for p in todo):
            continue
        started = time.monotonic()
        pre: dict[tuple[int, str], xgb.Booster] = {}
        for series in columns:
            for arm in ("era5", "cams", "short"):
                if arm == "cams" and technology[series] != "solar":
                    continue
                rows = training_rows(
                    series=series,
                    analysis=analysis,
                    observed=observed,
                    before=inputs.STAGE1_TRAIN_END,
                    arm=arm,
                )
                pre[series, arm] = fit(rows=rows, columns=columns[series], seed=seed)
        print(f"seed {seed}: pre models in {time.monotonic() - started:.0f} s", flush=True)

        for path in val_files:
            target = STAGE_DIR / "val" / f"seed{seed}" / path.name
            if target.exists():
                continue
            frame = pl.read_parquet(path)
            pieces = []
            for series, part in frame.partition_by("time_series_id", as_dict=True).items():
                sid = int(series[0])
                result = part.select(*inputs.KEY_COLUMNS).with_columns(
                    s1_era5=pl.Series(
                        predict(booster=pre[sid, "era5"], frame=part, columns=columns[sid])
                    ),
                    s1_cams=pl.Series(
                        predict(booster=pre[sid, "cams"], frame=part, columns=columns[sid])
                        if (sid, "cams") in pre
                        else np.full(part.height, np.nan, dtype=np.float32)
                    ),
                    s1_short=pl.Series(
                        predict(booster=pre[sid, "short"], frame=part, columns=columns[sid])
                    ),
                )
                pieces.append(result)
            target.parent.mkdir(parents=True, exist_ok=True)
            pl.concat(pieces).cast(
                {"s1_era5": pl.Float32, "s1_cams": pl.Float32, "s1_short": pl.Float32}
            ).write_parquet(target.with_suffix(".tmp"))
            target.with_suffix(".tmp").rename(target)

        ceiling_path = STAGE_DIR / "ceiling" / f"seed{seed}.parquet"
        if not ceiling_path.exists():
            window = analysis.filter(
                (pl.col("valid_time") >= inputs.VALIDATION_START)
                & (pl.col("valid_time") < datetime(2026, 7, 1, tzinfo=UTC))
            )
            pieces = []
            for series in columns:
                part = window.filter(pl.col("time_series_id") == series)
                cams_input = part.with_columns(shortwave=pl.col("shortwave_cams"))
                pieces.append(
                    part.select("time_series_id", "valid_time").with_columns(
                        s1_era5=pl.Series(
                            predict(
                                booster=pre[series, "era5"], frame=part, columns=columns[series]
                            )
                        ),
                        s1_cams=pl.Series(
                            predict(
                                booster=pre[series, "cams"],
                                frame=cams_input,
                                columns=columns[series],
                            )
                            if (series, "cams") in pre
                            else np.full(part.height, np.nan, dtype=np.float32)
                        ),
                    )
                )
            ceiling_path.parent.mkdir(parents=True, exist_ok=True)
            pl.concat(pieces).cast({"s1_era5": pl.Float32, "s1_cams": pl.Float32}).write_parquet(
                ceiling_path.with_suffix(".tmp")
            )
            ceiling_path.with_suffix(".tmp").rename(ceiling_path)

        oof_path = STAGE_DIR / "train_oof" / f"seed{seed}.parquet"
        if not oof_path.exists():
            pieces = []
            for month in OOF_MONTHS:
                next_month = datetime(
                    month.year + (month.month // 12), month.month % 12 + 1, 1, tzinfo=UTC
                )
                cutoff = month - timedelta(days=inputs.PURGE_DAYS)
                rows_in_month = train.filter(
                    (pl.col("valid_time") >= month) & (pl.col("valid_time") < next_month)
                )
                for series in columns:
                    fitted = fit(
                        rows=training_rows(
                            series=series,
                            analysis=analysis,
                            observed=observed,
                            before=cutoff,
                            arm="era5",
                        ),
                        columns=columns[series],
                        seed=seed,
                    )
                    part = rows_in_month.filter(pl.col("time_series_id") == series)
                    pieces.append(
                        part.select(*inputs.KEY_COLUMNS).with_columns(
                            s1_oof=pl.Series(
                                predict(booster=fitted, frame=part, columns=columns[series])
                            ).cast(pl.Float32),
                            month=pl.lit(month_label(month=month)),
                        )
                    )
                print(f"seed {seed}: out-of-fold {month_label(month=month)}", flush=True)
            oof_path.parent.mkdir(parents=True, exist_ok=True)
            pl.concat(pieces).write_parquet(oof_path.with_suffix(".tmp"))
            oof_path.with_suffix(".tmp").rename(oof_path)
        print(f"seed {seed} done in {time.monotonic() - started:.0f} s", flush=True)


def main() -> None:
    """Run the sanity fit or every fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("sanity", "all"), required=True)
    args = parser.parse_args()
    sites = inputs.site_table()
    analysis = pl.read_parquet(INPUT_DIR / "analysis_features.parquet")
    observed = observed_training_power(sites=sites)
    if args.stage == "sanity":
        report = sanity(sites=sites, analysis=analysis, observed=observed)
        STAGE_DIR.mkdir(parents=True, exist_ok=True)
        (STAGE_DIR / "sanity.md").write_text("\n".join(report) + "\n")
        print("\n".join(report))
        return
    fit_all(sites=sites, analysis=analysis, observed=observed)


if __name__ == "__main__":
    main()
