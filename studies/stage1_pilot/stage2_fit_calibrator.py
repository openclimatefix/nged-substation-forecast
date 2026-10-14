"""Fit stage 2's shared calibrator on out-of-fold stage 1a quantiles and apply it to every member.

One calibrator per technology and stage 1a variant, pooled over the technology's series, for seeds
0, 1, 2. Inputs: the member's 9 stage 1a quantiles and the lead in hours. A location model (XGBoost
quantile loss at level 0.5) and a scale model (squared error on the absolute residual of the
location model, floored at `MIN_SCALE`) give the calibrated quantile at level `t` as
`clip(location + scale * z_t, 0, 1)`, sorted. The fixed shape `z_t` is the level-`t` quantile of
the standardised residuals `(y - location) / scale` on the technology's and lead band's
forward-chained out-of-fold rows: for each month from 2024-10 to 2025-06 the calibrator is fitted on
the training months that end 14 days before the month starts. The final calibrator is fitted on
all 12 training months (2024-07 to 2025-06).

Outputs under `data/studies/per_study/stage2_pilot/calibrated/<variant>/`, skipped if they exist:
`val/seed<s>/init=<date>.parquet` (keys and the 9 calibrated quantile columns for every validation
row and member) and `shapes_seed<s>.parquet` (the shape by technology and lead band).

    uv run python studies/stage1_pilot/stage2_fit_calibrator.py
"""

import time
from datetime import timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import stage1_inputs as inputs
import xgboost as xgb
from stage2_common import (
    ENS_FEATURES_DIR,
    LEVEL_COLUMNS,
    LEVELS,
    MIN_SCALE,
    STAGE2_ROOT,
    band_name,
)
from stage2_fit_stage1a_quantile import OUT_DIR as STAGE1A_DIR
from stage2_fit_stage1a_quantile import TRAINING_MONTHS, VARIANTS

OUT_DIR: Final[Path] = STAGE2_ROOT / "calibrated"
FEATURES: Final[list[str]] = [*LEVEL_COLUMNS, "lead_hours"]
CHAIN_MONTHS: Final = TRAINING_MONTHS[3:]
"""2024-10 to 2025-06: the months whose calibrator is fitted only on earlier months."""
LOCATION_PARAMETERS: Final[dict[str, object]] = {
    "objective": "reg:quantileerror",
    "quantile_alpha": 0.5,
    "eta": 0.05,
    "max_depth": 4,
    "min_child_weight": 20,
    "subsample": 0.8,
    "tree_method": "hist",
    "device": "cuda",
    "nthread": 4,
}
SCALE_PARAMETERS: Final[dict[str, object]] = {
    "objective": "reg:squarederror",
    "eta": 0.05,
    "max_depth": 3,
    "min_child_weight": 50,
    "subsample": 0.8,
    "tree_method": "hist",
    "device": "cuda",
    "nthread": 4,
}
LOCATION_ROUNDS: Final[int] = 200
SCALE_ROUNDS: Final[int] = 150


def _lead(*, frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    )


def training_rows(*, variant: str, seed: int, sites: pl.DataFrame) -> pl.DataFrame:
    """Return the ENS control-member training rows with their out-of-fold stage 1a quantiles."""
    drops = pl.read_parquet(inputs.CLEANING_DROPS).rename({"time": "valid_time"})
    oof = pl.read_parquet(STAGE1A_DIR / variant / "train_oof" / f"seed{seed}.parquet")
    rows = (
        pl.read_parquet(ENS_FEATURES_DIR / "train.parquet")
        .select(*inputs.KEY_COLUMNS, "power")
        .filter(pl.col("power").is_finite())
        .join(drops, on=["time_series_id", "valid_time"], how="anti")
        .join(oof, on=list(inputs.KEY_COLUMNS), how="inner", validate="1:1")
        .join(
            sites.select("time_series_id", "technology", "effective_capacity_mw"),
            on="time_series_id",
        )
        .with_columns(target=pl.col("power") / pl.col("effective_capacity_mw"))
    )
    return _lead(frame=rows)


def _matrix(*, frame: pl.DataFrame) -> np.ndarray:
    return frame.select([pl.col(c).cast(pl.Float32) for c in FEATURES]).to_numpy()


def fit_calibrator(*, rows: pl.DataFrame, seed: int) -> tuple[xgb.Booster, xgb.Booster]:
    """Fit the location and scale models on rows that carry the features and `target`."""
    features = _matrix(frame=rows)
    target = rows["target"].to_numpy()
    location = xgb.train(
        {**LOCATION_PARAMETERS, "seed": seed},
        xgb.DMatrix(features, label=target),
        num_boost_round=LOCATION_ROUNDS,
    )
    residual = np.abs(target - location.predict(xgb.DMatrix(features)))
    scale = xgb.train(
        {**SCALE_PARAMETERS, "seed": seed},
        xgb.DMatrix(features, label=residual),
        num_boost_round=SCALE_ROUNDS,
    )
    return location, scale


def apply_calibrator(
    *, models: tuple[xgb.Booster, xgb.Booster], frame: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Return the location and the floored scale for every row."""
    matrix = xgb.DMatrix(_matrix(frame=frame))
    return models[0].predict(matrix), np.maximum(models[1].predict(matrix), MIN_SCALE)


def shapes(*, rows: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Return the fixed quantile shape by technology and lead band from forward-chained rows."""
    pieces = []
    for technology in ("solar", "wind"):
        part = rows.filter(pl.col("technology") == technology)
        residuals = []
        for month in CHAIN_MONTHS:
            next_month = month.replace(
                year=month.year + month.month // 12, month=month.month % 12 + 1
            )
            fit_rows = part.filter(pl.col("valid_time") < month - timedelta(days=inputs.PURGE_DAYS))
            test = part.filter(
                (pl.col("valid_time") >= month) & (pl.col("valid_time") < next_month)
            )
            location, scale = apply_calibrator(
                models=fit_calibrator(rows=fit_rows, seed=seed), frame=test
            )
            residuals.append(
                test.select("lead_hours").with_columns(
                    standardised=pl.Series((test["target"].to_numpy() - location) / scale)
                )
            )
        pooled = pl.concat(residuals)
        bands = band_name(lead_hours=pooled["lead_hours"].to_numpy())
        for name, _, _ in inputs.LEAD_BANDS:
            values = pooled["standardised"].to_numpy()[bands == name]
            pieces.append(
                pl.DataFrame(
                    {
                        "technology": technology,
                        "band": name,
                        **dict(
                            zip(LEVEL_COLUMNS, np.quantile(values, LEVELS).tolist(), strict=True)
                        ),
                        "rows": values.size,
                    }
                )
            )
    return pl.concat(pieces)


def calibrate_frame(
    *,
    frame: pl.DataFrame,
    models: dict[str, tuple[xgb.Booster, xgb.Booster]],
    shape: pl.DataFrame,
) -> pl.DataFrame:
    """Return the calibrated quantile columns for validation rows that carry stage 1a quantiles."""
    pieces = []
    for technology, part in frame.partition_by("technology", as_dict=True).items():
        location, scale = apply_calibrator(models=models[str(technology[0])], frame=part)
        bands = band_name(lead_hours=part["lead_hours"].to_numpy())
        table = {
            r["band"]: np.array([r[c] for c in LEVEL_COLUMNS])
            for r in shape.filter(pl.col("technology") == technology[0]).iter_rows(named=True)
        }
        z = np.stack([table[b] for b in bands])
        values = np.sort(np.clip(location[:, None] + scale[:, None] * z, 0.0, 1.0), axis=1)
        pieces.append(
            pl.concat(
                [
                    part.select(*inputs.KEY_COLUMNS),
                    pl.DataFrame(values.astype(np.float32), schema=list(LEVEL_COLUMNS)),
                ],
                how="horizontal",
            )
        )
    return pl.concat(pieces)


def main() -> None:
    """Fit and apply the calibrator for every variant and seed."""
    sites = inputs.site_table()
    technology = sites.select("time_series_id", "technology")
    val_files = sorted((ENS_FEATURES_DIR / "val").glob("init=*.parquet"))
    for variant in VARIANTS:
        for seed in inputs.SEEDS:
            shape_path = OUT_DIR / variant / f"shapes_seed{seed}.parquet"
            targets = [OUT_DIR / variant / "val" / f"seed{seed}" / p.name for p in val_files]
            if shape_path.exists() and all(t.exists() for t in targets):
                continue
            started = time.monotonic()
            rows = training_rows(variant=variant, seed=seed, sites=sites)
            shape = shapes(rows=rows, seed=seed)
            models = {
                tech: fit_calibrator(rows=rows.filter(pl.col("technology") == tech), seed=seed)
                for tech in ("solar", "wind")
            }
            shape_path.parent.mkdir(parents=True, exist_ok=True)
            shape.write_parquet(shape_path)
            print(
                f"{variant} seed {seed}: calibrators in {time.monotonic() - started:.0f} s",
                flush=True,
            )
            for path, target in zip(val_files, targets, strict=True):
                if target.exists():
                    continue
                stage1 = pl.read_parquet(STAGE1A_DIR / variant / "val" / f"seed{seed}" / path.name)
                frame = _lead(frame=stage1).join(technology, on="time_series_id")
                target.parent.mkdir(parents=True, exist_ok=True)
                calibrate_frame(frame=frame, models=models, shape=shape).write_parquet(
                    target.with_suffix(".tmp")
                )
                target.with_suffix(".tmp").rename(target)
            print(f"{variant} seed {seed}: done in {time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
