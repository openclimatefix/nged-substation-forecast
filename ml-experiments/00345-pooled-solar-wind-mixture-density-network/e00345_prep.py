# ruff: noqa: E501
"""Build the wind tables of experiment 345 in the unified column layout of `e00345_common`.

Outputs under `data/ml-experiments/00345-*/prep/`: `wind_train_metered.parquet` (W1 to W3, ERA5
weather and observed power over capacity before 2025-06-17, as experiment 313 trained on),
`wind_train_public.parquet` (the public farms of experiment 288 that qualify, hourly ERA5 weather
before 2024-04-01, no temperature), `wind_eval_<farm>_<era5|ens>.parquet` (W1 to W3's scored cases
and their 51 ensemble members, in case order), and `wind_cases.parquet` (experiment 285's ordered
cases). Public farms are labelled P00 and up; their names stay out of every file written here.

    PYTHONPATH=ml-experiments/00285-wind-effective-wind-adapters:ml-experiments/00310-metered-farm-long-lead-scoring \
        uv run --no-sync python ml-experiments/00345-*/e00345_prep.py
"""

import e00285_common as c285  # ty: ignore[unresolved-import]
import e00285_run as run285  # ty: ignore[unresolved-import]
import e00285_score as s285  # ty: ignore[unresolved-import]
import e00345_common as common
import numpy as np
import polars as pl
from studies.sources import REPO_DATA_DIR

ERA5_PUBLIC = REPO_DATA_DIR / "studies/downloads/reanalysis/ERA5/farm_cells_00288"


def unify(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the frame's weather columns renamed and expanded to the unified layout."""
    return frame.with_columns(
        temperature=pl.col("temperature_c") if "temperature_c" in frame.columns else None,
        wind10=pl.col("wind_speed_10m"),
        wind100=pl.col("wind_speed_100m"),
        dir_sin10=pl.col("wind_dir_sin_10m"),
        dir_cos10=pl.col("wind_dir_cos_10m"),
        dir_sin100=pl.col("wind_dir_sin_100m"),
        dir_cos100=pl.col("wind_dir_cos_100m"),
        hour_sin=(2 * np.pi * pl.col("hour") / 24.0).sin(),
        hour_cos=(2 * np.pi * pl.col("hour") / 24.0).cos(),
        doy_sin=(2 * np.pi * pl.col("doy") / 366.0).sin(),
        doy_cos=(2 * np.pi * pl.col("doy") / 366.0).cos(),
    ).with_columns(
        temp_missing=pl.col("temperature").is_null().cast(pl.Float32),
        is_wind=pl.lit(1.0, dtype=pl.Float32),
    )


UNIFIED = [
    "temperature",
    "wind10",
    "wind100",
    "dir_sin10",
    "dir_cos10",
    "dir_sin100",
    "dir_cos100",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "temp_missing",
    "is_wind",
]


def direction(*, u: pl.Expr, v: pl.Expr, height: str) -> list[pl.Expr]:
    """Return speed and the sine and cosine of the direction the wind blows from."""
    bearing = ((270.0 - pl.arctan2(v, u).degrees()) % 360.0).radians()
    return [
        (u**2 + v**2).sqrt().alias(f"wind_speed_{height}m"),
        bearing.sin().alias(f"wind_dir_sin_{height}m"),
        bearing.cos().alias(f"wind_dir_cos_{height}m"),
    ]


def public_rows() -> pl.DataFrame:
    """Return the public farms' hourly ERA5 weather joined to their clean power over capacity."""
    farms = pl.read_parquet(common.E288_WORK / "clean" / "farms.parquet").filter(
        pl.col("qualifies")
    )
    labels = {b: f"P{i:02d}" for i, b in enumerate(farms["elexon_bmu_id"].to_list())}
    era5 = (
        pl.read_parquet(ERA5_PUBLIC / "era5_farm_cells.parquet")
        .select(
            "cell_id",
            pl.col("time").alias("valid_time"),
            *direction(u=pl.col("u10"), v=pl.col("v10"), height="10"),
            *direction(u=pl.col("u100"), v=pl.col("v100"), height="100"),
        )
        .filter(pl.col("valid_time") < common.PUBLIC_END)
    )
    actual = pl.read_parquet(common.E288_WORK / "clean" / "actual_centred.parquet")
    pieces = []
    for farm in farms.iter_rows(named=True):
        rows = (
            era5.filter(pl.col("cell_id") == farm["cell_id"])
            .join(
                actual.filter(pl.col("elexon_bmu_id") == farm["elexon_bmu_id"]).select(
                    "valid_time", target="actual_norm"
                ),
                on="valid_time",
                how="inner",
            )
            .filter(pl.col("valid_time") >= farm["train_start"], pl.col("target").is_finite())
            .with_columns(farm=pl.lit(labels[farm["elexon_bmu_id"]]))
        )
        pieces.append(rows)
    frame = pl.concat(pieces).with_columns(
        hour=(pl.col("valid_time").dt.hour() + pl.col("valid_time").dt.minute() / 60).cast(
            pl.Float32
        ),
        doy=pl.col("valid_time").dt.ordinal_day().cast(pl.Float32),
    )
    return unify(frame=frame).select("farm", "valid_time", "target", *UNIFIED)


def main() -> None:
    """Write every wind table."""
    common.PREP_DIR.mkdir(parents=True, exist_ok=True)
    cases = s285.ordered_cases()
    cases.write_parquet(common.PREP_DIR / "wind_cases.parquet")
    metered = []
    for farm in common.WIND_FARMS:
        train, era5_cases, _, members, farm_cases = run285.farm_frames(farm=farm, need_ens=True)
        site_cases = cases.filter(pl.col("site") == farm)
        keys = list(c285.KEY_COLUMNS)
        if not farm_cases.select(keys).equals(site_cases.select(keys)):
            msg = f"{farm}: the case order differs from experiment 285's ordered cases"
            raise ValueError(msg)
        metered.append(
            unify(frame=train.filter(pl.col("valid_time") < common.WIND_TRAIN_END)).select(
                pl.lit(farm).alias("farm"), "valid_time", "target", *UNIFIED
            )
        )
        unify(frame=era5_cases).select(*UNIFIED).write_parquet(
            common.PREP_DIR / f"wind_eval_{farm}_era5.parquet"
        )
        assert members is not None
        unify(frame=members).select(*UNIFIED).write_parquet(
            common.PREP_DIR / f"wind_eval_{farm}_ens.parquet"
        )
        print(farm, train.height, era5_cases.height, members.height, flush=True)
    pl.concat(metered).write_parquet(common.PREP_DIR / "wind_train_metered.parquet")
    public = public_rows()
    public.write_parquet(common.PREP_DIR / "wind_train_public.parquet")
    print("public rows", public.height, "farms", public["farm"].n_unique())


if __name__ == "__main__":
    main()
