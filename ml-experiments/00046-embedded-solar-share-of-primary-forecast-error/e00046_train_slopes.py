"""Fit the training-window coefficient of the oracle shift, for the exploratory reads of C3 and C4.

Read only when the coefficient diagnostic of `e00046_stats.py` finds a pooled slope above -0.8 in
some lead band. For each series and lead band, the slope through the origin of the residual of the
`d1l` median (mean over the three seeds, on the ENS control-member training rows up to 2025-06-30)
on the oracle shift is fitted on daylight half-hours with a CAMS anomaly. The coefficient is minus
that slope. The training residuals are in-sample, so they understate the error and lean the
slope towards zero.

Writes `work/results/train_slopes.parquet` (every series and band) and
`work/results/band_coefficients.parquet` (`site`, `band`, `coefficient`: only the bands that the
trigger names, for `e00046_fit.py --band-coefficients-file`).

    uv run python ml-experiments/00046-*/e00046_train_slopes.py
"""

from typing import Final

import e00046_common as common
import e00046_fit as fit_module
import numpy as np
import polars as pl
import xgboost as xgb
from e00046_stats import TRIGGER_SLOPE


def training_slopes() -> pl.DataFrame:
    """Return the training-window slope per series and lead band.

    Returns:
        Columns `site`, `band` (index into `LEAD_BANDS`), `slope`, and `rows`.
    """
    series = common.series_table().filter(pl.col("site").is_in(list(common.SITES)))
    arrays = common.load_series_arrays(series=series)
    capacity = {r["site"]: r["C"] for r in common.capacities().iter_rows(named=True)}
    anomaly = pl.read_parquet(common.SOLAR_DIR / "cams_anomaly.parquet")
    daylight = pl.read_parquet(common.SOLAR_DIR / "climatology.parquet").select(
        "site", "month", "hh", "daylight"
    )
    train = pl.read_parquet(common.ENS_TRAIN_PATH).filter(
        pl.col("valid_time") <= common.LAST_TRAINING_VALID
    )
    columns = fit_module.input_columns(arm="d1l")
    pieces = []
    for row in series.iter_rows(named=True):
        site = row["site"]
        rows = fit_module.own_rows(row=row, frame=train, arrays=arrays[site])
        matrix = xgb.DMatrix(rows.select(columns).to_numpy().astype(np.float32))
        medians = []
        for seed in common.SEEDS:
            booster = xgb.Booster()
            booster.load_model(common.BOOSTER_DIR / f"d1l_seed{seed}_{site}.ubj")
            medians.append(booster.predict(matrix)[:, common.MEDIAN])
        effective = row["effective_capacity_mw"]
        frame = (
            rows.with_columns(
                site=pl.lit(site),
                residual=rows["power"] / effective - pl.Series(np.mean(medians, axis=0)),
                band=pl.Series(common.lead_band(lead_hours=rows["lead_hours"].to_numpy())),
            )
            .pipe(lambda f: common.with_cell(frame=f, time_column="valid_time"))
            .join(anomaly, on=["site", "valid_time"], how="inner")
            .join(daylight, on=["site", "month", "hh"], how="left")
            .filter(pl.col("daylight").fill_null(False) & pl.col("residual").is_finite())
            .with_columns(shift=capacity[site] * pl.col("anomaly_cams") / effective)
        )
        pieces.append(frame.select("site", "band", "residual", "shift"))
    return (
        pl.concat(pieces)
        .group_by("site", "band")
        .agg(
            slope=(pl.col("residual") * pl.col("shift")).sum() / (pl.col("shift") ** 2).sum(),
            rows=pl.len(),
        )
        .sort("site", "band")
    )


def main() -> None:
    """Fit the slopes and write the table of coefficients for the triggered lead bands."""
    slopes = training_slopes()
    diagnostic = pl.read_parquet(common.RESULTS_DIR / "diagnostic.parquet").filter(
        pl.col("split") == "lead band"
    )
    names = [name for name, _low, _high in common.LEAD_BANDS]
    triggered: Final[list[int]] = [
        names.index(r["group"])
        for r in diagnostic.iter_rows(named=True)
        if r["slope"] > TRIGGER_SLOPE
    ]
    print(f"triggered lead bands (pooled validation slope above {TRIGGER_SLOPE}): {triggered}")
    with pl.Config(tbl_rows=40):
        print("training-window slope of the d1l residual on the oracle shift")
        print(slopes)
    common.write_atomically(frame=slopes, target=common.RESULTS_DIR / "train_slopes.parquet")
    coefficients = (
        slopes.filter(pl.col("band").is_in(triggered))
        .select("site", "band", coefficient=-pl.col("slope"))
        .filter(pl.col("site").is_in(list(common.SITES)))
    )
    print("coefficients for the triggered bands")
    print(coefficients)
    common.write_atomically(
        frame=coefficients, target=common.RESULTS_DIR / "band_coefficients.parquet"
    )


if __name__ == "__main__":
    main()
