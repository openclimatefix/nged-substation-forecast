"""Calibrate the anomaly arms, run the stage 1 gate, and write the validation anomalies.

All fits use training-window valid times only (up to 2025-06-30 23:30 UTC). For each variant
(`t54`, `t54_late`, `t120`; see `e00053_ukv_inputs.py`) and each lead band (`BAND_NAMES`, lead
counted from the forecast origin) and site:

- `ens`: slope through the origin of the CAMS anomaly on the ENS ensemble-mean anomaly.
- `ukv`: the same on the UKV anomaly (native steps).
- `blend`: the CAMS anomaly regressed on both anomalies, two coefficients, no intercept.
- `ukv3h`: the same slope on the UKV anomaly subsampled to ENS's 3-hourly steps and interpolated
  as ENS is, treating the instantaneous values as window means (control).
- `ukv3w`: the same slope on the UKV anomaly formed from 3-hour window means (control).

Every anomaly is the source's `u` minus its own month-by-half-hour climatology (neighbouring local
months pooled, as in experiment 00046). The stage 1 gate is a 15-fold month-block cross-validation:
one fold per local calendar month of valid time from April 2024 to June 2025, fitting the
climatologies (the CAMS one included) and the calibration on the other months and scoring the
mean squared error against the CAMS anomaly on daylight rows pooled over the five verdict series.

    uv run python ml-experiments/00053-*/e00053_anomaly.py
"""

import json
from typing import Final

import e00053_common as common
import polars as pl

ARMS: Final[tuple[str, ...]] = ("ens", "ukv", "blend", "ukv3h", "ukv3w")
SOURCES: Final[tuple[str, ...]] = ("ens", "ukv", "ukv3h", "ukv3w")
MARGIN: Final[float] = 0.01
"""A candidate replaces ENS only if its cross-validated mean squared error is this share lower."""
FOLD_MONTHS: Final[tuple[tuple[int, int], ...]] = tuple(
    (2024 + (3 + k) // 12, (3 + k) % 12 + 1) for k in range(15)
)
"""(year, month) of the 15 training months, April 2024 to June 2025."""


def training_ens() -> pl.DataFrame:
    """Return the ENS ensemble-mean `u` of every training run (experiment 00046's file).

    Returns:
        Columns `site`, `power_fcst_init_time`, `valid_time`, `u_ens`.
    """
    return pl.concat(
        [
            pl.read_parquet(path, columns=["site", "power_fcst_init_time", "valid_time", "u_mean"])
            for path in sorted((common.E46_SOLAR / "ens_train").glob("init=*.parquet"))
        ]
    ).rename({"u_mean": "u_ens"})


def validation_ens() -> pl.DataFrame:
    """Return the ENS ensemble-mean anomaly of every validation run (experiment 00046's files).

    Returns:
        Columns `site`, `power_fcst_init_time`, `valid_time`, `a_ens` (mean over the members of the
        anomaly against the full-window ENS climatology).
    """
    return pl.concat(
        [
            pl.read_parquet(path)
            .group_by("site", "power_fcst_init_time", "valid_time")
            .agg(
                a_ens=pl.col("anomaly").mean(),
                a_ens_46=pl.col("anomaly_mean_calibrated").first(),
            )
            for path in sorted((common.E46_SOLAR / "ens_val").glob("init=*.parquet"))
        ]
    )


def with_block(*, frame: pl.DataFrame, column: str) -> pl.DataFrame:
    """Add the local (year, month) of a time column as `block` = year * 100 + month.

    Args:
        frame: Rows with a UTC datetime column.
        column: The column name.

    Returns:
        The frame with an Int32 `block`.
    """
    local = pl.col(column).dt.convert_time_zone("Europe/London")
    return frame.with_columns(block=(local.dt.year() * 100 + local.dt.month()).cast(pl.Int32))


def climatologies(*, cams: pl.DataFrame, rows: pl.DataFrame, exclude_block: int | None) -> dict:
    """Return the climatologies of CAMS and of each source, leaving out one month block.

    Args:
        cams: Columns `site`, `valid_time`, `u_cams`, `block`.
        rows: Columns `site`, `valid_time`, `block`, and one `u_<source>` column per source.
        exclude_block: A `block` to leave out, or `None` for the whole training window.

    Returns:
        Climatology tables keyed `cams` and by source name; columns `site`, `month`, `hh`, `clim`.
    """
    if exclude_block is not None:
        cams = cams.filter(pl.col("block") != exclude_block)
        rows = rows.filter(pl.col("block") != exclude_block)
    training_cams = cams.filter(pl.col("valid_time") <= common.LAST_TRAINING_VALID)
    result = {
        "cams": common.climatology(
            frame=training_cams, value="u_cams", time_column="valid_time", pool=True
        ).select("site", "month", "hh", "clim")
    }
    for source in SOURCES:
        unique = rows.select("site", "valid_time", f"u_{source}").unique(["site", "valid_time"])
        result[source] = common.climatology(
            frame=unique, value=f"u_{source}", time_column="valid_time", pool=True
        ).select("site", "month", "hh", "clim")
    return result


def anomalies(*, rows: pl.DataFrame, clims: dict, daylight: pl.DataFrame) -> pl.DataFrame:
    """Add the CAMS and source anomalies, the lead, the band, and the daylight flag to the rows.

    Args:
        rows: Columns `site`, `power_fcst_init_time`, `valid_time`, `u_cams`, `u_<source>`.
        clims: `climatologies()` output.
        daylight: Columns `site`, `month`, `hh`, `daylight`.

    Returns:
        The rows with `a_cams`, `a_<source>`, `lead_hours`, `band`, and `daylight`, dropping rows
        that lack any climatology cell.
    """
    out = common.with_cell(frame=rows, time_column="valid_time")
    for name, clim in clims.items():
        out = out.join(
            clim.rename({"clim": f"clim_{name}"}), on=["site", "month", "hh"], how="left"
        )
    lead = ((out["valid_time"] - out["power_fcst_init_time"]).dt.total_minutes() / 60.0).to_numpy()
    out = out.with_columns(
        a_cams=pl.col("u_cams") - pl.col("clim_cams"),
        lead_hours=pl.Series(lead),
        band=pl.Series(common.band_index(lead_hours=lead)),
        **{f"a_{s}": pl.col(f"u_{s}") - pl.col(f"clim_{s}") for s in SOURCES},
    ).join(daylight, on=["site", "month", "hh"], how="left")
    return out.filter(
        pl.col("daylight").fill_null(False)
        & pl.all_horizontal(pl.col(f"a_{s}").is_finite() for s in SOURCES)
    )


def fit(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Fit the slopes and blend weights per site and band.

    Args:
        rows: `anomalies()` output restricted to the fitting rows.

    Returns:
        Columns `site`, `band`, `n`, one `slope_<source>` per source, and the blend weights
        `w_ens` and `w_ukv`.
    """
    grouped = rows.group_by("site", "band").agg(
        n=pl.len(),
        see=(pl.col("a_ens") ** 2).sum(),
        suu=(pl.col("a_ukv") ** 2).sum(),
        s33=(pl.col("a_ukv3h") ** 2).sum(),
        sww=(pl.col("a_ukv3w") ** 2).sum(),
        seu=(pl.col("a_ens") * pl.col("a_ukv")).sum(),
        sec=(pl.col("a_ens") * pl.col("a_cams")).sum(),
        suc=(pl.col("a_ukv") * pl.col("a_cams")).sum(),
        s3c=(pl.col("a_ukv3h") * pl.col("a_cams")).sum(),
        swc=(pl.col("a_ukv3w") * pl.col("a_cams")).sum(),
    )
    det = pl.col("see") * pl.col("suu") - pl.col("seu") ** 2
    return grouped.select(
        "site",
        "band",
        "n",
        slope_ens=pl.col("sec") / pl.col("see"),
        slope_ukv=pl.col("suc") / pl.col("suu"),
        slope_ukv3h=pl.col("s3c") / pl.col("s33"),
        slope_ukv3w=pl.col("swc") / pl.col("sww"),
        w_ens=(pl.col("sec") * pl.col("suu") - pl.col("suc") * pl.col("seu")) / det,
        w_ukv=(pl.col("suc") * pl.col("see") - pl.col("sec") * pl.col("seu")) / det,
    )


def predict(*, rows: pl.DataFrame, coefficients: pl.DataFrame) -> pl.DataFrame:
    """Add the four arms' calibrated central anomalies (`c_<arm>`) to the rows.

    Args:
        rows: `anomalies()` output.
        coefficients: `fit()` output.

    Returns:
        The rows with one `c_<arm>` column per arm; rows without coefficients are dropped.
    """
    return rows.join(coefficients.drop("n"), on=["site", "band"], how="inner").with_columns(
        c_ens=pl.col("slope_ens") * pl.col("a_ens"),
        c_ukv=pl.col("slope_ukv") * pl.col("a_ukv"),
        c_ukv3h=pl.col("slope_ukv3h") * pl.col("a_ukv3h"),
        c_ukv3w=pl.col("slope_ukv3w") * pl.col("a_ukv3w"),
        c_blend=pl.col("w_ens") * pl.col("a_ens") + pl.col("w_ukv") * pl.col("a_ukv"),
    )


def cross_validate(
    *, rows: pl.DataFrame, cams: pl.DataFrame, daylight: pl.DataFrame, minimum_n: int = 50
) -> pl.DataFrame:
    """Run the 15-fold month-block cross-validation and return squared errors by arm and band.

    Args:
        rows: Training rows with `u_cams`, `u_<source>`, `block`.
        cams: CAMS `u` with `block`.
        daylight: The daylight mask.
        minimum_n: Cells with fewer fitting rows are left out of the fit.

    Returns:
        Columns `fold`, `band`, `arm`, `n`, `sse`, pooled over the five verdict series.
    """
    pieces = []
    for year, month in FOLD_MONTHS:
        block = year * 100 + month
        clims = climatologies(cams=cams, rows=rows, exclude_block=block)
        table = anomalies(rows=rows, clims=clims, daylight=daylight)
        train = table.filter(pl.col("block") != block)
        test = table.filter((pl.col("block") == block) & pl.col("site").is_in(common.VERDICT_SITES))
        coefficients = fit(rows=train).filter(pl.col("n") >= minimum_n)
        scored = predict(rows=test, coefficients=coefficients)
        pieces.extend(
            scored.group_by("band")
            .agg(n=pl.len(), sse=((pl.col("a_cams") - pl.col(f"c_{arm}")) ** 2).sum())
            .with_columns(fold=pl.lit(block), arm=pl.lit(arm))
            for arm in ARMS
        )
    return pl.concat(pieces)


def main() -> None:  # noqa: PLR0915
    """Run the gate and write the validation anomalies."""
    cams = with_block(
        frame=pl.read_parquet(common.E46_SOLAR / "cams_u.parquet").rename(
            {"time": "valid_time", "u": "u_cams"}
        ),
        column="valid_time",
    )
    daylight = pl.read_parquet(common.E46_SOLAR / "climatology.parquet").select(
        "site", "month", "hh", "daylight"
    )
    ens_train = training_ens()
    ukv = pl.read_parquet(common.WORK_DIR / "ukv_u.parquet")
    ens_clim_46 = pl.read_parquet(common.E46_SOLAR / "climatology.parquet").select(
        "site", "month", "hh", "clim_ens"
    )
    validation_ens_frame = validation_ens()
    selection: dict[str, dict] = {}
    all_cv, all_coefficients, all_validation = [], [], []
    for variant in ("t54", "t54_late", "t120"):
        own = ukv.filter(pl.col("variant") == variant).drop("variant")
        training_rows = (
            ens_train.join(own, on=["site", "power_fcst_init_time", "valid_time"], how="inner")
            .join(
                cams.select("site", "valid_time", "u_cams"),
                on=["site", "valid_time"],
                how="inner",
            )
            .pipe(lambda f: with_block(frame=f, column="valid_time"))
        )
        if variant == "t54_late":
            training_rows = training_rows.with_columns(
                u_ukv3h=pl.col("u_ukv"), u_ukv3w=pl.col("u_ukv")
            )
        else:
            training_rows = training_rows.filter(
                pl.col("u_ukv3h").is_not_null() & pl.col("u_ukv3w").is_not_null()
            )
        print(f"{variant}: {training_rows.height:,} training rows", flush=True)
        folds = cross_validate(rows=training_rows, cams=cams, daylight=daylight)
        cv = (
            folds.group_by("band", "arm")
            .agg(n=pl.col("n").sum(), sse=pl.col("sse").sum())
            .with_columns(mse=pl.col("sse") / pl.col("n"), variant=pl.lit(variant))
        )
        all_cv.append(cv)
        clims = climatologies(cams=cams, rows=training_rows, exclude_block=None)
        table = anomalies(rows=training_rows, clims=clims, daylight=daylight)
        coefficients = fit(rows=table).with_columns(variant=pl.lit(variant))
        all_coefficients.append(coefficients)
        # Validation rows: ENS anomaly from experiment 00046's files, UKV from this variant.
        validation = validation_ens_frame.join(
            own, on=["site", "power_fcst_init_time", "valid_time"], how="inner"
        ).join(cams.select("site", "valid_time", "u_cams"), on=["site", "valid_time"], how="inner")
        validation = common.with_cell(frame=validation, time_column="valid_time").join(
            ens_clim_46, on=["site", "month", "hh"], how="left"
        )
        validation = validation.with_columns(u_ens=pl.col("a_ens") + pl.col("clim_ens")).drop(
            "clim_ens", "month", "hh"
        )
        if variant == "t54_late":
            validation = validation.with_columns(u_ukv3h=pl.col("u_ukv"), u_ukv3w=pl.col("u_ukv"))
        else:
            validation = validation.filter(
                pl.col("u_ukv3h").is_not_null() & pl.col("u_ukv3w").is_not_null()
            )
        full_clims = climatologies(cams=cams, rows=training_rows, exclude_block=None)
        validation = predict(
            rows=anomalies(rows=validation, clims=full_clims, daylight=daylight),
            coefficients=coefficients,
        ).with_columns(variant=pl.lit(variant))
        check = validation.select(pl.corr("c_ens", "a_ens_46")).item()
        print(
            f"{variant}: validation rows {validation.height:,}; correlation of the ENS arm with "
            f"experiment 00046's calibrated ENS anomaly {check:.4f}",
            flush=True,
        )
        all_validation.append(validation)
        # Stage 1 gate.
        pooled = (
            cv.group_by("arm")
            .agg(sse=pl.col("sse").sum(), n=pl.col("n").sum())
            .with_columns(mse=pl.col("sse") / pl.col("n"))
        )
        selection[variant] = {"pooled_mse": dict(zip(pooled["arm"], pooled["mse"], strict=True))}
        for arm in ("ukv", "blend", "ukv3h", "ukv3w"):
            by_band = {}
            for band, name in enumerate(common.BAND_NAMES):
                part = cv.filter(pl.col("band") == band).select("arm", "mse")
                if part.height == 0:
                    continue
                mse = dict(zip(part["arm"], part["mse"], strict=True))
                by_band[name] = {
                    "mse_ens": mse["ens"],
                    f"mse_{arm}": mse[arm],
                    "relative": mse[arm] / mse["ens"] - 1.0,
                    "clears_margin": bool(mse[arm] <= (1.0 - MARGIN) * mse["ens"]),
                }
            pooled_mse = selection[variant]["pooled_mse"]
            selection[variant][arm] = {
                "by_band": by_band,
                "pooled_relative": pooled_mse[arm] / pooled_mse["ens"] - 1.0,
                "pooled_clears_margin": bool(pooled_mse[arm] <= (1.0 - MARGIN) * pooled_mse["ens"]),
            }
    cv_table = pl.concat(all_cv).sort("variant", "band", "arm")
    coefficient_table = pl.concat(all_coefficients).sort("variant", "site", "band")
    common.write_atomically(frame=cv_table, target=common.RESULTS_DIR / "cv_mse.parquet")
    common.write_atomically(
        frame=coefficient_table, target=common.RESULTS_DIR / "coefficients.parquet"
    )
    common.write_atomically(
        frame=pl.concat(all_validation, how="diagonal_relaxed"),
        target=common.WORK_DIR / "validation_anomalies.parquet",
    )
    (common.RESULTS_DIR / "selection.json").write_text(json.dumps(selection, indent=2))
    with pl.Config(tbl_rows=100):
        print(cv_table)
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
