"""Stage 1 of experiment 00050: how well do ENS, GEFS, and a blend forecast the CAMS solar anomaly?

The solar anomaly of a source is that source's unit PV shape `u` minus its own month-by-half-hour
climatology. Three calibrated forecasts of the CAMS anomaly are compared:

- `ens`: slope times the ENS ensemble-mean anomaly (experiment 00046's anomaly).
- `gefs`: slope times the GEFS ensemble-mean anomaly.
- `blend`: weight on the ENS anomaly plus weight on the GEFS anomaly, no intercept.

Each slope and weight pair is fitted per site and lead band by least squares through the origin on
daylight rows (a month-by-half-hour cell is daylight when the CAMS climatology exceeds 0.02).

- `cv` (training window only) is the selection gate. Fifteen month-block folds, one per valid month
  from April 2024 to June 2025, fit the three calibrations on the other 14 months and predict the
  held-out month. **Inside each fold, each source's climatology, the CAMS climatology, the
  daylight mask, the slopes, and the blend weights are all fitted on the other 14 months only.**
  Rows with a valid time in the held-out month, and rows of runs initialised in it, are removed from
  fitting, so a held-out target never
  appears in training at another lead. The scored rows are daylight rows at leads of 0 to 240
  hours in April to September (the months the validation scoring uses), pooled over the five
  verdict series. The arm with the lowest squared error is the best
  arm, and GEFS or the blend replaces ENS only when its error is at least 1% lower. The result is
  written to `results/selection.json`.
- `validate` fits the same calibrations on the whole training window, computes each arm's central
  anomaly on the validation runs (written to `solar/central_val.parquet`), and scores the
  correlation with the CAMS anomaly, the calibration slope, and the mean-squared-error skill, by
  lead band, pooled over the five verdict series and per series, with 95% intervals on the
  difference in correlation from a resample of the runs' initialisation months. It refuses to run
  until `cv` has written `selection.json`.

    uv run python ml-experiments/00050-*/e00050_anomaly.py cv
    uv run python ml-experiments/00050-*/e00050_anomaly.py validate
"""

import argparse
import json
from datetime import UTC, datetime
from typing import Final

import e00050_common as common
import e00050_gefs_inputs as inputs
import numpy as np
import polars as pl

N_RESAMPLES: Final[int] = 50_000
SEED: Final[int] = 0
BAND_NAMES: Final[tuple[str, ...]] = tuple(name for name, _low, _high in common.LEAD_BANDS)
ARM_COLUMNS: Final[dict[str, str]] = {
    "ens": "central_ens",
    "gefs": "central_gefs",
    "blend": "central_blend",
    "blend_fixed": "central_blend_fixed",
}
"""The arms scored on the validation year; `blend_fixed` (an equal-weight average of the two
calibrated anomalies) is exploratory and takes no part in the selection."""
SELECTION_ARMS: Final[tuple[str, ...]] = common.ARMS_ANOMALY
DET_FLOOR: Final[float] = 1e-12


def block_index(*, times: pl.Expr) -> pl.Expr:
    """Return the number of calendar months (UTC) from April 2024 to a valid time, as an Int32."""
    start = common.CV_FIRST_MONTH
    return ((times.dt.year() - start.year) * 12 + (times.dt.month() - start.month)).cast(pl.Int32)


def fit_calibrations(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Fit, per site and lead band, the ENS slope, the GEFS slope, and the blend weights.

    Args:
        rows: Daylight rows with `site`, `band`, `a_ens`, `a_gefs`, and `a_cams` (anomalies).

    Returns:
        One row per site and band with `slope_ens` and `slope_gefs` (each source alone, through the
        origin), `w_ens` and `w_gefs` (the two-regressor least-squares weights, no intercept), `n`,
        and `negative_weight` (true when either blend weight is below zero). Where the two
        regressors are collinear to within `DET_FLOOR`, the weights fall back to (`slope_ens`, 0).
    """
    sums = rows.group_by("site", "band").agg(
        ee=(pl.col("a_ens") ** 2).sum(),
        gg=(pl.col("a_gefs") ** 2).sum(),
        eg=(pl.col("a_ens") * pl.col("a_gefs")).sum(),
        ce=(pl.col("a_cams") * pl.col("a_ens")).sum(),
        cg=(pl.col("a_cams") * pl.col("a_gefs")).sum(),
        n=pl.len(),
    )
    det = pl.col("ee") * pl.col("gg") - pl.col("eg") ** 2
    slope_ens = pl.col("ce") / pl.col("ee")
    return (
        sums.with_columns(
            slope_ens=slope_ens,
            slope_gefs=pl.col("cg") / pl.col("gg"),
            w_ens=pl.when(det > DET_FLOOR)
            .then((pl.col("ce") * pl.col("gg") - pl.col("cg") * pl.col("eg")) / det)
            .otherwise(slope_ens),
            w_gefs=pl.when(det > DET_FLOOR)
            .then((pl.col("cg") * pl.col("ee") - pl.col("ce") * pl.col("eg")) / det)
            .otherwise(0.0),
        )
        .with_columns(negative_weight=(pl.col("w_ens") < 0.0) | (pl.col("w_gefs") < 0.0))
        .select(
            "site",
            "band",
            "slope_ens",
            "slope_gefs",
            "w_ens",
            "w_gefs",
            "n",
            "negative_weight",
        )
        .sort("site", "band")
    )


def apply_calibrations(*, rows: pl.DataFrame, calibrations: pl.DataFrame) -> pl.DataFrame:
    """Add the central anomaly of each arm to rows that hold `a_ens` and `a_gefs`.

    Args:
        rows: Rows with `site`, `band`, `a_ens`, and `a_gefs`.
        calibrations: `fit_calibrations()`.

    Returns:
        The rows with `central_ens`, `central_gefs`, `central_blend`, and `central_blend_fixed`
        (the mean of `central_ens` and `central_gefs`).
    """
    return (
        rows.join(
            calibrations.select("site", "band", "slope_ens", "slope_gefs", "w_ens", "w_gefs"),
            on=["site", "band"],
            how="left",
            validate="m:1",
            maintain_order="left",
        )
        .with_columns(
            central_ens=pl.col("slope_ens") * pl.col("a_ens"),
            central_gefs=pl.col("slope_gefs") * pl.col("a_gefs"),
            central_blend=pl.col("w_ens") * pl.col("a_ens") + pl.col("w_gefs") * pl.col("a_gefs"),
        )
        .with_columns(central_blend_fixed=(pl.col("central_ens") + pl.col("central_gefs")) / 2.0)
        .drop("slope_ens", "slope_gefs", "w_ens", "w_gefs")
    )


def read_training_rows(*, max_lead_hours: float | None) -> pl.DataFrame:
    """Return the training-window rows both sources and CAMS share.

    Args:
        max_lead_hours: Keep leads up to this many hours (the CV gate uses 240); all leads, to 351
            hours, when None, so the calibration of the 10+ d band has rows to fit.

    Returns:
        Columns `site`, `power_fcst_init_time`, `valid_time`, `lead_hours`, `band`, `block` (the
        CV fold the valid month belongs to), `u_ens`, `u_gefs`, and `u_cams`, with every `u` finite.

    Raises:
        FileNotFoundError: If either training window has no files yet.
    """
    inputs.require_complete(split="train")
    ens_files = sorted((common.SOURCE_SOLAR_DIR / "ens_train").glob("init=*.parquet"))
    gefs_files = sorted(common.SOLAR_DIR_TRAIN.glob("init=*.parquet"))
    if not ens_files or not gefs_files:
        msg = "build the ENS and GEFS training files first (e00050_gefs_inputs.py)"
        raise FileNotFoundError(msg)
    keys = ["site", "power_fcst_init_time", "valid_time"]
    ens = pl.concat([pl.read_parquet(p, columns=[*keys, "u_mean"]) for p in ens_files]).rename(
        {"u_mean": "u_ens"}
    )
    gefs = pl.concat([pl.read_parquet(p, columns=[*keys, "u_mean"]) for p in gefs_files]).rename(
        {"u_mean": "u_gefs"}
    )
    cams = pl.read_parquet(common.SOURCE_SOLAR_DIR / "cams_u.parquet").rename(
        {"time": "valid_time", "u": "u_cams"}
    )
    joined = ens.join(gefs, on=keys, how="inner", validate="1:1").join(
        cams, on=["site", "valid_time"], how="inner", validate="m:1"
    )
    lead = common.lead_hours_of(frame=joined)
    return (
        joined.with_columns(
            lead_hours=pl.Series(lead),
            band=pl.Series(common.lead_band(lead_hours=lead)),
            block=block_index(times=pl.col("valid_time")),
        )
        .filter(
            (pl.col("valid_time") <= common.LAST_TRAINING_VALID)
            & (pl.col("lead_hours") <= (max_lead_hours if max_lead_hours is not None else 1e9))
            & pl.col("u_ens").is_finite()
            & pl.col("u_gefs").is_finite()
            & pl.col("u_cams").is_finite()
        )
        .select(
            "site",
            "power_fcst_init_time",
            "valid_time",
            "lead_hours",
            "band",
            "block",
            "u_ens",
            "u_gefs",
            "u_cams",
        )
    )


def anomalies(
    *, rows: pl.DataFrame, clim_ens: pl.DataFrame, clim_gefs: pl.DataFrame, clim_cams: pl.DataFrame
) -> pl.DataFrame:
    """Subtract each source's climatology from its `u` and mark the daylight rows.

    Args:
        rows: Rows with `site`, `valid_time`, `u_ens`, `u_gefs`, and `u_cams`.
        clim_ens: A climatology table (`site`, `month`, `hh`, `clim`) for ENS.
        clim_gefs: The same for GEFS.
        clim_cams: The same for CAMS.

    Returns:
        The rows with `a_ens`, `a_gefs`, `a_cams`, and `daylight`, restricted to cells all three
        climatologies hold.
    """
    cells = common.with_cell(frame=rows, time_column="valid_time")
    for name, clim in (("ens", clim_ens), ("gefs", clim_gefs), ("cams", clim_cams)):
        cells = cells.join(
            clim.select("site", "month", "hh", **{f"clim_{name}": pl.col("clim")}),
            on=["site", "month", "hh"],
            how="inner",
            validate="m:1",
        )
    return cells.with_columns(
        a_ens=pl.col("u_ens") - pl.col("clim_ens"),
        a_gefs=pl.col("u_gefs") - pl.col("clim_gefs"),
        a_cams=pl.col("u_cams") - pl.col("clim_cams"),
        daylight=pl.col("clim_cams") > common.DAYLIGHT_THRESHOLD,
    ).drop("clim_ens", "clim_gefs", "clim_cams")


def fold_errors(*, rows: pl.DataFrame, cams: pl.DataFrame, fold: int) -> pl.DataFrame:
    """Fit on the other months and return the held-out month's squared errors by arm.

    Args:
        rows: `read_training_rows()`.
        cams: CAMS `u` over the whole CAMS history up to the last training valid time, with `site`,
            `valid_time`, `u`, and `block`.
        fold: The held-out `block`.

    Returns:
        One row per arm (the three calibrated arms plus `zero`, a zero anomaly) with `sse` and `n`,
        pooled over the verdict series.

    Raises:
        ValueError: If the held-out month leaks into the rows the fit sees.
    """
    init_block = block_index(times=pl.col("power_fcst_init_time"))
    train = rows.filter((pl.col("block") != fold) & (init_block != fold))
    test = rows.filter(pl.col("block") == fold)
    cams_train = cams.filter(pl.col("block") != fold)
    leaked = (train["block"] == fold).any() or (train.select((init_block == fold).any()).item())
    if leaked or common.latest_time(times=train["valid_time"]) > common.LAST_TRAINING_VALID:
        msg = "a fold must exclude runs initialised in, and valid times in, the held-out month"
        raise ValueError(msg)
    clims = {
        name: common.climatology(
            frame=frame.select("site", time=pl.col("valid_time"), u=pl.col(column)),
            value="u",
            time_column="time",
            pool=True,
        )
        for name, frame, column in (
            ("ens", train, "u_ens"),
            ("gefs", train, "u_gefs"),
            ("cams", cams_train.rename({"u": "u_cams"}), "u_cams"),
        )
    }
    lookup = {name: clim.select("site", "month", "hh", "clim") for name, clim in clims.items()}
    fit_rows = anomalies(
        rows=train, clim_ens=lookup["ens"], clim_gefs=lookup["gefs"], clim_cams=lookup["cams"]
    ).filter(pl.col("daylight"))
    calibrations = fit_calibrations(rows=fit_rows)
    scored = anomalies(
        rows=test, clim_ens=lookup["ens"], clim_gefs=lookup["gefs"], clim_cams=lookup["cams"]
    ).filter(
        pl.col("daylight")
        & pl.col("month").is_in(list(common.SCORED_MONTHS))
        & pl.col("site").is_in(list(common.VERDICT_SITES))
    )
    scored = apply_calibrations(rows=scored, calibrations=calibrations)
    rows_out = [
        {
            "arm": arm,
            "sse": float(((scored["a_cams"] - scored[f"central_{arm}"]) ** 2).sum()),
            "n": scored.height,
        }
        for arm in SELECTION_ARMS
    ]
    rows_out.append(
        {"arm": "zero", "sse": float((scored["a_cams"] ** 2).sum()), "n": scored.height}
    )
    return pl.DataFrame(rows_out).with_columns(fold=pl.lit(fold, dtype=pl.Int32))


def select_arm(*, errors: pl.DataFrame) -> dict[str, object]:
    """Apply the selection rule to the pooled squared errors of the folds.

    Args:
        errors: `fold_errors()` of every fold, concatenated.

    Returns:
        The selected arm (`ens` unless GEFS or the blend has an error at least 1% lower than
        ENS's), each arm's pooled mean squared error, the ratio to ENS's, and the margin.
    """
    pooled = errors.group_by("arm").agg(sse=pl.col("sse").sum(), n=pl.col("n").sum())
    mse = {row["arm"]: row["sse"] / row["n"] for row in pooled.iter_rows(named=True)}
    best = min(SELECTION_ARMS, key=lambda arm: mse[arm])
    selected = (
        best
        if best != "ens" and mse[best] <= (1.0 - common.SELECTION_MARGIN) * mse["ens"]
        else "ens"
    )
    return {
        "selected": selected,
        "best_by_error": best,
        "mean_squared_error": mse,
        "ratio_to_ens": {arm: mse[arm] / mse["ens"] for arm in mse},
        "margin": common.SELECTION_MARGIN,
        "rows": int(common.scalar(value=pooled["n"].max())),
        "folds": int(errors["fold"].n_unique()),
    }


def run_cv() -> None:
    """Run the 15-fold month-block cross-validation and write `selection.json`."""
    rows = read_training_rows(max_lead_hours=common.SELECTION_MAX_LEAD_HOURS)
    cams = (
        pl.read_parquet(common.SOURCE_SOLAR_DIR / "cams_u.parquet")
        .rename({"time": "valid_time"})
        .filter(pl.col("valid_time") <= common.LAST_TRAINING_VALID)
        .with_columns(block=block_index(times=pl.col("valid_time")))
    )
    print(f"{rows.height:,} shared rows, valid months {sorted(rows['block'].unique().to_list())}")
    folds = range(common.CV_FOLDS)
    errors = pl.concat([fold_errors(rows=rows, cams=cams, fold=fold) for fold in folds])
    with pl.Config(tbl_rows=100):
        print(errors.pivot(on="arm", index="fold", values="sse").sort("fold"))
    selection = select_arm(errors=errors)
    print(json.dumps(selection, indent=2))
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    common.SELECTION_PATH.write_text(json.dumps(selection, indent=2) + "\n")
    common.write_atomically(frame=errors, target=common.RESULTS_DIR / "cv_errors.parquet")


def full_window_calibrations() -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    """Fit the calibrations on the whole training window and return them with the climatologies.

    The ENS and CAMS climatologies and the daylight mask are experiment 00046's, and the GEFS
    climatology is `gefs_climatology.parquet`. The `ens` slopes are experiment 00046's
    `calibration.parquet`, so the `ens` arm is bit for bit experiment 00046's anomaly.

    Returns:
        The calibration table (`fit_calibrations()` columns, with `slope_ens` replaced by
        experiment 00046's) and the climatologies by source.
    """
    table = pl.read_parquet(common.SOURCE_SOLAR_DIR / "climatology.parquet")
    gefs = pl.read_parquet(common.SOLAR_DIR / "gefs_climatology.parquet")
    clims = {
        "ens": table.select("site", "month", "hh", clim=pl.col("clim_ens")),
        "cams": table.select("site", "month", "hh", clim=pl.col("clim_cams")),
        "gefs": gefs.select("site", "month", "hh", clim=pl.col("clim_gefs")),
    }
    rows = read_training_rows(max_lead_hours=None)
    fit_rows = anomalies(
        rows=rows, clim_ens=clims["ens"], clim_gefs=clims["gefs"], clim_cams=clims["cams"]
    )
    fit_rows = fit_rows.filter(pl.col("daylight"))
    calibrations = fit_calibrations(rows=fit_rows)
    source = pl.read_parquet(common.SOURCE_SOLAR_DIR / "calibration.parquet").select(
        "site", "band", slope_source=pl.col("slope")
    )
    calibrations = (
        calibrations.join(source, on=["site", "band"], how="left", validate="1:1")
        .with_columns(
            slope_ens=pl.col("slope_source"),
        )
        .drop("slope_source")
    )
    return calibrations, clims


def validation_centrals(
    *, calibrations: pl.DataFrame, clims: dict[str, pl.DataFrame]
) -> pl.DataFrame:
    """Return every arm's central anomaly on the validation runs.

    The ENS anomaly is the mean over the 51 members of experiment 00046's raw member anomaly. The
    GEFS anomaly is the GEFS ensemble-mean `u` minus the GEFS climatology. Where a run has no GEFS
    value, `has_gefs` is false and the GEFS, blend, and fixed-blend centrals equal the ENS central,
    which is what serving would do with a missing GEFS run.

    Args:
        calibrations: `full_window_calibrations()` table.
        clims: The climatologies by source.

    Returns:
        Columns `site`, `power_fcst_init_time`, `valid_time`, `band`, `has_gefs`, `a_ens`, `a_gefs`,
        and the four `central_*` columns.
    """
    ens = pl.concat(
        [
            pl.read_parquet(path)
            .group_by("site", "power_fcst_init_time", "valid_time")
            .agg(a_ens=pl.col("anomaly").mean())
            for path in sorted((common.SOURCE_SOLAR_DIR / "ens_val").glob("init=*.parquet"))
        ]
    )
    gefs = pl.concat(
        [pl.read_parquet(path) for path in sorted(common.SOLAR_DIR_VAL.glob("init=*.parquet"))]
    )
    gefs = (
        common.with_cell(frame=gefs, time_column="valid_time")
        .join(clims["gefs"], on=["site", "month", "hh"], how="left", validate="m:1")
        .select(
            "site",
            "power_fcst_init_time",
            "valid_time",
            a_gefs=pl.col("u_mean") - pl.col("clim"),
        )
    )
    joined = ens.join(
        gefs,
        on=["site", "power_fcst_init_time", "valid_time"],
        how="left",
        validate="1:1",
    )
    lead = common.lead_hours_of(frame=joined)
    joined = joined.with_columns(
        band=pl.Series(common.lead_band(lead_hours=lead)),
        has_gefs=pl.col("a_gefs").is_not_null() & pl.col("a_gefs").is_finite(),
    )
    centrals = apply_calibrations(
        rows=joined.with_columns(a_gefs=pl.col("a_gefs").fill_null(0.0).fill_nan(0.0)),
        calibrations=calibrations,
    )
    return centrals.with_columns(
        *(
            pl.when(pl.col("has_gefs"))
            .then(pl.col(column))
            .otherwise(pl.col("central_ens"))
            .alias(column)
            for column in ("central_gefs", "central_blend", "central_blend_fixed")
        )
    ).select(
        "site",
        "power_fcst_init_time",
        "valid_time",
        "band",
        "has_gefs",
        "a_ens",
        "a_gefs",
        *ARM_COLUMNS.values(),
    )


def correlation_from_sums(*, sums: np.ndarray) -> np.ndarray:
    """Return the Pearson correlation from sums of (n, sx, sy, sxx, syy, sxy) on the last axis."""
    n, sx, sy, sxx, syy, sxy = (sums[..., i] for i in range(6))
    numerator = n * sxy - sx * sy
    denominator = np.sqrt(np.maximum(n * sxx - sx**2, 0.0) * np.maximum(n * syy - sy**2, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return numerator / denominator


def block_sums(*, x: np.ndarray, y: np.ndarray, blocks: np.ndarray, n_blocks: int) -> np.ndarray:
    """Return the (n, sx, sy, sxx, syy, sxy) sums of x and y in each block, shape (blocks, 6)."""
    out = np.zeros((n_blocks, 6))
    for block in range(n_blocks):
        mask = blocks == block
        bx, by = x[mask], y[mask]
        out[block] = [
            bx.size,
            bx.sum(),
            by.sum(),
            (bx * bx).sum(),
            (by * by).sum(),
            (bx * by).sum(),
        ]
    return out


def correlation_difference_interval(
    *, sums_a: np.ndarray, sums_b: np.ndarray, level: float = 95.0
) -> tuple[float, float, float]:
    """Return corr(a) minus corr(b) and its interval from resampling whole blocks.

    Args:
        sums_a: Per-block sums of arm a, shape (blocks, 6).
        sums_b: Per-block sums of arm b, same blocks.
        level: Interval coverage in percent.

    Returns:
        The difference on all blocks, and the lower and upper bounds.
    """
    generator = np.random.default_rng(SEED)
    n_blocks = sums_a.shape[0]
    draws = generator.integers(0, n_blocks, size=(N_RESAMPLES, n_blocks))
    counts = (draws[:, :, None] == np.arange(n_blocks)).sum(axis=1)
    resampled = correlation_from_sums(sums=counts @ sums_a) - correlation_from_sums(
        sums=counts @ sums_b
    )
    tail = (100.0 - level) / 2.0
    lower, upper = np.nanpercentile(resampled, [tail, 100.0 - tail])
    point = float(
        correlation_from_sums(sums=sums_a.sum(axis=0))
        - correlation_from_sums(sums=sums_b.sum(axis=0))
    )
    return point, float(lower), float(upper)


def metric_rows(*, frame: pl.DataFrame, scope: str, band: str) -> list[dict[str, object]]:
    """Return the correlation, calibration slopes, and error skill of each arm on a set of rows.

    Args:
        frame: Scored rows with `anomaly_cams`, `a_ens`, `a_gefs`, and the `central_*` columns.
        scope: A label for the rows (`pooled` or a site).
        band: A lead band name or `all`.

    Returns:
        One row per arm and metric: `corr`, `slope_calibrated` (the slope through the origin of the
        CAMS anomaly on the calibrated central anomaly; 1 is ideal), `slope_raw` (the same on the
        uncalibrated source anomaly, for the sources that have one), and `mse_skill` (one minus the
        squared error over the squared CAMS anomaly).
    """
    y = frame["anomaly_cams"].to_numpy().astype(np.float64)
    out: list[dict[str, object]] = []
    raw_column = {"ens": "a_ens", "gefs": "a_gefs"}
    for arm, column in ARM_COLUMNS.items():
        x = frame[column].to_numpy().astype(np.float64)
        entries = {
            "corr": float(np.corrcoef(x, y)[0, 1]) if y.size > 2 else float("nan"),
            "slope_calibrated": float((x * y).sum() / (x * x).sum())
            if (x * x).sum()
            else float("nan"),
            "mse_skill": 1.0 - float(((y - x) ** 2).sum() / (y * y).sum()),
        }
        if arm in raw_column:
            raw = frame[raw_column[arm]].to_numpy().astype(np.float64)
            entries["slope_raw"] = float((raw * y).sum() / (raw * raw).sum())
        out.extend(
            {
                "scope": scope,
                "band": band,
                "arm": arm,
                "metric": metric,
                "value": value,
                "n": y.size,
            }
            for metric, value in entries.items()
        )
    return out


def three_hour_correlations(*, scored: pl.DataFrame) -> pl.DataFrame:
    """Return each arm's correlation after averaging to 3-hour windows ending at 3-hour labels.

    This repeats A1 free of any difference in how the sources are resampled to half-hours.

    Args:
        scored: Scored rows with `anomaly_cams` and the `central_*` columns.

    Returns:
        Columns `band`, `arm`, `corr`, and `n` (windows holding all six half-hours).
    """
    windows = (
        scored.with_columns(window=(pl.col("valid_time").dt.epoch("s") + 10_799) // 10_800)
        .group_by("site", "power_fcst_init_time", "window", "band")
        .agg(
            pl.col("anomaly_cams").mean(),
            *(pl.col(column).mean() for column in ARM_COLUMNS.values()),
            n=pl.len(),
        )
        .filter(pl.col("n") == 6)
    )
    rows = []
    for band in [*range(len(BAND_NAMES)), None]:
        part = windows if band is None else windows.filter(pl.col("band") == band)
        for arm, column in ARM_COLUMNS.items():
            rows.append(
                {
                    "band": "all" if band is None else BAND_NAMES[band],
                    "arm": arm,
                    "corr": float(part.select(pl.corr(column, "anomaly_cams")).item()),
                    "n": part.height,
                }
            )
    return pl.DataFrame(rows)


def correlation_intervals(*, verdict_rows: pl.DataFrame) -> pl.DataFrame:
    """Return, by lead band, the difference in correlation of GEFS and the blend from ENS.

    Args:
        verdict_rows: Scored rows of the verdict series with `anomaly_cams`, `init_block` (the
            initialisation month), `band`, and the `central_*` columns.

    Returns:
        One row per contrast and band with the difference and its 95% interval from resampling
        the initialisation months.
    """
    intervals = []
    init_blocks = verdict_rows["init_block"].to_numpy()
    unique_blocks = np.unique(init_blocks)
    remap = {block: i for i, block in enumerate(unique_blocks)}
    mapped = np.array([remap[b] for b in init_blocks])
    y_all = verdict_rows["anomaly_cams"].to_numpy().astype(np.float64)
    for band in [None, *range(len(BAND_NAMES))]:
        mask = (
            np.ones(y_all.size, dtype=bool)
            if band is None
            else verdict_rows["band"].to_numpy() == band
        )
        sums = {
            arm: block_sums(
                x=verdict_rows[column].to_numpy().astype(np.float64)[mask],
                y=y_all[mask],
                blocks=mapped[mask],
                n_blocks=len(unique_blocks),
            )
            for arm, column in ARM_COLUMNS.items()
        }
        for treatment in ("gefs", "blend"):
            point, lower, upper = correlation_difference_interval(
                sums_a=sums[treatment], sums_b=sums["ens"]
            )
            intervals.append(
                {
                    "contrast": f"{treatment} minus ens",
                    "band": "all" if band is None else BAND_NAMES[band],
                    "difference": point,
                    "lower_95": lower,
                    "upper_95": upper,
                    "blocks": len(unique_blocks),
                }
            )
    return pl.DataFrame(intervals)


def shared_rows(*, scored: pl.DataFrame) -> pl.DataFrame:
    """Keep the rows every arm can be scored on, and print how many rows had no GEFS value.

    Args:
        scored: Scored rows with `has_gefs`.

    Returns:
        The rows with a GEFS value, so that no arm is scored on a fallback to ENS.
    """
    shared = scored.filter(pl.col("has_gefs"))
    print(
        f"scored rows {scored.height:,}; rows with no GEFS value dropped so that every arm is "
        f"scored on the same rows: {scored.height - shared.height:,}"
    )
    return shared


def run_validate() -> None:
    """Score the three arms' anomalies on the validation year and write the tables."""
    if not common.SELECTION_PATH.exists():
        msg = "run `cv` first: the selection must be written before the validation scores are read"
        raise FileNotFoundError(msg)
    inputs.require_complete(split="val")
    calibrations, clims = full_window_calibrations()
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    common.write_atomically(
        frame=calibrations.select(
            "site", "band", "slope_ens", "slope_gefs", "w_ens", "w_gefs", "n", "negative_weight"
        ),
        target=common.RESULTS_DIR / "blend_weights.parquet",
    )
    with pl.Config(tbl_rows=60):
        print("Calibrations fitted on the whole training window (blend weights printed)")
        print(calibrations)
        flagged = calibrations.filter(pl.col("negative_weight"))
        print(
            f"blend weights below zero (kept, flagged): {flagged.height} of {calibrations.height}"
        )
    centrals = validation_centrals(calibrations=calibrations, clims=clims)
    common.write_atomically(frame=centrals, target=common.SOLAR_DIR / "central_val.parquet")
    print(
        f"validation rows {centrals.height:,}; without a GEFS value: "
        f"{centrals.filter(~pl.col('has_gefs')).height:,}"
    )
    series = common.series_table().filter(pl.col("site").is_in(common.SITES))
    arrays = common.load_series_arrays(series=series)
    scored = (
        common.scored_rows(series=series, arrays=arrays)
        .filter(pl.col("scored"))
        .select("site", "power_fcst_init_time", "valid_time", "band", "anomaly_cams")
        .join(
            centrals.drop("band"),
            on=["site", "power_fcst_init_time", "valid_time"],
            how="inner",
            validate="m:1",
        )
        .filter(pl.col("anomaly_cams").is_finite())
        .with_columns(init_block=block_index(times=pl.col("power_fcst_init_time")))
    )
    scored = shared_rows(scored=scored)
    rows: list[dict[str, object]] = []
    verdict_rows = scored.filter(pl.col("site").is_in(list(common.VERDICT_SITES)))
    for band in [None, *range(len(BAND_NAMES))]:
        band_name = "all" if band is None else BAND_NAMES[band]
        pooled = verdict_rows if band is None else verdict_rows.filter(pl.col("band") == band)
        rows += metric_rows(frame=pooled, scope="pooled", band=band_name)
        for site in common.SITES:
            own = pooled.filter(pl.col("site") == site)
            if own.height > 2:
                rows += metric_rows(frame=own, scope=site, band=band_name)
    scores = pl.DataFrame(rows)
    common.write_atomically(frame=scores, target=common.RESULTS_DIR / "anomaly_scores.parquet")
    pooled_scores = scores.filter(pl.col("scope") == "pooled")
    with pl.Config(tbl_rows=120, tbl_cols=12):
        for metric in ("corr", "slope_calibrated", "slope_raw", "mse_skill"):
            print(f"\nA: {metric}, pooled over the verdict series, by lead band")
            print(
                pooled_scores.filter(pl.col("metric") == metric).pivot(
                    on="arm", index="band", values="value"
                )
            )
        print("\nRows behind each pooled band")
        print(
            pooled_scores.filter((pl.col("metric") == "corr") & (pl.col("arm") == "ens")).select(
                "band", "n"
            )
        )
    interval_table = correlation_intervals(verdict_rows=verdict_rows)
    unique_blocks = verdict_rows["init_block"].unique()
    common.write_atomically(
        frame=interval_table, target=common.RESULTS_DIR / "anomaly_intervals.parquet"
    )
    with pl.Config(tbl_rows=40):
        print(
            f"\nDifference in correlation, 95% interval from {len(unique_blocks)} init-month blocks"
        )
        print(interval_table)
    three_hour = three_hour_correlations(scored=verdict_rows)
    common.write_atomically(
        frame=three_hour, target=common.RESULTS_DIR / "anomaly_three_hour.parquet"
    )
    with pl.Config(tbl_rows=40):
        print("\nExploratory: correlation of 3-hour mean anomalies")
        print(three_hour.pivot(on="arm", index="band", values="corr"))
    selection = json.loads(common.SELECTION_PATH.read_text())
    print(f"\nstage 1 selection (training window only): {selection['selected']}")


def main() -> None:
    """Run the requested stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("cv", "validate"))
    args = parser.parse_args()
    started = datetime.now(UTC)
    if args.stage == "cv":
        run_cv()
    else:
        run_validate()
    print(f"done in {(datetime.now(UTC) - started).total_seconds():.0f} s")


if __name__ == "__main__":
    main()
