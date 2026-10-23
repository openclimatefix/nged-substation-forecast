# ruff: noqa: E501
"""Rebuild Primary demand research default 1's convex chain out to 240 hours, one job per site.

The chain is experiment 136's arm A2 (convex model, Kalman filter), experiment 155's arm D (winter
night offset and recipe R), experiment 160's arm I (spread table read by interpolation between
lead-band centres), and experiment 164's fix O (upper offsets widened on forecast-cold rows).
Experiments 136 to 164 stop at 168 hours. Here the forecast window is set to 480 half-hours at run
time (`studies.dp_ens_weather.HORIZON_HALF_HOURS`), the spread table gets one lead band per day
for 168 to 240 hours, and fix O reuses experiment 164's fitted multipliers. Below 168 hours each
stage must equal the stored forecasts (the gates). The weather calibration keeps its last lead
band for every lead above 144 hours.

    PYTHONPATH=ml-experiments/00136-dp-bg-stage1-assembly-on-the-ensemble-mean:ml-experiments/00155-dp-bg-stage2-assembly-night-offset-and-error-correction \
    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00317-*/e00317_chain.py [--workers 6]
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from typing import Final

import e00136_run as run136  # ty: ignore[unresolved-import]
import e00155_common as c155  # ty: ignore[unresolved-import]
import e00155_corrections as corr  # ty: ignore[unresolved-import]
import e00317_common as common
import numpy as np
import polars as pl
import studies.dp_ens_weather as weather
import torch  # ty: ignore[unresolved-import]
from studies.dp_assimilation import DEFAULT_DELAY_HOURS
from studies.dp_gam import ResidualSpread

BASE_COLUMNS: Final[tuple[str, ...]] = c155.BASE_COLUMNS


def lead_band(*, lead_hours: np.ndarray) -> np.ndarray:
    """Return the table's lead band (0 to 6) of leads in hours."""
    edges = np.array(common.LEAD_BAND_UPPER_HOURS)
    return np.minimum(np.searchsorted(edges, lead_hours, side="left"), common.N_LEAD_BANDS - 1)


def fit_table(*, frame: pl.DataFrame, mean: np.ndarray) -> np.ndarray:
    """Return the centred residual table, shape (lead band, hour band, quarter, 11), of the tuning rows."""
    keep = (
        (frame["period"] == "tuning").to_numpy()
        & ~frame["switching_mask"].to_numpy()
        & np.isfinite(frame["actual"].to_numpy())
    )
    residual = (frame["actual"].to_numpy().astype(np.float64) - mean)[keep]
    band = frame["hour_band"].to_numpy().astype(np.int64)[keep]
    quarter = frame["quarter"].to_numpy().astype(np.int64)[keep]
    lead = lead_band(lead_hours=frame["lead_hours"].to_numpy()[keep])
    tables = []
    for index in range(common.N_LEAD_BANDS):
        chosen = lead == index
        if not chosen.any():
            msg = f"lead band {index} holds no tuning residual"
            raise ValueError(msg)
        tables.append(
            ResidualSpread.fit(
                residuals=residual[chosen], band=band[chosen], quarter=quarter[chosen], centre=True
            ).table
        )
    return np.stack(tables)


def interpolated_offsets(
    *, table: np.ndarray, band: np.ndarray, quarter: np.ndarray, lead_hours: np.ndarray
) -> np.ndarray:
    """Return each row's offsets, interpolated between the knots; below 168 h this is arm I of 160."""
    centres = np.array(common.LEAD_BAND_CENTRES_HOURS)
    index = np.array(common.KNOT_TABLE_INDEX)
    low = np.clip(np.searchsorted(centres, lead_hours, side="right") - 1, 0, len(centres) - 2)
    weight = np.clip((lead_hours - centres[low]) / (centres[low + 1] - centres[low]), 0.0, 1.0)
    return (1.0 - weight)[:, None] * table[index[low], band, quarter] + weight[:, None] * table[
        index[low + 1], band, quarter
    ]


def ensemble_temperature(*, frame: pl.DataFrame, site: str) -> np.ndarray:
    """Return each row's ensemble-mean 24-hour temperature (experiment 164's definition)."""
    with np.load(run136.common.STORE_DIR / f"{site}.npz") as stored:
        origins = stored["origins"]
        temperature = stored["temperature"].astype(np.float64)
    finite = np.isfinite(temperature)
    cumulative = np.zeros((*temperature.shape[:2], temperature.shape[2] + 1))
    count = np.zeros_like(cumulative)
    np.cumsum(np.where(finite, temperature, 0.0), axis=2, out=cumulative[:, :, 1:])
    np.cumsum(finite, axis=2, out=count[:, :, 1:])
    run = frame["run"].to_numpy().astype(np.int64)
    origin = frame["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy()
    if not (origins[run] == origin).all():
        msg = f"{site}: the run column does not index experiment 111's store"
        raise ValueError(msg)
    valid = frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    position = ((valid - origin) // np.timedelta64(30, "m")).astype(np.int64) - 1
    last = temperature.shape[2] - 1
    low = np.clip(position - common.WINDOW_HALF_HOURS, 0, last)
    high = np.clip(position + common.WINDOW_HALF_HOURS, 0, last)
    total = cumulative[run, :, high + 1] - cumulative[run, :, low]
    number = count[run, :, high + 1] - count[run, :, low]
    with np.errstate(invalid="ignore", divide="ignore"):
        per_member = np.where(number > 0, total / number, np.nan)
    return np.nanmean(per_member, axis=1)


def forecast_base(*, site: str) -> pl.DataFrame:
    """Return arm A2's frames (mean and keys) for the three periods out to 240 hours."""
    context = run136.load_site(site=site)
    forecaster = run136.ArmForecaster(arm="a2", context=context)
    first = forecaster.fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
    pieces = []
    for period in common.PERIODS:
        runs = [
            i for i in context.setup.indices(periods={period}) if context.setup.origins[i] >= first
        ]
        if runs:
            pieces.append(forecaster.forecast(runs=runs).with_columns(period=pl.lit(period)))
    return (
        pl.concat(pieces)
        .select([*BASE_COLUMNS, "period"])
        .sort(["power_fcst_init_time", "valid_time"])
    )


def offset_shift_fitted_short(*, frame: pl.DataFrame) -> np.ndarray:
    """Return experiment 155's winter night offset, fitted on rows of 168 hours or less.

    The ridge fit of experiment 155 uses every night row, so adding rows beyond 168 hours would
    change the coefficients and with them the forecasts below 168 hours. The coefficients here come
    from the rows of 168 hours or less only, and the shift is then applied to every row.
    """
    short = frame.filter(pl.col("lead_hours") <= common.STORED_HORIZON_HOURS)
    _, betas = corr.winter_offset_shift(frame=short, variant="winter")
    x, keep = corr.night_design(frame=frame, variant="winter")
    night = frame.filter(pl.Series(keep))
    month = night["power_fcst_init_time"].dt.truncate("1mo").to_numpy()
    shift_night = np.zeros(night.height)
    for cut, beta in betas.items():
        this = month == cut
        shift_night[this] = x[this] @ beta
    shift = np.zeros(frame.height)
    shift[np.flatnonzero(keep)] = shift_night
    return shift


def gate_against(*, built: pl.DataFrame, stored_path_of, columns: tuple[str, ...]) -> dict:  # noqa: ANN001
    """Return the largest absolute difference to a stored stage on its rows, and the row counts."""
    stored = pl.concat(
        [
            pl.read_parquet(stored_path_of(period)).select(["run", "valid_time", *columns])
            for period in common.PERIODS
        ]
    )
    short = built.filter(pl.col("lead_hours") <= common.STORED_HORIZON_HOURS)
    joined = short.select(["run", "valid_time", *columns]).join(
        stored, on=["run", "valid_time"], how="full", suffix="_stored", coalesce=True
    )
    both = joined.drop_nulls()
    missing = joined.height - both.height
    diffs = {
        c: float(
            np.abs(
                both[c].to_numpy().astype(np.float64)
                - both[f"{c}_stored"].to_numpy().astype(np.float64)
            ).max()
        )
        for c in columns
    }
    return {
        "max_abs_diff": diffs,
        "rows_built": short.height,
        "rows_stored": stored.height,
        "rows_unmatched": missing,
    }


def job(site: str) -> str:
    """Build the chain at one substation, run its gates, and write its frames."""
    torch.set_num_threads(1)
    started = time.monotonic()
    weather.HORIZON_HALF_HOURS = common.HORIZON_HALF_HOURS
    base = forecast_base(site=site)
    gates: dict = {
        "a2": gate_against(
            built=base,
            stored_path_of=lambda p: common.E136_PREDS / f"a2_{p}_{site}.parquet",
            columns=("mean",),
        )
    }
    # Experiment 155's arm D: winter night offset, then recipe R on the offset forecast's own errors.
    frame = corr.add_calendar(frame=base)
    mean_a = frame["mean"].to_numpy().astype(np.float64)
    offset = offset_shift_fitted_short(frame=frame)
    mean_b = mean_a + offset
    shift_d = offset + corr.recipe_r_shift(frame=frame, forecast=mean_b)
    mean_d = mean_a + shift_d
    frame = frame.with_columns(mean_d=pl.Series(mean_d))
    gates["arm_d"] = gate_against(
        built=frame.with_columns(mean=pl.col("mean_d")),
        stored_path_of=lambda p: common.E155_PREDS / f"d_{p}_{site}.parquet",
        columns=("mean",),
    )
    # Experiment 160's arm I with the table of arm A2 extended to seven lead bands.
    table = fit_table(frame=frame, mean=mean_a)
    stored_table = np.load(common.E160_WORK / f"table_{site}.npy")
    gates["table_first_four_bands"] = float(
        np.abs(table[:4].transpose(1, 2, 0, 3) - stored_table).max()
    )
    band = frame["hour_band"].to_numpy().astype(np.int64)
    quarter = frame["quarter"].to_numpy().astype(np.int64)
    lead = frame["lead_hours"].to_numpy().astype(np.float64)
    offsets = interpolated_offsets(table=table, band=band, quarter=quarter, lead_hours=lead)
    quantiles_i = mean_d[:, None] + offsets
    built_i = frame.with_columns(
        *[
            pl.Series(q, quantiles_i[:, k].astype(np.float32))
            for k, q in enumerate(common.QUANTILES)
        ],
        pl.Series("mean", mean_d.astype(np.float32)),
    )
    gates["arm_i"] = gate_against(
        built=built_i,
        stored_path_of=lambda p: common.E160_WORK / "dp_preds" / f"i_{p}_{site}.parquet",
        columns=common.QUANTILES,
    )
    # Experiment 164's fix O: stored multipliers on the upper offsets of forecast-cold rows.
    multipliers = json.loads((common.E164_WORK / "fit_parameters.json").read_text())["o"]
    ens_mean_c = ensemble_temperature(frame=frame, site=site)
    cold = np.nan_to_num(ens_mean_c, nan=99.0) < common.COLD_GATE_C
    scale = np.ones_like(quantiles_i)
    for k, level in enumerate(common.LEVELS.tolist()):
        if level in common.UPPER_FIX_LEVELS:
            scale[:, k] = np.where(cold, multipliers[str(level)], 1.0)
    quantiles_o = np.sort(mean_d[:, None] + offsets * scale, axis=1)
    built_o = built_i.with_columns(
        *[
            pl.Series(q, quantiles_o[:, k].astype(np.float32))
            for k, q in enumerate(common.QUANTILES)
        ],
        ens_mean_c=pl.Series(ens_mean_c),
    )
    gates["arm_o"] = gate_against(
        built=built_o,
        stored_path_of=lambda p: common.E164_WORK / "dp_preds" / f"o_{p}_{site}.parquet",
        columns=common.QUANTILES,
    )
    target = common.CHAIN_DIR / f"{site}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    built_o.drop("mean_d").write_parquet(temporary)
    temporary.rename(target)
    np.save(common.CHAIN_DIR / f"table_{site}.npy", table)
    common.write_json_atomic(path=common.CHAIN_DIR / f"gates_{site}.json", content=gates)
    return f"{site}: {built_o.height:,} rows, {time.monotonic() - started:.0f} s, gates {json.dumps(gates)[:600]}"


def main() -> None:
    """Build the chain at every substation in parallel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--sites", nargs="+", default=list(common.SCORED_SITES))
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(job, args.sites):
            print(line, flush=True)


if __name__ == "__main__":
    main()
