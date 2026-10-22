"""Re-issue experiment 136's and 155's forecasts at the production cadence and write the frames.

**Production issues a forecast every 6 hours from the newest ECMWF run, so the weather input is
12 to 30 hours old.** Experiments 136 and 155 issue one forecast a day, 9 hours after the 00:00 UTC
run. Here each run of those experiments is re-issued at four later origins, 3, 9, 15, and 21 hours
after the original (12:00, 18:00, 00:00, and 06:00 UTC). Each re-issue uses the same run's
ensemble-mean weather and the same frozen GAM fit, which is the newest run available at that origin.
Only two things change. The filter's state is read after the newest half-hour that ended at least
3.5 hours before the new origin, and the lead is counted from the new origin. The raw GAM mean is
experiment 136's stored mean minus its stored level and anomaly, so no GAM is refitted or re-run.

| Arm | Built from | Correction |
|---|---|---|
| `ap` | experiment 136's A2 raw mean | filter state at origin minus 3.5 hours |
| `dp` | `ap` | plus experiment 155's arm D shift (winter night offset and recipe R) |
| `ap0`, `dp0` | as `ap`, `dp` | filter state at the origin itself (no data delay; diagnostic) |
| `ai0`, `di0` | as `ap0`, `dp0` | the same, issued at the original origin (no data delay) |

The shift of arm D is experiment 155's `d` mean minus its `a` mean at the same run and valid time.
It reads only days before the run's original origin, so it is never later than what a re-issue
could see. Each arm's quantiles are its mean plus a spread table refitted on its own tuning-month
forecasts, pooled over the four origins, as experiment 155 refits each arm's spread.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00168-*/e00168_build.py [--workers 3]
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from pathlib import Path
from typing import Final

import e00168_common as common
import e00168_ens as ens
import e00168_filters as filters
import e00168_tall as tall
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.dp_assimilation import (
    DEFAULT_DELAY_HOURS,
    CorrectionTrace,
    correction_at,
    usable_cutoff_positions,
)
from studies.dp_ens import LeadHourSpread
from studies.dp_ens_weather import Calibration, proxy_cache
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_gam import QUANTILE_COLUMNS

OUT_DIR: Final[Path] = common.WORK_DIR / "dp_preds"
PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
SLOT_OFFSET_HOURS: Final[tuple[int, ...]] = (3, 9, 15, 21)
"""Hours after the original origin (09:00 UTC) of the four production origins."""
CONTROL_OFFSET_HOURS: Final[int] = 0
ARMS: Final[tuple[str, ...]] = ("ap", "dp", "ap0", "dp0")
BASE_KEEP: Final[tuple[str, ...]] = (
    "time_series_id", "run", "power_fcst_init_time", "valid_time", "level", "anomaly", "mean",
    "actual", "hour_band", "quarter", "switching_mask",
)  # fmt: skip


def out_path(*, arm: str, period: str, site: str) -> Path:
    """Return the file that holds an arm's forecasts for one period at one site."""
    return OUT_DIR / f"{arm}_{period}_{site}.parquet"


def read_tuned() -> dict:
    """Return experiment 111's tuned settings (solar shrink, filter values, spread factors)."""
    return json.loads((common.SOURCE_WORK_DIR / "tuned.json").read_text())


def load_calibration(*, site: str) -> Calibration:
    """Return experiment 111's stored calibration at a site, without its solar shrink."""
    with np.load(common.SOURCE_WORK_DIR / "calibration" / f"{site}.npz") as stored:
        return Calibration(
            temperature_shift_c=stored["shift"],
            irradiance_ratio=stored["ratio"],
            solar_shrink=stored["shrink"],
        )


def pooled_settings() -> filters.FilterSettings:
    """Return experiment 133's shared filter setting (POOL), as arm A2 uses."""
    hyper = np.array(json.loads(common.POOL_PATH.read_text())["hyper"])
    return filters.FilterSettings(
        q=float(np.exp(hyper[0])),
        phi=float(1.0 / (1.0 + np.exp(-hyper[1]))),
        sigma_z=float(np.exp(hyper[2])),
        sigma_eps=float(np.exp(hyper[3])),
        rho_a=0.0,
    )


def tall_residual_stream(*, cache: FeatureCache, fits: list[tall.TallFit], site: str) -> np.ndarray:
    """Return observed minus the chained T-all fits' forecast on a cache, `NaN` where none.

    Fit `i` supplies the rows after its cut and up to the next cut, so no residual is in-sample.
    """
    if not fits or any(later.cut <= earlier.cut for earlier, later in pairwise(fits)):
        msg = "the residual stream needs fits with strictly ascending cuts"
        raise ValueError(msg)
    features = cache.sites[site]
    stream = np.full(len(cache.end_times), np.nan)
    cuts = [
        int(np.searchsorted(cache.end_times, naive_utc(moment=fit.cut), side="right")) - 1
        for fit in fits
    ]
    ends = [*cuts[1:], len(stream) - 1]
    for fit, cut, end in zip(fits, cuts, ends, strict=True):
        rows = np.arange(cut + 1, end + 1)
        usable = (
            np.isfinite(features.smoothed_temperature_c[rows]).all(axis=1)
            & np.isfinite(features.pv_unit_shape[rows])
            & np.isfinite(features.temperature_c[rows])
            & np.isfinite(features.y[rows])
        )
        rows = rows[usable]
        if len(rows):
            stream[rows] = features.y[rows] - tall.predict_tall(cache=cache, fit=fit, index=rows)
    return stream


def filter_trace(*, cache: FeatureCache, site: str) -> CorrectionTrace:
    """Run arm A2's filter over the site's residual stream and return its trace."""
    calibration = load_calibration(site=site).with_solar_shrink(
        solar_shrink=read_tuned()["solar_shrink"]
    )
    setup = ens.load_setup(cache=cache, site=site).with_calibration(calibration=calibration)
    fits = tall.load_fits(cadence="monthly", site=site)
    proxy = proxy_cache(cache=cache, context=setup.context)
    stream = tall_residual_stream(cache=proxy, fits=fits, site=site)
    return filters.filter_stream(
        residual=stream, band=cache.calendar.band, settings=pooled_settings()
    ).trace


def read_base(*, site: str, period: str) -> pl.DataFrame:
    """Return arm A2's stored rows at a site and period with arm D's shift joined on."""
    a = pl.read_parquet(common.E136_WORK_DIR / "dp_preds" / f"a2_{period}_{site}.parquet")
    d = pl.read_parquet(common.E155_WORK_DIR / "dp_preds" / f"d_{period}_{site}.parquet")
    keys = ["run", "valid_time"]
    joined = a.select([*BASE_KEEP]).join(
        d.select(
            *keys,
            pl.col("mean").alias("mean_d_stored"),
            *[pl.col(q).alias(f"d_{q}") for q in QUANTILE_COLUMNS],
        ),
        on=keys,
        how="inner",
        validate="1:1",
    )
    if joined.height != a.height:
        msg = f"{site} {period}: arm D lacks {a.height - joined.height} of arm A's rows"
        raise ValueError(msg)
    quantiles = a.select(*keys, *[pl.col(q).alias(f"a_{q}") for q in QUANTILE_COLUMNS])
    return (
        joined.join(quantiles, on=keys, how="inner", validate="1:1")
        .with_columns(
            raw=(pl.col("mean").cast(pl.Float64) - pl.col("level") - pl.col("anomaly")),
            shift=(pl.col("mean_d_stored").cast(pl.Float64) - pl.col("mean").cast(pl.Float64)),
        )
        .sort(["run", "valid_time"])
    )


def reissue(
    *,
    base: pl.DataFrame,
    trace: CorrectionTrace,
    cache: FeatureCache,
    offset_hours: int,
    delay_hours: float,
) -> pl.DataFrame:
    """Re-issue every run of `base` at `offset_hours` after its origin, with the filter's state.

    Args:
        base: The stored rows, from `read_base`.
        trace: The filter's trace.
        cache: The feature cache (for the grid).
        offset_hours: Hours after the original origin; the rows that end after the new origin stay.
        delay_hours: The telemetry delay: the filter reads the newest half-hour that ended at least
            this long before the new origin.

    Returns:
        The rows with `origin`, `lead_hours`, `level_new`, `anomaly_new`, and the new means
        `mean_a` (filter only) and `mean_d` (filter plus arm D's shift).
    """
    shifted = base.with_columns(
        origin=pl.col("power_fcst_init_time") + pl.duration(hours=offset_hours)
    ).filter(pl.col("valid_time") > pl.col("origin"))
    origin_us = shifted["origin"].dt.replace_time_zone(None).to_numpy().astype("datetime64[us]")
    valid_us = shifted["valid_time"].dt.replace_time_zone(None).to_numpy().astype("datetime64[us]")
    target = np.searchsorted(cache.end_times, valid_us)
    if not np.array_equal(cache.end_times[target], valid_us):
        msg = "a valid time is not on the grid"
        raise ValueError(msg)
    cutoff = usable_cutoff_positions(
        end_times=cache.end_times, initialisation=origin_us, delay_hours=delay_hours
    )
    correction = correction_at(
        trace=trace,
        cutoff=cutoff,
        target=target,
        band=shifted["hour_band"].to_numpy().astype(np.int64),
    )
    return shifted.with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("origin")).dt.total_minutes() / 60.0,
        level_new=pl.Series(correction.level),
        anomaly_new=pl.Series(correction.anomaly),
    ).with_columns(
        mean_a=pl.col("raw") + pl.col("level_new") + pl.col("anomaly_new"),
        mean_d=pl.col("raw") + pl.col("level_new") + pl.col("anomaly_new") + pl.col("shift"),
    )


def fit_spread(*, frame: pl.DataFrame, mean_column: str) -> LeadHourSpread:
    """Fit the centred spread from a tuning frame's residuals outside the switching masks."""
    kept = frame.filter(~pl.col("switching_mask"))
    return LeadHourSpread.fit(
        residuals=(kept["actual"].cast(pl.Float64) - kept[mean_column]).to_numpy(),
        band=kept["hour_band"].to_numpy().astype(np.int64),
        quarter=kept["quarter"].to_numpy().astype(np.int64),
        lead_hours=kept["lead_hours"].to_numpy(),
    )


def with_quantiles(
    *, frame: pl.DataFrame, mean_column: str, spread: LeadHourSpread
) -> pl.DataFrame:
    """Return the frame with the 11 quantiles around its mean."""
    quantiles = spread.quantiles(
        mean=frame[mean_column].to_numpy(),
        band=frame["hour_band"].to_numpy().astype(np.int64),
        quarter=frame["quarter"].to_numpy().astype(np.int64),
        lead_hours=frame["lead_hours"].to_numpy(),
    )
    return frame.with_columns(
        pl.Series(column, quantiles[:, position].astype(np.float32))
        for position, column in enumerate(QUANTILE_COLUMNS)
    )


def control(*, base: dict[str, pl.DataFrame], trace: CorrectionTrace, cache: FeatureCache) -> dict:
    """Check that an offset of zero reproduces experiments 136's and 155's stored forecasts.

    The filter part must reproduce arm A's stored mean, arm D's stored mean, and, with a spread
    refitted on the tuning months, the stored quantiles.
    """
    issued = {
        period: reissue(
            base=base[period],
            trace=trace,
            cache=cache,
            offset_hours=CONTROL_OFFSET_HOURS,
            delay_hours=DEFAULT_DELAY_HOURS,
        )
        for period in PERIODS
    }
    out: dict = {}
    scored = issued["scored"]
    out["rows"] = scored.height
    out["max_abs_mean_a"] = float((scored["mean_a"] - scored["mean"]).abs().max())  # ty: ignore[invalid-argument-type]
    out["max_abs_mean_d"] = float((scored["mean_d"] - scored["mean_d_stored"]).abs().max())  # ty: ignore[invalid-argument-type]
    for arm, mean_column, prefix in (("a", "mean_a", "a_"), ("d", "mean_d", "d_")):
        spread = fit_spread(frame=issued["tuning"], mean_column=mean_column)
        rebuilt = with_quantiles(frame=scored, mean_column=mean_column, spread=spread)
        out[f"max_abs_quantile_{arm}"] = max(
            float((rebuilt[q] - scored[f"{prefix}{q}"]).abs().max())  # ty: ignore[invalid-argument-type]
            for q in QUANTILE_COLUMNS
        )
    return out


def _write(*, arm: str, period: str, site: str, frame: pl.DataFrame) -> None:
    target = out_path(arm=arm, period=period, site=site)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    frame.write_parquet(temporary)
    temporary.rename(target)


OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id", "run", "slot", "run_origin", "power_fcst_init_time", "valid_time",
    "lead_hours", "seed", "arm", *QUANTILE_COLUMNS, "actual", "mean", "hour_band", "quarter",
    "switching_mask",
)  # fmt: skip


def build_site(*, site: str) -> str:
    """Build every arm at one substation, check the control, and write the frames."""
    torch.set_num_threads(1)
    started = time.monotonic()
    cache = common.load_cache()
    trace = filter_trace(cache=cache, site=site)
    base = {period: read_base(site=site, period=period) for period in PERIODS}
    checks = control(base=base, trace=trace, cache=cache)
    common.write_json_atomic(
        path=common.WORK_DIR / "control" / f"{site}.json", content={"site": site, **checks}
    )
    rows = 0
    for arm, mean_column, delay, offsets in (
        ("ap", "mean_a", DEFAULT_DELAY_HOURS, SLOT_OFFSET_HOURS),
        ("dp", "mean_d", DEFAULT_DELAY_HOURS, SLOT_OFFSET_HOURS),
        ("ap0", "mean_a", 0.0, SLOT_OFFSET_HOURS),
        ("dp0", "mean_d", 0.0, SLOT_OFFSET_HOURS),
        ("ai0", "mean_a", 0.0, (CONTROL_OFFSET_HOURS,)),
        ("di0", "mean_d", 0.0, (CONTROL_OFFSET_HOURS,)),
    ):
        issued = {}
        for period in PERIODS:
            pieces = [
                reissue(
                    base=base[period],
                    trace=trace,
                    cache=cache,
                    offset_hours=hours,
                    delay_hours=delay,
                ).with_columns(slot=pl.lit(slot, dtype=pl.Int8))
                for slot, hours in enumerate(offsets)
            ]
            issued[period] = pl.concat(pieces)
        spread = fit_spread(frame=issued["tuning"], mean_column=mean_column)
        for period, frame in issued.items():
            final = (
                with_quantiles(frame=frame, mean_column=mean_column, spread=spread)
                .with_columns(
                    run_origin=pl.col("power_fcst_init_time"),
                    seed=pl.lit(0, dtype=pl.Int8),
                    arm=pl.lit(arm),
                    mean=pl.col(mean_column).cast(pl.Float32),
                )
                .drop("power_fcst_init_time")
                .rename({"origin": "power_fcst_init_time"})
                .select(list(OUTPUT_COLUMNS))
            )
            _write(arm=arm, period=period, site=site, frame=final)
            rows += final.height
    return f"{site}: {rows:,} rows, control {checks}, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Build every arm at every substation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--sites", nargs="+", default=list(common.SCORED_SITES))
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_build, args.sites):
            print(line, flush=True)


def _build(site: str) -> str:
    return build_site(site=site)


if __name__ == "__main__":
    main()
