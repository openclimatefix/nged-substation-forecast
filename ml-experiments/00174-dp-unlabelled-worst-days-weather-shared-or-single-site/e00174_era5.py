"""Re-run arm D's forecast with past weather in place of the ensemble mean (experiment 174).

Arm D is experiment 136's A2 (T-all monthly fits, the Kalman filter's level and anomaly) plus
experiment 155's winter night offset and 14-day mean-error profile. Here the T-all fit of each run
predicts from the feature cache's own ERA5 temperature and CAMS irradiance at the valid times, in
place of the calibrated ensemble mean. The filter's level and anomaly are the stored ones. The
offset and the profile are rebuilt from the new forecast's own errors with experiment 155's code.

| Arm | Weather | Spread |
|---|---|---|
| `ens_d` | ensemble mean | refitted on the arm's tuning months (reproduces experiment 155's D) |
| `era5_d` | ERA5 and CAMS | refitted on the arm's tuning months |
| `era5_d_spread_a` | ERA5 and CAMS | the ensemble arm A's spread table (the quantiles shift only) |
| `era5f_d` | ERA5 and CAMS | refitted; the Kalman filter also runs on the ERA5-driven residuals |

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00174-*/e00174_era5.py [--workers 4]
"""

import argparse
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import e00174_c136 as c136
import e00174_c155 as c155
import e00174_common as common
import e00174_corr as corr
import e00174_design  # noqa: F401  (the unpickler resolves fits through this module)
import e00174_filters as filters
import e00174_tall as tall
import numpy as np
import polars as pl
from studies.dp_assimilation import DEFAULT_DELAY_HOURS, correction_at, usable_cutoff_positions
from studies.dp_ens import choose_fit
from studies.dp_features import FeatureCache, naive_utc

ARMS = ("ens_d", "era5_d", "era5_d_spread_a", "era5f_d")


class _RenamingUnpickler(pickle.Unpickler):
    """Unpickle experiment 136's fits with this folder's copies of its design and tall modules."""

    def find_class(self, module: str, name: str) -> type:
        module = {"e00136_design": "e00174_design", "e00136_tall": "e00174_tall"}.get(
            module, module
        )
        return super().find_class(module, name)


def load_fits(*, site: str) -> list[tall.TallFit]:
    """Return experiment 136's monthly T-all fits at a site, ascending by cut."""
    path = common.E136_WORK_DIR / "tall_fits" / "monthly" / f"{site}.pkl"
    with path.open("rb") as handle:
        return _RenamingUnpickler(handle).load()


def read_ens_base(*, site: str) -> pl.DataFrame:
    """Return arm A's frame at a site with the filter's `level` and `anomaly`, all periods."""
    pieces = [
        pl.read_parquet(common.E136_WORK_DIR / "dp_preds" / f"a2_{period}_{site}.parquet")
        .select([*c155.BASE_COLUMNS, "level", "anomaly"])
        .with_columns(period=pl.lit(period))
        for period in c155.PERIODS
    ]
    return pl.concat(pieces).sort(["power_fcst_init_time", "valid_time"])


def era5_mean(*, base: pl.DataFrame, cache: FeatureCache, site: str) -> np.ndarray:
    """Return the T-all mean forecast driven by the cache's ERA5 and CAMS, plus the filter terms."""
    fits = load_fits(site=site)
    rows = np.searchsorted(
        cache.end_times, base["valid_time"].dt.replace_time_zone(None).to_numpy()
    )
    if not np.array_equal(
        cache.end_times[rows], base["valid_time"].dt.replace_time_zone(None).to_numpy()
    ):
        msg = f"{site}: a forecast row is not on the feature cache's grid"
        raise ValueError(msg)
    origins = base["power_fcst_init_time"].to_list()
    raw = np.full(base.height, np.nan)
    start = 0
    while start < base.height:
        stop = start
        while stop < base.height and origins[stop] == origins[start]:
            stop += 1
        fit = choose_fit(fits=fits, origin=origins[start])  # ty: ignore[invalid-argument-type]
        raw[start:stop] = tall.predict_tall(cache=cache, fit=fit, index=rows[start:stop])  # ty: ignore[invalid-argument-type]
        start = stop
    level = base["level"].to_numpy().astype(np.float64)
    anomaly = base["anomaly"].to_numpy().astype(np.float64)
    return raw + level + anomaly


def tall_residual_stream(*, cache: FeatureCache, fits: list[tall.TallFit], site: str) -> np.ndarray:
    """Return observed minus the chained T-all fits' forecast on a cache, `NaN` where none.

    Fit `i` supplies the rows after its cut and up to the next cut, so no residual is in-sample.
    This is experiment 136's `tall_residual_stream`.
    """
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


def era5_filter_terms(*, base: pl.DataFrame, cache: FeatureCache, site: str) -> np.ndarray:
    """Return the filter's `level + anomaly` per row, with the filter run on ERA5 residuals."""
    fits = load_fits(site=site)
    hyper = np.array(json.loads(c136.POOL_PATH.read_text())["hyper"])
    settings = filters.FilterSettings(
        q=float(np.exp(hyper[0])),
        phi=float(1.0 / (1.0 + np.exp(-hyper[1]))),
        sigma_z=float(np.exp(hyper[2])),
        sigma_eps=float(np.exp(hyper[3])),
        rho_a=0.0,
    )
    run = filters.filter_stream(
        residual=tall_residual_stream(cache=cache, fits=fits, site=site),
        band=cache.calendar.band,
        settings=settings,
    )
    rows = np.searchsorted(
        cache.end_times, base["valid_time"].dt.replace_time_zone(None).to_numpy()
    )
    origins = base["power_fcst_init_time"].to_list()
    out = np.zeros(base.height)
    start = 0
    while start < base.height:
        stop = start
        while stop < base.height and origins[stop] == origins[start]:
            stop += 1
        cutoff = usable_cutoff_positions(
            end_times=cache.end_times,
            initialisation=np.array([naive_utc(moment=origins[start])]),
            delay_hours=DEFAULT_DELAY_HOURS,
        )
        target = rows[start:stop]
        correction = correction_at(
            trace=run.trace,
            cutoff=np.full(len(target), cutoff[0]),
            target=target,
            band=cache.calendar.band[target].astype(np.int64),
        )
        out[start:stop] = correction.level + correction.anomaly
        start = stop
    return out


def d_arm(*, base: pl.DataFrame) -> pl.DataFrame:
    """Return arm D built from a base frame, as experiment 155's `build_arms` builds it."""
    base = corr.add_calendar(frame=base)
    mean_a = base["mean"].to_numpy().astype(np.float64)
    offset, _ = corr.winter_offset_shift(frame=base, variant="winter")
    mean_b = mean_a + offset
    shift_d = offset + corr.recipe_r_shift(frame=base, forecast=mean_b)
    return corr.apply_shift_and_refit_spread(frame=base, shift=shift_d)


def d_arm_with_spread_a(*, base: pl.DataFrame, ens_base: pl.DataFrame) -> pl.DataFrame:
    """Return arm D's mean on `base`, with the ensemble arm A's quantile offsets about the mean."""
    built = d_arm(base=base)
    return ens_base.with_columns(
        pl.Series("mean", built["mean"].to_numpy()),
        *[
            (pl.col(q) + (pl.Series(built["mean"]) - pl.col("mean"))).cast(pl.Float32)
            for q in c155.QUANTILES
        ],
    )


def _write(*, arm: str, site: str, frame: pl.DataFrame) -> None:
    for period in c155.PERIODS:
        target = common.preds_path(arm=arm, period=period, site=site)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        frame.filter(pl.col("period") == period).write_parquet(temporary)
        temporary.rename(target)


def _job(site: str) -> str:
    started = time.monotonic()
    cache = c155.load_cache()
    ens_base = read_ens_base(site=site)
    mean_era5 = era5_mean(base=ens_base, cache=cache, site=site)
    gap = np.abs(mean_era5 - ens_base["mean"].to_numpy())
    era5_base = ens_base.with_columns(
        pl.Series("mean", mean_era5.astype(np.float32)),
        *[
            (pl.col(q) + (pl.Series(mean_era5) - pl.col("mean"))).cast(pl.Float32)
            for q in c155.QUANTILES
        ],
    )
    ens_a = ens_base
    raw = mean_era5 - ens_base["level"].to_numpy() - ens_base["anomaly"].to_numpy()
    mean_f = raw + era5_filter_terms(base=ens_base, cache=cache, site=site)
    era5f_base = ens_base.with_columns(
        pl.Series("mean", mean_f.astype(np.float32)),
        *[
            (pl.col(q) + (pl.Series(mean_f) - pl.col("mean"))).cast(pl.Float32)
            for q in c155.QUANTILES
        ],
    )
    ens_d = d_arm(base=ens_base)
    era5_d = d_arm(base=era5_base)
    era5_d_spread_a = d_arm_with_spread_a(base=era5_base, ens_base=ens_a)
    era5f_d = d_arm(base=era5f_base)
    for arm, frame in (
        ("ens_d", ens_d),
        ("era5_d", era5_d),
        ("era5_d_spread_a", era5_d_spread_a),
        ("era5f_d", era5f_d),
    ):
        _write(arm=arm, site=site, frame=frame)
    return (
        f"{site}: {ens_base.height:,} rows, mean |ERA5 - ENS| mean forecast "
        f"{np.nanmean(gap):.3f} points, {time.monotonic() - started:.0f} s"
    )


def main() -> None:
    """Build the three arms at every substation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--sites", nargs="+", default=list(c155.SCORED_SITES))
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_job, args.sites):
            print(line, flush=True)


if __name__ == "__main__":
    main()
