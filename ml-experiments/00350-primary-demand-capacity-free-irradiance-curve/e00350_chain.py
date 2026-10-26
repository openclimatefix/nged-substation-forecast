# ruff: noqa: E501
"""Run the convex chain of Primary demand research default 3 on each arm's fits, one job per site.

The chain is experiment 317's: the convex fits driven by the calibrated ensemble mean (experiment
136's arm A2, Kalman filter with the POOL noise), experiment 155's arm D (winter night offset and
recipe R), experiment 160's arm I (the spread table read by interpolation between lead-band
centres), and experiment 164's fix O (upper offsets widened on forecast-cold rows, with
experiment 164's stored multipliers). Only the convex fits differ between the arms (see
`e00350_fits.py`); the filter, the offset, recipe R, and the spread table are refitted on each
arm's own residuals. The neural-network head, the solar summary, and the heuristic blend are not
refitted.

Arm A is also the reproduction gate: its stage outputs below 168 hours must equal the stored
forecasts of experiments 136, 155, 160, and 164, and the first 168 hours of experiment 317's chain.

    PYTHONPATH=ml-experiments/00136-dp-bg-stage1-assembly-on-the-ensemble-mean:ml-experiments/00155-dp-bg-stage2-assembly-night-offset-and-error-correction:ml-experiments/00317-primary-demand-default-1-at-7-to-10-days:ml-experiments/00350-primary-demand-capacity-free-irradiance-curve \
    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00350-*/e00350_chain.py [--workers 6]
"""

import argparse
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from multiprocessing import get_context

import e00136_ens as ens  # ty: ignore[unresolved-import]
import e00136_filters as filters  # ty: ignore[unresolved-import]
import e00136_run as run136  # ty: ignore[unresolved-import]
import e00136_tall as tall  # ty: ignore[unresolved-import]
import e00155_corrections as corr  # ty: ignore[unresolved-import]
import e00317_chain as chain317  # ty: ignore[unresolved-import]
import e00317_common as c317  # ty: ignore[unresolved-import]
import e00350_common as common
import e00350_curve as curve
import numpy as np
import polars as pl
import studies.dp_ens_weather as weather
import torch  # ty: ignore[unresolved-import]
from studies.dp_assimilation import DEFAULT_DELAY_HOURS
from studies.dp_ens_weather import proxy_cache
from studies.dp_features import FeatureCache, naive_utc


def predict_any(*, driven: FeatureCache, fit, index: np.ndarray) -> np.ndarray:  # noqa: ANN001
    """Predict net power in points with a T-all fit (arm A) or a curve fit (arms B and C)."""
    if isinstance(fit, curve.CurveFit):
        return curve.predict_curve(cache=driven, fit=fit, index=index)
    return tall.predict_tall(cache=driven, fit=fit, index=index)


ens.predict_mean = predict_any  # `forecast_mean_run` looks `predict_mean` up in the module


def load_fits(*, arm: str, site: str) -> list:
    """Return an arm's stored monthly fits at a site, ascending by cut."""
    return pickle.loads((common.FITS_DIR / arm / f"{site}.pkl").read_bytes())


def residual_stream(*, cache: FeatureCache, fits: list, site: str) -> np.ndarray:
    """Return observed minus the chained fits' forecast on the proxy cache, `NaN` where none.

    Fit `i` supplies the rows after its cut and up to the next cut, as experiment 136's
    `tall_residual_stream` does, so no residual is in-sample.
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
            stream[rows] = features.y[rows] - predict_any(driven=cache, fit=fit, index=rows)
    return stream


class ArmForecaster:
    """Forecasts one arm at one substation: filter on the arm's residuals, spread on tuning months."""

    def __init__(self, *, arm: str, context: run136.SiteContext) -> None:
        """Run the filter on the arm's residual stream and fit the spread on the tuning months."""
        self.arm = arm
        self.context = context
        self.fits = load_fits(arm=arm, site=context.site)
        proxy = proxy_cache(cache=context.cache, context=context.setup.context)
        stream = residual_stream(cache=proxy, fits=self.fits, site=context.site)
        self.run = filters.filter_stream(
            residual=stream,
            band=context.cache.calendar.band,
            settings=run136.settings_of_hyper(
                hyper=run136.noise_hyper(noise="pool", site=context.site)
            ),
        )
        first = self.fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
        self.tuning = [
            i
            for i in context.setup.indices(periods={"tuning"})
            if context.setup.origins[i] >= first
        ]
        self.spread = ens.fit_spread_from_frame(
            frame=self._frame(runs=self.tuning, spread=ens.ZERO_SPREAD)
        )

    def _frame(self, *, runs: list[int], spread) -> pl.DataFrame:  # noqa: ANN001
        return ens.forecast_frame(
            cache=self.context.cache,
            setup=self.context.setup,
            fits=self.fits,
            spread=spread,
            trace=self.run.trace,
            run_indices=runs,
            arm=self.arm,
        )

    def forecast(self, *, runs: list[int]) -> pl.DataFrame:
        """Forecast the runs with the fitted spread, every factor 1."""
        return self._frame(runs=runs, spread=self.spread)


def forecast_base(*, arm: str, site: str) -> pl.DataFrame:
    """Return the arm's convex-stage frames (mean and keys) for the three periods to 240 hours."""
    context = run136.load_site(site=site)
    forecaster = ArmForecaster(arm=arm, context=context)
    first = forecaster.fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
    pieces = []
    for period in c317.PERIODS:
        runs = [
            i for i in context.setup.indices(periods={period}) if context.setup.origins[i] >= first
        ]
        if runs:
            pieces.append(forecaster.forecast(runs=runs).with_columns(period=pl.lit(period)))
    return (
        pl.concat(pieces)
        .select([*chain317.BASE_COLUMNS, "period"])
        .sort(["power_fcst_init_time", "valid_time"])
    )


def job(job_spec: tuple[str, str]) -> str:
    """Build the chain of one arm at one site, run its gates (arm A), and write its frame."""
    arm, site = job_spec
    torch.set_num_threads(1)
    started = time.monotonic()
    weather.HORIZON_HALF_HOURS = c317.HORIZON_HALF_HOURS
    base = forecast_base(arm=arm, site=site)
    gates: dict = {}
    if arm == "A":
        stored = pickle.loads(tall.fit_path(cadence="monthly", site=site).read_bytes())
        refit = load_fits(arm="A", site=site)
        gates["theta_max_abs_diff_to_stored_fits"] = max(
            float(np.abs(a.fit.theta - b.fit.theta).max())
            for a, b in zip(refit, stored, strict=True)
        )
        gates["a2"] = chain317.gate_against(
            built=base,
            stored_path_of=lambda p: c317.E136_PREDS / f"a2_{p}_{site}.parquet",
            columns=("mean",),
        )
    frame = corr.add_calendar(frame=base)
    mean_a = frame["mean"].to_numpy().astype(np.float64)
    offset = chain317.offset_shift_fitted_short(frame=frame)
    mean_b = mean_a + offset
    shift_d = offset + corr.recipe_r_shift(frame=frame, forecast=mean_b)
    mean_d = mean_a + shift_d
    frame = frame.with_columns(mean_d=pl.Series(mean_d))
    if arm == "A":
        gates["arm_d"] = chain317.gate_against(
            built=frame.with_columns(mean=pl.col("mean_d")),
            stored_path_of=lambda p: c317.E155_PREDS / f"d_{p}_{site}.parquet",
            columns=("mean",),
        )
    table = chain317.fit_table(frame=frame, mean=mean_a)
    band = frame["hour_band"].to_numpy().astype(np.int64)
    quarter = frame["quarter"].to_numpy().astype(np.int64)
    lead = frame["lead_hours"].to_numpy().astype(np.float64)
    offsets = chain317.interpolated_offsets(
        table=table, band=band, quarter=quarter, lead_hours=lead
    )
    quantiles_i = mean_d[:, None] + offsets
    built_i = frame.with_columns(
        *[pl.Series(q, quantiles_i[:, k].astype(np.float32)) for k, q in enumerate(c317.QUANTILES)],
        pl.Series("mean", mean_d.astype(np.float32)),
    )
    if arm == "A":
        gates["arm_i"] = chain317.gate_against(
            built=built_i,
            stored_path_of=lambda p: c317.E160_WORK / "dp_preds" / f"i_{p}_{site}.parquet",
            columns=c317.QUANTILES,
        )
    multipliers = json.loads((c317.E164_WORK / "fit_parameters.json").read_text())["o"]
    ens_mean_c = chain317.ensemble_temperature(frame=frame, site=site)
    cold = np.nan_to_num(ens_mean_c, nan=99.0) < c317.COLD_GATE_C
    scale = np.ones_like(quantiles_i)
    for k, level in enumerate(c317.LEVELS.tolist()):
        if level in c317.UPPER_FIX_LEVELS:
            scale[:, k] = np.where(cold, multipliers[str(level)], 1.0)
    quantiles_o = np.sort(mean_d[:, None] + offsets * scale, axis=1)
    built_o = built_i.with_columns(
        *[pl.Series(q, quantiles_o[:, k].astype(np.float32)) for k, q in enumerate(c317.QUANTILES)],
        ens_mean_c=pl.Series(ens_mean_c),
    )
    if arm == "A":
        gates["arm_o"] = chain317.gate_against(
            built=built_o,
            stored_path_of=lambda p: c317.E164_WORK / "dp_preds" / f"o_{p}_{site}.parquet",
            columns=c317.QUANTILES,
        )
        stored_chain = pl.read_parquet(c317.CHAIN_DIR / f"{site}.parquet").select(
            "run", "valid_time", *c317.QUANTILES
        )
        joined = built_o.select("run", "valid_time", *c317.QUANTILES).join(
            stored_chain, on=["run", "valid_time"], how="full", suffix="_stored", coalesce=True
        )
        both = joined.drop_nulls()
        gates["chain_317_all_leads"] = {
            "rows_built": built_o.height,
            "rows_stored": stored_chain.height,
            "rows_unmatched": joined.height - both.height,
            "max_abs_diff": max(
                float(np.abs(both[q].to_numpy() - both[f"{q}_stored"].to_numpy()).max())
                for q in c317.QUANTILES
            ),
        }
    target = common.CHAIN_DIR / arm / f"{site}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    built_o.drop("mean_d").write_parquet(temporary)
    temporary.rename(target)
    common.write_json_atomic(path=common.CHAIN_DIR / arm / f"gates_{site}.json", content=gates)
    return f"{arm} {site}: {built_o.height:,} rows, {time.monotonic() - started:.0f} s, gates {json.dumps(gates)[:700]}"


def main() -> None:
    """Build the chain of every arm at every substation in parallel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--arms", nargs="+", default=list(common.ARMS))
    parser.add_argument("--sites", nargs="+", default=list(common.SCORED_SITES))
    args = parser.parse_args()
    jobs = [(arm, site) for arm in args.arms for site in args.sites]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        for line in pool.map(job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
