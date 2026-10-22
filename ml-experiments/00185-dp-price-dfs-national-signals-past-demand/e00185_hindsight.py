"""Experiment 185, Part 3: a national signal as a day-level covariate in the Kalman level state.

**Not forecastable: the covariate is read for the target day itself, in hindsight.** The script
replays experiment 136's arm A2 (T-all fits, POOL filter noise, monthly cadence) and experiment
155's arm D (night offset plus recipe R on A2's own errors) with the code both experiments used,
copied into this folder. The only change is the covariate. For each hour band the filter's residual
stream has `beta_band * z(day)` removed before the filter runs, and each forecast adds
`beta_band * z(target day)` back to the mean and every quantile. The coefficients come from the
tuning months (odd months of 2024-07 to 2025-06) only. The spread table is refitted on the tuning
months, as in experiment 155. The `control` variant has all coefficients at zero and must reproduce
the stored arm D.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00185-*/e00185_hindsight.py \
        --workers 4
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00185_common as common
import e00185_x136_common as c136
import e00185_x136_design as design136
import e00185_x136_ens as ens
import e00185_x136_filters as filters
import e00185_x136_run as run136
import e00185_x136_tall as tall136
import e00185_x155_common as c155
import e00185_x155_corrections as corr
import e00185_x155_run as run155
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.dp_assimilation import DEFAULT_DELAY_HOURS
from studies.dp_ens_weather import proxy_cache

# Experiment 136's stored fits are pickles that name its modules, so the copies answer to them.
sys.modules["e00136_common"] = c136
sys.modules["e00136_design"] = design136
sys.modules["e00136_ens"] = ens
sys.modules["e00136_filters"] = filters
sys.modules["e00136_tall"] = tall136

OUT_DIR: Final[Path] = common.WORK_DIR / "dp_preds"
WINSOR_SD: Final[float] = 4.0
"""A covariate is clipped to plus or minus this many standard deviations, fixed before any run."""
VARIANTS: Final[dict[str, tuple[str, ...]]] = {
    "control": (),
    "national_demand": ("nd_mean", "nd_eve"),
    "price": ("price_mean", "price_eve"),
}
TUNING_FIRST: Final = pl.date(2024, 7, 1)


def standardised_signals(*, columns: tuple[str, ...]) -> dict[object, np.ndarray]:
    """Return each local date's clipped z-scores of the covariate columns (tuning-year moments)."""
    table = pl.read_parquet(common.WORK_DIR / "national_signals.parquet").sort("local_date")
    tuning = table.filter(pl.col("local_date") < pl.date(2025, 7, 1))
    z = np.column_stack(
        [
            np.clip(
                (table[c].to_numpy() - tuning[c].mean()) / tuning[c].std(), -WINSOR_SD, WINSOR_SD
            )
            for c in columns
        ]
    )
    return dict(zip(table["local_date"].to_list(), z, strict=True))


def grid_z(*, end_times: np.ndarray, signals: dict[object, np.ndarray], k: int) -> np.ndarray:
    """Return the covariate at every row of the grid (zero before the table starts)."""
    start = pl.Series(end_times.astype("datetime64[us]") - np.timedelta64(30, "m"))
    dates = start.dt.replace_time_zone("UTC").dt.convert_time_zone(common.LOCAL_TZ).dt.date()
    z = np.zeros((len(end_times), k))
    for i, day in enumerate(dates.to_list()):
        row = signals.get(day)
        if row is not None:
            z[i] = row
    return z


def band_coefficients(
    *, stream: np.ndarray, z: np.ndarray, band: np.ndarray, tuning: np.ndarray
) -> np.ndarray:
    """Return each hour band's least-squares slopes of the stream on the covariates."""
    beta = np.zeros((filters.N_BANDS, z.shape[1]))
    for b in range(filters.N_BANDS):
        rows = np.flatnonzero((band == b) & tuning & np.isfinite(stream))
        if len(rows) < 100 or z.shape[1] == 0:
            continue
        x = np.column_stack([np.ones(len(rows)), z[rows]])
        solution = np.linalg.lstsq(x, stream[rows], rcond=None)[0]
        beta[b] = solution[1:]
    return beta


class CovariateForecaster:
    """Arm A2 with the covariate: the filter runs on the adjusted stream, forecasts add it back."""

    def __init__(self, *, context: run136.SiteContext, columns: tuple[str, ...]) -> None:
        """Fit the coefficients on the tuning months, run the filter, and fit the spread."""
        self.context = context
        self.columns = columns
        self.fits = run136.load_arm_fits(arm="a2", site=context.site)
        proxy = proxy_cache(cache=context.cache, context=context.setup.context)
        stream = run136.tall_residual_stream(cache=proxy, fits=self.fits, site=context.site)
        end_times = context.cache.end_times
        band = context.cache.calendar.band
        self.signals = standardised_signals(columns=columns) if columns else {}
        k = len(columns)
        self.z = grid_z(end_times=end_times, signals=self.signals, k=k)
        month = pl.Series(end_times.astype("datetime64[us]")).dt
        tuning = np.array(
            [
                (y, m) in c136.TUNING_MONTHS
                for y, m in zip(month.year().to_list(), month.month().to_list(), strict=True)
            ]
        )
        self.beta = band_coefficients(stream=stream, z=self.z, band=band, tuning=tuning)
        adjusted = stream - (self.beta[band] * self.z).sum(axis=1) if k else stream
        self.run = filters.filter_stream(
            residual=adjusted,
            band=band,
            settings=run136.settings_of_hyper(
                hyper=run136.noise_hyper(noise="pool", site=context.site)
            ),
        )
        first = self.fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
        self.first = first
        tuning_runs = [
            i
            for i in context.setup.indices(periods={"tuning"})
            if context.setup.origins[i] >= first
        ]
        self.spread = ens.fit_spread_from_frame(
            frame=self._frame(runs=tuning_runs, spread=ens.ZERO_SPREAD)
        )

    def _shift(self, *, frame: pl.DataFrame) -> pl.DataFrame:
        """Add `beta_band * z(target day)` to the mean and every quantile."""
        if not self.columns:
            return frame
        valid = frame["valid_time"].to_numpy()
        z = grid_z(end_times=valid, signals=self.signals, k=len(self.columns))
        band = frame["hour_band"].to_numpy()
        delta = (self.beta[band] * z).sum(axis=1)
        names = ["mean", *c155.QUANTILES]
        return frame.with_columns(
            *[(pl.col(n) + pl.Series(delta)).cast(pl.Float32).alias(n) for n in names]
        )

    def _frame(self, *, runs: list[int], spread) -> pl.DataFrame:  # noqa: ANN001
        frame = ens.forecast_frame(
            cache=self.context.cache,
            setup=self.context.setup,
            fits=self.fits,
            spread=spread,
            trace=self.run.trace,
            run_indices=runs,
            arm="a2",
        )
        return self._shift(frame=frame)

    def base_frame(self) -> pl.DataFrame:
        """Return arm A's frame (all three periods) in experiment 155's layout."""
        pieces = []
        for period in c155.PERIODS:
            runs = [
                i
                for i in self.context.setup.indices(periods={period})
                if self.context.setup.origins[i] >= self.first
            ]
            if runs:
                pieces.append(
                    self._frame(runs=runs, spread=self.spread)
                    .select(list(c155.BASE_COLUMNS))
                    .with_columns(period=pl.lit(period))
                )
        return pl.concat(pieces).sort(["power_fcst_init_time", "valid_time"])


def write(*, variant: str, arm: str, period: str, site: str, frame: pl.DataFrame) -> None:
    """Write a variant's arm for one period and site, atomically."""
    target = OUT_DIR / f"{variant}_{arm}_{period}_{site}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    part = frame.filter(pl.col("period") == period).with_columns(arm=pl.lit(arm))
    part.select(list(run155.OUTPUT_COLUMNS)).write_parquet(temporary)
    temporary.rename(target)


def job(job_spec: tuple[str, str]) -> str:
    """Replay arm A2 and arm D for one variant and one substation."""
    torch.set_num_threads(1)
    variant, site = job_spec
    started = time.monotonic()
    context = run136.load_site(site=site)
    forecaster = CovariateForecaster(context=context, columns=VARIANTS[variant])
    base = corr.add_calendar(frame=forecaster.base_frame())
    arms = run155.build_arms(site=site, frame=base)
    for arm in ("a", "d"):
        frame = arms[arm]
        for period in c155.PERIODS:
            write(variant=variant, arm=arm, period=period, site=site, frame=frame)
    beta = np.array2string(forecaster.beta, precision=3)
    return f"{variant} {site}: {time.monotonic() - started:.0f} s, beta per band {beta}"


def main() -> None:
    """Replay every variant at every substation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--sites", nargs="+", default=list(c155.SCORED_SITES))
    args = parser.parse_args()
    jobs = [(v, s) for v in args.variants for s in args.sites]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
