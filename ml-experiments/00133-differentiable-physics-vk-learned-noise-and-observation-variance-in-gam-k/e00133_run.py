"""Forecast every arm of experiment 133 at every substation and write the prediction frames.

Stages, in order:

1. `rho` chooses the online variance's random-walk variance per half-hour from `RHO_GRID`, one
   value for all substations, by the tuning-month quantile score at 12 to 36 hours of the VK0 arm.
2. `factors` fits each arm's spread on the tuning months, scores the spread factor grid per lead
   band, and writes the per-site sums; the factor of each arm and lead band is the grid value with
   the lowest mean quantile score at the five decision substations, as `e00111_tune.py` chooses.
3. `forecast` forecasts the tuning months, the replicate months, and the 292 scored runs, and
   writes one parquet file per arm, period, and substation under `work/dp_preds/`.

The tail arms (`vk1`, `trail`) scale each row's spread by a factor computed from the filter's
state at the run's cutoff. The spread tables are fitted on residuals divided by the same factor,
and each quantile is then `mean + factor * (quantile - mean)`.

    uv run python ml-experiments/00133-*/e00133_run.py rho
    uv run python ml-experiments/00133-*/e00133_run.py factors [--workers 6]
    uv run python ml-experiments/00133-*/e00133_run.py forecast [--workers 6]
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00133_arms as arms
import e00133_common as common
import e00133_ens as ens
import e00133_filters as filters
import e00133_select as select
import numpy as np
import polars as pl
import torch
from studies.dp_assimilation import DEFAULT_DELAY_HOURS
from studies.dp_ens import LeadHourSpread
from studies.dp_ens_weather import N_LEAD_BANDS, lead_band_of_hours
from studies.dp_gam import QUANTILE_COLUMNS

FACTOR_GRID: Final[tuple[float, ...]] = (0.9, 1.0, 1.15, 1.3)
PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
OUT_DIR: Final[Path] = common.WORK_DIR / "dp_preds"
RUN_DIR: Final[Path] = common.WORK_DIR / "run"


def out_path(*, arm: str, period: str, site: str) -> Path:
    """Return the file that holds an arm's forecasts for one period at one site."""
    return OUT_DIR / f"{arm}_{period}_{site}.parquet"


def arm_hyper(*, arm: str, site: str) -> np.ndarray:
    """Return the four unconstrained values an arm starts from at a site."""
    if arm == "fix":
        return arms.fix_hyper(site=site)
    if arm == "pool":
        return np.array(json.loads((select.SELECT_DIR / "pool.json").read_text())["hyper"])
    return np.array(json.loads((select.SELECT_DIR / f"step2_{site}.json").read_text())["hyper"])


def chosen_rho() -> float:
    """Return the stored random-walk variance of the online variance."""
    return float(json.loads((RUN_DIR / "rho.json").read_text())["rho_a"])


def arm_settings(*, arm: str, site: str, rho_a: float | None = None) -> filters.FilterSettings:
    """Return the filter setting of an arm; `vk0` and `vk1` carry the online variance."""
    online = arm in {"vk0", "vk1"}
    rho = (chosen_rho() if rho_a is None else rho_a) if online else 0.0
    return arms.settings_of_hyper(hyper=arm_hyper(arm=arm, site=site), rho_a=rho)


def band_scores(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the sum and count of quantile scores per calibration lead band, shape (4, 2)."""
    kept = frame.filter(~pl.col("switching_mask"))
    quantiles = kept.select(list(QUANTILE_COLUMNS)).to_numpy()
    scores = ens.quantile_score(
        actual=kept["actual"].to_numpy().astype(np.float64), quantiles=quantiles
    )
    band = lead_band_of_hours(lead_hours=kept["lead_hours"].to_numpy())
    out = np.zeros((N_LEAD_BANDS, 2))
    for index in range(N_LEAD_BANDS):
        out[index] = (scores[band == index].sum(), (band == index).sum())
    return out


class ArmForecaster:
    """Forecasts one arm at one substation, with the spread fitted on the tuning months."""

    def __init__(self, *, arm: str, context: arms.SiteContext, rho_a: float | None = None) -> None:
        """Run the filter and fit the arm's spread on the tuning months.

        Args:
            arm: The arm's name.
            context: The substation's stored inputs.
            rho_a: The online variance's random-walk variance, for the `rho` stage.
        """
        self.arm = arm
        self.context = context
        self.run = filters.filter_stream(
            residual=context.stream,
            band=context.cache.calendar.band,
            settings=arm_settings(arm=arm, site=context.site, rho_a=rho_a),
        )
        self.kind = arms.TAIL_ARMS.get(arm)
        self.tuning = arms.tuning_runs(context=context)
        base = self._frame(runs=self.tuning, spread=ens.ZERO_SPREAD, scale=False)
        rows = arms.rows_of_frame(context=context, frame=base)
        self.reference = (
            arms.tail_reference(run=self.run, kind=self.kind, frame_positions=rows)
            if self.kind
            else None
        )
        self.spread = ens.fit_spread_from_frame(frame=self._for_spread(frame=base, rows=rows))

    def _frame(self, *, runs: list[int], spread: LeadHourSpread, scale: bool) -> pl.DataFrame:
        frame = ens.forecast_frame(
            cache=self.context.cache,
            setup=self.context.setup,
            fits=self.context.fits,
            spread=spread,
            trace=self.run.trace,
            run_indices=runs,
            arm=self.arm,
        )
        if not (scale and self.kind):
            return frame
        rows = arms.rows_of_frame(context=self.context, frame=frame)
        assert self.reference is not None
        factor = arms.tail_factors(
            run=self.run, kind=self.kind, reference=self.reference, rows=rows
        )
        statistic = arms._statistic(run=self.run, kind=self.kind)
        from e00133_filters import value_at_cutoffs

        raw = value_at_cutoffs(
            positions=self.run.trace.positions,
            values=statistic,
            cutoff=rows["cutoff"],
            band=rows["band"],
        )
        mean = frame["mean"].to_numpy().astype(np.float64)
        scaled = {
            column: (mean + factor * (frame[column].to_numpy().astype(np.float64) - mean)).astype(
                np.float32
            )
            for column in QUANTILE_COLUMNS
        }
        return frame.with_columns(
            [pl.Series(name, values) for name, values in scaled.items()]
            + [pl.Series("tail_factor", factor.astype(np.float32))]
            + [pl.Series("tail_statistic", raw.astype(np.float32))]
        )

    def _for_spread(self, *, frame: pl.DataFrame, rows: dict) -> pl.DataFrame:
        if not self.kind:
            return frame
        assert self.reference is not None
        factor = arms.tail_factors(
            run=self.run, kind=self.kind, reference=self.reference, rows=rows
        )
        mean = frame["mean"].to_numpy().astype(np.float64)
        actual = frame["actual"].to_numpy().astype(np.float64)
        adjusted = mean + (actual - mean) / factor
        return frame.with_columns(pl.Series("actual", adjusted.astype(np.float32)))

    def forecast(self, *, runs: list[int], factors: list[float]) -> pl.DataFrame:
        """Forecast the runs with the spread at the given factors per lead band."""
        return self._frame(runs=runs, spread=self.spread.with_factors(factors=factors), scale=True)

    def tuning_score(self) -> float:
        """Return the mean quantile score at 12 to 36 hours on the tuning months, factors 1."""
        return select.day_ahead_score(frame=self.forecast(runs=self.tuning, factors=[1.0] * 4))


def _rho_job(job: tuple[str, float]) -> tuple[str, float, float]:
    torch.set_num_threads(1)
    site, rho = job
    context = arms.load_site(site=site)
    return site, rho, ArmForecaster(arm="vk0", context=context, rho_a=rho).tuning_score()


def stage_rho(*, workers: int) -> None:
    """Choose the online variance's random-walk variance on the tuning months."""
    jobs = [(site, rho) for site in common.SCORED_SITES for rho in arms.RHO_GRID]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_rho_job, jobs))
    totals = dict.fromkeys(arms.RHO_GRID, 0.0)
    for _, rho, score in results:
        totals[rho] += score / len(common.SCORED_SITES)
    best = min(totals, key=totals.__getitem__)
    common.write_json_atomic(
        path=RUN_DIR / "rho.json",
        content={
            "rho_a": best,
            "mean_scores": {str(rho): score for rho, score in totals.items()},
            "per_site": [{"site": s, "rho_a": r, "score": v} for s, r, v in results],
        },
    )
    print(f"rho_a {best:.2e}; mean tuning scores {totals}", flush=True)


def _factor_job(job: tuple[str, str]) -> dict:
    torch.set_num_threads(1)
    arm, site = job
    context = arms.load_site(site=site)
    forecaster = ArmForecaster(arm=arm, context=context)
    scores = {
        str(factor): band_scores(
            frame=forecaster.forecast(runs=forecaster.tuning, factors=[factor] * N_LEAD_BANDS)
        ).tolist()
        for factor in FACTOR_GRID
    }
    return {"arm": arm, "site": site, "scores": scores}


def stage_factors(*, workers: int) -> None:
    """Score the spread factor grid per arm and substation, and choose the factors per arm."""
    jobs = [(arm, site) for arm in arms.ARM_NAMES for site in common.SCORED_SITES]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_factor_job, jobs))
    chosen = {}
    for arm in arms.ARM_NAMES:
        totals = np.zeros((len(FACTOR_GRID), N_LEAD_BANDS, 2))
        for result in results:
            if result["arm"] == arm and result["site"] in common.DECISION_SITES:
                for position, factor in enumerate(FACTOR_GRID):
                    totals[position] += np.array(result["scores"][str(factor)])
        mean = totals[:, :, 0] / totals[:, :, 1]
        chosen[arm] = np.array(FACTOR_GRID)[mean.argmin(axis=0)].tolist()
    common.write_json_atomic(
        path=RUN_DIR / "factors.json",
        content={
            "factors": chosen,
            "scores": {f"{r['arm']}_{r['site']}": r["scores"] for r in results},
        },
    )
    print(f"factors: {chosen}", flush=True)


def _forecast_job(job: tuple[str, str]) -> str:
    torch.set_num_threads(1)
    arm, site = job
    started = time.monotonic()
    context = arms.load_site(site=site)
    forecaster = ArmForecaster(arm=arm, context=context)
    factors = json.loads((RUN_DIR / "factors.json").read_text())["factors"][arm]
    first = context.fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
    rows = 0
    for period in PERIODS:
        runs = [
            i for i in context.setup.indices(periods={period}) if context.setup.origins[i] >= first
        ]
        if not runs:
            continue
        frame = forecaster.forecast(runs=runs, factors=factors)
        target = out_path(arm=arm, period=period, site=site)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        frame.write_parquet(temporary)
        temporary.rename(target)
        rows += frame.height
    return f"{arm} {site}: {rows:,} rows, {time.monotonic() - started:.0f} s"


def stage_forecast(*, workers: int) -> None:
    """Forecast every arm and substation."""
    jobs = [(arm, site) for arm in arms.ARM_NAMES for site in common.SCORED_SITES]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for line in pool.map(_forecast_job, jobs):
            print(line, flush=True)


def main() -> None:
    """Run one stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("rho", "factors", "forecast"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    {"rho": stage_rho, "factors": stage_factors, "forecast": stage_forecast}[args.stage](
        workers=args.workers
    )


if __name__ == "__main__":
    main()
