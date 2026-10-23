# ruff: noqa: E501, D107
"""Fit every arm of experiment 293 in both settings and write each job's scored-year quantiles.

Every arm is P2 of experiment 270 (6 forecast bins by 3 ensemble-spread bins, 11 residual quantiles
per cell, from the tuning months) with a different plant: tilt 35 or 20 degrees, a hard or soft cap,
a cap at the tuning 0.995 or 0.999 quantile of output, or the pooled leave-one-farm-out shape of
experiment 282. The scale is the unclipped least-squares scale for every arm. `own_history` trains
on the scored farm's tuning months; `zero_history` on the other five farms', with the scale and cap
the median over those five farms. One job is one (setting, farm, arm); its file is written
atomically and skipped when it exists.

    systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00293-*/e00293_fit.py
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import itertools
import multiprocessing
from functools import cache
from pathlib import Path

import e00293_common as common
import numpy as np
import polars as pl
from studies.pv_plant_fit import ResponseGrid, ResponseTable, block_folds, fit_response_table
from studies.tail_calibration import level_column

N_PROCESSES = 4
RESPONSE_GRID = ResponseGrid(
    irradiance_w_m2=np.arange(0.0, 1101.0, 50.0), elevation_deg=np.arange(5.0, 66.0, 5.0)
)
PENALTIES = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0)
CV_FOLDS = 5


@cache
def farm(site: str) -> common.Farm:
    """Load a farm once per process."""
    return common.load_farm(site)


@cache
def shapes(site: str) -> dict[str, np.ndarray]:
    """Load the farm's recomputed unit responses and members' shortwave."""
    with np.load(common.SHAPES_DIR / f"{site}.npz") as data:
        return {key: data[key] for key in data.files}


@cache
def pooled_table(site: str) -> ResponseTable:
    """Return the pooled shape fitted to the other five farms' tuning rows (experiment 282, check 7)."""
    frame = pl.read_parquet(common.E282_FRAME).sort("site", "time")
    others = frame.filter((pl.col("site") != site) & pl.col("tuning"))
    ghi = others["ghi_w_m2"].to_numpy()
    elevation = others["elevation_deg"].to_numpy()
    output = others["output"].to_numpy()
    prior = others["plant_fleet"].to_numpy()
    folds = block_folds(block_of_row=others["month_index"].to_numpy(), n_folds=CV_FOLDS)
    errors = {}
    for penalty in PENALTIES:
        error = 0.0
        for held in folds:
            table = fit_response_table(
                grid=RESPONSE_GRID,
                irradiance_w_m2=ghi[~held],
                elevation_deg=elevation[~held],
                output=output[~held],
                prior_output=prior[~held],
                penalty=penalty,
            )
            predicted = table.predict(irradiance_w_m2=ghi[held], elevation_deg=elevation[held])
            error += float(np.sum((output[held] - predicted) ** 2))
        errors[penalty] = error
    best = min(errors, key=errors.__getitem__)
    print(f"pooled shape for farm {site}: penalty {best}", flush=True)
    return fit_response_table(
        grid=RESPONSE_GRID,
        irradiance_w_m2=ghi,
        elevation_deg=elevation,
        output=output,
        prior_output=prior,
        penalty=best,
    )


def unit_columns(*, site: str, shape: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the farm's unit response to the mean weather and the members' response spread."""
    f, s = farm(site), shapes(site)
    if shape == "pooled":
        table = pooled_table(site)
        elevation = f.frame["elevation_deg"].to_numpy()
        mean = table.predict(
            irradiance_w_m2=f.frame["ghi_mean"].to_numpy(), elevation_deg=elevation
        )
        members = s["ghi_members"].astype(np.float64)
        flat = table.predict(
            irradiance_w_m2=members.ravel(),
            elevation_deg=np.broadcast_to(elevation[:, None], members.shape).ravel(),
        )
        return np.asarray(mean), flat.reshape(members.shape).std(axis=1)
    tilt = shape.removeprefix("tilt")
    return s[f"unit_mean_{tilt}"], s[f"unit_std_{tilt}"]


class Fitted:
    """A variant's own-farm scale and cap, from the farm's tuning months."""

    def __init__(self, *, site: str, variant: common.Variant) -> None:
        f = farm(site)
        unit, _ = unit_columns(site=site, shape=variant.shape)
        output = f.frame["output"].to_numpy().astype(np.float64)
        self.scale = common.fit_scale(unit=unit[f.tuning], output=output[f.tuning])
        self.cap = float(np.quantile(output[f.tuning], variant.cap_quantile))


class Job:
    """The training rows and scored rows of one (setting, scored farm, variant), with P2's cells."""

    def __init__(self, *, setting: str, site: str, variant: common.Variant) -> None:
        self.site, self.variant = site, variant
        others = [s for s in common.FARMS if s != site]
        if setting == "own_history":
            own = Fitted(site=site, variant=variant)
            self.scale, self.cap = own.scale, own.cap
            train_sites = [site]
        else:
            fitted = [Fitted(site=s, variant=variant) for s in others]
            self.scale = float(np.median([x.scale for x in fitted]))
            self.cap = float(np.median([x.cap for x in fitted]))
            train_sites = others
        train = [self.frame(site=s) for s in train_sites]
        self.train = pl.concat(
            [t.filter(pl.Series(farm(s).tuning)) for t, s in zip(train, train_sites, strict=True)]
        )
        self.test = self.frame(site=site).filter(pl.Series(farm(site).scored))
        self.fit_cells()

    def frame(self, *, site: str) -> pl.DataFrame:
        """Return the farm's rows with the plant forecast `mu` and the members' spread `plant_spread`."""
        unit, std = unit_columns(site=site, shape=self.variant.shape)
        soft = self.variant.soft
        mu = common.clip(power=self.scale * unit, cap=self.cap, soft=soft)
        # The clip on the spread column is kept for parity with experiment 270; its effect is negligible.
        spread = common.clip(power=self.scale * std, cap=self.cap, soft=soft)
        return (
            farm(site)
            .frame.select("time", "init_time", "lead_hours", "output")
            .with_columns(mu=pl.Series(mu), plant_spread=pl.Series(spread))
        )

    def fit_cells(self) -> None:
        """Fit P2's cells: 6 forecast bins by 3 spread bins, 11 residual quantiles each."""
        mu = self.train["mu"].to_numpy()
        spread = self.train["plant_spread"].to_numpy()
        y = self.train["output"].to_numpy().astype(float)
        four_edges, fallback = common.bin_offsets(mu=mu, output=y)
        usable = mu > common.MIN_FORECAST_POWER
        self.mu_edges = np.quantile(mu[usable], np.linspace(0, 1, common.N_MU_BINS + 1)[1:-1])
        self.spread_edges = np.quantile(
            spread[usable], np.linspace(0, 1, common.N_SPREAD_BINS + 1)[1:-1]
        )
        mu_bin, sp_bin = np.digitize(mu, self.mu_edges), np.digitize(spread, self.spread_edges)
        coarse = np.digitize(mu, four_edges)
        residual = y - mu
        self.cells = np.empty((common.N_MU_BINS, common.N_SPREAD_BINS, len(common.LEVELS)))
        for i in range(common.N_MU_BINS):
            for j in range(common.N_SPREAD_BINS):
                rows = (mu_bin == i) & (sp_bin == j)
                self.cells[i, j] = (
                    np.quantile(residual[rows], common.LEVELS)
                    if rows.sum() >= common.MIN_BIN_ROWS
                    else fallback[int(np.bincount(coarse[rows]).argmax()) if rows.any() else 0]
                )

    def quantiles(self) -> np.ndarray:
        """Return the scored rows' sorted quantiles, clamped to [0, cap]."""
        mu = self.test["mu"].to_numpy()
        mu_bin = np.digitize(mu, self.mu_edges)
        sp_bin = np.digitize(self.test["plant_spread"].to_numpy(), self.spread_edges)
        q = np.sort(mu[:, None] + self.cells[mu_bin, sp_bin], axis=1)
        return np.minimum(np.maximum(q, 0.0), self.cap)


def out_path(*, setting: str, site: str, arm: str) -> Path:
    """Return a job's output file."""
    return common.PRED_DIR / setting / f"{arm}_s0_{site}.parquet"


def run_group(task: tuple[str, str]) -> str:
    """Run every arm of one (setting, farm), skipping jobs whose file exists."""
    setting, site = task
    for variant in common.VARIANTS:
        path = out_path(setting=setting, site=site, arm=variant.name)
        if setting not in variant.settings or path.exists():
            continue
        job = Job(setting=setting, site=site, variant=variant)
        q = job.quantiles()
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = job.test.select("time", "init_time", "lead_hours").with_columns(
            time_series_id=pl.lit(site),
            arm=pl.lit(variant.name),
            seed=pl.lit(0),
            setting=pl.lit(setting),
            actual=job.test["output"].cast(pl.Float64),
            scale=pl.lit(job.scale),
            cap=pl.lit(job.cap),
            **{level_column(level=lv): pl.Series(q[:, i]) for i, lv in enumerate(common.LEVELS)},
        )
        tmp = path.with_suffix(".tmp")
        frame.write_parquet(tmp)
        tmp.replace(path)
        print(
            f"{setting} {site} {variant.name} scale={job.scale:.3f} cap={job.cap:.3f}", flush=True
        )
    return f"{setting} {site}"


def main() -> None:
    """Run all (setting, farm) groups in parallel."""
    tasks = list(itertools.product(common.SETTINGS, common.FARMS))
    with multiprocessing.get_context("spawn").Pool(N_PROCESSES) as pool:
        for done in pool.imap_unordered(run_group, tasks):
            print("group finished", done, flush=True)


if __name__ == "__main__":
    main()
