r"""Refit BG2's N1 XGBoost twin with and without input-family interactions (experiment 305).

Copies the N1 recipe of experiment 87 (`e00087_fit_demand.py`: control-member rows, 11 quantile
levels, 500 rounds, eta 0.05, depth 6, half-life 0.5 years, monthly refits) and changes only the
device (CPU) and, per variant, XGBoost's `interaction_constraints`. A refit at the first of a month
forecasts that month's 09:00 UTC runs, control member only, on the half-hours in experiment 266's
master frame. Each (site, refit, variant) writes a booster and a parquet of the 11 quantiles in
points of effective capacity, atomically, so a stopped run resumes.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \\
        uv run --no-sync python ml-experiments/00305-*/e00305_fit.py --workers 3 --threads 2
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00305_common as common
import e00305_recipe as fit
import numpy as np
import polars as pl
import xgboost as xgb
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00305-summer-gap-what-bg2-knows-that-arm-d-does-not"
)
MASTER: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00266-summer-gap-diagnostics-solar-term" / "work" / "master.parquet"
)
SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D16", "D14")
REFITS: Final[tuple[datetime, ...]] = tuple(datetime(2026, m, 1, tzinfo=UTC) for m in (4, 5, 6))
SOLAR: Final[tuple[str, ...]] = ("shortwave", "solar_zenith", "solar_azimuth")
TEMPERATURE: Final[tuple[str, ...]] = ("temperature_2m", "dew_point_temperature_2m")
LAGS: Final[tuple[str, ...]] = common.LAG_COLUMNS
SHARED: Final[tuple[str, ...]] = (
    "longwave", "pressure_surface", "precipitation", "wind_speed_10m",
    "hour_local", "day_of_week_local", "covid_indicator", "days_to_christmas", "lead_hours",
)  # fmt: skip
VARIANTS: Final[dict[str, list[list[str]] | None]] = {
    "full": None,
    "no_solar_x_temp_or_lags": [[*SOLAR, *SHARED], [*TEMPERATURE, *LAGS, *SHARED]],
    "no_solar_x_lags": [[*SOLAR, *TEMPERATURE, *SHARED], [*TEMPERATURE, *LAGS, *SHARED]],
    "no_solar_x_temp": [[*SOLAR, *LAGS, *SHARED], [*TEMPERATURE, *LAGS, *SHARED]],
    "additive_post_hoc": None,
}
ADDITIVE: Final[str] = "additive_post_hoc"
"""Post hoc variant: depth-1 trees, so no interaction of any kind, with 4 times the rounds."""
SCALE: Final[float] = 100.0


def _constraints(*, groups: list[list[str]] | None) -> list[list[str]] | None:
    """Return the column-name groups of `interaction_constraints`; every column must be in one."""
    if groups is None:
        return None
    columns = fit.input_columns(a7=False)
    unknown = set().union(*map(set, groups)) - set(columns)
    missing = set(columns) - set().union(*map(set, groups))
    if unknown or missing:
        msg = f"constraint groups: unknown {unknown}, missing {missing}"
        raise ValueError(msg)
    return [list(group) for group in groups]


def run_one(*, site: str, refit: datetime, variant: str, threads: int) -> str:
    """Fit one booster and write its quantiles for the month's master rows."""
    out = DATA_DIR / "work" / "preds" / variant / f"{site}_{refit:%Y-%m}.parquet"
    inputs_file = DATA_DIR / "work" / "inputs" / f"{site}_{refit:%Y-%m}.parquet"
    if out.exists() and (variant != "full" or inputs_file.exists()):
        return f"{variant} {site} {refit:%Y-%m}: exists"
    series, arrays = fit._series_and_arrays()
    row = next(r for r in series.iter_rows(named=True) if r["site"] == site)
    config = fit.CONFIGS["n1h"]
    known = fit.known_rows(
        rows=fit._training_rows_cached(row=row, arrays=arrays[site], config=config),
        refit=refit,
        config=config,
    )
    params = {**fit.parameters(config=config, seed=0, nthread=threads), "device": "cpu"}
    if variant == ADDITIVE:
        params["max_depth"] = 1
    constraints = _constraints(groups=VARIANTS[variant])
    if constraints is not None:
        params["interaction_constraints"] = constraints
    train = fit.train_inputs(known=known, row=row, refit=refit, config=config)
    names = fit.input_columns(a7=False) if constraints is not None else None
    train.feature_names = names
    model = xgb.train(
        params, train, num_boost_round=config.rounds * (4 if variant == ADDITIVE else 1)
    )
    model.save_model(str(DATA_DIR / "work" / "models" / variant / f"{site}_{refit:%Y-%m}.ubj"))
    master = pl.read_parquet(
        MASTER, columns=["site", "run", "power_fcst_init_time", "valid_time", "init_month"]
    )
    wanted = master.filter((pl.col("site") == site) & (pl.col("init_month") == f"{refit:%Y-%m}"))
    pieces = []
    inputs = []
    for origin in wanted["power_fcst_init_time"].unique().sort().to_list():
        frame = pl.read_parquet(common.demand_frame_path(run=origin - common.PUBLICATION_DELAY))
        prepared = fit.prepared_run(
            run_rows=frame.filter(pl.col("time_series_id") == row["time_series_id"]),
            row=row,
            arrays=arrays[site],
            a7=False,
        ).filter(pl.col("ensemble_member") == 0)
        raw = model.predict(
            xgb.DMatrix(
                prepared.select(fit.input_columns(a7=False)).to_numpy().astype(np.float32),
                feature_names=names,
            )
        )
        quantiles = np.sort(raw, axis=1) * SCALE
        piece = prepared.select("valid_time").with_columns(power_fcst_init_time=pl.lit(origin))
        for i, level in enumerate(common.LEVELS):
            piece = piece.with_columns(pl.Series(f"q{round(level * 100):02d}", quantiles[:, i]))
        pieces.append(piece)
        inputs.append(
            prepared.select("valid_time", *fit.input_columns(a7=False)).with_columns(
                power_fcst_init_time=pl.lit(origin).cast(wanted.schema["power_fcst_init_time"])
            )
        )
    table = (
        pl.concat(pieces)
        .with_columns(
            site=pl.lit(site),
            power_fcst_init_time=pl.col("power_fcst_init_time").cast(
                wanted.schema["power_fcst_init_time"]
            ),
        )
        .join(
            wanted.select("site", "power_fcst_init_time", "valid_time"),
            on=["site", "power_fcst_init_time", "valid_time"],
            how="semi",
        )
    )
    if variant == "full":
        common.atomic_write(
            frame=pl.concat(inputs).with_columns(site=pl.lit(site)),
            target=inputs_file,
        )
    common.atomic_write(frame=table, target=out)
    return f"{variant} {site} {refit:%Y-%m}: {known.height:,} rows, {table.height:,} predictions"


def _job(args: tuple[str, datetime, str, int]) -> str:
    site, refit, variant, threads = args
    return run_one(site=site, refit=refit, variant=variant, threads=threads)


def main() -> None:
    """Run every refit that lacks its file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--sites", nargs="+", default=list(SITES))
    parser.add_argument("--refits", type=int, default=3)
    args = parser.parse_args()
    for variant in args.variants:
        (DATA_DIR / "work" / "models" / variant).mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "work" / "preds" / variant).mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "work" / "inputs").mkdir(parents=True, exist_ok=True)
    jobs = [
        (site, refit, variant, args.threads)
        for variant in args.variants
        for refit in REFITS[: args.refits]
        for site in args.sites
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
