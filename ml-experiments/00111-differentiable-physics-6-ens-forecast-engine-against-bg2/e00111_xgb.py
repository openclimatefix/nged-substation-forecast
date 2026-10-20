"""Fit experiment 111's XGBoost arms and write each refit's 11 quantiles per daily run.

Every arm is the BG2 recipe's N1 model (the ENS control member's rows, a recency half-life of 0.5
years anchored at the refit, `reg:quantileerror` at 11 levels, depth 6) on the 51 members of each
daily run pooled into 11 mixture quantiles. They differ in the inputs and the starting point:

| Arm | Lags | Rows | Starting point (base margin) | Refits |
|---|---|---|---|---|
| `xnl` | none | all leads | XGBoost's estimate | monthly, 500 rounds, eta 0.05 |
| `xnl_z` | none | all leads | zero | the same |
| `xnl_o` | none | all leads | median label plus experiment 64's offset (offset sites) | the same |
| `xlag_d` | 3 | leads to 168 h | XGBoost's estimate | monthly, 150 rounds, eta 0.15 |
| `xlag_z` | 3 | leads to 168 h | zero | the same |
| `xlag_o` | 3 | leads to 168 h | median label plus the solar offset (offset sites) | the same |
| `gam_margin` | 3 | leads to 168 h | the DP's forecast, forward-chained | the same |

The margin arms refit monthly, not weekly as the plan said, because the plan's compute estimate was
20 times too low. They train on leads to 168 hours because the DP's forecast covers 168 hours; the
lagless arms keep SIMP1's rows. A refit at time `t` trains on the control member's rows with a
valid time at least 3.5 hours before `t` and a run that began at or before `t`, from each site's
`first_usable` instant (experiment 103's start cut). It forecasts the runs of the tuning,
replicate, and scored periods whose origin lies from `t` to the next refit.

The margin of a `gam_margin` training row is the DP's control-member forecast of its run, from
the fit of the latest monthly cut at or before the run (`studies.dp_ens.margin_frame`). At forecast
time each member's rows take that member's DP forecast. Existing files stay, so a stopped run
resumes.

    uv run python ml-experiments/00111-*/e00111_xgb.py --arm xnl [--workers 3] [--cores 6]
        [--sites D5 D10] [--smoke]
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00111_arms as arms
import e00111_common as common
import e00111_ens as ens
import e00111_xgb_data as data
import numpy as np
import polars as pl
import xgboost as xgb
from studies.dp_ens import margin_frame
from studies.dp_ens_weather import lead_band_of_hours  # noqa: F401
from studies.ens_members import mixture_quantiles

LEVELS: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)
LEVEL_COLUMNS: Final[tuple[str, ...]] = (
    "q01",
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
    "q99",
)
QUANTILE_PARAMETERS: Final[dict[str, object]] = {
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "colsample_bynode": 1.0,
    "tree_method": "hist",
    "device": "cuda",
    "objective": "reg:quantileerror",
    "quantile_alpha": LEVELS,
}
HALF_LIFE_DAYS: Final[float] = 0.5 * 365.25
REFIT_CUTOFF_AGE: Final[timedelta] = timedelta(hours=3.5)
FIRST_REFIT: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
LAST_MONTHLY_REFIT: Final[datetime] = datetime(2026, 6, 1, tzinfo=UTC)
LAST_WEEKLY_REFIT: Final[datetime] = datetime(2026, 6, 22, tzinfo=UTC)
SERVED_PERIODS: Final[frozenset[str]] = frozenset({"tuning", "replicate", "scored"})
OFFSET_SITES: Final[tuple[str, ...]] = ("D9", "D10", "D11", "D14", "D16")
SMOKE_ROUNDS: Final[int] = 20
SEED: Final[int] = 0
ANOMALY_FILE: Final[Path] = (
    common.WORK_DIR.parents[1]
    / "00087-bg2-best-guess-assembled-with-drop-one-ablations"
    / "work"
    / "offset"
    / "anomaly.parquet"
)
E64_DIR: Final[Path] = (
    common.WORK_DIR.parents[1]
    / ("00064-pvdis-5-net-demand-as-demand-minus-forecast-solar")
    / "work"
)
LEAD_BAND_EDGES: Final[tuple[float, ...]] = (24.0, 48.0, 144.0, 240.0)
SEASON_OF_MONTH: Final[dict[int, str]] = {
    4: "Apr-Jun",
    5: "Apr-Jun",
    6: "Apr-Jun",
    7: "Jul-Sep",
    8: "Jul-Sep",
    9: "Jul-Sep",
}
NO_SKILL_BAND: Final[int] = 4


@dataclass(frozen=True)
class Arm:
    """One XGBoost arm.

    Attributes:
        name: The arm's name, used for folders.
        lagged: Whether the three origin lags are inputs.
        start: `default`, `zero`, `offset`, or `gam`.
        max_train_lead: The longest lead of a training row in hours, or `None` for all leads.
        rounds: Boosting rounds.
        eta: Learning rate.
        schedule: `monthly` or `weekly`.
    """

    name: str
    lagged: bool
    start: str
    max_train_lead: float | None
    rounds: int
    eta: float
    schedule: str


ARMS: Final[dict[str, Arm]] = {
    "xnl": Arm("xnl", False, "default", None, 500, 0.05, "monthly"),
    "xnl_z": Arm("xnl_z", False, "zero", None, 500, 0.05, "monthly"),
    "xnl_o": Arm("xnl_o", False, "offset", None, 500, 0.05, "monthly"),
    "xlag_d": Arm("xlag_d", True, "default", data.MAX_LEAD_HOURS, 150, 0.15, "monthly"),
    "xlag_z": Arm("xlag_z", True, "zero", data.MAX_LEAD_HOURS, 150, 0.15, "monthly"),
    "xlag_o": Arm("xlag_o", True, "offset", data.MAX_LEAD_HOURS, 150, 0.15, "monthly"),
    "gam_margin": Arm("gam_margin", True, "gam", data.MAX_LEAD_HOURS, 150, 0.15, "monthly"),
}


def refit_times(*, schedule: str) -> list[datetime]:
    """Return an arm's refit times from 2024-07-01."""
    if schedule == "monthly":
        return [t for t in common.monthly_cuts(last=LAST_MONTHLY_REFIT) if t >= FIRST_REFIT]
    times = []
    moment = FIRST_REFIT
    while moment <= LAST_WEEKLY_REFIT:
        times.append(moment)
        moment += timedelta(days=7)
    return times


def served(
    *, refits: list[datetime], origins: dict[datetime, datetime]
) -> dict[datetime, list[datetime]]:
    """Return each refit and the runs whose origin lies from it to the next refit.

    Args:
        refits: The refit times, ascending.
        origins: Each served run's initialisation time mapped to its `power_fcst_init_time`.

    Returns:
        Each refit mapped to the run initialisation times it forecasts.
    """
    out: dict[datetime, list[datetime]] = {refit: [] for refit in refits}
    for run, origin in sorted(origins.items()):
        earlier = [refit for refit in refits if refit <= origin]
        if earlier:
            out[max(earlier)].append(run)
    return {refit: runs for refit, runs in out.items() if runs}


def known_rows(*, rows: pl.DataFrame, refit: datetime, first_usable: datetime) -> pl.DataFrame:
    """Return the rows a refit may see, and check that none comes from its future.

    Raises:
        ValueError: If no row is usable, or a kept row is newer than the refit allows.
    """
    latest_valid = refit - REFIT_CUTOFF_AGE
    known = rows.filter(
        (pl.col("valid_time") <= latest_valid)
        & (pl.col("power_fcst_init_time") <= refit)
        & (pl.col("valid_time") > first_usable)
    )
    too_new = known.filter(
        (pl.col("valid_time") > latest_valid) | (pl.col("power_fcst_init_time") > refit)
    ).height
    if known.is_empty() or too_new:
        msg = f"refit {refit}: no usable rows or a row from after the refit"
        raise ValueError(msg)
    return known


def lead_band(*, lead_hours: np.ndarray) -> np.ndarray:
    """Return the index of experiment 87's lead band (0-24 h, 1-2 d, 2-6 d, 6-10 d, 10+ d)."""
    return np.searchsorted(np.array(LEAD_BAND_EDGES), lead_hours, side="right")


class OffsetTables:
    """Experiment 64's solar-offset weights, held capacity, and the experiment 87 anomaly table."""

    def __init__(self, *, site: str, time_series_id: int, effective_capacity_mw: float) -> None:
        """Load the tables for one offset site."""
        self.site = site
        self.effective_capacity_mw = effective_capacity_mw
        weights = pl.read_parquet(E64_DIR / "weights.parquet").filter(
            (pl.col("variant") == "all") & (pl.col("site") == site)
        )
        self.weights = weights
        self.held_mw = float(
            pl.read_parquet(E64_DIR / "capacity.parquet").filter(pl.col("site") == site)[
                "capacity_mw"
            ][0]
        )
        self.anomaly = pl.read_parquet(ANOMALY_FILE).filter(
            pl.col("time_series_id") == time_series_id
        )

    def with_anomaly(self, *, frame: pl.DataFrame) -> pl.DataFrame:
        """Keep the rows that have a solar anomaly.

        Experiment 87's table lacks the anomaly of the rows of the last two weeks of June 2025
        (0.9% of the training rows), which are left out of the offset arms' training.
        """
        return frame.join(
            self.anomaly.select(*data.KEYS), on=list(data.KEYS), how="semi", maintain_order="left"
        )

    def shift(self, *, frame: pl.DataFrame) -> np.ndarray:
        """Return `w * held capacity / effective capacity * anomaly` for each row of the frame.

        The held capacity is used for training and prediction rows alike; experiment 87's capacity
        vintages of the scored year are not used.
        """
        joined = frame.select(*data.KEYS).join(
            self.anomaly, on=list(data.KEYS), how="left", validate="m:1", maintain_order="left"
        )
        if joined["anomaly"].null_count():
            msg = f"{joined['anomaly'].null_count()} rows have no solar anomaly at {self.site}"
            raise ValueError(msg)
        lead = (
            (frame["valid_time"] - frame["power_fcst_init_time"]).dt.total_minutes() / 60.0
        ).to_numpy()
        month = frame["valid_time"].dt.convert_time_zone("Europe/London").dt.month().to_numpy()
        band = lead_band(lead_hours=lead)
        table = (
            pl.DataFrame({"band": band.astype(np.int64), "month": month.astype(np.int64)})
            .with_columns(
                season=pl.col("month").replace_strict(
                    SEASON_OF_MONTH, default=None, return_dtype=pl.String
                )
            )
            .join(
                self.weights.select(pl.col("band").cast(pl.Int64), "season", "slope_applied"),
                on=["band", "season"],
                how="left",
                validate="m:1",
                maintain_order="left",
            )
        )
        weight = np.where(
            band >= NO_SKILL_BAND, 0.0, table["slope_applied"].fill_null(0.0).to_numpy()
        )
        return weight * (self.held_mw / self.effective_capacity_mw) * joined["anomaly"].to_numpy()


def parameters(*, arm: Arm, nthread: int) -> dict[str, object]:
    """Return the XGBoost parameters of an arm."""
    return {**QUANTILE_PARAMETERS, "seed": SEED, "eta": arm.eta, "nthread": nthread}


def tile(*, vector: np.ndarray) -> np.ndarray:
    """Return a base margin with the same value in each of the 11 quantile columns."""
    return np.tile(vector[:, None], (1, len(LEVELS))).astype(np.float32)


class SiteData:
    """A site's rows, power grid, margins, and offset, loaded once per worker."""

    def __init__(self, *, site: str) -> None:
        """Load the site's control rows, power array, DP fits, and offset tables."""
        self.cache = common.load_cache()
        self.site = site
        table = self.cache.sites[site].table
        self.table_row = {
            "time_series_id": table.time_series_id,
            "latitude": table.latitude,
            "longitude": table.longitude,
        }
        self.capacity = table.effective_capacity_mw
        self.first_usable = table.first_usable
        self.power = data.power_array(
            time_series_id=table.time_series_id, effective_capacity_mw=self.capacity
        )
        control = data.control_rows(table_row=self.table_row)
        self.control = control.filter(pl.col("power").is_not_null() & pl.col("power").is_finite())
        self.offset = (
            OffsetTables(
                site=site, time_series_id=table.time_series_id, effective_capacity_mw=self.capacity
            )
            if site in OFFSET_SITES
            else None
        )
        self.setup = ens.load_setup(cache=self.cache, site=site).with_calibration(
            calibration=arms.load_calibration(site=site)
        )
        self.fits = arms.load_arm_fits(arm=arms.ARMS["gam0"], site=site)
        self._margins: pl.DataFrame | None = None

    def with_lags(self, *, frame: pl.DataFrame) -> pl.DataFrame:
        """Add BG2's three origin lags to a frame of runs."""
        target = data.grid_index(times=frame["valid_time"])
        origin = data.grid_index(times=frame["power_fcst_init_time"])
        lags = data.origin_lags(power=self.power, target=target, origin=origin)
        return frame.with_columns(
            pl.DataFrame(lags, schema=list(data.LAG_COLUMNS), orient="row").cast(pl.Float32)
        )

    def control_margins(self) -> pl.DataFrame:
        """Return the DP's control-member margin (fraction) for every run's 168-hour window."""
        if self._margins is None:
            frames = []
            for index, run in enumerate(self.setup.runs):
                origin = self.setup.origins[index]
                if origin < self.fits[0].cut + timedelta(hours=3.5):
                    continue
                if len(self.cache.end_times) - run.origin_position - 1 < ens.MIN_WINDOW_HALF_HOURS:
                    continue
                if index < 9:
                    continue
                frames.append(
                    margin_frame(
                        cache=self.cache,
                        context=self.setup.context,
                        run=run,
                        origin=origin,
                        fits=self.fits,
                    ).select("power_fcst_init_time", "valid_time", "margin_fraction")
                )
            self._margins = pl.concat(frames)
        return self._margins


def start_margin(
    *,
    arm: Arm,
    site_data: SiteData,
    frame: pl.DataFrame,
    level: float | None,
    margin: np.ndarray | None,
) -> np.ndarray | None:
    """Return the base margin of rows, or `None` for XGBoost's own estimate.

    Args:
        arm: The arm.
        site_data: The site's data.
        frame: The rows (for the offset).
        level: The median of the refit's training labels.
        margin: The DP's forecast fraction for each row, for `gam`.

    Returns:
        Shape (rows, 11), or `None`.
    """
    if arm.start == "default":
        return None
    if arm.start == "zero":
        return tile(vector=np.zeros(frame.height))
    if arm.start == "gam":
        return tile(vector=np.asarray(margin))
    if site_data.offset is None:
        return None
    return tile(vector=(level or 0.0) + site_data.offset.shift(frame=frame))


def fit_refit(
    *, arm: Arm, site_data: SiteData, refit: datetime, nthread: int, rounds: int
) -> tuple[xgb.Booster, float, pl.DataFrame]:
    """Fit one refit and return the booster, the median label, and the training rows used."""
    known = known_rows(rows=site_data.control, refit=refit, first_usable=site_data.first_usable)
    if arm.max_train_lead is not None:
        known = known.filter(pl.col("lead_hours") <= arm.max_train_lead)
    if arm.start == "offset" and site_data.offset is not None:
        known = site_data.offset.with_anomaly(frame=known)
    if arm.lagged:
        known = site_data.with_lags(frame=known)
    margin = None
    if arm.start == "gam":
        margins = site_data.control_margins()
        known = known.join(
            margins,
            on=["power_fcst_init_time", "valid_time"],
            how="inner",
            validate="m:1",
            maintain_order="left",
        )
        margin = known["margin_fraction"].to_numpy()
    label = (known["power"] / site_data.capacity).to_numpy()
    level = float(np.median(label))
    age_days = (refit - known["valid_time"]).dt.total_seconds() / 86_400
    matrix = xgb.DMatrix(
        known.select(data.input_columns(lagged=arm.lagged)).to_numpy().astype(np.float32),
        label=label,
        weight=0.5 ** (age_days.to_numpy() / HALF_LIFE_DAYS),
        base_margin=start_margin(
            arm=arm, site_data=site_data, frame=known, level=level, margin=margin
        ),
    )
    booster = xgb.train(parameters(arm=arm, nthread=nthread), matrix, num_boost_round=rounds)
    return booster, level, known


def predict_run(
    *, arm: Arm, site_data: SiteData, booster: xgb.Booster, level: float, run: datetime
) -> pl.DataFrame:
    """Return one run's mixture quantiles in points, keyed by series, origin, and valid time."""
    frame = data.run_rows(run=run, table_row=site_data.table_row)
    if arm.start == "gam":
        last = site_data.cache.end_times[-1].astype("datetime64[us]").astype(datetime)
        frame = frame.filter(pl.col("valid_time") <= last.replace(tzinfo=UTC))
    if arm.lagged:
        frame = site_data.with_lags(frame=frame)
    margin = None
    if arm.start == "gam":
        margin = member_margins(site_data=site_data, run=run, frame=frame)
    matrix = xgb.DMatrix(
        frame.select(data.input_columns(lagged=arm.lagged)).to_numpy().astype(np.float32),
        base_margin=start_margin(
            arm=arm, site_data=site_data, frame=frame, level=level, margin=margin
        ),
    )
    raw = booster.predict(matrix)
    members = frame["ensemble_member"].n_unique()
    cube = raw.reshape(-1, members, len(LEVELS))
    keys = frame.select(*data.KEYS).unique(maintain_order=True)
    return keys.hstack(
        pl.DataFrame(100.0 * mixture_quantiles(member_quantiles=cube), schema=list(LEVEL_COLUMNS))
    )


def member_margins(*, site_data: SiteData, run: datetime, frame: pl.DataFrame) -> np.ndarray:
    """Return each member row's DP forecast fraction, driven by that member's weather."""
    from studies.dp_ens import member_means

    index = common.run_dates().index(run)
    result = member_means(
        cache=site_data.cache,
        context=site_data.setup.context,
        run=site_data.setup.runs[index],
        origin=site_data.setup.origins[index],
        fits=site_data.fits,
    )
    times = pl.Series(site_data.cache.end_times[result.window]).dt.replace_time_zone("UTC")
    table = pl.DataFrame(
        {
            "ensemble_member": np.repeat(np.array(result.members, dtype=np.int8), len(times)),
            "valid_time": pl.concat([times] * len(result.members)),
            "margin": (result.means / 100.0).ravel(),
        }
    )
    joined = frame.select("ensemble_member", "valid_time").join(
        table,
        on=["ensemble_member", "valid_time"],
        how="left",
        validate="m:1",
        maintain_order="left",
    )
    if joined["margin"].null_count():
        missing = joined["margin"].null_count()
        msg = f"{missing} member rows have no DP margin in the {run:%Y-%m-%d} run"
        raise ValueError(msg)
    return joined["margin"].to_numpy()


def job_path(*, arm: str, site: str, refit: datetime) -> Path:
    """Return the file one refit's predictions for one site are written to."""
    return common.WORK_DIR / "xgb_preds" / arm / f"{site}_{refit:%Y%m%dT%H%M}.parquet"


GPU_LOG: Final[Path] = common.WORK_DIR / "gpu_hours.jsonl"


def log_seconds(*, job: str, seconds: float) -> None:
    """Append a job's wall-clock seconds to the GPU-hour log (one small write per job)."""
    with GPU_LOG.open("a") as handle:
        handle.write(json.dumps({"job": job, "seconds": round(seconds, 1)}) + "\n")


_SITES: dict[str, SiteData] = {}


def run_job(job: tuple[str, str, datetime, list[datetime], int, int, str]) -> str:
    """Fit one refit for one site and write the runs it serves."""
    arm_name, site, refit, runs, nthread, rounds_override, folder = job
    arm = ARMS[arm_name]
    target = job_path(arm=folder, site=site, refit=refit)
    if target.exists():
        return f"{arm_name} {site} {refit:%Y-%m-%d}: exists"
    started = time.monotonic()
    if site not in _SITES:
        _SITES[site] = SiteData(site=site)
    site_data = _SITES[site]
    rounds = rounds_override or arm.rounds
    booster, level, known = fit_refit(
        arm=arm, site_data=site_data, refit=refit, nthread=nthread, rounds=rounds
    )
    pieces = [
        predict_run(arm=arm, site_data=site_data, booster=booster, level=level, run=run)
        for run in runs
    ]
    frame = pl.concat(pieces).with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
        / 60.0,
        seed=pl.lit(SEED).cast(pl.Int8),
        arm=pl.lit(arm_name),
        refit=pl.lit(refit),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    frame.write_parquet(temporary)
    temporary.rename(target)
    seconds = time.monotonic() - started
    log_seconds(job=f"{arm_name} {site} {refit:%Y-%m-%d}", seconds=seconds)
    return (
        f"{arm_name} {site} {refit:%Y-%m-%d}: {known.height:,} rows, {len(runs)} runs, "
        f"{seconds:.0f} s"
    )


def main() -> None:
    """Fit the requested arm's refits for the requested sites."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--sites", nargs="+", default=list(common.ALL_SITES))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--cores", type=int, default=6)
    parser.add_argument("--smoke", action="store_true", help="2 refits, 3 runs, 20 rounds")
    parser.add_argument("--only-refit", help="fit only this refit, as YYYY-MM-DD")
    parser.add_argument("--first", help="fit only refits on or after YYYY-MM-DD")
    parser.add_argument("--before", help="fit only refits before YYYY-MM-DD")
    args = parser.parse_args()
    arm = ARMS[args.arm]
    origins = {
        run: common.origin_of(run=run)
        for run in common.run_dates()
        if common.period_of_run(run=run) in SERVED_PERIODS
    }
    jobs_by_refit = served(refits=refit_times(schedule=arm.schedule), origins=origins)
    nthread = max(1, args.cores // args.workers)
    items = list(jobs_by_refit.items())
    if args.first:
        items = [(refit, runs) for refit, runs in items if f"{refit:%Y-%m-%d}" >= args.first]
    if args.before:
        items = [(refit, runs) for refit, runs in items if f"{refit:%Y-%m-%d}" < args.before]
    if args.only_refit:
        items = [(refit, runs) for refit, runs in items if f"{refit:%Y-%m-%d}" == args.only_refit]
    rounds = 0
    folder = f"{args.arm}_smoke" if args.smoke else args.arm
    if args.smoke:
        items = [(refit, runs[:3]) for refit, runs in items[:2]]
        rounds = SMOKE_ROUNDS
    jobs = [
        (args.arm, site, refit, runs, nthread, rounds, folder)
        for refit, runs in items
        for site in args.sites
    ]
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, line in enumerate(pool.map(run_job, jobs), start=1):
            print(f"[{done}/{len(jobs)}] {line}", flush=True)
    print(f"done in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
