"""Fit experiment 00031's arms and write their validation quantiles.

Each refitted arm is experiment 00013's N1 (the 00006 columns, the lead in hours, a recency
half-life of 2 years, 500 rounds, the 9-level quantile loss, the ENS control member from 2024-04-01
to 2025-06-30 23:30, `colsample` 1.0) plus the arm's capacity columns or base margin. The validation
quantiles are the weighted quantiles of the 51 members' 9 quantiles each. The 14-day correction and
the spread factor are applied later by `e00031_score.py`.

Arms: `a1` (F1 to F3), `a2` (F6 and F7 from the data-only estimate), `a3` (F6 to F8 from the
estimate that uses the register), `a4` (a linear PV offset on the directional Primaries), `a6` (A3
with the estimate shuffled across origin dates), `a7` (PV_Live generation at the valid time, GSPs
only). The arm `pens` writes the ensemble-mean `p_unit` at each validation valid time, for the
sunshine-shaped correction.

Output: `work/arms/<arm>_seed<s>.parquet`, keyed by series, run, valid time. Existing files are
kept.

    uv run python ml-experiments/00031-embedded-pv-features-primary-demand/e00031_fit.py \
        --arm a3 --seeds 0 1 2 [--smoke]
"""

import argparse
import time
from pathlib import Path
from typing import Final, Literal

import e00031_common as common
import e00031_ecr as ecr
import e00031_pv as pv
import numpy as np
import polars as pl
import xgboost as xgb

ARMS_DIR: Final[Path] = common.WORK_DIR / "arms"
FEATURES_PATH: Final[Path] = common.WORK_DIR / "features" / "origin_features.parquet"
KEYS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")
ArmType = Literal["a1", "a2", "a3", "a4", "a6", "a7"]
CAPACITY_STATISTICS: Final[tuple[str, ...]] = (
    "cap_slope_28d",
    "sunny_minus_cloudy_28d",
    "cap_change_yoy",
)
ARM_EXTRA_COLUMNS: Final[dict[ArmType, tuple[str, ...]]] = {
    "a1": CAPACITY_STATISTICS,
    "a2": ("pv_hat", "lag_gross_2d"),
    "a3": ("pv_hat", "lag_gross_2d", "pv_ecr_hat"),
    "a4": (),
    "a6": ("pv_hat", "lag_gross_2d", "pv_ecr_hat"),
    "a7": ("pv_live_share",),
}
ARM_CAPACITY_COLUMN: Final[dict[ArmType, str | None]] = {
    "a1": None,
    "a2": "cap_data_only",
    "a3": "cap_with_register",
    "a4": "cap_with_register",
    "a6": "cap_shuffled",
    "a7": None,
}
GSP_SITES: Final[tuple[str, ...]] = ("D1", "D6")
PV_LIVE_NAMES: Final[dict[str, str]] = {"D1": "BICF_1", "D6": "WALP_B"}
PV_LIVE_PATH: Final[Path] = (
    common.ML_EXPERIMENTS_DATA_DIR.parent
    / "studies"
    / "downloads"
    / "market"
    / "pv_live_gsp"
    / "pv_live_gsp.parquet"
)
LAG_BACK_STEPS: Final[int] = pv.FEATURE_DELAY_STEPS
"""The 48-hour lag of `lag_gross_2d`, in half-hours."""


def input_columns(*, arm: ArmType) -> list[str]:
    """Return an arm's input columns in matrix order."""
    return [
        *common.feature_columns(calendar=True),
        *common.N1_EXTRA_COLUMNS,
        *common.LAG_COLUMNS,
        *ARM_EXTRA_COLUMNS[arm],
    ]


def add_lags(*, frame: pl.DataFrame, power: np.ndarray) -> pl.DataFrame:
    """Add the three origin lags to a frame with a run's origin and valid times."""
    lags = common.origin_lags(
        power=power,
        target=np.asarray(common.grid_index(times=frame["valid_time"])),
        origin=np.asarray(common.grid_index(times=frame["power_fcst_init_time"])),
    )
    return frame.with_columns(
        pl.DataFrame(lags, schema=list(common.LAG_COLUMNS), orient="row").cast(pl.Float32)
    )


def shuffled_capacity(*, origin_table: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Permute `cap_with_register` across origins within the calendar month and the data split.

    Args:
        origin_table: One site's origin features, with `power_fcst_init_time`.
        seed: The arm's seed.

    Returns:
        The table with `cap_shuffled`. An origin's value comes from another origin of the same
        calendar month, in the training or the validation period alike.
    """
    rng = np.random.default_rng(seed)
    table = origin_table.with_columns(
        group=pl.col("power_fcst_init_time").dt.month().cast(pl.Int32) * 2
        + (pl.col("power_fcst_init_time") >= common.VALIDATION_START).cast(pl.Int32)
    )
    shuffled = np.full(table.height, np.nan)
    values = table["cap_with_register"].to_numpy()
    for group in table["group"].unique():
        positions = np.flatnonzero((table["group"] == group).to_numpy())
        shuffled[positions] = values[rng.permutation(positions)]
    return table.with_columns(cap_shuffled=pl.Series(shuffled)).drop("group")


def pv_live_share(*, site: str) -> np.ndarray:
    """Return PV_Live generation over the GSP's effective capacity on the grid, NaN if missing."""
    series = common.series_table().filter(pl.col("site") == site)
    capacity = series["effective_capacity_mw"][0]
    grid = pl.DataFrame({"valid_time": common.grid_times()}).with_columns(
        start=pl.col("valid_time") - common.STEP
    )
    live = pl.read_parquet(PV_LIVE_PATH).filter(pl.col("gsp_name") == PV_LIVE_NAMES[site])
    joined = grid.join(
        live.select(start="time", generation_mw="generation_mw"),
        on="start",
        how="left",
        validate="1:1",
        maintain_order="left",
    )
    return (joined["generation_mw"] / capacity).to_numpy().astype(np.float32)


class SiteInputs:
    """Everything an arm needs about one site, built once."""

    def __init__(self, *, row: dict, power: np.ndarray, arm: ArmType, seed: int) -> None:
        """Load a site's grid, origin features, and arm-specific series.

        Args:
            row: The site's row of `common.series_table()`.
            power: The site's observed power over effective capacity on the grid.
            arm: The arm.
            seed: The arm's seed, used for the shuffled control only.
        """
        self.row = row
        self.power = power
        self.grid = pv.site_grid(
            site=row["site"], latitude=row["latitude"], longitude=row["longitude"]
        )
        table = (
            pl.read_parquet(FEATURES_PATH)
            .filter(pl.col("site") == row["site"])
            .drop("time_series_id", "site", "role")
        )
        self.origin_table = shuffled_capacity(origin_table=table, seed=seed)
        self.live = pv_live_share(site=row["site"]) if arm == "a7" else None

    def p_ens(self, *, frame: pl.DataFrame) -> np.ndarray:
        """Return `p_unit` from each row's ENS `shortwave` at its valid time."""
        index = np.asarray(common.grid_index(times=frame["valid_time"]))
        return pv.unit_power(
            ghi_w_m2=frame["shortwave"].to_numpy().astype(np.float64),
            zenith_deg=self.grid.zenith_deg[index],
            azimuth_deg=self.grid.azimuth_deg[index],
            day_of_year=self.grid.day_of_year[index],
        )


def add_arm_columns(*, frame: pl.DataFrame, inputs: SiteInputs, arm: ArmType) -> pl.DataFrame:
    """Add an arm's extra columns, and `margin` for the offset arm, to rows of one site.

    Args:
        frame: Rows with `power_fcst_init_time`, `valid_time`, and `shortwave`.
        inputs: The site's inputs.
        arm: The arm.

    Returns:
        The frame with the arm's `ARM_EXTRA_COLUMNS` and, for `a4`, `margin` (minus the offset).
    """
    capacity_column = ARM_CAPACITY_COLUMN[arm]
    joined = frame.join(
        inputs.origin_table,
        on="power_fcst_init_time",
        how="left",
        validate="m:1",
        maintain_order="left",
    )
    if arm == "a7":
        assert inputs.live is not None
        index = np.asarray(common.grid_index(times=joined["valid_time"]))
        return joined.with_columns(
            pv_live_share=pl.Series(common.take(values=inputs.live, index=index))
        )
    if capacity_column is None:
        return joined
    capacity = joined[capacity_column].to_numpy().astype(np.float64)
    p_ens = inputs.p_ens(frame=joined)
    target = np.asarray(common.grid_index(times=joined["valid_time"]))
    origin = np.asarray(common.grid_index(times=joined["power_fcst_init_time"]))
    newest = origin - LAG_BACK_STEPS
    lag_index = target - common.STEPS_PER_DAY * np.ceil(
        (target - newest) / common.STEPS_PER_DAY
    ).astype(int)
    lag_power = common.take(values=inputs.power, index=lag_index)
    lag_pv = common.take(values=inputs.grid.cams_unit_power, index=lag_index)
    columns = {
        "pv_hat": capacity * p_ens,
        "lag_gross_2d": lag_power + capacity * lag_pv,
        "pv_ecr_hat": joined["register_share"].to_numpy() * p_ens,
        "margin": -capacity * p_ens,
    }
    return joined.with_columns(
        **{name: pl.Series(values).cast(pl.Float32) for name, values in columns.items()}
    )


def mixture_quantiles(*, member_quantiles: np.ndarray) -> np.ndarray:
    """Pool members' quantiles into the mixture's 9 quantiles.

    Args:
        member_quantiles: Shape (cases, members, 9), each member's quantiles at `LEVELS`.

    Returns:
        Shape (cases, 9): the weighted quantiles of the pooled values, each member weighted equally
        and each level by its atom mass, sorted along the levels so none cross.
    """
    cases, members, levels = member_quantiles.shape
    values = member_quantiles.reshape(cases, members * levels)
    weights = np.tile(common.ATOM_MASSES, members) / members
    order = np.argsort(values, axis=1)
    sorted_values = np.take_along_axis(values, order, axis=1)
    cumulative = np.cumsum(weights[order], axis=1)
    picks = (cumulative[:, :, None] >= (common.LEVELS - 1e-9)).argmax(axis=1)
    pooled = np.take_along_axis(sorted_values, picks, axis=1)
    return np.sort(pooled, axis=1).astype(np.float32)


def _matrix(*, frame: pl.DataFrame, arm: ArmType) -> xgb.DMatrix:
    """Return the feature matrix of an arm, with the base margin for the offset arm."""
    matrix = xgb.DMatrix(frame.select(input_columns(arm=arm)).to_numpy().astype(np.float32))
    if arm == "a4":
        margin = np.nan_to_num(frame["margin"].to_numpy().astype(np.float32))
        matrix.set_base_margin(np.tile(margin[:, None], (1, len(common.LEVELS))).ravel())
    return matrix


def fit_series(*, inputs: SiteInputs, train: pl.DataFrame, arm: ArmType, seed: int) -> xgb.Booster:
    """Fit one series' quantile model for an arm and seed."""
    if train["ensemble_member"].unique().to_list() != [0]:
        msg = "the training frame must hold the control member (0) only"
        raise ValueError(msg)
    row = inputs.row
    own = train.filter(
        (pl.col("time_series_id") == row["time_series_id"])
        & pl.col("power").is_finite()
        & (pl.col("valid_time") <= common.LAST_TRAINING_VALID)
    )
    own = common.with_derived(frame=own, latitude=row["latitude"], longitude=row["longitude"])
    own = add_lags(frame=own, power=inputs.power)
    own = add_arm_columns(frame=own, inputs=inputs, arm=arm)
    age_days = (common.LAST_TRAINING_VALID - own["valid_time"]).dt.total_seconds() / 86_400
    weights = 0.5 ** (age_days.to_numpy() / common.RECENCY_HALF_LIFE_DAYS)
    matrix = _matrix(frame=own, arm=arm)
    matrix.set_label((own["power"] / row["effective_capacity_mw"]).to_numpy())
    matrix.set_weight(weights)
    return xgb.train(
        {**common.QUANTILE_PARAMETERS, "seed": seed}, matrix, num_boost_round=common.N_ROUNDS
    )


def _validation_run(*, path: Path, inputs: SiteInputs, arm: ArmType) -> pl.DataFrame:
    """Return one validation run's rows for a site, with lags and the arm's columns."""
    row = inputs.row
    run = pl.read_parquet(path).filter(pl.col("time_series_id") == row["time_series_id"])
    run = common.with_derived(frame=run, latitude=row["latitude"], longitude=row["longitude"])
    run = add_lags(frame=run, power=inputs.power).sort("valid_time", "ensemble_member")
    members = run["ensemble_member"].n_unique()
    if run.height != run.select(*KEYS).n_unique() * members:
        msg = "every valid time must hold every member"
        raise ValueError(msg)
    if run["ensemble_member"].min() != 0:
        msg = "member 0, the control, must be the first member"
        raise ValueError(msg)
    return add_arm_columns(frame=run, inputs=inputs, arm=arm)


def predict_series(*, inputs: SiteInputs, model: xgb.Booster, arm: ArmType) -> pl.DataFrame:
    """Return the mixture quantiles of every validation run."""
    mixtures = []
    for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet")):
        run = _validation_run(path=path, inputs=inputs, arm=arm)
        members = run["ensemble_member"].n_unique()
        raw = model.predict(_matrix(frame=run, arm=arm)).reshape(-1, members, len(common.LEVELS))
        keys = run.select(*KEYS).unique(maintain_order=True)
        mixtures.append(
            keys.with_columns(
                pl.DataFrame(
                    mixture_quantiles(member_quantiles=raw),
                    schema=list(common.LEVEL_COLUMNS),
                    orient="row",
                )
            )
        )
    return pl.concat(mixtures)


def ensemble_mean_unit_power(*, inputs: SiteInputs) -> pl.DataFrame:
    """Return, for every validation run and valid time, the mean `p_unit` over the members."""
    pieces = []
    for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet")):
        row = inputs.row
        run = pl.read_parquet(path).filter(pl.col("time_series_id") == row["time_series_id"])
        run = run.with_columns(p_ens=pl.Series(inputs.p_ens(frame=run)))
        pieces.append(
            run.group_by(*KEYS, maintain_order=True).agg(p_ens_mean=pl.col("p_ens").mean())
        )
    return pl.concat(pieces)


def sites_for(*, arm: str, series: pl.DataFrame) -> pl.DataFrame:
    """Return the series an arm is fitted on."""
    if arm == "a4":
        directional = ecr.directional_sites(series=series)
        return series.filter(
            (pl.col("substation_type") == "Primary") & pl.col("site").is_in(directional)
        )
    if arm == "a7":
        return series.filter(pl.col("site").is_in(GSP_SITES))
    return series


def main() -> None:
    """Fit and write an arm for the requested seeds."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=[*ARM_EXTRA_COLUMNS, "pens"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(common.SEEDS))
    parser.add_argument("--smoke", action="store_true", help="two series, in a scratch folder")
    args = parser.parse_args()
    out_dir = ARMS_DIR.with_name("arms_smoke") if args.smoke else ARMS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    full_series = common.series_table()
    series = sites_for(arm=args.arm, series=full_series)
    if args.smoke:
        series = series.head(2)
    power = common.load_power(series=full_series)
    if args.arm == "pens":
        path = out_dir / "pens.parquet"
        if not path.exists():
            pieces = [
                ensemble_mean_unit_power(
                    inputs=SiteInputs(row=row, power=power[row["site"]], arm="a1", seed=0)
                )
                for row in series.iter_rows(named=True)
            ]
            pl.concat(pieces).write_parquet(path)
        return
    train = pl.read_parquet(common.ENS_TRAIN_PATH)
    for seed in args.seeds:
        path = out_dir / f"{args.arm}_seed{seed}.parquet"
        if path.exists():
            continue
        pieces = []
        for row in series.iter_rows(named=True):
            started = time.monotonic()
            inputs = SiteInputs(row=row, power=power[row["site"]], arm=args.arm, seed=seed)
            model = fit_series(inputs=inputs, train=train, arm=args.arm, seed=seed)
            pieces.append(predict_series(inputs=inputs, model=model, arm=args.arm))
            print(
                f"{args.arm} seed {seed} {row['site']}: {time.monotonic() - started:.0f} s",
                flush=True,
            )
        frame = pl.concat(pieces).with_columns(seed=pl.lit(seed).cast(pl.Int8))
        tmp = out_dir / f"{args.arm}_seed{seed}.tmp"
        frame.write_parquet(tmp)
        tmp.rename(path)


if __name__ == "__main__":
    main()
