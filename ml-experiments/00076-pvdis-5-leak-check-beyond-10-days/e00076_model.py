"""Training matrices, validation runs, and shifted forecasts shared by the fits of experiment 00076.

Copied from experiment 00064's `e00064_model.py`, with one added arm, `am0` (see `training_matrix`).
The rest of this docstring describes experiment 00064's arms.

Every arm is experiment 00046's `d1l` recipe (experiment 00033's N1: half-life 2 years, one
nine-level quantile model per series and seed, trained on the ENS control member) with the embedded
solar term attached in one of four ways. All quantities are in power over the series' effective
capacity, so the solar term is `scale * shape` with `scale = capacity_mw / effective_capacity_mw`.

- `sf`: the stored `d1l` quantiles plus `w * scale * anomaly`, where `anomaly` is the
  ensemble-mean calibrated ENS anomaly and `w` the fitted weight (negative: a sunnier day than
  usual lowers net power). Predict-only, so the sign convention is "added", not "subtracted".
- `a`: one extra input column, `scale * (clim_cams + anomaly)`, the calibrated shape. In training
  the anomaly is the ensemble mean; at inference each member row has its own calibrated anomaly.
- `am`: XGBoost `base_margin` of `w * scale * anomaly` (ensemble-mean anomaly in training and at
  inference; the climatology is left to the trees), repeated for the nine quantile outputs. A
  model given a margin starts from the margin alone, not from XGBoost's estimated `base_score`.
- `b`: trained on gross demand (power plus `scale` times the CAMS shape) with gross lags (each net
  lag plus `scale` times the unit shape at the lagged time, see `e00076_inputs.py`); at inference
  each member's gross quantiles are shifted down by `scale * (clim_cams + anomaly)`.

The placebo twins replace the ensemble-mean calibrated anomaly by `anomaly_mean_calibrated_twin`
(a member's own deviation from the mean is kept on top at inference for `a_twin`).

Choices where the plan is silent: a month outside April to September has a weight of zero; a
missing anomaly, climatology, or capacity counts as zero in the shift; a missing lag shape drops a
training row of `b` and leaves a NaN gross lag at inference.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00076_common as common
import numpy as np
import polars as pl
import xgboost as xgb

RADIATION_COLUMNS: Final[tuple[str, ...]] = ("shortwave", "longwave")
CLOUD_PROXY_COLUMNS: Final[tuple[str, ...]] = (
    "dew_point_temperature_2m",
    "precipitation",
    "precipitation_type",
    "pressure_surface",
    "pressure_msl",
)
"""The cloud-proxy inputs; `d1l` holds those in experiment 00006's frozen column list."""
SOLAR_TERM_COLUMN: Final[str] = "solar_term"
WEIGHT_VARIANTS: Final[dict[str, tuple[str, ...]]] = {
    "all": (),
    "no_lags": common.LAG_COLUMNS,
    "no_cloud_proxies": CLOUD_PROXY_COLUMNS,
    "neither": (*common.LAG_COLUMNS, *CLOUD_PROXY_COLUMNS),
}
"""The four `d1l` variants of the slope diagnostic, each mapped to the columns it drops."""
N_QUANTILES: Final[int] = len(common.LEVELS)
ANOMALY_COLUMN: Final[str] = "anomaly_mean_calibrated"
TWIN_COLUMN: Final[str] = "anomaly_mean_calibrated_twin"
SHAPE_LAG_NAMES: Final[tuple[str, ...]] = common.FIT_LAG_SHAPE_COLUMNS

NO_SKILL_BAND: Final[int] = 4
"""Lead bands from this index on (10+ days) get a weight of zero.

The training-window correlation of the ENS anomaly with the CAMS anomaly is 0.02 to 0.04 there
(experiment 46), so the applied-shift slope is a ratio of near-zero numbers (it ranges from -8.2
to +6.2 across the series). The rule was fixed from the collected weights, which use training rows
only, before any arm was scored.
"""


def d1l_columns(*, drop: tuple[str, ...] = ()) -> list[str]:
    """Return the input columns of `d1l` in matrix order, less any columns in `drop`.

    Args:
        drop: Columns to remove; the ones `d1l` does not hold are ignored.

    Returns:
        The names (15 when `drop` is empty).

    Raises:
        ValueError: If the full list does not hold 15 columns.
    """
    columns = [
        *common.feature_columns(calendar=True),
        *common.N1_EXTRA_COLUMNS,
        *common.LAG_COLUMNS,
    ]
    columns = [c for c in columns if c not in RADIATION_COLUMNS]
    if len(columns) != 15:
        msg = f"d1l has {len(columns)} columns, expected 15"
        raise ValueError(msg)
    return [c for c in columns if c not in drop]


@dataclass(frozen=True)
class SeriesContext:
    """One series' table row, arrays, and capacity.

    Attributes:
        row: The series' row of `series_table()`.
        arrays: Its arrays on the dense grid.
        capacity_mw: Its embedded-solar capacity, MW.
    """

    row: dict
    arrays: common.SeriesArrays
    capacity_mw: float

    @property
    def site(self) -> str:
        """Return the anonymised label, such as `D9`."""
        return self.row["site"]

    @property
    def scale(self) -> float:
        """Return the capacity over the effective capacity."""
        return self.capacity_mw / self.row["effective_capacity_mw"]


def load_context(*, site: str, capacity_scale: float = 1.0) -> SeriesContext:
    """Load a series' table row, arrays, and capacity.

    Args:
        site: The anonymised label.
        capacity_scale: Multiplier on the capacity (zero for the zero-shift checks).

    Returns:
        The context.
    """
    series = common.series_table().filter(pl.col("site") == site)
    arrays = common.load_series_arrays(series=series)
    capacity = common.capacities().filter(pl.col("site") == site)["capacity_mw"][0]
    return SeriesContext(
        row=series.row(0, named=True), arrays=arrays[site], capacity_mw=capacity * capacity_scale
    )


def add_lags(*, frame: pl.DataFrame, power: np.ndarray) -> pl.DataFrame:
    """Add the three net-power lags to a frame with a run's origin and valid times."""
    lags = common.origin_lags(
        power=power,
        target=np.asarray(common.grid_index(times=frame["valid_time"])),
        origin=np.asarray(common.grid_index(times=frame["power_fcst_init_time"])),
    )
    return frame.with_columns(
        pl.DataFrame(lags, schema=list(common.LAG_COLUMNS), orient="row").cast(pl.Float32)
    )


def weights_per_row(
    *, weights: pl.DataFrame, site: str, band: np.ndarray, month: np.ndarray
) -> np.ndarray:
    """Return each row's fitted weight, zero where its band and season has none.

    Args:
        weights: The `weights.parquet` table restricted to the variant used (`slope_applied`).
        site: The series label.
        band: Each row's lead-band index.
        month: Each row's local calendar month.

    Returns:
        A float64 array with the applied-shift slope of the row's lead band and season.
    """
    frame = (
        pl.DataFrame({"band": band.astype(np.int64), "month": month.astype(np.int64)})
        .with_columns(season=common.season_of_month_expr(month=pl.col("month")))
        .join(
            weights.filter(pl.col("site") == site).select(
                pl.col("band").cast(pl.Int64), "season", "slope_applied"
            ),
            on=["band", "season"],
            how="left",
            validate="m:1",
            maintain_order="left",
        )
    )
    cells = weights.filter(pl.col("site") == site)
    if cells.height != len(common.LEAD_BANDS) * 2:
        msg = f"{site} has {cells.height} weight cells, expected {len(common.LEAD_BANDS) * 2}"
        raise ValueError(msg)
    slope = frame["slope_applied"].fill_null(0.0).to_numpy()
    return np.where(band >= NO_SKILL_BAND, 0.0, slope)


def read_weights(*, path: Path = common.WEIGHTS_FILE) -> pl.DataFrame:
    """Return the table of weights that feeds `sf` and `am`: the "all inputs" variant."""
    return pl.read_parquet(path).filter(pl.col("variant") == "all")


def training_rows(*, ctx: SeriesContext) -> pl.DataFrame:
    """Return one series' training rows with lags, the solar inputs, and the placebo anomaly.

    Args:
        ctx: The series' context.

    Returns:
        The finite-power rows of experiment 00003's training file for the series (valid time after
        the origin), with the derived columns, `lead_hours`, the three net lags, and from
        `e00076_inputs.py`: `anomaly_mean_calibrated`, `anomaly_mean_calibrated_twin`, `clim_cams`,
        `shape_calibrated`, `u_cams`, `month` (local), `band`, `daylight`, `anomaly_cams`, and the
        three `shape_lag_*` columns.

    Raises:
        ValueError: If a row is later than the last training valid time or has no solar inputs.
    """
    series_id = ctx.row["time_series_id"]
    train = (
        pl.scan_parquet(common.ENS_TRAIN_PATH)
        .filter(
            (pl.col("time_series_id") == series_id)
            & (pl.col("valid_time") <= common.LAST_TRAINING_VALID)
        )
        .collect()
    )
    own = train.filter(
        pl.col("power").is_finite() & (pl.col("valid_time") > pl.col("power_fcst_init_time"))
    )
    own = common.with_derived(
        frame=own, latitude=ctx.row["latitude"], longitude=ctx.row["longitude"]
    ).with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    )
    own = add_lags(frame=own, power=ctx.arrays.power)
    common.require_training_window(times=own["valid_time"], what="training rows")
    keys = list(common.KEYS)
    shape = pl.read_parquet(common.INPUTS_DIR / "train_shape.parquet").filter(
        pl.col("time_series_id") == series_id
    )
    twin = pl.read_parquet(common.INPUTS_DIR / "placebo_train.parquet").filter(
        pl.col("time_series_id") == series_id
    )
    lag_shapes = pl.read_parquet(common.INPUTS_DIR / "lag_shapes_train.parquet").filter(
        pl.col("time_series_id") == series_id
    )
    for table, name in ((shape, "train_shape"), (twin, "placebo_train")):
        common.require_training_window(times=table["valid_time"], what=name)
    joined = (
        own.join(
            shape.select(
                *keys,
                ANOMALY_COLUMN,
                "clim_cams",
                "shape_calibrated",
                "u_cams",
                "month",
                "band",
                "daylight",
                "anomaly_cams",
            ),
            on=keys,
            how="left",
            validate="m:1",
            maintain_order="left",
        )
        .join(
            twin.select(*keys, TWIN_COLUMN),
            on=keys,
            how="left",
            validate="m:1",
            maintain_order="left",
        )
        .join(
            lag_shapes.select(*keys, *SHAPE_LAG_NAMES),
            on=keys,
            how="left",
            validate="m:1",
            maintain_order="left",
        )
    )
    if joined["shape_calibrated"].null_count() or joined[TWIN_COLUMN].null_count():
        msg = f"{ctx.site}: some training rows have no solar inputs"
        raise ValueError(msg)
    return joined


def gross_training_rows(*, rows: pl.DataFrame, ctx: SeriesContext) -> pl.DataFrame:
    """Turn net training rows into gross rows for arm `b`.

    The gross power is the power plus the capacity times the CAMS unit shape at the valid time.
    Each lag column becomes the net lag plus the scale times the unit shape at the lagged time.
    A row is dropped when its CAMS shape is missing, or when a lag has a net value and no shape.

    Args:
        rows: `training_rows()`.
        ctx: The series' context.

    Returns:
        The kept rows with `power` replaced by the gross power in MW and the lags made gross.
    """
    kept = rows.filter(pl.col("u_cams").is_not_null())
    for lag, shape in zip(common.LAG_COLUMNS, SHAPE_LAG_NAMES, strict=True):
        kept = kept.filter(~(pl.col(lag).is_finite() & pl.col(shape).is_null()))
    return kept.with_columns(
        power=pl.col("power") + ctx.capacity_mw * pl.col("u_cams"),
        **{
            lag: pl.col(lag) + ctx.scale * pl.col(shape)
            for lag, shape in zip(common.LAG_COLUMNS, SHAPE_LAG_NAMES, strict=True)
        },
    )


def margin_rows(*, weight: np.ndarray, anomaly: np.ndarray, scale: float) -> np.ndarray:
    """Return `am`'s base margin, shape (rows, 9): `w * scale * anomaly` for every quantile.

    The climatology never enters; only the calibrated anomaly is multiplied.

    Args:
        weight: The fitted weight of each row.
        anomaly: The ensemble-mean calibrated anomaly of each row.
        scale: The capacity over the effective capacity.

    Returns:
        A float32 array of shape (rows, 9).
    """
    return np.tile((weight * scale * anomaly)[:, None], (1, N_QUANTILES)).astype(np.float32)


def training_matrix(
    *,
    arm: str,
    rows: pl.DataFrame,
    ctx: SeriesContext,
    columns: list[str],
    weights: pl.DataFrame | None,
) -> xgb.DMatrix:
    """Return one arm's recency-weighted training matrix for one series.

    Args:
        arm: `d1l`, `a`, `am`, `am0`, `b`, `a_twin`, or `am_twin`. `am0` has a base margin of
            zero in every row, so it starts from zero like `am` but carries no solar term.
        rows: `training_rows()`.
        ctx: The series' context.
        columns: The input columns before any solar column (`SOLAR_TERM_COLUMN` is added for `a`).
        weights: The `read_weights()` table, needed by `am` and `am_twin`.

    Returns:
        The matrix: features, the label (power over effective capacity, gross for `b`), the
        recency weight (half-life 2 years from the last training valid time), and for `am` the
        base margin.

    Raises:
        ValueError: If `am` is asked for without weights.
    """
    base = arm.removesuffix("_twin")
    anomaly = rows[TWIN_COLUMN if arm.endswith("_twin") else ANOMALY_COLUMN].to_numpy()
    margin = None
    if base == "a":
        rows = rows.with_columns(
            **{SOLAR_TERM_COLUMN: ctx.scale * (rows["clim_cams"] + pl.Series(anomaly))}
        )
        columns = [*columns, SOLAR_TERM_COLUMN]
    elif base == "am":
        if weights is None:
            msg = "arm am needs the table of weights"
            raise ValueError(msg)
        weight = weights_per_row(
            weights=weights,
            site=ctx.site,
            band=rows["band"].to_numpy(),
            month=rows["month"].to_numpy(),
        )
        margin = margin_rows(weight=weight, anomaly=anomaly, scale=ctx.scale)
    elif base == "am0":
        margin = np.zeros((rows.height, N_QUANTILES), dtype=np.float32)
    elif base == "b":
        rows = gross_training_rows(rows=rows, ctx=ctx)
    age_days = (common.LAST_TRAINING_VALID - rows["valid_time"]).dt.total_seconds() / 86_400
    matrix = xgb.DMatrix(
        rows.select(columns).to_numpy().astype(np.float32),
        label=(rows["power"] / ctx.row["effective_capacity_mw"]).to_numpy(),
        weight=0.5 ** (age_days.to_numpy() / common.RECENCY_HALF_LIFE_DAYS),
        base_margin=margin,
    )
    if matrix.num_col() != len(columns):
        msg = f"matrix has {matrix.num_col()} columns, expected {len(columns)}"
        raise ValueError(msg)
    return matrix


def fit_booster(
    *,
    matrix: xgb.DMatrix,
    seed: int,
    n_rounds: int,
    device: str,
    extra_parameters: dict[str, float] | None = None,
) -> xgb.Booster:
    """Fit one nine-level quantile model with experiment 00033's N1 parameters.

    Args:
        matrix: The training matrix.
        seed: The fitting seed.
        n_rounds: Boosting rounds.
        device: `cuda` or `cpu`.
        extra_parameters: Parameters that override the N1 set (the smoke run sets `eta` to zero).

    Returns:
        The fitted booster.
    """
    return xgb.train(
        {**common.QUANTILE_PARAMETERS, "seed": seed, "device": device, **(extra_parameters or {})},
        matrix,
        num_boost_round=n_rounds,
    )


def save_booster(*, booster: xgb.Booster, path: Path) -> None:
    """Save a booster through a temporary name so a crash leaves no half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp.ubj")
    booster.save_model(temporary)
    temporary.rename(path)


def anomaly_matrix(
    *, anomaly_file: pl.DataFrame, site: str, valid_times: pl.Series, column: str
) -> np.ndarray:
    """Return a run's ENS anomaly as an array of shape (cases, members), 0 where unknown.

    Args:
        anomaly_file: A validation run's `ens_val` file of experiment 00046.
        site: The site label.
        valid_times: The run's sorted distinct valid times, in case order.
        column: `anomaly`, `anomaly_calibrated`, or `anomaly_mean_calibrated`.

    Returns:
        The anomalies per case and member, members in ascending order, NaN replaced by zero.

    Raises:
        ValueError: If the file's valid times differ from the run's.
    """
    own = anomaly_file.filter(pl.col("site") == site).sort("valid_time", "ensemble_member")
    if not own["valid_time"].unique(maintain_order=True).equals(valid_times):
        msg = f"the anomaly file of {site} does not cover the run's valid times"
        raise ValueError(msg)
    values = own[column].to_numpy().astype(np.float64).reshape(valid_times.len(), -1)
    return np.nan_to_num(values, nan=0.0)


@dataclass(frozen=True)
class RunSolar:
    """The solar inputs of one validation run, one value per case (valid time) unless noted.

    Attributes:
        clim: The CAMS climatology of the valid time's month and half-hour (0 where missing).
        anomaly_mean: The ensemble-mean calibrated anomaly.
        anomaly_member: The calibrated anomaly of each member, shape (cases, members).
        anomaly_twin: The placebo twin of `anomaly_mean`.
        weight: The fitted weight of the case's lead band and season (0 when none).
        temperature_mean: The member-mean 2 m temperature, degrees C.
    """

    clim: np.ndarray
    anomaly_mean: np.ndarray
    anomaly_member: np.ndarray
    anomaly_twin: np.ndarray
    weight: np.ndarray
    temperature_mean: np.ndarray


@dataclass(frozen=True)
class RunData:
    """One validation run of one series.

    Attributes:
        file_name: The run file's name, such as `init=2025-07-01.parquet`.
        keys: The distinct `KEYS`, one row per case, sorted by valid time.
        members: One row per case and member, sorted by valid time then member, with the derived
            columns, the net lags, and the three `shape_lag_*` columns.
        n_members: Members per case.
        solar: The run's solar inputs.
    """

    file_name: str
    keys: pl.DataFrame
    members: pl.DataFrame
    n_members: int
    solar: RunSolar


def validation_runs(
    *,
    ctx: SeriesContext,
    weights: pl.DataFrame | None,
    max_runs: int | None = None,
) -> Iterator[RunData]:
    """Yield every validation run of one series with its lags and solar inputs.

    Args:
        ctx: The series' context.
        weights: The `read_weights()` table; a weight of zero everywhere when `None`.
        max_runs: Only the first this many runs when given.

    Yields:
        One `RunData` per run file.

    Raises:
        ValueError: If a valid time does not hold all 51 members.
    """
    series_id = ctx.row["time_series_id"]
    climatology = pl.read_parquet(common.E46_SOLAR_DIR / "climatology.parquet").filter(
        pl.col("site") == ctx.site
    )
    twin = pl.read_parquet(common.INPUTS_DIR / "placebo_val.parquet").filter(
        pl.col("time_series_id") == series_id
    )
    lag_shapes = pl.read_parquet(common.INPUTS_DIR / "lag_shapes_val.parquet").filter(
        pl.col("time_series_id") == series_id
    )
    keys_list = list(common.KEYS)
    paths = sorted(common.VALIDATION_DIR.glob("init=*.parquet"))
    for path in paths[:max_runs]:
        run = pl.read_parquet(path).filter(pl.col("time_series_id") == series_id)
        run = common.with_derived(
            frame=run, latitude=ctx.row["latitude"], longitude=ctx.row["longitude"]
        )
        run = add_lags(frame=run, power=ctx.arrays.power).sort("valid_time", "ensemble_member")
        members = run["ensemble_member"].n_unique()
        keys = run.select(*keys_list).unique(maintain_order=True)
        if run.height != keys.height * members or members != common.MEMBERS:
            msg = "every valid time must hold all 51 members"
            raise ValueError(msg)
        run = run.join(
            lag_shapes.select(*keys_list, *SHAPE_LAG_NAMES),
            on=keys_list,
            how="left",
            validate="m:1",
            maintain_order="left",
        )
        times = keys["valid_time"]
        anomaly_file = pl.read_parquet(common.E46_SOLAR_DIR / "ens_val" / path.name)
        member = anomaly_matrix(
            anomaly_file=anomaly_file, site=ctx.site, valid_times=times, column="anomaly_calibrated"
        )
        mean = anomaly_matrix(
            anomaly_file=anomaly_file,
            site=ctx.site,
            valid_times=times,
            column="anomaly_mean_calibrated",
        )[:, 0]
        cells = common.with_cell(frame=keys, time_column="valid_time").join(
            climatology.select("month", "hh", "clim_cams"),
            on=["month", "hh"],
            how="left",
            validate="m:1",
            maintain_order="left",
        )
        twin_values = keys.join(
            twin.select(*keys_list, TWIN_COLUMN),
            on=keys_list,
            how="left",
            validate="m:1",
            maintain_order="left",
        )[TWIN_COLUMN]
        if twin_values.null_count():
            msg = f"{ctx.site}: the placebo table does not cover {path.name}"
            raise ValueError(msg)
        lead = ((times - keys["power_fcst_init_time"]).dt.total_minutes() / 60.0).to_numpy()
        weight = (
            weights_per_row(
                weights=weights,
                site=ctx.site,
                band=common.lead_band(lead_hours=lead),
                month=cells["month"].to_numpy(),
            )
            if weights is not None
            else np.zeros(keys.height)
        )
        solar = RunSolar(
            clim=cells["clim_cams"].fill_null(0.0).to_numpy(),
            anomaly_mean=mean,
            anomaly_member=member,
            anomaly_twin=np.nan_to_num(twin_values.to_numpy(), nan=0.0),
            weight=weight,
            temperature_mean=run["temperature_2m"]
            .to_numpy()
            .astype(np.float64)
            .reshape(keys.height, members)
            .mean(axis=1),
        )
        yield RunData(file_name=path.name, keys=keys, members=run, n_members=members, solar=solar)


def predict_members(
    *,
    booster: xgb.Booster,
    members: pl.DataFrame,
    columns: list[str],
    margin: np.ndarray | None = None,
    n_members: int = common.MEMBERS,
) -> np.ndarray:
    """Return the per-member quantiles of a run.

    Args:
        booster: The fitted nine-level model.
        members: The run's member rows, sorted by valid time then member, holding every column in
            `columns`.
        columns: The input columns in matrix order.
        margin: Per-case base margin of shape (cases, 9) repeated across members, for `am`.
        n_members: Members per case.

    Returns:
        Shape (cases, members, 9), float32.
    """
    base_margin = None if margin is None else np.repeat(margin, n_members, axis=0)
    matrix = xgb.DMatrix(
        members.select(columns).to_numpy().astype(np.float32), base_margin=base_margin
    )
    if base_margin is not None and matrix.get_base_margin().shape != base_margin.shape:
        msg = "the base margin was not set on the inference matrix"
        raise ValueError(msg)
    raw = booster.predict(matrix)
    return raw.reshape(-1, n_members, N_QUANTILES).astype(np.float32)


def with_solar_term(*, run: RunData, ctx: SeriesContext, twin: bool) -> pl.DataFrame:
    """Return the member rows with arm `a`'s input column.

    Args:
        run: The validation run.
        ctx: The series' context.
        twin: Replace the ensemble-mean anomaly by its placebo twin, keeping each member's own
            deviation from the mean.

    Returns:
        `run.members` plus `SOLAR_TERM_COLUMN`, `scale * (clim_cams + calibrated anomaly)` per row.
    """
    solar = run.solar
    anomaly = solar.anomaly_member
    if twin:
        anomaly = solar.anomaly_twin[:, None] + (anomaly - solar.anomaly_mean[:, None])
    shape = solar.clim[:, None] + anomaly
    return run.members.with_columns(**{SOLAR_TERM_COLUMN: pl.Series((ctx.scale * shape).ravel())})


def with_gross_lags(*, run: RunData, ctx: SeriesContext) -> pl.DataFrame:
    """Return the member rows with arm `b`'s gross lags (NaN where a lag shape is missing)."""
    return run.members.with_columns(
        **{
            lag: pl.col(lag) + ctx.scale * pl.col(shape)
            for lag, shape in zip(common.LAG_COLUMNS, SHAPE_LAG_NAMES, strict=True)
        }
    )


def added_pool(*, member_quantiles: np.ndarray, member_shift: np.ndarray) -> np.ndarray:
    """Add each member's shift to its quantiles, then pool the members.

    Args:
        member_quantiles: Shape (cases, members, 9), float32.
        member_shift: Shape (cases, members), in power over effective capacity.

    Returns:
        The pooled nine quantiles, shape (cases, 9). A shift of zero returns exactly the pooled
        unshifted quantiles.
    """
    shifted = member_quantiles + member_shift.astype(np.float32)[:, :, None]
    return common.mixture_quantiles(member_quantiles=shifted)


def sf_shifts(
    *, run: RunData, ctx: SeriesContext, reference_temperature: float
) -> dict[str, np.ndarray]:
    """Return the member shifts (cases, members) of each predict-only arm on `d1l`.

    The shift is added to the quantiles: `w * scale * anomaly` with the weight `w` (negative).

    Args:
        run: The validation run.
        ctx: The series' context.
        reference_temperature: `T_ref` of the temperature derating, degrees C.

    Returns:
        `sf` (ensemble-mean anomaly), `sf_member` (each member's own), `sf_twin` (the placebo
        twin), and `sf_temp` (`sf` times `1 - 0.004 * (T_cell - T_ref)`).
    """
    solar = run.solar
    factor = solar.weight * ctx.scale
    members = run.n_members
    shape_mean = solar.clim + solar.anomaly_mean
    cell_temperature = solar.temperature_mean + common.CELL_TEMPERATURE_RISE_K * shape_mean
    derating = 1.0 - common.DERATING_PER_KELVIN * (cell_temperature - reference_temperature)
    return {
        "sf": np.repeat((factor * solar.anomaly_mean)[:, None], members, axis=1),
        "sf_member": factor[:, None] * solar.anomaly_member,
        "sf_twin": np.repeat((factor * solar.anomaly_twin)[:, None], members, axis=1),
        "sf_temp": np.repeat((factor * solar.anomaly_mean * derating)[:, None], members, axis=1),
    }


def b_shifts(*, run: RunData, ctx: SeriesContext) -> dict[str, np.ndarray]:
    """Return the member shifts (cases, members) that turn `b`'s gross quantiles into net.

    Args:
        run: The validation run.
        ctx: The series' context.

    Returns:
        `base` (minus the scale times the ensemble-mean calibrated shape), `member` (each member's
        own shape), and `twin` (the placebo twin's shape), added to the gross quantiles.
    """
    solar = run.solar
    members = run.n_members
    return {
        "base": np.repeat((-ctx.scale * (solar.clim + solar.anomaly_mean))[:, None], members, 1),
        "member": -ctx.scale * (solar.clim[:, None] + solar.anomaly_member),
        "twin": np.repeat((-ctx.scale * (solar.clim + solar.anomaly_twin))[:, None], members, 1),
    }


def variant_frame(
    *, keys: pl.DataFrame, quantiles: np.ndarray, variant: str, seed: int
) -> pl.DataFrame:
    """Return a run's pooled quantiles as rows keyed by run and valid time, for one variant."""
    return keys.with_columns(variant=pl.lit(variant), seed=pl.lit(seed).cast(pl.Int8)).hstack(
        pl.DataFrame(quantiles.astype(np.float32), schema=list(common.LEVEL_COLUMNS))
    )
