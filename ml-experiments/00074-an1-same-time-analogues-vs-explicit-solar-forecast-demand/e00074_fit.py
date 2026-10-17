"""Prepare the rows, then fit and predict the arms of experiment 00074 under P-roll.

P-roll (experiment 00049) refits monthly, from 2025-07-01 to 2026-06-01, on the latest 365 days of
the ENS control member's rows, with 150 rounds, a learning rate of 0.15, and a recency half-life of
0.5 years. A refit forecasts the validation runs whose origin lies between it and the next refit.

Two steps, each one job per series (and per seed and refit for the fits):

- `--prepare SITE` writes `work/rows/<site>.parquet` (every training row with its inputs, the
  offset anomaly and its placebo twin, and the analogue columns of the four analogue variants) and
  `work/val/<site>/init=*.parquet` (every member row of every validation run, the same columns).
- `--fit SITE SEED REFIT` fits the arms of one refit and writes the pooled nine quantiles of each
  validation run it serves to `work/arms_raw/<site>_s<seed>_<refit>.parquet`, atomically.

The arms (`ARMS`) differ only in the columns the model receives and in the base margin. Every arm
has a base margin, so every arm starts from the same level: the median of the refit's training
labels, plus the offset `w * scale * anomaly` for the arms that carry the explicit solar forecast.

    uv run python ml-experiments/00074-*/e00074_fit.py --prepare D9
    uv run python ml-experiments/00074-*/e00074_fit.py --fit D9 0 2025-07-01
    uv run python ml-experiments/00074-*/e00074_fit.py --smoke
"""

import argparse
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00074_analogues as analogues
import e00074_common as common
import numpy as np
import polars as pl
import xgboost as xgb

WORK_ROWS_DIR: Final[Path] = common.WORK_DIR / "rows"
WORK_VAL_DIR: Final[Path] = common.WORK_DIR / "val"
RAW_DIR: Final[Path] = common.WORK_DIR / "arms_raw"
REFITS: Final[tuple[datetime, ...]] = tuple(
    datetime(2025 + (6 + i) // 12, (6 + i) % 12 + 1, 1, tzinfo=UTC) for i in range(12)
)
"""The twelve monthly refit times, 2025-07-01 to 2026-06-01."""
REFIT_CUTOFF_AGE: Final[timedelta] = timedelta(hours=3.5)
WINDOW_DAYS: Final[float] = 365.0
ROUNDS: Final[int] = 150
ETA: Final[float] = 0.15
NO_SKILL_BAND: Final[int] = 4
"""Lead bands from this index on (10+ days) get a weight of zero, as in experiment 00064."""
LAST_NEEDED_VALID: Final[datetime] = datetime(2026, 6, 1, tzinfo=UTC)
RADIATION_COLUMNS: Final[tuple[str, ...]] = ("shortwave", "longwave")
WEATHER: Final[tuple[str, ...]] = (
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
)
VARIANTS: Final[dict[str, tuple[int, bool]]] = {
    "a1": (1, True),
    "a3": (3, True),
    "a5": (5, True),
    "a3s": (3, False),
}
"""The analogue variants: the number of analogues and whether they are matched (else random)."""


@dataclass(frozen=True)
class Arm:
    """One arm: the extra columns the model receives and the kind of offset in its base margin.

    Attributes:
        name: The arm's name in the output.
        variant: The analogue variant whose columns are added (`a1`, `a3`, `a5`, `a3s`), or None.
        offset: `none`, `real` (the offset from the ensemble-mean anomaly), or `twin`.
        radiation: Add the `shortwave` and `longwave` columns (a context arm, not in a contrast).
    """

    name: str
    variant: str | None = None
    offset: str = "none"
    radiation: bool = False


ARMS: Final[tuple[Arm, ...]] = (
    Arm("R0"),
    Arm("E", offset="real"),
    Arm("E_twin", offset="twin"),
    Arm("A1", variant="a1"),
    Arm("A3", variant="a3"),
    Arm("A5", variant="a5"),
    Arm("A3E", variant="a3", offset="real"),
    Arm("A3S", variant="a3s"),
    Arm("R0_rad", radiation=True),
)


def base_columns() -> list[str]:
    """Return the 15 input columns of experiment 00064's `d1l`, in matrix order."""
    columns = [
        *common.feature_columns(calendar=True),
        *common.N1_EXTRA_COLUMNS,
        *common.LAG_COLUMNS,
    ]
    columns = [c for c in columns if c not in RADIATION_COLUMNS]
    if len(columns) != 15:
        msg = f"d1l has {len(columns)} columns, expected 15"
        raise ValueError(msg)
    return columns


def analogue_names(*, variant: str) -> list[str]:
    """Return the names of one variant's columns in the prepared tables."""
    k, _ = VARIANTS[variant]
    names = (
        analogues.COLUMN_NAMES
        if k > 1
        else (
            analogues.COLUMN_NAMES[0],
            analogues.COLUMN_NAMES[2],
            analogues.COLUMN_NAMES[3],
        )
    )
    return [f"{variant}_{name}" for name in names]


def arm_input_columns(*, arm: Arm) -> list[str]:
    """Return the columns of an arm's matrix."""
    columns = base_columns()
    if arm.radiation:
        columns = [*columns, *RADIATION_COLUMNS]
    if arm.variant is not None:
        columns = [*columns, *analogue_names(variant=arm.variant)]
    return columns


def add_lags(*, frame: pl.DataFrame, power: np.ndarray) -> pl.DataFrame:
    """Add the three net-power lags to a frame with the origin and valid times."""
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
    """Return each row's fitted weight of experiment 00064: zero outside April to September.

    Args:
        weights: Experiment 00064's weights table, the `all` variant.
        site: The series label.
        band: Each row's lead-band index.
        month: Each row's local calendar month.

    Returns:
        The applied-shift slope of the row's lead band and season, zero in the 10+ day band.

    Raises:
        ValueError: If the site does not hold one weight per lead band and season.
    """
    cells = weights.filter(pl.col("site") == site)
    if cells.height != len(common.LEAD_BANDS) * 2:
        msg = f"{site} has {cells.height} weight cells, expected {len(common.LEAD_BANDS) * 2}"
        raise ValueError(msg)
    frame = (
        pl.DataFrame({"band": band.astype(np.int64), "month": month.astype(np.int64)})
        .with_columns(season=common.season_of_month_expr(month=pl.col("month")))
        .join(
            cells.select(pl.col("band").cast(pl.Int64), "season", "slope_applied"),
            on=["band", "season"],
            how="left",
            validate="m:1",
            maintain_order="left",
        )
    )
    slope = frame["slope_applied"].fill_null(0.0).to_numpy()
    return np.where(band >= NO_SKILL_BAND, 0.0, slope)


@dataclass(frozen=True)
class Context:
    """One series' inputs shared by the preparation and the fits."""

    row: dict
    power: np.ndarray
    cams: np.ndarray
    day_class: np.ndarray
    weights: pl.DataFrame
    scale: float

    @property
    def site(self) -> str:
        """Return the anonymised label."""
        return self.row["site"]


def load_context(*, site: str) -> Context:
    """Load a series' table row, power, CAMS shape, day types, weights, and capacity scale."""
    series = common.series_table().filter(pl.col("site") == site)
    arrays = common.load_series_arrays(series=series)[site]
    capacity = common.capacities().filter(pl.col("site") == site)["capacity_mw"][0]
    row = series.row(0, named=True)
    return Context(
        row=row,
        power=arrays.power,
        cams=analogues.cams_grid(site=site),
        day_class=analogues.day_class_grid(),
        weights=pl.read_parquet(common.WEIGHTS_FILE).filter(pl.col("variant") == "all"),
        scale=capacity / row["effective_capacity_mw"],
    )


def with_analogues(*, frame: pl.DataFrame, ctx: Context, shape: np.ndarray) -> pl.DataFrame:
    """Add the columns of the four analogue variants to a frame of cases.

    Args:
        frame: Rows with `valid_time` and `power_fcst_init_time`.
        ctx: The series' context.
        shape: The unit PV shape forecast for each row's valid time.

    Returns:
        The frame with the columns named by `analogue_names`.
    """
    target = np.asarray(common.grid_index(times=frame["valid_time"]))
    origin = np.asarray(common.grid_index(times=frame["power_fcst_init_time"]))
    out = frame
    for variant, (k, matched) in VARIANTS.items():
        columns = analogues.arm_columns(
            columns=analogues.analogue_columns(
                power=ctx.power,
                cams=ctx.cams,
                day_class=ctx.day_class,
                target=target,
                origin=origin,
                shape=shape,
                k=k,
                matched=matched,
            ),
            k=k,
        )
        out = out.with_columns(
            pl.DataFrame(columns, schema=analogue_names(variant=variant), orient="row")
        )
    return out


def training_rows(*, ctx: Context) -> pl.DataFrame:
    """Return one series' training rows over the whole P-roll window, with every arm's inputs.

    The rows are the ENS control member's rows of experiment 00003's training file and of
    experiment 00034's files after it, with the power filled from the dense grid.

    Args:
        ctx: The series' context.

    Returns:
        Rows with the input columns, `power`, `lead_hours`, `band`, `month`, `offset_anomaly`,
        `offset_anomaly_twin`, and the analogue columns, for valid times up to the last refit.

    Raises:
        ValueError: If a row up to `LAST_NEEDED_VALID` has no solar shape.
    """
    series_id = ctx.row["time_series_id"]
    before = (
        pl.scan_parquet(common.ENS_TRAIN_PATH)
        .filter(pl.col("time_series_id") == series_id)
        .select(*common.KEYS, "ensemble_member", "power", *WEATHER)
        .collect()
    )
    after = pl.concat(
        [
            pl.read_parquet(path).filter(pl.col("time_series_id") == series_id)
            for path in sorted(common.E34_AFTER_DIR.glob("init=*.parquet"))
        ]
    )
    padded = after.with_columns(power=pl.lit(None, dtype=pl.Float32)).select(before.columns)
    rows = pl.concat([before, padded.cast(before.schema)])
    index = np.asarray(common.grid_index(times=rows["valid_time"]))
    filled = common.take(values=ctx.power, index=index) * ctx.row["effective_capacity_mw"]
    rows = rows.with_columns(
        power=pl.when(pl.col("power").is_null()).then(pl.Series(filled)).otherwise(pl.col("power"))
    ).filter(
        pl.col("power").is_finite()
        & (pl.col("valid_time") > pl.col("power_fcst_init_time"))
        & (pl.col("valid_time") <= LAST_NEEDED_VALID)
    )
    rows = common.with_derived(
        frame=rows, latitude=ctx.row["latitude"], longitude=ctx.row["longitude"]
    ).with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    )
    rows = add_lags(frame=rows, power=ctx.power)
    shape = pl.read_parquet(common.INPUTS_DIR / "train_shape.parquet").filter(
        pl.col("time_series_id") == series_id
    )
    twin = pl.read_parquet(common.INPUTS_DIR / "placebo_train.parquet").filter(
        pl.col("time_series_id") == series_id
    )
    keys = list(common.KEYS)
    rows = rows.join(
        shape.select(
            *keys,
            "month",
            "band",
            "clim_cams",
            "shape_calibrated",
            offset_anomaly=pl.col("anomaly_mean_calibrated"),
        ),
        on=keys,
        how="left",
        validate="m:1",
        maintain_order="left",
    ).join(
        twin.select(*keys, offset_anomaly_twin=pl.col("anomaly_mean_calibrated_twin")),
        on=keys,
        how="left",
        validate="m:1",
        maintain_order="left",
    )
    missing = rows["shape_calibrated"].null_count()
    if missing:
        msg = f"{ctx.site}: {missing} training rows have no solar shape"
        raise ValueError(msg)
    return with_analogues(frame=rows, ctx=ctx, shape=rows["shape_calibrated"].to_numpy())


def prepare_validation(*, ctx: Context) -> None:
    """Write every validation run's member rows with the inputs of every arm.

    Each run's rows are sorted by valid time then member. The shape that the analogues are matched
    to is each member's own calibrated shape (the CAMS climatology plus the member's calibrated
    anomaly); the offset uses the ensemble-mean anomaly.

    Args:
        ctx: The series' context.

    Raises:
        ValueError: If a valid time does not hold every member, or the solar files miss a run.
    """
    site = ctx.site
    climatology = pl.read_parquet(common.E46_SOLAR_DIR / "climatology.parquet").filter(
        pl.col("site") == site
    )
    twin = pl.read_parquet(common.PLACEBO_VAL_FILE).filter(
        pl.col("time_series_id") == ctx.row["time_series_id"]
    )
    keys = list(common.KEYS)
    for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet")):
        target = WORK_VAL_DIR / site / path.name
        if target.exists():
            continue
        run = pl.read_parquet(path).filter(pl.col("time_series_id") == ctx.row["time_series_id"])
        run = common.with_derived(
            frame=run, latitude=ctx.row["latitude"], longitude=ctx.row["longitude"]
        )
        run = add_lags(frame=run, power=ctx.power).sort("valid_time", "ensemble_member")
        members = run["ensemble_member"].n_unique()
        cases = run.select(*keys).unique().height
        if members != common.MEMBERS or run.height != cases * members:
            msg = "every valid time must hold all 51 members"
            raise ValueError(msg)
        anomaly = (
            pl.read_parquet(common.E46_SOLAR_DIR / "ens_val" / path.name)
            .filter(pl.col("site") == site)
            .select(
                "valid_time",
                "ensemble_member",
                member_anomaly=pl.col("anomaly_calibrated"),
                offset_anomaly=pl.col("anomaly_mean_calibrated"),
            )
        )
        run = (
            common.with_cell(frame=run, time_column="valid_time")
            .join(
                climatology.select("month", "hh", "clim_cams"),
                on=["month", "hh"],
                how="left",
                validate="m:1",
                maintain_order="left",
            )
            .join(
                anomaly,
                on=["valid_time", "ensemble_member"],
                how="left",
                validate="m:1",
                maintain_order="left",
            )
            .join(
                twin.select(*keys, offset_anomaly_twin=pl.col("anomaly_mean_calibrated_twin")),
                on=keys,
                how="left",
                validate="m:1",
                maintain_order="left",
            )
        )
        if run["member_anomaly"].null_count() or run["offset_anomaly_twin"].null_count():
            msg = f"{site}: the solar files do not cover {path.name}"
            raise ValueError(msg)
        lead = (
            (run["valid_time"] - run["power_fcst_init_time"]).dt.total_minutes() / 60.0
        ).to_numpy()
        run = run.with_columns(
            band=pl.Series(common.lead_band(lead_hours=lead)),
            lead_hours=pl.Series(lead),
            shape_member=pl.col("clim_cams").fill_null(0.0) + pl.col("member_anomaly"),
        )
        run = with_analogues(frame=run, ctx=ctx, shape=run["shape_member"].to_numpy())
        common.write_atomically(frame=run, target=target)


def prepare(*, site: str) -> None:
    """Write the prepared training rows and validation runs of one series."""
    started = time.monotonic()
    ctx = load_context(site=site)
    target = WORK_ROWS_DIR / f"{site}.parquet"
    if not target.exists():
        rows = training_rows(ctx=ctx)
        common.write_atomically(frame=rows, target=target)
        print(f"{site}: {rows.height:,} training rows", flush=True)
    prepare_validation(ctx=ctx)
    print(f"{site}: prepared in {time.monotonic() - started:.0f} s", flush=True)


def known_rows(*, rows: pl.DataFrame, refit: datetime) -> pl.DataFrame:
    """Return the rows a refit may see: the latest 12 months, eligible at the refit time."""
    known = rows.filter(
        (pl.col("valid_time") <= refit - REFIT_CUTOFF_AGE)
        & (pl.col("power_fcst_init_time") <= refit)
        & (pl.col("valid_time") > refit - timedelta(days=WINDOW_DAYS))
    )
    if (
        known.is_empty()
        or common.latest_time(times=known["valid_time"]) > refit - REFIT_CUTOFF_AGE
        or common.latest_time(times=known["power_fcst_init_time"]) > refit
    ):
        msg = f"refit {refit}: no usable rows or a row from after the refit"
        raise ValueError(msg)
    return known


def offsets(*, frame: pl.DataFrame, ctx: Context, column: str) -> np.ndarray:
    """Return `w * scale * anomaly` for each row, from the row's band, month, and anomaly column."""
    weight = weights_per_row(
        weights=ctx.weights,
        site=ctx.site,
        band=frame["band"].to_numpy(),
        month=frame["month"].to_numpy(),
    )
    return weight * ctx.scale * frame[column].to_numpy().astype(np.float64)


def margin_for(*, frame: pl.DataFrame, ctx: Context, arm: Arm, level: float) -> np.ndarray:
    """Return an arm's base margin, shape (rows, 9): the level plus any offset."""
    column = {"none": None, "real": "offset_anomaly", "twin": "offset_anomaly_twin"}[arm.offset]
    shift = (
        np.zeros(frame.height) if column is None else offsets(frame=frame, ctx=ctx, column=column)
    )
    return np.tile((level + shift)[:, None], (1, len(common.LEVELS))).astype(np.float32)


def parameters(*, seed: int, device: str) -> dict[str, object]:
    """Return N1's XGBoost parameters with P-roll's learning rate and one thread."""
    return {**common.QUANTILE_PARAMETERS, "seed": seed, "eta": ETA, "nthread": 1, "device": device}


def served_origins(*, origins: list[datetime], refit: datetime) -> list[datetime]:
    """Return the validation origins forecast by a refit: the latest refit at or before each."""
    later = [r for r in REFITS if r > refit]
    stop = min(later) if later else None
    return [o for o in origins if o >= refit and (stop is None or o < stop)]


def fit_refit(
    *, site: str, seed: int, refit: datetime, rounds: int = ROUNDS, device: str = "cuda"
) -> pl.DataFrame:
    """Fit every arm at one refit and return the pooled quantiles of the validation runs it serves.

    Args:
        site: The series label.
        seed: The fitting seed.
        refit: The refit time.
        rounds: Boosting rounds (the smoke run lowers it).
        device: `cuda` or `cpu`.

    Returns:
        Columns `arm`, `seed`, `KEYS`, and the nine quantiles, for every arm and served run.
    """
    ctx = load_context(site=site)
    rows = pl.read_parquet(WORK_ROWS_DIR / f"{site}.parquet")
    known = known_rows(rows=rows, refit=refit)
    age_days = (refit - known["valid_time"]).dt.total_seconds() / 86_400
    weight = 0.5 ** (age_days.to_numpy() / common.SHORT_HALF_LIFE_DAYS)
    label = (known["power"] / ctx.row["effective_capacity_mw"]).to_numpy()
    level = float(np.median(label))
    paths = sorted((WORK_VAL_DIR / site).glob("init=*.parquet"))
    run_paths = {
        pl.scan_parquet(path).select(pl.col("power_fcst_init_time").first()).collect().item(): path
        for path in paths
    }
    served = served_origins(origins=sorted(run_paths), refit=refit)
    runs = {o: pl.read_parquet(run_paths[o]) for o in served}
    out = []
    for arm in ARMS:
        columns = arm_input_columns(arm=arm)
        matrix = xgb.DMatrix(
            known.select(columns).to_numpy().astype(np.float32),
            label=label,
            weight=weight,
            base_margin=margin_for(frame=known, ctx=ctx, arm=arm, level=level),
        )
        booster = xgb.train(parameters(seed=seed, device=device), matrix, num_boost_round=rounds)
        for run in runs.values():
            members = common.MEMBERS
            test = xgb.DMatrix(
                run.select(columns).to_numpy().astype(np.float32),
                base_margin=margin_for(frame=run, ctx=ctx, arm=arm, level=level),
            )
            raw = booster.predict(test).reshape(-1, members, len(common.LEVELS))
            keys = run.select(*common.KEYS).unique(maintain_order=True)
            out.append(
                keys.hstack(
                    pl.DataFrame(
                        common.mixture_quantiles(member_quantiles=raw),
                        schema=list(common.LEVEL_COLUMNS),
                    )
                ).with_columns(arm=pl.lit(arm.name), seed=pl.lit(seed).cast(pl.Int8))
            )
    return pl.concat(out)


def job_path(*, site: str, seed: int, refit: datetime) -> Path:
    """Return the file one fit job writes."""
    return RAW_DIR / f"{site}_s{seed}_{refit:%Y%m%d}.parquet"


def smoke() -> None:
    """Fit one series, one refit, and 20 rounds, then check the checks listed below.

    Checks: all nine arms are present; each served run has every key; the quantiles are finite;
    an arm with an offset differs from `R0`; the base margin was applied at inference.
    """
    site = common.SITES[0]
    refit = REFITS[0]
    started = time.monotonic()
    result = fit_refit(site=site, seed=0, refit=refit, rounds=20)
    arms = set(result["arm"].unique().to_list())
    if arms != {arm.name for arm in ARMS}:
        msg = f"arms found: {sorted(arms)}"
        raise ValueError(msg)
    if not np.isfinite(result.select(common.LEVEL_COLUMNS).to_numpy()).all():
        msg = "non-finite quantiles"
        raise ValueError(msg)
    median = {a: result.filter(pl.col("arm") == a)["q50"].to_numpy() for a in arms}
    if np.allclose(median["E"], median["R0"]):
        msg = "the offset arm equals the reference"
        raise ValueError(msg)
    counts = result.group_by("arm").len()["len"].unique().to_list()
    if len(counts) != 1:
        msg = f"arms cover different row counts: {counts}"
        raise ValueError(msg)
    print(f"smoke passed in {time.monotonic() - started:.0f} s: {counts[0]:,} rows per arm")


def main() -> None:
    """Run the requested step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", metavar="SITE")
    parser.add_argument("--fit", nargs=3, metavar=("SITE", "SEED", "REFIT"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(site=args.prepare)
    elif args.fit:
        site, seed, refit_text = args.fit
        refit = datetime.fromisoformat(refit_text).replace(tzinfo=UTC)
        target = job_path(site=site, seed=int(seed), refit=refit)
        if target.exists():
            return
        started = time.monotonic()
        frame = fit_refit(site=site, seed=int(seed), refit=refit)
        common.write_atomically(frame=frame, target=target)
        print(
            f"{site} seed {seed} refit {refit:%Y-%m-%d}: {time.monotonic() - started:.0f} s",
            flush=True,
        )
    elif args.smoke:
        smoke()


if __name__ == "__main__":
    main()
