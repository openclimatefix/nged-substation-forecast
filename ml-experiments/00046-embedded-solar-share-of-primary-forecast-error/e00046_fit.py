"""Fit the demand arms `d0` and `d1l` of experiment 00046 and write their shifted variants.

Both arms are experiment 00033's N1 recipe (half-life 2 years, trained on the ENS control
member, one quantile model per series and seed). `d0` keeps the 13 weather and calendar columns,
`lead_hours`, and the 3 power lags. `d1l` drops `shortwave` and `longwave`. Each job writes one file
per arm, seed, and site to `work/arms_raw/`, and the booster beside it in `work/boosters/`.

The file holds the pooled 9 quantiles of the 51-member mixture, one block of rows per variant:

- `base`: the unshifted forecast (both arms).
- `ens_shift`, `ens_shift_headline` (`d1l` only): each member's quantiles are shifted by minus the
  capacity times the member's calibrated ENS solar anomaly over the series' effective capacity, and
  then the members are pooled. The capacity is `C` or `headline` of experiment 00040.
- `ens_mean_shift` (`d1l` only, exploratory): every member gets the shift of the calibrated
  ensemble-mean anomaly.
- `cams_shift`, `cams_shift_headline` (`d1l` only): every pooled quantile is shifted by minus the
  capacity times the CAMS anomaly at the valid time (the oracle).

A shift is zero where the anomaly is missing. `--predict-only --coefficient K --suffix NAME` reloads
the saved boosters and writes the shifted variants with the capacity multiplied by `K`, with no
refit.

    uv run python ml-experiments/00046-*/e00046_fit.py --arm d1l --seed 0 [--sites D5 D9 ...]
"""

import argparse
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Final, Literal

import e00046_common as common
import numpy as np
import polars as pl
import xgboost as xgb

ArmType = Literal["d0", "d1l"]
RADIATION_COLUMNS: Final[tuple[str, ...]] = ("shortwave", "longwave")
BASE_COLUMN_COUNT: Final[int] = 13 + 1 + 3
"""Weather and calendar columns, `lead_hours`, and the three power lags of `d0`."""
VARIANTS_D1L: Final[tuple[str, ...]] = (
    "base",
    "ens_shift",
    "cams_shift",
    "ens_shift_headline",
    "cams_shift_headline",
    "ens_mean_shift",
)


def input_columns(*, arm: ArmType) -> list[str]:
    """Return an arm's input columns in matrix order.

    Args:
        arm: `d0` keeps every column; `d1l` drops the two radiation columns.

    Returns:
        The column names.

    Raises:
        ValueError: If the count differs from 17 for `d0` or 15 for `d1l`.
    """
    columns = [
        *common.feature_columns(calendar=True),
        *common.N1_EXTRA_COLUMNS,
        *common.LAG_COLUMNS,
    ]
    if arm == "d1l":
        columns = [c for c in columns if c not in RADIATION_COLUMNS]
    expected = BASE_COLUMN_COUNT - (len(RADIATION_COLUMNS) if arm == "d1l" else 0)
    if len(columns) != expected:
        msg = f"arm {arm} has {len(columns)} columns, expected {expected}"
        raise ValueError(msg)
    return columns


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


def own_rows(*, row: dict, frame: pl.DataFrame, arrays: common.SeriesArrays) -> pl.DataFrame:
    """Return one series' finite-power rows with derived columns, `lead_hours`, and lags."""
    own = frame.filter(pl.col("time_series_id") == row["time_series_id"])
    own = own.filter(
        pl.col("power").is_finite() & (pl.col("valid_time") > pl.col("power_fcst_init_time"))
    )
    own = common.with_derived(frame=own, latitude=row["latitude"], longitude=row["longitude"])
    own = own.with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    )
    return add_lags(frame=own, power=arrays.power)


def train_matrix(
    *, rows: pl.DataFrame, row: dict, anchor: datetime, half_life_days: float, columns: list[str]
) -> xgb.DMatrix:
    """Return the recency-weighted training matrix of one series' rows."""
    age_days = (anchor - rows["valid_time"]).dt.total_seconds() / 86_400
    return xgb.DMatrix(
        rows.select(columns).to_numpy().astype(np.float32),
        label=(rows["power"] / row["effective_capacity_mw"]).to_numpy(),
        weight=0.5 ** (age_days.to_numpy() / half_life_days),
    )


def fit(*, matrix: xgb.DMatrix, seed: int) -> xgb.Booster:
    """Fit one nine-level quantile model."""
    return xgb.train(
        {**common.QUANTILE_PARAMETERS, "seed": seed}, matrix, num_boost_round=common.N_ROUNDS
    )


def shifted_pool(*, member_quantiles: np.ndarray, member_shift: np.ndarray) -> np.ndarray:
    """Subtract each member's shift from its quantiles, then pool the members.

    Args:
        member_quantiles: Shape (cases, members, 9), float32.
        member_shift: Shape (cases, members), in the forecast's units (power over capacity).

    Returns:
        The pooled nine quantiles, shape (cases, 9). A shift of zero returns exactly the pooled
        unshifted quantiles.
    """
    shifted = member_quantiles - member_shift.astype(np.float32)[:, :, None]
    return common.mixture_quantiles(member_quantiles=shifted)


def anomaly_matrix(
    *, anomaly_file: pl.DataFrame, site: str, valid_times: pl.Series, column: str
) -> np.ndarray:
    """Return a run's ENS anomaly as an array of shape (cases, members), 0 where unknown.

    Args:
        anomaly_file: A validation run's `ens_val` file.
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


def cams_anomaly_vector(
    *, cams_anomaly: pl.DataFrame, site: str, valid_times: pl.Series
) -> np.ndarray:
    """Return the CAMS anomaly at each of a run's valid times, 0 where CAMS is missing."""
    own = cams_anomaly.filter(pl.col("site") == site)
    joined = pl.DataFrame({"valid_time": valid_times}).join(
        own.select("valid_time", "anomaly_cams"), on="valid_time", how="left", maintain_order="left"
    )
    return np.nan_to_num(joined["anomaly_cams"].to_numpy().astype(np.float64), nan=0.0)


def capacity_cases(*, member_quantiles: np.ndarray) -> int:
    """Return the number of cases in an array of member quantiles."""
    return int(member_quantiles.shape[0])


def variant_tables(
    *,
    arm: ArmType,
    member_quantiles: np.ndarray,
    run_context: dict,
    coefficients: dict[str, float],
    names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Return the pooled quantiles of each requested variant of one run.

    Args:
        arm: The arm; shifted variants exist for `d1l` only.
        member_quantiles: Shape (cases, members, 9).
        run_context: `effective_capacity_mw`, `ens` (a dict of anomaly arrays by column name),
            `cams` (shape (cases,)), and optionally `scale` (shape (cases,)), a multiplier on the
            capacity for each case (one when absent).
        coefficients: The capacity in MW for each of `C` and `headline`.
        names: The variants to build.

    Returns:
        Each variant's pooled quantiles, shape (cases, 9).
    """
    capacity = run_context["effective_capacity_mw"]
    out: dict[str, np.ndarray] = {}
    for name in names:
        if name == "base":
            out[name] = common.mixture_quantiles(member_quantiles=member_quantiles)
            continue
        mw = coefficients["headline" if name.endswith("headline") else "C"]
        scale = run_context.get("scale", np.ones(capacity_cases(member_quantiles=member_quantiles)))
        if name.startswith("cams_shift"):
            shift = mw * scale * run_context["cams"] / capacity
            out[name] = (
                common.mixture_quantiles(member_quantiles=member_quantiles)
                - shift.astype(np.float32)[:, None]
            )
        else:
            column = "anomaly_mean_calibrated" if name == "ens_mean_shift" else "anomaly_calibrated"
            shift = mw * scale[:, None] * run_context["ens"][column] / capacity
            out[name] = shifted_pool(member_quantiles=member_quantiles, member_shift=shift)
    return out


def member_quantile_runs(
    *, row: dict, arrays: common.SeriesArrays, booster: xgb.Booster, columns: list[str]
) -> Iterator[tuple[str, pl.DataFrame, np.ndarray]]:
    """Yield, for each validation run, its keys, valid times, and per-member quantiles.

    Args:
        row: The series' table row.
        arrays: The series' arrays.
        booster: The fitted quantile model.
        columns: The arm's input columns.

    Yields:
        The run file's date label, its distinct sorted keys, and the member quantiles of shape
        (cases, members, 9), float32.
    """
    for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet")):
        run = pl.read_parquet(path).filter(pl.col("time_series_id") == row["time_series_id"])
        run = common.with_derived(frame=run, latitude=row["latitude"], longitude=row["longitude"])
        run = add_lags(frame=run, power=arrays.power).sort("valid_time", "ensemble_member")
        members = run["ensemble_member"].n_unique()
        keys = run.select(*common.KEYS).unique(maintain_order=True)
        if run.height != keys.height * members or members != common.MEMBERS:
            msg = "every valid time must hold all 51 members"
            raise ValueError(msg)
        raw = booster.predict(xgb.DMatrix(run.select(columns).to_numpy().astype(np.float32)))
        yield path.name, keys, raw.reshape(-1, members, len(common.LEVELS)).astype(np.float32)


def write_site(
    *,
    arm: ArmType,
    seed: int,
    row: dict,
    arrays: common.SeriesArrays,
    booster: xgb.Booster,
    capacity: dict[str, float],
    names: tuple[str, ...],
    suffix: str,
    target: Path,
    capacity_scale: float,
    band_coefficients: dict[int, float] | None = None,
) -> None:
    """Predict every validation run of one series and write the requested variants.

    Args:
        arm: The arm.
        seed: The fitting seed.
        row: The series' table row.
        arrays: The series' arrays.
        booster: The fitted quantile model.
        capacity: `C` and `headline`, in MW.
        names: The variants to write.
        suffix: Appended to each variant's name (`ens_shift_<suffix>`) when not empty.
        target: The output file.
        capacity_scale: The coefficient multiplying both capacities (1.0 for the planned arms).
        band_coefficients: When given, a further multiplier for each lead band (an index into
            `LEAD_BANDS`) of this series; a band absent from the dict keeps 1.0.
    """
    cams_anomaly = pl.read_parquet(common.SOLAR_DIR / "cams_anomaly.parquet")
    columns = input_columns(arm=arm)
    scaled = {name: capacity_scale * mw for name, mw in capacity.items()}
    pieces = []
    for file_name, keys, member_quantiles in member_quantile_runs(
        row=row, arrays=arrays, booster=booster, columns=columns
    ):
        times = keys["valid_time"]
        context: dict = {"effective_capacity_mw": row["effective_capacity_mw"]}
        if band_coefficients is not None:
            lead = ((times - keys["power_fcst_init_time"]).dt.total_minutes() / 60.0).to_numpy()
            context["scale"] = np.array(
                [band_coefficients.get(int(b), 1.0) for b in common.lead_band(lead_hours=lead)]
            )
        if arm == "d1l" and names != ("base",):
            anomaly_file = pl.read_parquet(common.SOLAR_DIR / "ens_val" / file_name)
            context["ens"] = {
                column: anomaly_matrix(
                    anomaly_file=anomaly_file, site=row["site"], valid_times=times, column=column
                )
                for column in ("anomaly_calibrated", "anomaly_mean_calibrated")
            }
            context["cams"] = cams_anomaly_vector(
                cams_anomaly=cams_anomaly, site=row["site"], valid_times=times
            )
        tables = variant_tables(
            arm=arm,
            member_quantiles=member_quantiles,
            run_context=context,
            coefficients=scaled,
            names=names,
        )
        for name, quantiles in tables.items():
            pieces.append(
                keys.with_columns(
                    variant=pl.lit(f"{name}_{suffix}" if suffix else name),
                    seed=pl.lit(seed).cast(pl.Int8),
                ).hstack(
                    pl.DataFrame(quantiles.astype(np.float32), schema=list(common.LEVEL_COLUMNS))
                )
            )
    common.write_atomically(frame=pl.concat(pieces), target=target)


def band_coefficients_for(*, path: Path | None, site: str) -> dict[int, float] | None:
    """Return one site's lead-band multipliers from a `site, band, coefficient` parquet, if any."""
    if path is None:
        return None
    return dict(
        pl.read_parquet(path)
        .filter(pl.col("site") == site)
        .select("band", "coefficient")
        .iter_rows()
    )


def main() -> None:
    """Fit (or reload) the boosters of one arm and seed and write the sites' variants."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("d0", "d1l"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sites", nargs="+", default=list(common.SITES))
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--coefficient", type=float, default=1.0)
    parser.add_argument("--suffix", default="")
    parser.add_argument(
        "--band-coefficients-file",
        type=Path,
        default=None,
        help="parquet with site, band (index), and coefficient columns; multiplies the capacity",
    )
    parser.add_argument("--out-dir", type=Path, default=common.RAW_ARMS_DIR)
    args = parser.parse_args()
    arm: ArmType = args.arm
    if args.predict_only and not args.suffix:
        parser.error("--predict-only needs --suffix")
    series = common.series_table().filter(pl.col("site").is_in(args.sites))
    arrays = common.load_series_arrays(series=series)
    capacities = {r["site"]: r for r in common.capacities().iter_rows(named=True)}
    train = (
        None
        if args.predict_only
        else pl.read_parquet(common.ENS_TRAIN_PATH).filter(
            pl.col("valid_time") <= common.LAST_TRAINING_VALID
        )
    )
    columns = input_columns(arm=arm)
    names = ("base",) if arm == "d0" else VARIANTS_D1L
    if args.predict_only:
        names = ("ens_shift", "cams_shift")
    for row in series.iter_rows(named=True):
        site = row["site"]
        started = time.monotonic()
        stem = f"{arm}_seed{args.seed}_{site}"
        booster_path = common.BOOSTER_DIR / f"{stem}.ubj"
        target = args.out_dir / (
            f"{stem}_{args.suffix}.parquet" if args.suffix else f"{stem}.parquet"
        )
        if target.exists():
            continue
        if booster_path.exists():
            booster = xgb.Booster()
            booster.load_model(booster_path)
        else:
            if train is None:
                msg = f"no saved booster at {booster_path}"
                raise FileNotFoundError(msg)
            rows = own_rows(row=row, frame=train, arrays=arrays[site])
            if common.latest_time(times=rows["valid_time"]) > common.LAST_TRAINING_VALID:
                msg = "training rows must end at the last training valid time"
                raise ValueError(msg)
            matrix = train_matrix(
                rows=rows,
                row=row,
                anchor=common.LAST_TRAINING_VALID,
                half_life_days=common.RECENCY_HALF_LIFE_DAYS,
                columns=columns,
            )
            assert matrix.num_col() == len(columns)
            booster = fit(matrix=matrix, seed=args.seed)
            booster_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = booster_path.with_name(f"{stem}.tmp.ubj")
            booster.save_model(temporary)
            temporary.rename(booster_path)
        capacity = {"C": capacities[site]["C"], "headline": capacities[site]["headline"]}
        write_site(
            arm=arm,
            seed=args.seed,
            row=row,
            arrays=arrays[site],
            booster=booster,
            capacity=capacity,
            names=names,
            suffix=args.suffix,
            target=target,
            capacity_scale=args.coefficient,
            band_coefficients=band_coefficients_for(path=args.band_coefficients_file, site=site),
        )
        print(f"{arm} seed {args.seed} {site}: {time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
