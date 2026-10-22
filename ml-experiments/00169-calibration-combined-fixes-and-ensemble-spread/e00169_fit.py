"""Fit the ensemble-spread scale model on the tuning months and write every arm's forecasts.

| Arm | Quantiles |
|---|---|
| `ref` | experiment 160's arm I, as experiment 164 stored it |
| `a` | experiment 164's fix O: the upper offsets widened on forecast-cold days |
| `at` | `a`, with the q01, q05, and q10 of experiment 164's fix T |
| `sp` | `a`, with each offset times a non-decreasing function of the two ensemble spreads |
| `sp0` | `sp` with every spread coefficient set to zero: the fitted intercept alone (a control) |

The mean forecast is never changed. The scale of one level's offset is
`c + sum_k a_k * max(z1 - k1_k, 0) + sum_k b_k * max(z2 - k2_k, 0)`, where `z1` is the standard
deviation across members of the 24-hour mean temperature and `z2` the same for downward shortwave
radiation, and `a_k` and `b_k` are non-negative. The pinball loss is linear in the coefficients, so
each level is a linear programme, fitted on the tuning months only.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00169-*/e00169_fit.py [--workers 6]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Final

import e00169_common as common
import numpy as np
import polars as pl
from scipy import sparse
from scipy.optimize import linprog

FIT_LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.1, 0.25, 0.75, 0.9, 0.95, 0.99)
KNOT_PERCENTILES: Final[tuple[float, ...]] = (25.0, 50.0, 75.0, 90.0)
MAX_FIT_ROWS: Final[int] = 150_000
MINIMUM_OFFSET: Final[float] = 1e-3
INTERCEPT_RANGE: Final[tuple[float, float]] = (0.3, 3.0)
COEFFICIENT_MAX: Final[float] = 5.0
SEED: Final[int] = 169
LOWER_TAIL_ARMS: Final[tuple[str, ...]] = ("q01", "q05", "q10")


def ensemble_columns(*, frame: pl.DataFrame, site: str) -> pl.DataFrame:
    """Return the frame with the ensemble temperature mean and the two spread columns.

    The window is 24 half-hours either side of the row's valid time, clipped to the run's window.
    Each spread is the standard deviation across the 51 members of each member's window mean.
    """
    with np.load(common.E111_STORE_DIR / f"{site}.npz") as stored:
        origins = stored["origins"]
        fields = {
            "temperature": stored["temperature"].astype(np.float64),
            "shortwave": stored["shortwave"].astype(np.float64),
        }
    run = frame["run"].to_numpy().astype(np.int64)
    origin = frame["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy()
    if not (origins[run] == origin).all():
        msg = f"{site}: the run column does not index experiment 111's store"
        raise ValueError(msg)
    valid = frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    position = ((valid - origin) // np.timedelta64(30, "m")).astype(np.int64) - 1
    out = {}
    for name, values in fields.items():
        finite = np.isfinite(values)
        cumulative = np.zeros((*values.shape[:2], values.shape[2] + 1))
        count = np.zeros_like(cumulative)
        np.cumsum(np.where(finite, values, 0.0), axis=2, out=cumulative[:, :, 1:])
        np.cumsum(finite, axis=2, out=count[:, :, 1:])
        last = values.shape[2] - 1
        low = np.clip(position - common.WINDOW_HALF_HOURS, 0, last)
        high = np.clip(position + common.WINDOW_HALF_HOURS, 0, last)
        total = cumulative[run, :, high + 1] - cumulative[run, :, low]
        number = count[run, :, high + 1] - count[run, :, low]
        with np.errstate(invalid="ignore", divide="ignore"):
            per_member = np.where(number > 0, total / number, np.nan)
        out[name] = (np.nanmean(per_member, axis=1), np.nanstd(per_member, axis=1))
    return frame.with_columns(
        ens_mean_c=pl.Series(out["temperature"][0]),
        ens_std_c=pl.Series(out["temperature"][1]),
        ens_sw_std=pl.Series(out["shortwave"][1]),
    )


def load_arm_a(*, period: str, site: str) -> pl.DataFrame:
    """Return experiment 164's fix O forecasts with the three ensemble columns."""
    frame = pl.read_parquet(common.stored_path(arm="o", period=period, site=site)).select(
        list(common.BASE_COLUMNS)
    )
    return ensemble_columns(frame=frame, site=site)


def fit_rows(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Keep the rows a fit may use: a reading, outside the switching events, lead of 168 hours."""
    return frame.filter(
        ~pl.col("switching_mask")
        & pl.col("actual").is_finite()
        & (pl.col("lead_hours") <= common.HORIZON_HOURS)
        & pl.col("ens_std_c").is_finite()
        & pl.col("ens_sw_std").is_finite()
    )


def basis(*, z1: np.ndarray, z2: np.ndarray, knots: dict) -> np.ndarray:
    """Return the design matrix: a constant, then one hinge per knot for each spread."""
    columns = [np.ones_like(z1)]
    columns += [np.maximum(z1 / knots["scale1"] - k, 0.0) for k in knots["k1"]]
    columns += [np.maximum(z2 / knots["scale2"] - k, 0.0) for k in knots["k2"]]
    return np.stack(columns, axis=1)


def offsets_of(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the frame's quantile offsets from the mean, in points, shape (rows, 11)."""
    quantiles = frame.select(list(common.QUANTILES)).to_numpy().astype(np.float64)
    return quantiles - frame["mean"].to_numpy().astype(np.float64)[:, None]


def fit_level(*, level: float, residual, offset, design) -> np.ndarray:  # noqa: ANN001
    """Return the coefficients that minimise the pinball loss of `mean + offset * (design @ w)`.

    With `ratio = residual / offset`, the loss of a row is `abs(offset)` times the pinball loss of
    `ratio - design @ w`, at level `level` where the offset is positive and `1 - level` where it
    is negative. Positive and negative parts of the error are the linear programme's slack columns.
    """
    n_rows, n_terms = design.shape
    tau = np.where(offset > 0, level, 1.0 - level)
    weight = np.abs(offset)
    cost = np.concatenate([np.zeros(n_terms), weight * tau, weight * (1.0 - tau)])
    identity = sparse.identity(n_rows, format="csr")
    equality = sparse.hstack([sparse.csr_matrix(design), identity, -identity], format="csr")
    bounds = [INTERCEPT_RANGE] + [(0.0, COEFFICIENT_MAX)] * (n_terms - 1)
    bounds += [(0.0, None)] * (2 * n_rows)
    result = linprog(
        cost,
        A_eq=equality,
        b_eq=residual / offset,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        msg = f"level {level}: {result.message}"
        raise RuntimeError(msg)
    return result.x[:n_terms]


def _fit_job(task: tuple) -> tuple[float, list[float]]:
    level, residual, offset, design = task
    return level, fit_level(level=level, residual=residual, offset=offset, design=design).tolist()


def apply_scale(*, frame: pl.DataFrame, scale: np.ndarray) -> pl.DataFrame:
    """Return the frame with quantiles rebuilt as mean plus scaled offsets, sorted per row."""
    mean = frame["mean"].to_numpy().astype(np.float64)
    quantiles = np.sort(mean[:, None] + offsets_of(frame=frame) * scale, axis=1)
    return frame.with_columns(
        pl.Series(q, quantiles[:, position].astype(np.float32))
        for position, q in enumerate(common.QUANTILES)
    )


def write_arm(*, frame: pl.DataFrame, arm: str, period: str, site: str) -> None:
    """Write one arm's forecasts atomically."""
    target = common.out_path(arm=arm, period=period, site=site)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    frame.with_columns(arm=pl.lit(arm)).select(list(common.OUTPUT_COLUMNS)).write_parquet(temporary)
    temporary.rename(target)


def main() -> None:
    """Fit on the tuning months, build the arms for every period and substation, write them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    frames = {
        (period, site): load_arm_a(period=period, site=site)
        for period in common.PERIODS
        for site in common.SCORED_SITES
    }
    tuning = pl.concat([fit_rows(frame=frames["tuning", s]) for s in common.SCORED_SITES])
    z1 = tuning["ens_std_c"].to_numpy()
    z2 = tuning["ens_sw_std"].to_numpy()
    knots = {
        "scale1": float(np.percentile(z1, 90.0)),
        "scale2": float(np.percentile(z2, 90.0)),
    }
    knots["k1"] = [float(np.percentile(z1, p) / knots["scale1"]) for p in KNOT_PERCENTILES]
    knots["k2"] = [float(np.percentile(z2, p) / knots["scale2"]) for p in KNOT_PERCENTILES]
    rng = np.random.default_rng(SEED)
    chosen = np.sort(
        rng.choice(tuning.height, size=min(MAX_FIT_ROWS, tuning.height), replace=False)
    )
    sample = tuning[chosen]
    residual = (sample["actual"] - sample["mean"]).to_numpy().astype(np.float64)
    offsets = offsets_of(frame=sample)
    design = basis(z1=z1[chosen], z2=z2[chosen], knots=knots)
    tasks = []
    for level in FIT_LEVELS:
        position = common.QUANTILES.index(f"q{round(level * 100):02d}")
        usable = np.abs(offsets[:, position]) > MINIMUM_OFFSET
        tasks.append((level, residual[usable], offsets[usable, position], design[usable]))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        coefficients = dict(pool.map(_fit_job, tasks))
    parameters = {
        "knots": knots,
        "coefficients": {str(level): coefficients[level] for level in FIT_LEVELS},
        "tuning_rows": tuning.height,
        "fit_rows": len(chosen),
    }
    common.write_json_atomic(path=common.WORK_DIR / "fit_parameters.json", content=parameters)
    print(parameters, flush=True)

    for (period, site), frame in frames.items():
        scale = np.ones((frame.height, len(common.QUANTILES)))
        z1_all = np.nan_to_num(frame["ens_std_c"].to_numpy(), nan=0.0)
        z2_all = np.nan_to_num(frame["ens_sw_std"].to_numpy(), nan=0.0)
        design_all = basis(z1=z1_all, z2=z2_all, knots=knots)
        for level in FIT_LEVELS:
            position = common.QUANTILES.index(f"q{round(level * 100):02d}")
            scale[:, position] = design_all @ np.array(coefficients[level])
        reference = pl.read_parquet(common.stored_path(arm="ref", period=period, site=site)).select(
            list(common.BASE_COLUMNS)
        )
        tail = pl.read_parquet(common.stored_path(arm="t", period=period, site=site))
        at = frame.with_columns(*[tail[q] for q in LOWER_TAIL_ARMS])
        scale0 = np.ones_like(scale)
        for level in FIT_LEVELS:
            position = common.QUANTILES.index(f"q{round(level * 100):02d}")
            scale0[:, position] = coefficients[level][0]
        arms = {
            "ref": ensemble_columns(frame=reference, site=site),
            "a": frame,
            "at": at,
            "sp": apply_scale(frame=frame, scale=scale),
            "sp0": apply_scale(frame=frame, scale=scale0),
        }
        for arm, built in arms.items():
            write_arm(frame=built, arm=arm, period=period, site=site)
        print(period, site, "written", flush=True)


if __name__ == "__main__":
    main()
