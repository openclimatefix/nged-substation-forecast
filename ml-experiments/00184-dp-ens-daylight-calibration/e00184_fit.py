# ruff: noqa: E501
"""Fit the factors of the member arms and the scale model of arm S on the tuning months, build every arm.

| Arm | Quantiles |
|---|---|
| `a` | experiment 164's fix O (experiment 169's arm A), as stored |
| `m` | 51 members through the solar term; arm A's offsets, times one factor per lead band, around each |
| `mc` | `m` with the members' responses centred on zero (control) |
| `f` | `m` with every weather input from the member |
| `s` | daylight offsets of arm A times a scale from the members' clear-sky-index spread statistics |
| `sp` | arm S's coefficients on statistics from a random row of the same site, month, and hour band |
| `mr` | placebo: arm A's offsets times a factor per lead band, with every member's response zero |
| `mp` | placebo: `m` with each row's 51 member responses taken from a random row of the same site, month, and hour band |

Only daylight rows (solar elevation of 5 degrees or more) change. Every coefficient and factor is
fitted on the tuning months.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00184-*/e00184_fit.py
"""

import argparse
from typing import Final

import e00184_common as common
import e00184_data as data
import numpy as np
import polars as pl
from scipy import sparse
from scipy.optimize import linprog
from studies.dp_ens_weather import N_LEAD_BANDS, lead_band_of_hours
from studies.ens_members import mixture_quantiles

ARMS: Final[tuple[str, ...]] = ("a", "m", "mc", "f", "s", "sp", "mr", "mp")
FACTOR_GRID: Final[np.ndarray] = np.round(np.arange(0.5, 1.3001, 0.05), 2)
FACTOR_FIT_ROWS: Final[int] = 20_000
SCALE_FIT_ROWS: Final[int] = 150_000
FIT_LEVELS: Final[tuple[str, ...]] = ("q01", "q05", "q10", "q25", "q75", "q90", "q95", "q99")
KNOT_PERCENTILES: Final[tuple[float, ...]] = (25.0, 50.0, 75.0, 90.0)
MINIMUM_OFFSET: Final[float] = 1e-3
INTERCEPT_RANGE: Final[tuple[float, float]] = (0.3, 3.0)
COEFFICIENT_MAX: Final[float] = 5.0
SEED: Final[int] = 184
CHUNK: Final[int] = 4000
ARM_DIR = common.WORK_DIR / "arms"


def fill_missing(*, delta: np.ndarray) -> np.ndarray:
    """Return the member deltas with an unusable member's entry set to the row's mean of the rest."""
    mean = np.nanmean(delta, axis=1, keepdims=True)
    return np.where(np.isfinite(delta), delta, mean)


def mixture(
    *, mean: np.ndarray, delta: np.ndarray, offsets: np.ndarray, factor: np.ndarray
) -> np.ndarray:
    """Return the 11 mixture quantiles of the members' forecasts, each with scaled offsets around it.

    Args:
        mean: Arm A's mean forecast, shape (rows,).
        delta: Each member's response, shape (rows, members), with no missing value.
        offsets: Arm A's 11 quantiles minus its mean, shape (rows, 11).
        factor: The offsets' factor per row, shape (rows,).
    """
    out = np.empty((len(mean), len(data.QUANTILES)))
    for start in range(0, len(mean), CHUNK):
        part = slice(start, start + CHUNK)
        centres = (mean[part, None] + delta[part])[:, :, None]
        members = centres + factor[part, None, None] * offsets[part, None, :]
        out[part] = mixture_quantiles(member_quantiles=members)
    return out


def pinball_rows(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return each row's pinball loss averaged over the 11 quantiles."""
    error = actual[:, None] - quantiles
    return np.mean(np.maximum(data.LEVELS * error, (data.LEVELS - 1.0) * error), axis=1)


def fit_factors(*, frame: pl.DataFrame, delta: np.ndarray, centre: bool) -> list[float]:
    """Return the offsets' factor per calibration lead band that minimises the tuning daylight loss."""
    usable = data.usable_rows(frame=frame) & (frame["kind"] == "daylight").to_numpy()
    lead_band = lead_band_of_hours(lead_hours=frame["lead_hours"].to_numpy())
    mean = frame["mean"].to_numpy().astype(np.float64)
    offsets = frame.select(list(data.QUANTILES)).to_numpy().astype(np.float64) - mean[:, None]
    actual = frame["actual"].to_numpy().astype(np.float64)
    rng = np.random.default_rng(SEED)
    factors = []
    for band in range(N_LEAD_BANDS):
        rows = np.flatnonzero(usable & (lead_band == band))
        rows = np.sort(rng.choice(rows, size=min(FACTOR_FIT_ROWS, len(rows)), replace=False))
        filled = fill_missing(delta=delta[rows])
        if centre:
            filled = filled - filled.mean(axis=1, keepdims=True)
        losses = [
            float(
                pinball_rows(
                    actual=actual[rows],
                    quantiles=mixture(
                        mean=mean[rows],
                        delta=filled,
                        offsets=offsets[rows],
                        factor=np.full(len(rows), f),
                    ),
                ).mean()
            )
            for f in FACTOR_GRID
        ]
        factors.append(float(FACTOR_GRID[int(np.argmin(losses))]))
        print("band", band, "factor", factors[-1], "loss", min(losses), flush=True)
    return factors


def apply_member_arm(
    *, frame: pl.DataFrame, delta: np.ndarray, factors: list[float], centre: bool
) -> np.ndarray:
    """Return an arm's 11 quantiles: the mixture on daylight rows and arm A's elsewhere."""
    out = frame.select(list(data.QUANTILES)).to_numpy().astype(np.float64)
    daylight = np.flatnonzero(
        (frame["kind"] == "daylight").to_numpy() & (frame["n_members"] > 0).to_numpy()
    )
    mean = frame["mean"].to_numpy().astype(np.float64)
    offsets = out - mean[:, None]
    lead_band = lead_band_of_hours(lead_hours=frame["lead_hours"].to_numpy())
    filled = fill_missing(delta=delta[daylight])
    if centre:
        filled = filled - filled.mean(axis=1, keepdims=True)
    factor = np.array(factors)[lead_band[daylight]]
    out[daylight] = mixture(
        mean=mean[daylight], delta=filled, offsets=offsets[daylight], factor=factor
    )
    return out


def basis(*, stats: np.ndarray, knots: dict) -> np.ndarray:
    """Return the design matrix: a constant, then one hinge per knot for each of the three statistics."""
    columns = [np.ones(len(stats))]
    for k, name in enumerate(data.STAT_COLUMNS):
        scale = knots[name]["scale"]
        columns += [np.maximum(stats[:, k] / scale - knot, 0.0) for knot in knots[name]["knots"]]
    return np.stack(columns, axis=1)


def fit_level(
    *, level: float, residual: np.ndarray, offset: np.ndarray, design: np.ndarray
) -> np.ndarray:
    """Return the coefficients minimising the pinball loss of `mean + offset * (design @ w)`.

    With `ratio = residual / offset`, a row's loss is `abs(offset)` times the pinball loss of
    `ratio - design @ w`, at level `level` where the offset is positive and `1 - level` where it is
    negative (the method of experiment 169).
    """
    n_rows, n_terms = design.shape
    tau = np.where(offset > 0, level, 1.0 - level)
    weight = np.abs(offset)
    cost = np.concatenate([np.zeros(n_terms), weight * tau, weight * (1.0 - tau)])
    identity = sparse.identity(n_rows, format="csr")
    equality = sparse.hstack([sparse.csr_matrix(design), identity, -identity], format="csr")
    bounds = (
        [INTERCEPT_RANGE] + [(0.0, COEFFICIENT_MAX)] * (n_terms - 1) + [(0.0, None)] * (2 * n_rows)
    )
    result = linprog(cost, A_eq=equality, b_eq=residual / offset, bounds=bounds, method="highs")
    if not result.success:
        msg = f"level {level}: {result.message}"
        raise RuntimeError(msg)
    return result.x[:n_terms]


def fit_scale_model(*, frame: pl.DataFrame) -> dict:
    """Fit arm S's coefficients on the tuning daylight rows and return them with the knots."""
    usable = data.usable_rows(frame=frame) & (frame["kind"] == "daylight").to_numpy()
    stats = frame.select(list(data.STAT_COLUMNS)).to_numpy().astype(np.float64)
    usable &= np.isfinite(stats).all(axis=1)
    knots = {}
    for k, name in enumerate(data.STAT_COLUMNS):
        values = stats[usable, k]
        scale = float(np.percentile(values, 90.0)) or 1.0
        knots[name] = {
            "scale": scale,
            "knots": [float(np.percentile(values, p)) / scale for p in KNOT_PERCENTILES],
        }
    rng = np.random.default_rng(SEED)
    rows = np.sort(
        rng.choice(
            np.flatnonzero(usable), size=min(SCALE_FIT_ROWS, int(usable.sum())), replace=False
        )
    )
    mean = frame["mean"].to_numpy().astype(np.float64)[rows]
    actual = frame["actual"].to_numpy().astype(np.float64)[rows]
    quantiles = frame.select(list(data.QUANTILES)).to_numpy().astype(np.float64)[rows]
    design = basis(stats=stats[rows], knots=knots)
    coefficients = {}
    for name in FIT_LEVELS:
        position = data.QUANTILES.index(name)
        offset = quantiles[:, position] - mean
        keep = np.abs(offset) > MINIMUM_OFFSET
        level = data.LEVELS[position]
        coefficients[name] = fit_level(
            level=float(level),
            residual=(actual - mean)[keep],
            offset=offset[keep],
            design=design[keep],
        ).tolist()
        print("scale model", name, np.round(coefficients[name], 3).tolist(), flush=True)
    return {"knots": knots, "coefficients": coefficients, "fit_rows": len(rows)}


def apply_scale_arm(*, frame: pl.DataFrame, model: dict, shuffled: bool) -> np.ndarray:
    """Return an arm's 11 quantiles: arm A's with daylight offsets rescaled by the fitted model."""
    quantiles = frame.select(list(data.QUANTILES)).to_numpy().astype(np.float64)
    mean = frame["mean"].to_numpy().astype(np.float64)
    stats = frame.select(list(data.STAT_COLUMNS)).to_numpy().astype(np.float64)
    daylight = (frame["kind"] == "daylight").to_numpy() & np.isfinite(stats).all(axis=1)
    if shuffled:
        stats = shuffle_statistics(frame=frame, stats=stats, daylight=daylight)
    design = basis(stats=np.nan_to_num(stats), knots=model["knots"])
    scale = np.ones_like(quantiles)
    for name, coefficients in model["coefficients"].items():
        position = data.QUANTILES.index(name)
        scale[:, position] = np.where(daylight, design @ np.array(coefficients), 1.0)
    return np.sort(mean[:, None] + (quantiles - mean[:, None]) * scale, axis=1)


def shuffle_statistics(
    *, frame: pl.DataFrame, stats: np.ndarray, daylight: np.ndarray
) -> np.ndarray:
    """Return the statistics with each daylight row's triple taken from a random row of its group.

    A group is the same site, calendar month, and hour band, so the placebo keeps the season and
    the time of day and drops the match between the day's cloud and the day's statistics.
    """
    group = (
        frame["site"].cast(pl.String)
        + "|"
        + frame["valid_time"].dt.month().cast(pl.String)
        + "|"
        + frame["hour_band"].cast(pl.String)
    ).to_numpy()
    rng = np.random.default_rng(SEED)
    out = stats.copy()
    for key in np.unique(group[daylight]):
        rows = np.flatnonzero(daylight & (group == key))
        out[rows] = stats[rng.permutation(rows)]
    return out


def shuffle_rows(*, frame: pl.DataFrame, delta: np.ndarray) -> np.ndarray:
    """Return the member responses with each daylight row's 51 values taken from a random row of its group."""
    daylight = (frame["kind"] == "daylight").to_numpy()
    group = (
        frame["site"].cast(pl.String)
        + "|"
        + frame["valid_time"].dt.month().cast(pl.String)
        + "|"
        + frame["hour_band"].cast(pl.String)
    ).to_numpy()
    rng = np.random.default_rng(SEED + 1)
    out = delta.copy()
    for key in np.unique(group[daylight]):
        rows = np.flatnonzero(daylight & (group == key))
        out[rows] = delta[rng.permutation(rows)]
    return out


def main() -> None:
    """Fit on the tuning months and write every arm for every period."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    cache = common.load_cache()
    periods = {
        p: data.load_period(period=p, cache=cache) for p in ("tuning", "replicate", "scored")
    }
    tuning, delta_m, delta_f = periods["tuning"]
    parameters = {
        "m": fit_factors(frame=tuning, delta=delta_m, centre=False),
        "mc": fit_factors(frame=tuning, delta=delta_m, centre=True),
        "f": fit_factors(frame=tuning, delta=delta_f, centre=False),
        "mr": fit_factors(frame=tuning, delta=np.zeros_like(delta_m), centre=False),
        "mp": fit_factors(
            frame=tuning, delta=shuffle_rows(frame=tuning, delta=delta_m), centre=False
        ),
        "scale_model": fit_scale_model(frame=tuning),
    }
    common.write_json_atomic(path=common.WORK_DIR / "fit_parameters.json", content=parameters)
    ARM_DIR.mkdir(parents=True, exist_ok=True)
    for period, (frame, dm, df) in periods.items():
        built = {
            "a": frame.select(list(data.QUANTILES)).to_numpy().astype(np.float64),
            "m": apply_member_arm(frame=frame, delta=dm, factors=parameters["m"], centre=False),
            "mc": apply_member_arm(frame=frame, delta=dm, factors=parameters["mc"], centre=True),
            "f": apply_member_arm(frame=frame, delta=df, factors=parameters["f"], centre=False),
            "s": apply_scale_arm(frame=frame, model=parameters["scale_model"], shuffled=False),
            "sp": apply_scale_arm(frame=frame, model=parameters["scale_model"], shuffled=True),
            "mr": apply_member_arm(
                frame=frame, delta=np.zeros_like(dm), factors=parameters["mr"], centre=False
            ),
            "mp": apply_member_arm(
                frame=frame,
                delta=shuffle_rows(frame=frame, delta=dm),
                factors=parameters["mp"],
                centre=False,
            ),
        }
        frame.write_parquet(ARM_DIR / f"{period}_frame.parquet")
        for arm, quantiles in built.items():
            np.save(ARM_DIR / f"{period}_{arm}.npy", np.sort(quantiles, axis=1).astype(np.float32))
        # Member responses of the daylight rows, for the figures.
        np.save(ARM_DIR / f"{period}_delta_m.npy", dm)
        print(period, "written", frame.height, flush=True)


if __name__ == "__main__":
    main()
