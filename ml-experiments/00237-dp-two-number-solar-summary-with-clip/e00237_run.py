"""Does the two-number solar summary keep its gain when the solar term is clipped?

For D5 and D9 and each condition (`noclip`: experiment 184's arm A and member responses; `clip`:
experiment 214's as-of 60%/50% clip chain and its member responses from `e00237_members.py`),
this script builds arm A, arm M (the 51 member responses), and arm Q2 (the mean and standard
deviation of the 51 responses) as experiment 193 does, and scores their daylight quantile score.
The offset and spread factors of M and Q2 are fitted on the tuning months of D5 and D9 together,
separately for each condition, so the two conditions are fitted by the same rule on the same rows.

The clip chain's forecast is experiment 214's: the monthly T-all fit with the clip at the ceiling
that experiment 214 chose as-of for the origin's month (the fixed 60% clip in the tuning and
replicate months, where it chose 60% at every cut).

    systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00237-*/e00237_run.py
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Final

import e00237_common as common
import e00237_data as data
import numpy as np
import polars as pl
from scipy.stats import norm
from studies.dp_ens_weather import N_LEAD_BANDS, lead_band_of_hours
from studies.ens_members import mixture_quantiles

SITES: Final[tuple[str, ...]] = ("D5", "D9")
PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
N_MEMBERS: Final[int] = 51
Z_POINTS: Final[np.ndarray] = norm.ppf((np.arange(N_MEMBERS) + 0.5) / N_MEMBERS)
F_GRID: Final[np.ndarray] = np.round(np.arange(0.5, 1.3001, 0.05), 2)
S_GRID: Final[np.ndarray] = np.round(np.arange(0.5, 1.5001, 0.1), 2)
FIT_ROWS: Final[int] = 20_000
CHUNK: Final[int] = 4000
SEED: Final[int] = 237
KEY: Final[list[str]] = ["run", "valid_time"]
DAYLIGHT_ELEVATION_DEG: Final[float] = 5.0
ARM_NAMES: Final[tuple[str, ...]] = ("a", "m", "q2", "sg", "rs")
OUT_DIR: Final[Path] = common.WORK_DIR / "report"
ARM_FILE: Final[str] = "d_{period}_{site}.parquet"


def e184_frame(*, period: str) -> tuple[pl.DataFrame, np.ndarray]:
    """Return experiment 184's rows of D5 and D9 for a period and their member responses."""
    arms = common.E184_WORK_DIR / "arms"
    frame = pl.read_parquet(arms / f"{period}_frame.parquet").with_row_index("row")
    delta = np.load(arms / f"{period}_delta_m.npy")
    keep = frame.filter(pl.col("site").is_in(list(SITES)))
    return keep.drop("row"), delta[keep["row"].to_numpy()].astype(np.float64)


def clip_arm_file(*, arm: str, period: str, site: str) -> Path:
    """Return experiment 214's stage-2 forecast file of a clip arm."""
    return (
        common.E214_WORK_DIR
        / "arms"
        / arm
        / "dp_preds_s2"
        / ARM_FILE.format(period=period, site=site)
    )


def member_deltas(*, arm: str, period: str, site: str) -> pl.DataFrame:
    """Return an arm's member responses at a site, keyed by run and valid time."""
    stored = np.load(common.WORK_DIR / "members" / arm / f"{site}_{period}.npz")
    keys = pl.DataFrame(
        {
            "run": stored["run"],
            "valid_time": pl.Series(stored["valid_time"]).dt.replace_time_zone("UTC"),
        }
    )
    delta = stored["delta_m"].astype(np.float64)
    return keys.with_columns(delta=pl.Series(delta.tolist(), dtype=pl.Array(pl.Float64, N_MEMBERS)))


def asof_choices(*, site: str) -> dict[str, str]:
    """Return experiment 214's scored-year choice at a site: cut date (month start) to arm."""
    stored = json.loads((common.E214_WORK_DIR / "asof.json").read_text())
    return {c["cut"]: f"k{c['ceiling']}" for c in stored["scored"][site]["choices"]}


def clip_condition(
    *, frame: pl.DataFrame, delta: np.ndarray, period: str
) -> tuple[pl.DataFrame, np.ndarray]:
    """Return arm A's rows and member responses of the clip chain, in the order of `frame`."""
    frames, deltas = [], []
    for site in SITES:
        rows = frame.filter(pl.col("site") == site).with_row_index("pos")
        if period == "scored":
            choice = asof_choices(site=site)
        pieces = []
        for arm in ("k50", "k60"):
            stage2 = pl.read_parquet(clip_arm_file(arm=arm, period=period, site=site)).select(
                "run",
                "valid_time",
                "power_fcst_init_time",
                *data.QUANTILES,
                "actual",
                "mean",
            )
            joined = rows.select("pos", *KEY).join(stage2, on=KEY, how="inner", validate="1:1")
            joined = joined.join(
                member_deltas(arm=arm, period=period, site=site), on=KEY, validate="1:1"
            )
            if period == "scored":
                month = joined["power_fcst_init_time"].dt.truncate("1mo").dt.strftime("%Y-%m-%d")
                chosen = month.map_elements(
                    lambda m, c=choice: c.get(m, "none"), return_dtype=pl.String
                )
                joined = joined.filter(chosen == arm)
            elif arm != "k60":
                continue
            pieces.append(joined.with_columns(arm=pl.lit(arm)))
        merged = pl.concat(pieces).sort("pos")
        if merged.height != rows.height:
            msg = f"{site} {period}: {merged.height} clip rows for {rows.height} rows"
            raise ValueError(msg)
        keep = rows.sort("pos")
        replaced = keep.drop([*data.QUANTILES, "actual", "mean"]).with_columns(
            *[merged[c] for c in (*data.QUANTILES, "actual", "mean")]
        )
        frames.append(replaced)
        deltas.append(merged["delta"].to_numpy().astype(np.float64))
    out = pl.concat(frames)
    return out, np.concatenate(deltas)


def order_like(
    *, frame: pl.DataFrame, delta: np.ndarray, reference: pl.DataFrame
) -> tuple[pl.DataFrame, np.ndarray]:
    """Return a frame and its deltas re-ordered to follow the reference rows' keys and sites."""
    index = (
        reference.select("site", *KEY)
        .with_row_index("target")
        .join(
            frame.select("site", *KEY).with_row_index("source"), on=["site", *KEY], validate="1:1"
        )
        .sort("target")
    )
    source = index["source"].to_numpy()
    return frame[source], delta[source]


def summaries(*, delta: np.ndarray) -> dict[str, np.ndarray]:
    """Return the mean and standard deviation of each row's member responses."""
    with np.errstate(all="ignore"):
        return {"mu": np.nanmean(delta, axis=1), "sigma": np.nanstd(delta, axis=1)}


def gaussian_points(*, stats: dict[str, np.ndarray], spread: float) -> np.ndarray:
    """Return 51 points per row: `mu + spread * sigma * z`."""
    return stats["mu"][:, None] + spread * stats["sigma"][:, None] * Z_POINTS[None, :]


def mixture(*, mean: np.ndarray, delta: np.ndarray, offsets: np.ndarray, f: float) -> np.ndarray:
    """Return the 11 mixture quantiles of 51 centres `mean + delta`, each with offsets times `f`."""
    out = np.empty((len(mean), len(data.QUANTILES)))
    for start in range(0, len(mean), CHUNK):
        part = slice(start, start + CHUNK)
        centres = (mean[part, None] + delta[part])[:, :, None]
        out[part] = mixture_quantiles(member_quantiles=centres + f * offsets[part, None, :])
    return out


def pinball_rows(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return each row's pinball loss averaged over the 11 quantiles."""
    error = actual[:, None] - quantiles
    return np.mean(np.maximum(data.LEVELS * error, (data.LEVELS - 1.0) * error), axis=1)


def daylight(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the rows an arm changes: daylight (sun 5 degrees up) with at least one member."""
    return ((frame["elevation"] >= DAYLIGHT_ELEVATION_DEG) & (frame["n_members"] > 0)).to_numpy()


def fit_factors(
    *,
    frame: pl.DataFrame,
    make_points: Callable[[float], np.ndarray],
    fit_spread: bool,
) -> list[tuple[float, float]]:
    """Return `(f, s)` per lead band minimising the tuning daylight loss."""
    usable = data.usable_rows(frame=frame) & daylight(frame=frame)
    band = lead_band_of_hours(lead_hours=frame["lead_hours"].to_numpy())
    mean = frame["mean"].to_numpy().astype(np.float64)
    offsets = frame.select(list(data.QUANTILES)).to_numpy().astype(np.float64) - mean[:, None]
    actual = frame["actual"].to_numpy().astype(np.float64)
    rng = np.random.default_rng(SEED)
    out = []
    for b in range(N_LEAD_BANDS):
        rows = np.flatnonzero(usable & (band == b))
        rows = np.sort(rng.choice(rows, size=min(FIT_ROWS, len(rows)), replace=False))

        def loss(f: float, s: float, rows: np.ndarray = rows) -> float:
            q = mixture(mean=mean[rows], delta=make_points(s)[rows], offsets=offsets[rows], f=f)
            return float(pinball_rows(actual=actual[rows], quantiles=q).mean())

        f, s = 1.0, 1.0
        f = float(F_GRID[int(np.argmin([loss(g, s) for g in F_GRID]))])
        if fit_spread:
            s = float(S_GRID[int(np.argmin([loss(f, g) for g in S_GRID]))])
            f = float(F_GRID[int(np.argmin([loss(g, s) for g in F_GRID]))])
        out.append((f, s))
    return out


def apply_arm(
    *,
    frame: pl.DataFrame,
    make_points: Callable[[float], np.ndarray],
    factors: list[tuple[float, float]],
) -> np.ndarray:
    """Return an arm's 11 quantiles: the mixture on daylight rows and arm A's elsewhere."""
    out = frame.select(list(data.QUANTILES)).to_numpy().astype(np.float64)
    rows = np.flatnonzero(daylight(frame=frame))
    mean = frame["mean"].to_numpy().astype(np.float64)
    band = lead_band_of_hours(lead_hours=frame["lead_hours"].to_numpy())
    offsets = out - mean[:, None]
    for b, (f, s) in enumerate(factors):
        chosen = rows[band[rows] == b]
        if len(chosen):
            out[chosen] = mixture(
                mean=mean[chosen], delta=make_points(s)[chosen], offsets=offsets[chosen], f=f
            )
    return out


def condition_arms(
    *, tuning: tuple[pl.DataFrame, np.ndarray], periods: dict[str, tuple[pl.DataFrame, np.ndarray]]
) -> tuple[dict[str, list[tuple[float, float]]], dict[str, dict[str, np.ndarray]]]:
    """Fit M and Q2 on the tuning rows and return every period's arm quantiles."""
    frame, delta = tuning
    stats = summaries(delta=delta)
    factors = {
        "m": fit_factors(frame=frame, make_points=lambda s: np.nan_to_num(delta), fit_spread=False),
        "q2": fit_factors(
            frame=frame,
            make_points=lambda s: np.nan_to_num(gaussian_points(stats=stats, spread=s)),
            fit_spread=True,
        ),
        "sg": fit_factors(
            frame=frame,
            make_points=lambda s: np.nan_to_num(
                gaussian_points(stats={**stats, "mu": np.zeros_like(stats["mu"])}, spread=s)
            ),
            fit_spread=True,
        ),
    }
    factors["rs"] = fit_factors(
        frame=frame, make_points=lambda s: np.zeros_like(delta), fit_spread=False
    )
    arms: dict[str, dict[str, np.ndarray]] = {}
    for period, (rows, d) in periods.items():
        st = summaries(delta=d)
        arms[period] = {
            "a": rows.select(list(data.QUANTILES)).to_numpy().astype(np.float64),
            "m": apply_arm(
                frame=rows, make_points=lambda s, d=d: np.nan_to_num(d), factors=factors["m"]
            ),
            "q2": apply_arm(
                frame=rows,
                make_points=lambda s, st=st: np.nan_to_num(gaussian_points(stats=st, spread=s)),
                factors=factors["q2"],
            ),
            "rs": apply_arm(
                frame=rows,
                make_points=lambda s, d=d: np.zeros_like(d),
                factors=factors["rs"],
            ),
            "sg": apply_arm(
                frame=rows,
                make_points=lambda s, st=st: np.nan_to_num(
                    gaussian_points(stats={**st, "mu": np.zeros_like(st["mu"])}, spread=s)
                ),
                factors=factors["sg"],
            ),
        }
    return factors, arms


def main() -> None:
    """Build both conditions, fit, and write the per-row losses for `e00237_score.py`."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = {period: e184_frame(period=period) for period in PERIODS}
    conditions = {"noclip": base}
    clip = {}
    for period in PERIODS:
        frame, delta = base[period]
        clip[period] = clip_condition(frame=frame, delta=delta, period=period)
        clip[period] = order_like(frame=clip[period][0], delta=clip[period][1], reference=frame)
    conditions["clip"] = clip
    summary: dict = {}
    for name, periods in conditions.items():
        factors, arms = condition_arms(tuning=periods["tuning"], periods=periods)
        summary[name] = {"factors": factors}
        print(name, "factors", factors, flush=True)
        for period, (rows, _) in periods.items():
            actual = rows["actual"].to_numpy().astype(np.float64)
            losses = rows.select(
                "site",
                "run",
                "valid_time",
                "lead_hours",
                "month",
                "elevation",
                "unit",
                "n_members",
                "actual",
                "switching_mask",
                "csi_raw",
            )
            for arm, raw in arms[period].items():
                q = np.sort(raw, axis=1)
                losses = losses.with_columns(
                    **{
                        f"qs_{arm}": pl.Series(2.0 * pinball_rows(actual=actual, quantiles=q)),
                        f"abs_{arm}": pl.Series(np.abs(actual - q[:, 5])),
                        f"q01_{arm}": pl.Series(q[:, 0]),
                        f"q05_{arm}": pl.Series(q[:, 1]),
                        f"q95_{arm}": pl.Series(q[:, 9]),
                        f"q99_{arm}": pl.Series(q[:, 10]),
                    }
                )
            losses.write_parquet(OUT_DIR / f"{name}_{period}.parquet")
    (OUT_DIR / "factors.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
