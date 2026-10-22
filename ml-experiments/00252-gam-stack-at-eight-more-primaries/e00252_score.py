"""Score arm A0 (and its uncorrected base) against BG2, climatology, and the manual heuristic.

Rows: the scored year's runs of `fold_a_daily_gap_v1`, outside the hand-filtered load-shift events,
with a reading, leads up to 168 hours, and a BG2 forecast. Every score is the quantile score (twice
the mean pinball loss over the 11 quantiles) in points of effective capacity. Intervals come from a
paired bootstrap that resamples whole valid-time months, with the same months drawn at every
substation. Writes `work/<source>/scores.json` and prints every number the write-up quotes.

    uv run --no-sync python ml-experiments/00252-*/e00252_score.py [--arms d a]
"""

import argparse
from typing import Final

import e00252_common as common
import numpy as np
import polars as pl
from studies.dp_calendar import HOLIDAY_PERIODS
from studies.dp_features import FeatureCache
from studies.sources import ML_EXPERIMENTS_DATA_DIR

QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
SUN_UP: Final[float] = 0.01
HORIZON_HOURS: Final[float] = 168.0
N_RESAMPLES: Final[int] = 20_000
INTERVAL: Final[tuple[float, float]] = (2.5, 97.5)
PREDICTIONS: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00087-bg2-best-guess-assembled-with-drop-one-ablations"
    / "predictions"
)
REFERENCES: Final[dict[str, str]] = {
    "bg2": "p_s6_bg2",
    "climatology": "ref_climatology",
    "manual": "ref_manual_heuristic",
}
SLICES: Final[tuple[str, ...]] = (
    "all", "sunny", "winter", "winter_night", "cold_night", "christmas", "bank_holiday",
)  # fmt: skip
KEY: Final[list[str]] = ["power_fcst_init_time", "valid_time"]


def score(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return the quantile score of each row: twice the pinball loss averaged over the levels."""
    error = actual[:, None] - quantiles
    return 2.0 * np.mean(np.maximum(LEVELS * error, (LEVELS - 1.0) * error), axis=1)


def reference_frame(*, name: str, site: str) -> pl.DataFrame:
    """Return a stored reference at a site, quantiles in points of effective capacity."""
    frame = (
        pl.scan_parquet(PREDICTIONS / REFERENCES[name] / "predictions.parquet")
        .filter(pl.col("site") == site)
        .collect()
    )
    return frame.select(
        *KEY,
        *[(pl.col(q) * 100.0).alias(f"{name}_{q}") for q in QUANTILES],
    )


def site_frame(*, cache: FeatureCache, arm: str, site: str) -> pl.DataFrame:
    """Return the arm's scored rows at a site joined to the references, with the slice flags."""
    frame = pl.read_parquet(common.arm_file(arm=arm, period="scored", site=site)).filter(
        ~pl.col("switching_mask")
        & pl.col("actual").is_finite()
        & (pl.col("lead_hours") <= HORIZON_HOURS)
    )
    for name in REFERENCES:
        frame = frame.join(
            reference_frame(name=name, site=site), on=KEY, how="left", validate="1:1"
        )
    features = cache.sites[site]
    index = np.searchsorted(
        cache.end_times, frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    )
    unit = features.pv_unit_shape[index]
    local_month = cache.calendar.local_date[index].astype("datetime64[M]").astype(np.int64) % 12 + 1
    holiday = cache.calendar.holiday_period[index].astype(np.int64)
    daylight = unit > SUN_UP
    edge = float(np.quantile(unit[daylight], 2 / 3))
    winter = np.isin(local_month, [11, 12, 1, 2, 3])
    night = features.elevation_deg[index] < -5
    christmas = [HOLIDAY_PERIODS.index(n) for n in ("Christmas Day", "Boxing Day")]
    return frame.with_columns(
        site=pl.lit(site),
        month=frame["valid_time"].dt.convert_time_zone("Europe/London").dt.strftime("%Y-%m"),
        all=pl.lit(True),
        sunny=pl.Series(unit >= edge),
        winter=pl.Series(winter),
        winter_night=pl.Series(winter & night),
        cold_night=pl.Series(night & (features.temperature_c[index] < 2.0)),
        christmas=pl.Series(np.isin(holiday, christmas)),
        bank_holiday=pl.Series(holiday >= 0),
    )


def bootstrap(*, sums: np.ndarray, counts: np.ndarray, seed: int) -> dict:
    """Resample months: return per-site and pooled means with 95% intervals.

    Args:
        sums: Sum of the per-row difference, shape (sites, months).
        counts: Rows, shape (sites, months).
        seed: The random seed.

    Returns:
        `site` (point, low, high per site) and `pooled` (the mean of the site means).
    """
    rng = np.random.default_rng(seed)
    n_months = sums.shape[1]
    draws = rng.integers(0, n_months, size=(N_RESAMPLES, n_months))
    site_means = sums[:, draws].sum(axis=2) / np.maximum(counts[:, draws].sum(axis=2), 1)
    pooled = site_means.mean(axis=0)
    point = sums.sum(axis=1) / counts.sum(axis=1)
    low, high = np.percentile(site_means, INTERVAL, axis=1)
    return {
        "site": [(float(p), float(a), float(b)) for p, a, b in zip(point, low, high, strict=True)],
        "pooled": (
            float(point.mean()),
            float(np.percentile(pooled, INTERVAL[0])),
            float(np.percentile(pooled, INTERVAL[1])),
        ),
    }


def contrast(*, frames: dict[str, pl.DataFrame], column: str, flag: str) -> dict:
    """Return the paired month-resampled contrast of a per-row column, over a slice."""
    months = sorted({m for f in frames.values() for m in f["month"].unique().to_list()})
    sites = list(frames)
    sums = np.zeros((len(sites), len(months)))
    counts = np.zeros_like(sums)
    for i, site in enumerate(sites):
        part = frames[site].filter(pl.col(flag) & pl.col(column).is_finite())
        grouped = part.group_by("month").agg(s=pl.col(column).sum(), n=pl.len())
        for month, total, count in grouped.iter_rows():
            sums[i, months.index(month)] = total
            counts[i, months.index(month)] = count
    return bootstrap(sums=sums, counts=counts, seed=252) | {"sites": sites}


def add_scores(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add each forecast's per-row quantile score, in points, and the point-forecast errors."""
    actual = frame["actual"].to_numpy().astype(np.float64)
    columns = {"a0": QUANTILES}
    built = [
        pl.Series(
            "s_a0",
            score(
                actual=actual, quantiles=frame.select(list(QUANTILES)).to_numpy().astype(np.float64)
            ),
        )
    ]
    for name in REFERENCES:
        quantiles = frame.select([f"{name}_{q}" for q in QUANTILES]).to_numpy().astype(np.float64)
        built.append(pl.Series(f"s_{name}", score(actual=actual, quantiles=quantiles)))
    del columns
    return (
        frame.with_columns(*built)
        .with_columns(
            d_bg2=pl.col("s_a0") - pl.col("s_bg2"),
            e_mean=(pl.col("actual") - pl.col("mean")).abs(),
            e_bg2=(pl.col("actual") - pl.col("bg2_q50")).abs(),
        )
        .with_columns(
            d_mean_bg2=pl.col("e_mean") - pl.col("e_bg2"),
            d_clim=pl.col("s_a0") - pl.col("s_climatology"),
            d_manual=pl.col("s_a0") - pl.col("s_manual"),
        )
    )


def main() -> None:
    """Score every substation and write the tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="d")
    parser.add_argument("--sites", nargs="+", default=list(common.SCORED_SITES))
    args = parser.parse_args()
    cache = common.load_cache()
    frames = {}
    for site in args.sites:
        frame = add_scores(frame=site_frame(cache=cache, arm=args.arm, site=site))
        frames[site] = frame.filter(pl.col("bg2_q50").is_not_null())
    out: dict = {"arm": args.arm, "sites": args.sites, "rows": {}, "absolute": {}, "checks": {}}
    for site, frame in frames.items():
        out["rows"][site] = frame.height
        shared = frame.filter(
            pl.col("s_climatology").is_not_null() & pl.col("s_manual").is_not_null()
        )
        out["absolute"][site] = {
            "rows": shared.height,
            **{name: float(shared[f"s_{name}"].mean()) for name in ("a0", *REFERENCES)},
        }
        actual = frame["actual"].to_numpy().astype(np.float64)
        cold = frame.filter(pl.col("cold_night"))
        out["checks"][site] = {
            "median_minus_mean": float((frame["q50"] - frame["mean"]).mean()),
            "mae_mean_forecast": float(frame["e_mean"].mean()),
            "mae_bg2_median": float(frame["e_bg2"].mean()),
            "q95_cover_cold_night_a0": float((cold["actual"] <= cold["q95"]).mean())
            if cold.height
            else None,
            "q95_cover_cold_night_bg2": float((cold["actual"] <= cold["bg2_q95"]).mean())
            if cold.height
            else None,
            "tails": {
                q: {
                    "cover_a0": float((actual <= frame[q].to_numpy()).mean()),
                    "cover_bg2": float((actual <= frame[f"bg2_{q}"].to_numpy()).mean()),
                }
                for q in ("q01", "q05", "q95", "q99")
            },
        }
    out["contrast"] = {
        flag: {
            col: contrast(frames=frames, column=col, flag=flag)
            for col in ("d_bg2", "d_mean_bg2", "d_clim", "d_manual")
        }
        for flag in SLICES
    }
    leave_out = {}
    for left in args.sites:
        rest = {s: f for s, f in frames.items() if s != left}
        leave_out[left] = contrast(frames=rest, column="d_bg2", flag="all")["pooled"]
    out["leave_one_out_pooled_d_bg2"] = leave_out
    common.write_json_atomic(path=common.WORK_DIR / f"scores_{args.arm}.json", content=out)
    print_report(out=out)


def print_report(*, out: dict) -> None:
    """Print every number the write-up quotes."""
    sites = out["sites"]
    print("arm", out["arm"])
    for flag, tables in out["contrast"].items():
        for col, res in tables.items():
            pooled = tuple(round(x, 3) for x in res["pooled"])
            per = {
                s: tuple(round(x, 3) for x in r) for s, r in zip(sites, res["site"], strict=True)
            }
            print(f"{flag}|{col}|pooled {pooled}|{per}")
    for site in sites:
        print(site, out["rows"][site], {k: round(v, 3) for k, v in out["absolute"][site].items()})
        print(site, "checks", out["checks"][site])
    print(
        "leave one out (pooled A0 minus BG2)",
        {k: tuple(round(x, 3) for x in v) for k, v in out["leave_one_out_pooled_d_bg2"].items()},
    )


if __name__ == "__main__":
    main()
