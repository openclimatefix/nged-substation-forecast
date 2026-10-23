"""Score the arms of experiment 280 against BG2 and against the control, on experiment 178's rows.

Rows: the scored year's runs of `fold_a_daily_gap_v1`, outside the switching events, with a reading,
leads up to 168 hours. A sunny daylight half-hour is in the top third of its substation's daylight
half-hours (fleet unit shape above 0.01), ranked by the fixed fleet shape on CAMS irradiance, so
every arm is scored on identical rows. Writes `work/scores.json`.

    uv run --no-sync python ml-experiments/00180-*/e00280_score.py --arms c p k
"""

import argparse
from typing import Final

import e00280_common as common
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache
from studies.sources import ML_EXPERIMENTS_DATA_DIR

QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
SUN_UP: Final[float] = 0.01
HORIZON_HOURS: Final[float] = 168.0
N_BOOTSTRAP: Final[int] = 4000
INTERVAL_LEVEL: Final[float] = 99.0
"""Five planned contrasts, so a Bonferroni level of 99%."""
BG2_PATH: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00087-bg2-best-guess-assembled-with-drop-one-ablations"
    / "predictions"
    / "p_s6_bg2"
    / "predictions.parquet"
)
SLICES: Final[tuple[str, ...]] = ("all", "sunny", "aj_day", "aj_night")
SITES_PRIMARY: Final[tuple[str, ...]] = ("D9", "D16")
"""The two substations where the real fit supports a much lower capacity than the pinned one."""
CONFIGURATIONS: Final[dict[str, dict[str, str]]] = {
    "off": {},
    "asm": {"D5": "k5"},
    "k59": {"D5": "k5", "D9": "k9"},
}
"""Each configuration names the clip suffix of the arms read at a substation; the other
substations read the unclipped arm. `asm` is the assembly of experiment 180 and decides."""
ARMS: Final[tuple[str, ...]] = ("c0", "e0", "e1", "e2", "f1", "s1", "s2")


def pinball(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return each row's pinball loss averaged over the 11 quantiles."""
    error = actual[:, None] - quantiles
    return np.mean(np.maximum(LEVELS * error, (LEVELS - 1.0) * error), axis=1)


def arm_frame(
    *, cache: FeatureCache, arm: str, site: str, period: str, bg2: pl.DataFrame | None
) -> pl.DataFrame:
    """Return an arm's rows at a site in MW, with the slice flags, optionally joined to BG2."""
    path = common.WORK_DIR / "arms" / arm / "dp_preds_s2" / f"d_{period}_{site}.parquet"
    capacity = float(pl.read_parquet(BG2_PATH).filter(pl.col("site") == site)["capacity_mw"][0])
    frame = pl.read_parquet(path).filter(
        ~pl.col("switching_mask")
        & pl.col("actual").is_finite()
        & (pl.col("lead_hours") <= HORIZON_HOURS)
    )
    frame = frame.with_columns([pl.col(c) * capacity / 100.0 for c in (*QUANTILES, "actual")])
    key = ["power_fcst_init_time", "valid_time"]
    if bg2 is not None:
        b = bg2.filter(pl.col("site") == site).select(
            *key, **{f"b_{c}": pl.col(c) * capacity for c in QUANTILES}
        )
        frame = frame.join(b, on=key, how="inner", validate="1:1")
    ends = cache.end_times
    valid = frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    index = np.searchsorted(ends, valid)
    features = cache.sites[site]
    unit = features.pv_unit_shape[index]
    return frame.with_columns(
        unit=pl.Series(unit),
        month=frame["valid_time"].dt.convert_time_zone("Europe/London").dt.strftime("%Y-%m"),
        elevation=pl.Series(features.elevation_deg[index]),
        temperature=pl.Series(features.temperature_c[index]),
        local_month=pl.Series(
            cache.calendar.local_date[index].astype("datetime64[M]").astype(np.int64) % 12 + 1
        ),
        half_hour=pl.Series(cache.calendar.half_hour[index].astype(np.int64)),
        holiday=pl.Series(cache.calendar.holiday_period[index].astype(np.int64)),
    )


def add_slices(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add one boolean column per slice, from the fixed fleet shape and the calendar."""
    day = frame.filter(pl.col("unit") > SUN_UP)["unit"].to_numpy()
    edge = float(np.quantile(day, 2 / 3))
    spring = pl.col("local_month").is_in([4, 5, 6])
    return frame.with_columns(
        all=pl.lit(True),
        sunny=pl.col("unit") >= edge,
        aj_day=spring & (pl.col("unit") > SUN_UP),
        aj_night=spring & (pl.col("elevation") < -5),
    ).with_columns(
        abs_err=(pl.col("actual") - pl.col("q50")).abs(),
        pin=pl.Series(
            pinball(
                actual=frame["actual"].to_numpy().astype(np.float64),
                quantiles=frame.select(list(QUANTILES)).to_numpy().astype(np.float64),
            )
        ),
    )


GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "primary": SITES_PRIMARY,
    "d5": ("D5",),
    "others": ("D10", "D11", "D14"),
    "decision": common.DECISION_SITES,
}
"""Equal-weight site groups. D9 and D16 are the planned primary pair, D5 is read separately, and
D10, D11, and D14 are the substations where the real fit does not pick the capacity cut."""


def arm_name(*, arm: str, site: str, configuration: str) -> str:
    """Return the stored arm whose frame a configuration reads at a substation."""
    suffix = CONFIGURATIONS[configuration].get(site)
    return arm if suffix is None else f"{arm}_{suffix}"


def week_table(*, frames: dict[str, pl.DataFrame], value: str, slice_name: str) -> tuple:
    """Return per-site sums and counts of `value` in each ISO week of the forecast init time."""
    weeks = sorted(
        {
            w
            for f in frames.values()
            for w in f["power_fcst_init_time"].dt.truncate("1w").unique().to_list()
        }
    )
    sums = np.zeros((len(frames), len(weeks)))
    counts = np.zeros((len(frames), len(weeks)))
    for i, frame in enumerate(frames.values()):
        grouped = (
            frame.filter(pl.col(slice_name))
            .group_by(pl.col("power_fcst_init_time").dt.truncate("1w").alias("week"))
            .agg(total=pl.col(value).sum(), n=pl.len())
        )
        for week, total, n in grouped.iter_rows():
            sums[i, weeks.index(week)] = total
            counts[i, weeks.index(week)] = n
    return sums, counts


def interval(*, sums: np.ndarray, counts: np.ndarray, seed: int = 0) -> dict[str, float]:
    """Return the equal-site-weight mean and its week-block bootstrap interval."""
    rng = np.random.default_rng(seed)
    point = float(np.mean(sums.sum(axis=1) / counts.sum(axis=1)))
    index = rng.integers(0, sums.shape[1], size=(N_BOOTSTRAP, sums.shape[1]))
    draws = np.mean(sums[:, index].sum(axis=2) / counts[:, index].sum(axis=2), axis=0)
    tail = (100.0 - INTERVAL_LEVEL) / 2.0
    low, high = np.percentile(draws, [tail, 100.0 - tail])
    return {"mean": point, "low": float(low), "high": float(high)}


def main() -> None:
    """Score every configuration, arm, slice, and site group, and write `work/scores.json`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    args = parser.parse_args()
    cache = common.load_cache()
    bg2 = pl.read_parquet(BG2_PATH)
    sites = common.SCORED_SITES
    out: dict = {}
    for configuration in CONFIGURATIONS:
        frames: dict[str, dict[str, pl.DataFrame]] = {}
        for arm in args.arms:
            frames[arm] = {}
            for site in sites:
                name = arm_name(arm=arm, site=site, configuration=configuration)
                if not (
                    common.WORK_DIR / "arms" / name / "dp_preds_s2" / f"d_scored_{site}.parquet"
                ).exists():
                    continue
                frames[arm][site] = add_slices(
                    frame=arm_frame(cache=cache, arm=name, site=site, period="scored", bg2=bg2)
                )
        control = frames.get("c0", {})
        for arm, by_site in frames.items():
            for group, members in GROUPS.items():
                if not all(s in by_site and s in control for s in members):
                    continue
                for slice_name in SLICES:
                    key = f"{configuration}|{arm}|{group}|{slice_name}"
                    diffs = {}
                    for s in members:
                        if by_site[s].height != control[s].height:
                            msg = f"{arm} and c0 differ in row count at {s}"
                            raise ValueError(msg)
                        diffs[s] = control[s].select(
                            "power_fcst_init_time",
                            pl.col(slice_name),
                            d_mae=by_site[s]["abs_err"] - control[s]["abs_err"],
                            d_bg2=by_site[s]["abs_err"]
                            - (by_site[s]["actual"] - by_site[s]["b_q50"]).abs(),
                            d_pin=by_site[s]["pin"] - control[s]["pin"],
                            mae=by_site[s]["abs_err"],
                        )
                    entry = {}
                    for value in ("d_mae", "d_bg2", "d_pin", "mae"):
                        sums, counts = week_table(frames=diffs, value=value, slice_name=slice_name)
                        entry[value] = interval(sums=sums, counts=counts)
                    out[key] = entry
    target = common.WORK_DIR / "scores.json"
    common.write_json_atomic(path=target, content=out)
    print(f"wrote {target} with {len(out)} entries")


if __name__ == "__main__":
    main()
