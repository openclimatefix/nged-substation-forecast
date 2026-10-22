"""Score the arms of experiment 180 against BG2 and against the control, on experiment 178's rows.

Rows: the scored year's runs of `fold_a_daily_gap_v1`, outside the switching events, with a reading,
leads up to 168 hours. A sunny daylight half-hour is in the top third of its substation's daylight
half-hours (fleet unit shape above 0.01), ranked by the fixed fleet shape on CAMS irradiance, so
every arm is scored on identical rows. Writes `work/scores.json`.

    uv run --no-sync python ml-experiments/00180-*/e00214_score.py --arms c p k
"""

import argparse
from typing import Final

import e00214_common as common
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_month_totals
from studies.dp_calendar import HOLIDAY_PERIODS
from studies.dp_features import FeatureCache
from studies.sources import ML_EXPERIMENTS_DATA_DIR

QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
SUN_UP: Final[float] = 0.01
HORIZON_HOURS: Final[float] = 168.0
INTERVAL_LEVEL: Final[float] = 98.75
BG2_PATH: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00087-bg2-best-guess-assembled-with-drop-one-ablations"
    / "predictions"
    / "p_s6_bg2"
    / "predictions.parquet"
)
SLICES: Final[tuple[str, ...]] = (
    "all", "sunny", "winter", "winter_night", "cold_night", "e7_step", "christmas", "bank_holiday",
)  # fmt: skip


def pinball(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return each row's pinball loss averaged over the 11 quantiles."""
    error = actual[:, None] - quantiles
    return np.mean(np.maximum(LEVELS * error, (LEVELS - 1.0) * error), axis=1)


def arm_frame(
    *, cache: FeatureCache, arm: str, site: str, period: str, bg2: pl.DataFrame | None
) -> pl.DataFrame:
    """Return an arm's rows at a site in MW, with the slice flags, optionally joined to BG2."""
    path = common.arm_file(arm=arm, period=period, site=site)
    capacity = float(pl.read_parquet(BG2_PATH).filter(pl.col("site") == site)["capacity_mw"][0])
    frame = pl.read_parquet(path).filter(
        ~pl.col("switching_mask")
        & pl.col("actual").is_finite()
        & (pl.col("lead_hours") <= HORIZON_HOURS)
    )
    frame = frame.with_columns(
        [pl.col(c) * capacity / 100.0 for c in (*QUANTILES, "actual", "mean")]
    )
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
    """Add one boolean column per slice."""
    day = frame.filter(pl.col("unit") > SUN_UP)["unit"].to_numpy()
    edge = float(np.quantile(day, 2 / 3))
    winter = pl.col("local_month").is_in([11, 12, 1, 2, 3])
    night = pl.col("elevation") < -5
    christmas = [HOLIDAY_PERIODS.index(n) for n in ("Christmas Day", "Boxing Day")]
    return frame.with_columns(
        all=pl.lit(True),
        sunny=pl.col("unit") >= edge,
        winter=winter,
        winter_night=winter & night,
        cold_night=night & (pl.col("temperature") < 2.0),
        e7_step=pl.col("half_hour").is_in([1, 2, 3, 4, 5, 6]),
        christmas=pl.col("holiday").is_in(christmas),
        bank_holiday=pl.col("holiday") >= 0,
    ).with_columns(
        abs_err=(pl.col("actual") - pl.col("q50")).abs(),
        pin=pl.Series(
            pinball(
                actual=frame["actual"].to_numpy().astype(np.float64),
                quantiles=frame.select(list(QUANTILES)).to_numpy().astype(np.float64),
            )
        ),
    )


def interval(*, frame: pl.DataFrame, column: str) -> tuple[float, float, float]:
    """Return the mean of a column and its month-block interval at `INTERVAL_LEVEL`."""
    monthly = frame.group_by("month").agg(total=pl.col(column).sum(), count=pl.len()).sort("month")
    sums = monthly["total"].to_numpy()[None, :]
    counts = monthly["count"].to_numpy().astype(np.float64)
    low, high = bootstrap_month_totals(month_sums=sums, month_counts=counts, level=INTERVAL_LEVEL)
    return float(sums.sum() / counts.sum()), float(low), float(high)


def main() -> None:
    """Score every arm at every substation and write the tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["c", "p"])
    parser.add_argument("--sites", nargs="+", default=list(common.SCORED_SITES))
    parser.add_argument("--period", default="scored")
    args = parser.parse_args()
    cache = common.load_cache()
    bg2 = pl.read_parquet(BG2_PATH) if args.period == "scored" else None
    out: dict = {}
    for site in args.sites:
        control = add_slices(
            frame=arm_frame(cache=cache, arm="c", site=site, period=args.period, bg2=bg2)
        )
        key = ["power_fcst_init_time", "valid_time"]
        for arm in args.arms:
            frame = add_slices(
                frame=arm_frame(cache=cache, arm=arm, site=site, period=args.period, bg2=bg2)
            )
            if frame.height != control.height:
                msg = f"{arm} at {site} has {frame.height} rows, the control {control.height}"
                raise ValueError(msg)
            joined = frame.join(
                control.select(*key, c_abs=pl.col("abs_err"), c_pin=pl.col("pin")),
                on=key,
                how="inner",
                validate="1:1",
            )
            if bg2 is not None:
                b = joined.select([f"b_{q}" for q in QUANTILES]).to_numpy().astype(np.float64)
                joined = joined.with_columns(
                    bg2_abs=(pl.col("actual") - pl.col("b_q50")).abs(),
                    bg2_pin=pl.Series(
                        pinball(actual=joined["actual"].to_numpy().astype(np.float64), quantiles=b)
                    ),
                )
            for name in SLICES:
                part = joined.filter(pl.col(name))
                if part.height < 100:
                    continue
                cell = {
                    "rows": part.height,
                    "mae": float(part["abs_err"].to_numpy().mean()),
                    "pin": float(part["pin"].to_numpy().mean()),
                }
                diff = part.with_columns(
                    d=pl.col("abs_err") - pl.col("c_abs"), dp=pl.col("pin") - pl.col("c_pin")
                )
                cell["mae_minus_control"] = interval(frame=diff, column="d")
                cell["pin_minus_control"] = interval(frame=diff, column="dp")
                if bg2 is not None:
                    gap = part.with_columns(g=pl.col("abs_err") - pl.col("bg2_abs"))
                    cell["bg2_mae"] = float(part["bg2_abs"].to_numpy().mean())
                    cell["gap_to_bg2"] = interval(frame=gap, column="g")
                out[f"{arm}|{site}|{name}"] = cell
    target = common.WORK_DIR / f"scores_{args.period}.json"
    existing = {}
    if target.exists():
        import json

        existing = json.loads(target.read_text())
    existing.update(out)
    common.write_json_atomic(path=target, content=existing)
    for key_, cell in out.items():
        if key_.split("|")[2] in ("sunny", "all"):
            print(
                key_,
                {
                    k: (
                        round(v, 4)
                        if isinstance(v, float)
                        else (tuple(round(x, 4) for x in v) if isinstance(v, tuple) else v)
                    )
                    for k, v in cell.items()
                },
            )


if __name__ == "__main__":
    main()
