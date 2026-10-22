"""Experiment 214, stage 2: choose each Primary's clip ceiling as-of, splice the arms, and score.

At the first day of each month, the ceiling is the arm (control, or a clip at 50% to 90% of held
capacity) with the lowest mean absolute error of q50 on the Primary's sunny daylight half-hours
over the 12 months before the cut (an expanding window from 2024-07 with at least 3 months for the
replicate months), using only forecast origins whose 168-hour window ended before the cut. The
sunny threshold is the top third of the Primary's daylight unit shape before 2024-07-01. The
forecasts of the cut's own month then come from the chosen arm.

Writes `work/asof.json` (choices, scores, intervals, checks).

    uv run --no-sync python ml-experiments/00214-*/e00214_asof.py
"""

from datetime import UTC, datetime, timedelta
from typing import Final

import e00214_common as common
import e00214_score as score
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_month_totals
from studies.dp_features import FeatureCache

CEILINGS: Final[tuple[int, ...]] = (50, 60, 70, 80, 90, 100)
"""Ceilings in percent of held capacity; 100 is the control (no clip)."""
SWEEP_SITES: Final[tuple[str, ...]] = ("D5", "D9")
FIXED_SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
REST_SITES: Final[tuple[str, ...]] = ("D10", "D11", "D14", "D16")
LEVEL: Final[float] = 98.33
"""Three planned contrasts, so the Bonferroni level is 100 - 5 / 3 percent."""
WINDOW_MONTHS: Final[int] = 12
MIN_WINDOW_MONTHS: Final[int] = 3
FIRST_FORECAST: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
LEAD_WINDOW: Final[timedelta] = timedelta(hours=168)
KEY: Final[list[str]] = ["power_fcst_init_time", "valid_time"]
TAILS: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
TAIL_LEVEL: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}


def arm_of(*, ceiling: int) -> str:
    """Return the arm name of a ceiling: the control `c` at 100, else `k<percent>`."""
    return "c" if ceiling == 100 else f"k{ceiling}"


def month_start(*, year: int, month: int) -> datetime:
    """Return the first instant of a UTC month."""
    return datetime(year, month, 1, tzinfo=UTC)


def add_months(*, moment: datetime, months: int) -> datetime:
    """Return the first of the month `months` after the first-of-month `moment`."""
    index = moment.year * 12 + moment.month - 1 + months
    return month_start(year=index // 12, month=index % 12 + 1)


def load(
    *, cache: FeatureCache, site: str, arm: str, period: str, bg2: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Return an arm's rows in MW at a site and period, with the unit shape and the abs error."""
    frame = score.arm_frame(cache=cache, arm=arm, site=site, period=period, bg2=bg2)
    return frame.with_columns(
        abs_err=(pl.col("actual") - pl.col("q50")).abs(),
        origin_month=pl.col("power_fcst_init_time")
        .dt.convert_time_zone("UTC")
        .dt.truncate("1mo")
        .dt.replace_time_zone("UTC"),
    )


def sunny_edge(*, cache: FeatureCache, site: str) -> float:
    """Return the top-third threshold of the daylight unit shape before 2024-07-01."""
    features = cache.sites[site]
    before = cache.end_times < np.datetime64("2024-07-01")
    unit = features.pv_unit_shape[before]
    return float(np.quantile(unit[unit > score.SUN_UP], 2 / 3))


def window_mean(
    *, table: pl.DataFrame, arm: str, start: datetime, cut: datetime
) -> tuple[float, int, int]:
    """Return the arm's mean sunny error over origins in the window, rows, and distinct months."""
    part = table.filter(
        (pl.col("arm") == arm)
        & (pl.col("power_fcst_init_time") >= start)
        & (pl.col("power_fcst_init_time") + LEAD_WINDOW <= cut)
    )
    if part.height == 0:
        return float("nan"), 0, 0
    return float(part["abs_err"].to_numpy().mean()), part.height, part["origin_month"].n_unique()


def choose(*, table: pl.DataFrame, cut: datetime, expanding: bool) -> dict:
    """Return the as-of choice at a cut: the ceiling and each candidate's window error."""
    start = FIRST_FORECAST if expanding else add_months(moment=cut, months=-WINDOW_MONTHS)
    start = max(start, FIRST_FORECAST)
    errors = {}
    months = 0
    for ceiling in CEILINGS:
        mean, _rows, n_months = window_mean(
            table=table, arm=arm_of(ceiling=ceiling), start=start, cut=cut
        )
        errors[ceiling] = mean
        months = max(months, n_months)
    best = min(CEILINGS, key=lambda c: (round(errors[c], 9), -c))
    return {
        "cut": cut.date().isoformat(),
        "ceiling": best,
        "window_months": months,
        "errors": errors,
    }


def selection_table(*, cache: FeatureCache, site: str, arms: list[str]) -> pl.DataFrame:
    """Return every arm's sunny rows (as-of threshold) from all three periods, stacked."""
    edge = sunny_edge(cache=cache, site=site)
    parts = []
    for arm in arms:
        for period in ("tuning", "replicate", "scored"):
            frame = load(cache=cache, site=site, arm=arm, period=period)
            parts.append(
                frame.filter(pl.col("unit") >= edge).select(
                    *KEY, "abs_err", "origin_month", arm=pl.lit(arm)
                )
            )
    return pl.concat(parts)


def splice(*, frames: dict[int, pl.DataFrame], choices: dict[str, int]) -> pl.DataFrame:
    """Return rows where each origin month takes the arm of its cut's chosen ceiling."""
    parts = []
    for ceiling, frame in frames.items():
        months = [m for m, c in choices.items() if c == ceiling]
        if months:
            parts.append(frame.filter(pl.col("origin_month").dt.strftime("%Y-%m-%d").is_in(months)))
    return pl.concat(parts)


def interval(*, frame: pl.DataFrame, column: str) -> list[float]:
    """Return the mean of a column and its month-block interval at `LEVEL`."""
    monthly = frame.group_by("month").agg(total=pl.col(column).sum(), count=pl.len()).sort("month")
    sums = monthly["total"].to_numpy()[None, :]
    counts = monthly["count"].to_numpy().astype(np.float64)
    low, high = bootstrap_month_totals(month_sums=sums, month_counts=counts, level=LEVEL)
    return [float(sums.sum() / counts.sum()), float(low), float(high)]


def tail_row(*, frame: pl.DataFrame, suffix: str = "") -> dict:
    """Return coverage and pinball loss at each tail quantile on the rows of a frame."""
    out = {}
    actual = frame["actual"].to_numpy()
    for name in TAILS:
        q = frame[name].to_numpy()
        level = TAIL_LEVEL[name]
        error = actual - q
        out[name] = {
            "coverage": float(np.mean(actual <= q)),
            "pinball": float(np.mean(np.maximum(level * error, (level - 1.0) * error))),
        }
    return out


def contrast(
    *, arm: pl.DataFrame, control: pl.DataFrame, bg2: pl.DataFrame | None
) -> tuple[dict, pl.DataFrame]:
    """Return the sunny and all-rows scores of an arm against the control (same rows)."""
    arm_s = score.add_slices(frame=arm)
    control_s = score.add_slices(frame=control)
    if arm_s.height != control_s.height:
        msg = f"arm has {arm_s.height} rows, control {control_s.height}"
        raise ValueError(msg)
    joined = arm_s.join(
        control_s.select(
            *KEY,
            c_abs=pl.col("abs_err"),
            c_pin=pl.col("pin"),
            c_mean_abs=(pl.col("actual") - pl.col("mean")).abs(),
        ),
        on=KEY,
        how="inner",
        validate="1:1",
    ).with_columns(mean_abs=(pl.col("actual") - pl.col("mean")).abs())
    out: dict = {}
    for name in ("all", "sunny"):
        part = joined.filter(pl.col(name))
        diff = part.with_columns(
            d=pl.col("abs_err") - pl.col("c_abs"),
            dp=pl.col("pin") - pl.col("c_pin"),
            dm=pl.col("mean_abs") - pl.col("c_mean_abs"),
        )
        cell = {
            "rows": part.height,
            "mae": float(part["abs_err"].to_numpy().mean()),
            "control_mae": float(part["c_abs"].to_numpy().mean()),
            "pin": float(part["pin"].to_numpy().mean()),
            "control_pin": float(part["c_pin"].to_numpy().mean()),
            "mae_minus_control": interval(frame=diff, column="d"),
            "pin_minus_control": interval(frame=diff, column="dp"),
            "mean_abs_minus_control": interval(frame=diff, column="dm"),
            "median_minus_mean_mw": float((part["q50"] - part["mean"]).to_numpy().mean()),
            "tails": tail_row(frame=part),
            "control_tails": tail_row(frame=control_s.filter(pl.col(name))),
        }
        if "b_q50" in part.columns:
            gap = part.with_columns(
                g=pl.col("abs_err") - (pl.col("actual") - pl.col("b_q50")).abs()
            )
            cell["gap_to_bg2"] = interval(frame=gap, column="g")
            cell["bg2_mae"] = float((part["actual"] - part["b_q50"]).abs().to_numpy().mean())
        out[name] = cell
    return out, joined


def pooled(*, joined_by_site: dict[str, pl.DataFrame]) -> dict:
    """Return the pooled sunny contrast over sites, months totalled across sites."""
    parts = [
        j.filter(pl.col("sunny")).with_columns(d=pl.col("abs_err") - pl.col("c_abs"))
        for j in joined_by_site.values()
    ]
    stacked = pl.concat([p.select("month", "d") for p in parts])
    return {"sites": list(joined_by_site), "mae_minus_control": interval(frame=stacked, column="d")}


def main() -> None:
    """Write `work/asof.json`."""
    cache = common.load_cache()
    scored_cuts = [
        month_start(year=2025 if m >= 7 else 2026, month=m) for m in (*range(7, 13), *range(1, 7))
    ]
    replicate_cuts = [month_start(year=y, month=m) for y, m in sorted(common.REPLICATE_MONTHS)]
    result: dict = {"scored": {}, "replicate": {}, "fixed": {}, "sweep": {}}
    joined_fixed: dict[str, pl.DataFrame] = {}
    bg2 = pl.read_parquet(score.BG2_PATH)
    for site in FIXED_SITES:
        control = load(cache=cache, site=site, arm="c", period="scored", bg2=bg2)
        fixed_arm = load(cache=cache, site=site, arm="k60", period="scored", bg2=bg2)
        stats, joined = contrast(arm=fixed_arm, control=control, bg2=bg2)
        result["fixed"][site] = stats
        joined_fixed[site] = joined
    result["fixed"]["rest_pooled"] = pooled(joined_by_site={s: joined_fixed[s] for s in REST_SITES})
    for site in SWEEP_SITES:
        arms = [arm_of(ceiling=c) for c in CEILINGS]
        table = selection_table(cache=cache, site=site, arms=arms)
        for period, cuts, expanding in (
            ("scored", scored_cuts, False),
            ("replicate", replicate_cuts, True),
        ):
            choices = {}
            records = []
            for cut in cuts:
                record = choose(table=table, cut=cut, expanding=expanding)
                if expanding and record["window_months"] < MIN_WINDOW_MONTHS:
                    continue
                records.append(record)
                choices[cut.strftime("%Y-%m-%d")] = record["ceiling"]
            frame_bg2 = bg2 if period == "scored" else None
            frames = {
                c: load(cache=cache, site=site, arm=arm_of(ceiling=c), period=period, bg2=frame_bg2)
                for c in set(choices.values()) | {100, 60}
            }
            keep = [pl.col("origin_month").dt.strftime("%Y-%m-%d").is_in(list(choices))]
            control = frames[100].filter(*keep)
            spliced = splice(frames=frames, choices=choices)
            stats, _ = contrast(arm=spliced, control=control, bg2=None)
            fixed_frame = frames[60].filter(*keep)
            fixed_stats, _ = contrast(arm=fixed_frame, control=control, bg2=None)
            result[period][site] = {
                "choices": records,
                "as_of": stats,
                "fixed_60": fixed_stats,
            }
        # The sweep: every fixed ceiling against the control, per period (figure of the ceiling).
        for period in ("tuning", "replicate", "scored"):
            control = load(cache=cache, site=site, arm="c", period=period)
            row = {}
            for ceiling in CEILINGS[:-1]:
                frame = load(cache=cache, site=site, arm=arm_of(ceiling=ceiling), period=period)
                stats, _ = contrast(arm=frame, control=control, bg2=None)
                row[ceiling] = stats["sunny"]["mae_minus_control"]
            result["sweep"][f"{site}|{period}"] = row
        print(site, "done", flush=True)
    common.write_json_atomic(path=common.WORK_DIR / "asof.json", content=result)


if __name__ == "__main__":
    main()
