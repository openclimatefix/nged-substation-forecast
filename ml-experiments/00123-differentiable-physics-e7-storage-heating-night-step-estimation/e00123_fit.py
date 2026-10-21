"""The real fits of experiment 123, run in stages that follow the plan's order.

Stages (each refuses to run before its precondition):

- `design`: winters 2023-24 and 2024-25 only (the design and replicate winters).
- `headline`: winters 2021-22 to 2024-25. Needs `gate.json` (the gate has run).
- `all`: winters 2021-22 to 2025-26, the selection winter included. Needs `gate.json` and the
  key `pr1397_merge_sha` in `decisions.json`.

For each stage it writes `fit_<stage>.parquet` (one row per series and winter set), the per-winter
steps, the cross-checks R and E7-K, the cold-snap raw contrast, and `pooled_<stage>.json`.
"""

import argparse
import json
from typing import Final

import e00123_common as common
import e00123_estimate as estimate
import e00123_pool as pool
import numpy as np
import polars as pl

STAGES: Final[dict[str, tuple[int, ...]]] = {
    "design": common.DESIGN_WINTERS,
    "headline": common.HEADLINE_WINTERS,
    "all": common.WINTERS,
}
POOLED_SETS: Final[dict[str, tuple[str, ...]]] = {
    "six decision series (headline)": common.DECISION_SERIES,
    "D5, D10, D16 (selected for showing the step)": common.SELECTED_SERIES,
    "D9, D11, D14 (not selected)": common.UNSELECTED_DECISION_SERIES,
    "E7-X (eight further Primaries)": common.EXTENSION_SERIES,
    "all 14 series": common.FITTED_SERIES,
}
ARMS: Final[tuple[str, ...]] = ("b0", "b0_k", "t_all_k")
"""Experiment 114's stored predictions used for R (`b0`) and E7-K (`b0_k`, `t_all_k`)."""
R_SUMMER_YEARS: Final[tuple[int, ...]] = (2023, 2024, 2025)
COLD_SNAP_SHARE: Final[float] = 0.10
MIN_COLD_SNAP_NIGHTS: Final[int] = 5
SEED: Final[int] = 2026


def require(*, stage: str) -> None:
    """Raise unless the stage's precondition holds.

    Args:
        stage: `design`, `headline`, or `all`.

    Raises:
        PermissionError: If the gate has not run, or the scored winter is still locked.
    """
    if stage != "design" and not (common.WORK_DIR / "gate.json").exists():
        msg = "the gate has not run: winters 2021-22 and 2022-23 stay unopened until it has"
        raise PermissionError(msg)
    if stage == "all" and not common.scored_winter_unlocked():
        msg = "winter 2025-26 stays locked until decisions.json pins PR 1397's merge commit"
        raise PermissionError(msg)


def residual_table(*, info: common.SeriesInfo, arm: str, tags: tuple[str, ...]) -> pl.DataFrame:
    """Return a night table whose `d` is the stored B0 (or filtered) residual difference.

    Args:
        info: The series.
        arm: One of `ARMS`.
        tags: Prediction-set prefixes to read (`tuning`, `replicate`, and `scored` once unlocked).

    Returns:
        A night table in the layout of `build_night_table`, built from the residual in MW.
    """
    if "scored" in tags and not common.scored_winter_unlocked():
        msg = "experiment 114's scored-year residuals are locked until PR 1397 has merged"
        raise PermissionError(msg)
    frames = [
        pl.read_parquet(common.EXPERIMENT_114_DIR / f"{tag}_{arm}" / "predictions.parquet").filter(
            pl.col("time_series_id") == info.time_series_id
        )
        for tag in tags
    ]
    scale = info.effective_capacity_mw / 100.0
    residual = (
        pl.concat(frames)
        .filter(pl.col("valid_time") < common.HOLDOUT_START)
        .select(
            time=pl.col("valid_time"),
            power=(pl.col("actual").cast(pl.Float64) - pl.col("q50").cast(pl.Float64)) * scale,
        )
        .unique(subset="time", keep="first")
        .sort("time")
    )
    return common.build_night_table(info=info, power=residual)


def cold_snap_nights(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return each evening of the six decision series flagged cold-snap, by experiment 122's rule.

    A day is cold-snap if it lies in the coldest 10% of days of its July-to-June year by the daily
    mean ERA5 temperature averaged over the six decision series. A night takes the flag of the
    morning's day (the day that contains its window rows).

    Args:
        table: `night_tables.parquet`.

    Returns:
        Columns `evening` and `cold_snap`.
    """
    mean = (
        table.filter(pl.col("site").is_in(list(common.DECISION_SERIES)))
        .group_by("evening")
        .agg(daily_mean_c=pl.col("daily_mean_c").mean())
        .drop_nulls()
        .with_columns(
            winter_july=pl.when(pl.col("evening").dt.month() >= 7)
            .then(pl.col("evening").dt.year())
            .otherwise(pl.col("evening").dt.year() - 1)
        )
    )
    return mean.with_columns(
        cold_snap=pl.col("daily_mean_c")
        <= pl.col("daily_mean_c").quantile(COLD_SNAP_SHARE).over("winter_july")
    ).select("evening", "cold_snap")


def cold_snap_summary(*, table: pl.DataFrame, nights: estimate.Nights, flags: pl.DataFrame) -> dict:
    """Return the raw cold-snap contrast per winter, for November to March and November to February.

    Args:
        table: `night_tables.parquet` (to read the evening dates).
        nights: One series' nights.
        flags: The output of `cold_snap_nights`.

    Returns:
        Per winter: `nights`, `contrast_nov_mar`, `contrast_nov_feb`, and `note`.
    """
    del table
    flag_map = dict(zip(flags["evening"].to_list(), flags["cold_snap"].to_list(), strict=True))
    result: dict[int, dict[str, float | str]] = {}
    evenings = nights.evening.astype("datetime64[D]").tolist()
    months = np.array([day.month for day in evenings])
    snap = np.array([bool(flag_map.get(day, False)) for day in evenings])
    d_summer = float(nights.summer_d.mean())
    for winter in nights.winters:
        mask = (nights.winter == winter) & snap
        feb = mask & (months != 3)
        result[winter] = {
            "nights": float(mask.sum()),
            "contrast_nov_mar": float(nights.d[mask].mean() - d_summer)
            if mask.any()
            else float("nan"),
            "contrast_nov_feb": float(nights.d[feb].mean() - d_summer)
            if feb.any()
            else float("nan"),
            "note": "too few days" if mask.sum() < MIN_COLD_SNAP_NIGHTS else "ok",
        }
    return result


def fit_one(
    *, table: pl.DataFrame, info: common.SeriesInfo, winters: tuple[int, ...], seed: int
) -> estimate.SeriesResult:
    """Fit D for one series over the given winters."""
    nights = estimate.load_nights(
        table=table,
        site=info.site,
        capacity_mw=info.effective_capacity_mw,
        winters=winters,
        summer_years=common.SUMMER_YEARS,
    )
    return estimate.fit_series(nights=nights, rng=np.random.default_rng(seed))


def result_row(*, label: str, info: common.SeriesInfo, result: estimate.SeriesResult) -> dict:
    """Return one table row for a fit, with the step also in MW."""
    scale = info.effective_capacity_mw / 100.0
    return {
        "site": info.site,
        "winter_set": label,
        "winters": ",".join(str(w) for w in result.winters),
        "n_winters": len(result.winters),
        "n_nights": result.n_nights,
        "slope": result.slope.value,
        "slope_low": result.slope.low,
        "slope_high": result.slope.high,
        "flat": result.flat.value,
        "flat_low": result.flat.low,
        "flat_high": result.flat.high,
        "step": result.step.value,
        "step_low": result.step.low,
        "step_high": result.step.high,
        "step_sd": result.step.sd,
        "step_mw": result.step.value * scale,
        "detectable_interval": result.step.low > 0.0,
    }


def pooled_sets(*, rows: pl.DataFrame, label: str) -> list[dict]:
    """Pool each of the five sets from the rows of one winter set."""
    rng = np.random.default_rng(SEED)
    out = []
    own = rows.filter(pl.col("winter_set") == label)
    for name, sites in POOLED_SETS.items():
        picked = own.filter(pl.col("site").is_in(list(sites)))
        pooled = pool.pool(
            values=picked["step"].to_numpy(), standard_errors=picked["step_sd"].to_numpy(), rng=rng
        )
        out.append({"set": name, "winter_set": label, **pooled.__dict__})
    return out


def main() -> None:
    """Run one stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=sorted(STAGES))
    stage = parser.parse_args().stage
    require(stage=stage)
    common.cap_threads()
    infos = common.series_infos()
    table = pl.read_parquet(common.WORK_DIR / "night_tables.parquet")
    winters = STAGES[stage]
    rows: list[dict] = []
    for index, site in enumerate(common.FITTED_SERIES):
        info = infos[site]
        label = f"{stage}: " + ", ".join(str(w) for w in winters)
        usable = tuple(w for w in winters if w in _winters_of(table=table, site=site))
        rows.append(
            result_row(
                label=label,
                info=info,
                result=fit_one(table=table, info=info, winters=usable, seed=SEED + index),
            )
        )
        rows.extend(
            result_row(
                label=f"single winter {winter}",
                info=info,
                result=fit_one(table=table, info=info, winters=(winter,), seed=SEED + index),
            )
            for winter in usable
        )
    frame = pl.DataFrame(rows)
    frame.write_parquet(common.WORK_DIR / f"fit_{stage}.parquet")
    pooled = pooled_sets(rows=frame, label=f"{stage}: " + ", ".join(str(w) for w in winters))
    (common.WORK_DIR / f"pooled_{stage}.json").write_text(json.dumps(pooled, indent=2))
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=2, tbl_width_chars=250):
        print(frame.filter(pl.col("winter_set").str.starts_with(stage)).drop("winters"))
        print(pl.DataFrame(pooled).drop("winter_set"))


def _winters_of(*, table: pl.DataFrame, site: str) -> set[int]:
    """Return the winters in which a series has usable nights."""
    usable = table.filter(
        (pl.col("site") == site) & (pl.col("season") == "gmt") & pl.col("d").is_not_null()
    )
    return {int(w) for w in usable["winter"].unique().to_list() if w is not None}


if __name__ == "__main__":
    main()
