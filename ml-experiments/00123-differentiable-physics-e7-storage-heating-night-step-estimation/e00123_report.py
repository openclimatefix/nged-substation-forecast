"""Print every table and number that the experiment 123 page quotes, and save them as JSON.

Run `uv run python e00123_report.py headline` (winters 2021-22 to 2024-25, plus the design-winter
cross-checks R and E7-K) or `... all` (adds the selection winter, once PR 1397 has merged). The page
quotes only what this script prints; the charts read `numbers_<stage>.json`.
"""

import argparse
import json
from datetime import timedelta
from typing import Final
from zoneinfo import ZoneInfo

import e00123_common as common
import e00123_estimate as estimate
import e00123_fit as fit
import e00123_pool as pool
import numpy as np
import polars as pl

START_FRACTION: Final[float] = 0.5
PROFILE_BOOTSTRAP: Final[int] = 400
MIN_BST_NIGHTS: Final[int] = 10
PROFILE_START_MINUTES: Final[tuple[int, ...]] = tuple(range(22 * 60, 24 * 60, 30)) + tuple(
    range(0, 8 * 60, 30)
)
"""Start minutes (of the evening date, then the morning date) of the half-hours profiled."""


def stage_winters(*, stage: str) -> tuple[int, ...]:
    """Return the winters of a stage."""
    return fit.STAGES[stage]


def load_stage_nights(
    *, table: pl.DataFrame, infos: dict[str, common.SeriesInfo], winters: tuple[int, ...]
) -> dict[str, estimate.Nights]:
    """Return each fitted series' nights over the usable winters of the stage."""
    out = {}
    for site in common.FITTED_SERIES:
        usable = tuple(w for w in winters if w in fit._winters_of(table=table, site=site))
        out[site] = estimate.load_nights(
            table=table,
            site=site,
            capacity_mw=infos[site].effective_capacity_mw,
            winters=usable,
            summer_years=common.SUMMER_YEARS,
        )
    return out


def line_and_raw(*, nights: estimate.Nights) -> dict[str, float]:
    """Return D's point fit read at the cold nights' mean `c`, and the raw cold-night contrast."""
    point = estimate.weighted_fit(
        nights=nights,
        winter_weights=np.ones((1, len(nights.d))),
        summer_weights=np.ones((1, len(nights.summer_d))),
    )
    raw = estimate.raw_cold_contrast(nights=nights)
    line = float(point.flat[0] + point.slope[0] * raw["mean_cold"]) if raw["nights"] else np.nan
    return {
        "cold_nights": raw["nights"],
        "mean_cold_c": raw["mean_cold"],
        "raw_contrast": raw["contrast"],
        "d_line_at_mean_cold": line,
        "gap_raw_minus_line": raw["contrast"] - line if raw["nights"] else np.nan,
        "d_summer": float(nights.summer_d.mean()),
        "bounded_slope": float(max(point.slope[0], 0.0)),
        "unbounded_slope": float(point.slope[0]),
    }


def residual_cross_checks(
    *, infos: dict[str, common.SeriesInfo], tags: tuple[str, ...]
) -> list[dict]:
    """Fit D to experiment 114's stored residuals (R, and E7-K) at the six decision series."""
    rows = []
    for site in common.DECISION_SERIES:
        info = infos[site]
        own = pl.read_parquet(common.WORK_DIR / "night_tables.parquet").filter(
            pl.col("site") == site
        )
        for arm in fit.ARMS:
            try:
                table = fit.residual_table(info=info, arm=arm, tags=tags)
            except FileNotFoundError, pl.exceptions.ComputeError, ValueError:
                continue
            usable = tuple(
                w
                for w in (
                    (*common.DESIGN_WINTERS, common.SELECTION_WINTER)
                    if "scored" in tags
                    else common.DESIGN_WINTERS
                )
                if w in fit._winters_of(table=table, site=site)
            )
            if not usable:
                continue
            nights = estimate.load_nights(
                table=table,
                site=site,
                capacity_mw=info.effective_capacity_mw,
                winters=usable,
                summer_years=fit.R_SUMMER_YEARS,
            )
            if len(nights.summer_d) == 0:
                continue
            result = estimate.fit_series(nights=nights, rng=np.random.default_rng(fit.SEED))
            reference = estimate.fit_series(
                nights=estimate.load_nights(
                    table=own,
                    site=site,
                    capacity_mw=info.effective_capacity_mw,
                    winters=usable,
                    summer_years=common.SUMMER_YEARS,
                ),
                rng=np.random.default_rng(fit.SEED),
            )
            rows.append(
                {
                    "site": site,
                    "arm": arm,
                    "winters": ",".join(str(w) for w in usable),
                    "step": result.step.value,
                    "step_low": result.step.low,
                    "step_high": result.step.high,
                    "slope": result.slope.value,
                    "d_step_same_winters": reference.step.value,
                    "ratio_to_d": result.step.value / reference.step.value,
                }
            )
    return rows


def leave_one_series_out(*, steps: pl.DataFrame) -> list[dict]:
    """Return the six-series pooled mean with each series left out in turn."""
    rng = np.random.default_rng(fit.SEED)
    out = []
    own = steps.filter(pl.col("site").is_in(list(common.DECISION_SERIES)))
    full = pool.pool(
        values=own["step"].to_numpy(), standard_errors=own["step_sd"].to_numpy(), rng=rng
    )
    for site in common.DECISION_SERIES:
        rest = own.filter(pl.col("site") != site)
        pooled = pool.pool(
            values=rest["step"].to_numpy(), standard_errors=rest["step_sd"].to_numpy(), rng=rng
        )
        out.append(
            {
                "left_out": site,
                "mean": pooled.mean,
                "k": pooled.k,
                "inside_full_hk": bool(full.low_hk <= pooled.mean <= full.high_hk),
            }
        )
    return out


def night_end(*, winters: tuple[int, ...]) -> common.date:
    """Return the first evening a stage does not read: the start of the next unread winter."""
    if common.SELECTION_WINTER in winters:
        return common.date(2026, 6, 29)
    return common.date(common.SELECTION_WINTER, 10, 26)


def timing_rows(*, table: pl.DataFrame, winters: tuple[int, ...]) -> list[dict]:
    """Return cold BST night counts per series, and `delta` where there are 10 or more."""
    rows = []
    rng = np.random.default_rng(fit.SEED)
    for site in common.FITTED_SERIES:
        own = table.filter(
            (pl.col("site") == site)
            & pl.col("season").is_in(["bst_other", "summer"])
            & ~pl.col("excluded")
            & (pl.col("daily_mean_c") < common.COLD_BST_NIGHT_C)
            & pl.col("d").is_not_null()
            & pl.col("d_utc").is_not_null()
            & (pl.col("evening") < night_end(winters=winters))
        )
        count = own.height
        row: dict[str, object] = {"site": site, "cold_bst_nights": count}
        if count >= MIN_BST_NIGHTS:
            gap = (own["d"] - own["d_utc"]).to_numpy()
            days = (own["evening"] - common.date(2019, 1, 1)).dt.total_days().to_numpy()
            block = days // common.BLOCK_DAYS
            labels = np.unique(block)
            members = {label: gap[block == label] for label in labels}
            draws = np.array(
                [
                    np.concatenate(
                        [members[label] for label in rng.choice(labels, len(labels))]
                    ).mean()
                    for _ in range(1000)
                ]
            )
            low, high = np.quantile(draws, [0.05, 0.95])
            row |= {
                "delta": float(gap.mean()),
                "low": float(low),
                "high": float(high),
                "label": "local" if low > 0 else "UTC" if high < 0 else "unresolved",
            }
        else:
            row["label"] = "cannot tell"
        rows.append(row)
    return rows


TARGET_SLOT: Final[int] = 9
"""Index in `PROFILE_START_MINUTES` of the half-hour starting 02:30."""
FIRST_NIGHT_SLOT: Final[int] = 4
"""Index of the half-hour starting 00:00, the first candidate for the start of the step."""


def start_slot(*, cold_mean: np.ndarray, summer_mean: np.ndarray) -> float:
    """Return the index of the first half-hour whose excess exceeds half the excess at 02:30."""
    excess = cold_mean - summer_mean
    if excess[TARGET_SLOT] <= 0:
        return float("nan")
    above = np.flatnonzero(excess[FIRST_NIGHT_SLOT:] > START_FRACTION * excess[TARGET_SLOT])
    return float(above[0] + FIRST_NIGHT_SLOT) if above.size else float("nan")


def start_times(
    *, infos: dict[str, common.SeriesInfo], table: pl.DataFrame, winters: tuple[int, ...]
) -> list[dict]:
    """Return each series' start time and its mean profile on cold GMT nights minus the summer mean.

    The profile is each half-hour's mean demand minus the night's baseline, in points. The start is
    the first half-hour (from 00:00) whose cold-minus-summer mean exceeds half of its 02:30 value.
    """
    ids = [infos[s].time_series_id for s in common.FITTED_SERIES]
    power = common.load_power(time_series_ids=ids)
    out = []
    rng = np.random.default_rng(fit.SEED)
    for site in common.FITTED_SERIES:
        info = infos[site]
        own_power = power.filter(pl.col("time_series_id") == info.time_series_id)
        lookup = dict(zip(own_power["time"].to_list(), own_power["power"].to_list(), strict=True))
        scale = 100.0 / info.effective_capacity_mw
        nights = table.filter(
            (pl.col("site") == site)
            & ~pl.col("excluded")
            & pl.col("d").is_not_null()
            & (
                ((pl.col("season") == "gmt") & pl.col("winter").is_in(list(winters)))
                | (pl.col("season") == "summer")
            )
        )
        profiles, kinds = [], []
        for row in nights.iter_rows(named=True):
            evening = row["evening"]
            base = common.datetime.combine(evening, common.time.min).replace(tzinfo=common.UTC)
            values = []
            for minute in PROFILE_START_MINUTES:
                start = base + timedelta(minutes=minute if minute >= 22 * 60 else 24 * 60 + minute)
                values.append(lookup.get(start + timedelta(minutes=30)))
            present = [v for v in values if v is not None]
            if len(present) != len(values):
                continue
            level = sum(present[:4]) / 4.0
            profiles.append([(v - level) * scale for v in present])
            cold = row["season"] == "gmt" and row["cold_c"] is not None and row["cold_c"] >= 7.5
            kinds.append("cold" if cold else "summer" if row["season"] == "summer" else "other")
        array = np.array(profiles)
        kind = np.array(kinds)
        if (kind == "cold").sum() < MIN_BST_NIGHTS or (kind == "summer").sum() < MIN_BST_NIGHTS:
            out.append({"site": site, "start": "cannot tell"})
            continue
        cold_rows = array[kind == "cold"]
        summer_rows = array[kind == "summer"]
        point = start_slot(cold_mean=cold_rows.mean(axis=0), summer_mean=summer_rows.mean(axis=0))
        draws = []
        for _ in range(PROFILE_BOOTSTRAP):
            a = cold_rows[rng.integers(0, len(cold_rows), len(cold_rows))].mean(axis=0)
            b = summer_rows[rng.integers(0, len(summer_rows), len(summer_rows))].mean(axis=0)
            draws.append(start_slot(cold_mean=a, summer_mean=b))
        valid = [d for d in draws if np.isfinite(d)]
        minutes = lambda slot: PROFILE_START_MINUTES[int(slot)] % (24 * 60)  # noqa: E731
        out.append(
            {
                "site": site,
                "start_half_hour_start_minute": minutes(point) if np.isfinite(point) else None,
                "low_half_hours": float(np.quantile(valid, 0.05) - point) if valid else None,
                "high_half_hours": float(np.quantile(valid, 0.95) - point) if valid else None,
                "profile_cold_minus_summer": (
                    cold_rows.mean(axis=0) - summer_rows.mean(axis=0)
                ).tolist(),
                "cold_nights": len(cold_rows),
            }
        )
    return out


JUMP_FIRST_SLOT: Final[int] = 3
"""The jump search starts at the transition into the half-hour starting 23:30."""
JUMP_LAST_SLOT: Final[int] = 14
"""... and ends at the transition into the half-hour starting 03:00."""


def local_profile(*, lookup: dict, evening: common.date, scale: float) -> list[float] | None:
    """Return one night's local-clock profile in points, or None if a half-hour is missing."""
    midnight = common.datetime.combine(evening, common.time.min).replace(
        tzinfo=ZoneInfo(common.LOCAL_TIME_ZONE)
    )
    values = []
    for minute in PROFILE_START_MINUTES:
        offset = minute if minute >= 22 * 60 else 24 * 60 + minute
        start = (midnight + timedelta(minutes=offset)).astimezone(common.UTC)
        values.append(lookup.get(start + timedelta(minutes=30)))
    present = [v for v in values if v is not None]
    return [v * scale for v in present] if len(present) == len(values) else None


def jump_location(*, profiles: np.ndarray) -> int:
    """Return the index of the half-hour entered by the largest mean increase of demand."""
    increments = np.diff(profiles.mean(axis=0))
    window = increments[JUMP_FIRST_SLOT - 1 : JUMP_LAST_SLOT]
    return int(np.argmax(window)) + JUMP_FIRST_SLOT


def bst_jump_rows(
    *, infos: dict[str, common.SeriesInfo], table: pl.DataFrame, winters: tuple[int, ...]
) -> list[dict]:
    """Post hoc: where is the largest jump on cold BST nights, against cold GMT nights?

    Both are read on the local clock. A step fixed to the local clock jumps at the same local
    half-hour on both kinds of night; a step fixed to UTC jumps two half-hours later on BST nights.
    """
    ids = [infos[s].time_series_id for s in common.FITTED_SERIES]
    power = common.load_power(time_series_ids=ids)
    rows = []
    for site in common.FITTED_SERIES:
        info = infos[site]
        own = power.filter(pl.col("time_series_id") == info.time_series_id)
        lookup = dict(zip(own["time"].to_list(), own["power"].to_list(), strict=True))
        scale = 100.0 / info.effective_capacity_mw
        nights = table.filter(
            (pl.col("site") == site)
            & ~pl.col("excluded")
            & (pl.col("daily_mean_c") < common.COLD_BST_NIGHT_C)
            & pl.col("season").is_in(["gmt", "bst_other"])
            & (pl.col("winter").is_null() | (pl.col("winter") <= max(winters)))
            & (pl.col("evening") < night_end(winters=winters))
        )
        groups: dict[str, list[list[float]]] = {"gmt": [], "bst_other": []}
        for row in nights.iter_rows(named=True):
            profile = local_profile(lookup=lookup, evening=row["evening"], scale=scale)
            if profile is not None:
                groups[row["season"]].append(profile)
        if len(groups["bst_other"]) < MIN_BST_NIGHTS or len(groups["gmt"]) < MIN_BST_NIGHTS:
            rows.append({"site": site, "label": "cannot tell"})
            continue
        gmt = jump_location(profiles=np.array(groups["gmt"]))
        bst = jump_location(profiles=np.array(groups["bst_other"]))
        rows.append(
            {
                "site": site,
                "gmt_jump_start_minute": PROFILE_START_MINUTES[gmt] % (24 * 60),
                "bst_jump_start_minute": PROFILE_START_MINUTES[bst] % (24 * 60),
                "shift_half_hours": bst - gmt,
                "bst_nights": len(groups["bst_other"]),
                "gmt_nights": len(groups["gmt"]),
            }
        )
    return rows


def hand_filtered_rows(
    *, infos: dict[str, common.SeriesInfo], winters: tuple[int, ...]
) -> list[dict]:
    """Sensitivity: refit the six decision series without experiment 103's switching events.

    Readings inside the hand-filtered switching events (the `switching_mask` of the DP feature
    cache) are dropped before the nights are built, on top of the rule-based flag.
    """
    from studies.dp_features import default_input_paths, load_or_build_feature_cache

    cache = load_or_build_feature_cache(
        paths=default_input_paths(),
        cache_dir=common.ML_EXPERIMENTS_DATA_DIR
        / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
        / "cache",
    )
    power = common.load_power(
        time_series_ids=[infos[s].time_series_id for s in common.DECISION_SERIES]
    )
    rows = []
    for site in common.DECISION_SERIES:
        info = infos[site]
        masked = pl.Series(cache.end_times[cache.sites[site].switching_mask]).dt.replace_time_zone(
            "UTC"
        )
        own = power.filter(pl.col("time_series_id") == info.time_series_id).select("time", "power")
        kept = own.filter(~pl.col("time").is_in(masked.implode()))
        table = common.build_night_table(info=info, power=kept)
        usable = tuple(w for w in winters if w in fit._winters_of(table=table, site=site))
        result = fit.fit_one(table=table, info=info, winters=usable, seed=fit.SEED)
        rows.append(
            {
                "site": site,
                "readings_dropped": own.height - kept.height,
                "step": result.step.value,
                "step_low": result.step.low,
                "step_high": result.step.high,
                "slope": result.slope.value,
            }
        )
    return rows


def raw_cold_step_rows(*, infos: dict[str, common.SeriesInfo], table: pl.DataFrame) -> list[dict]:
    """Return the raw half-hour-ending-01:00 step on cold GMT nights, in MW and in points.

    Cold nights are GMT nights of the design winters whose daily mean is at most 3 C. The step is
    the half-hour ending at 01:00 minus the mean of the 22:00 to 23:30 half-hours, in the series'
    units; it reads the stored cleaned power directly and needs no fit.
    """
    ids = [infos[s].time_series_id for s in common.SELECTED_SERIES]
    power = common.load_power(time_series_ids=ids)
    rows = []
    for site in common.SELECTED_SERIES:
        info = infos[site]
        own = power.filter(pl.col("time_series_id") == info.time_series_id)
        lookup = dict(zip(own["time"].to_list(), own["power"].to_list(), strict=True))
        nights = table.filter(
            (pl.col("site") == site)
            & (pl.col("season") == "gmt")
            & ~pl.col("excluded")
            & pl.col("winter").is_in(list(common.DESIGN_WINTERS))
            & (pl.col("daily_mean_c") <= 3.0)
        )["evening"].to_list()
        steps = []
        for evening in nights:
            base = common.datetime.combine(evening, common.time.min).replace(tzinfo=common.UTC)
            baseline = [
                lookup.get(base + timedelta(hours=22, minutes=30 * k + 30)) for k in range(4)
            ]
            after = lookup.get(base + timedelta(hours=25))
            if after is not None and all(v is not None for v in baseline):
                steps.append(after - sum(v for v in baseline if v is not None) / 4.0)
        mean = float(np.mean(steps))
        rows.append(
            {
                "site": site,
                "cold_nights": len(steps),
                "step_mw": mean,
                "step_points": mean * 100.0 / info.effective_capacity_mw,
                "effective_capacity_mw": info.effective_capacity_mw,
            }
        )
    return rows


def main() -> None:
    """Print the report for one stage and save its numbers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["headline", "all"])
    stage = parser.parse_args().stage
    fit.require(stage=stage)
    winters = stage_winters(stage=stage)
    infos = common.series_infos()
    table = pl.read_parquet(common.WORK_DIR / "night_tables.parquet")
    label = f"{stage}: " + ", ".join(str(w) for w in winters)
    fits = pl.read_parquet(common.WORK_DIR / f"fit_{stage}.parquet")
    headline = fits.filter(pl.col("winter_set") == label)
    nights = load_stage_nights(table=table, infos=infos, winters=winters)
    raw = pl.DataFrame(
        [{"site": s, **line_and_raw(nights=nights[s])} for s in common.FITTED_SERIES]
    )
    flags = fit.cold_snap_nights(table=table)
    snap = [
        {"site": s, "winter": w, **values}
        for s in common.DECISION_SERIES
        for w, values in fit.cold_snap_summary(table=table, nights=nights[s], flags=flags).items()
    ]
    tags = ("tuning", "replicate", *(("scored",) if stage == "all" else ()))
    cross = residual_cross_checks(infos=infos, tags=tags)
    numbers = {
        "stage": stage,
        "winters": list(winters),
        "unpooled": headline.to_dicts(),
        "per_winter": fits.filter(pl.col("winter_set").str.starts_with("single")).to_dicts(),
        "raw": raw.to_dicts(),
        "cold_snap": snap,
        "cross_checks": cross,
        "pooled": json.loads((common.WORK_DIR / f"pooled_{stage}.json").read_text()),
        "leave_one_out": leave_one_series_out(steps=headline),
        "timing": timing_rows(table=table, winters=winters),
        "bst_jump": bst_jump_rows(infos=infos, table=table, winters=winters),
        "hand_filtered": hand_filtered_rows(infos=infos, winters=winters),
        "raw_cold_step": raw_cold_step_rows(infos=infos, table=table),
        "start_times": start_times(infos=infos, table=table, winters=winters),
        "gate": json.loads((common.WORK_DIR / "gate.json").read_text()),
    }
    (common.WORK_DIR / f"numbers_{stage}.json").write_text(
        json.dumps(numbers, indent=1, default=str)
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=2, tbl_width_chars=250):
        for name in (
            "unpooled",
            "raw",
            "cold_snap",
            "cross_checks",
            "leave_one_out",
            "timing",
            "bst_jump",
            "hand_filtered",
            "raw_cold_step",
        ):
            print(f"\n== {name} ==")
            print(pl.DataFrame(numbers[name]))
        print("\n== pooled ==")
        print(pl.DataFrame(numbers["pooled"]).drop("winter_set"))
        print("\n== per winter (single-winter fits, step in points) ==")
        print(
            pl.DataFrame(numbers["per_winter"])
            .select("site", "winter_set", "step", "step_low", "step_high", "n_nights")
            .sort("site", "winter_set")
        )
        print("\n== start times ==")
        print(
            pl.DataFrame(
                [
                    {k: v for k, v in r.items() if k != "profile_cold_minus_summer"}
                    for r in numbers["start_times"]
                ]
            )
        )


if __name__ == "__main__":
    main()
