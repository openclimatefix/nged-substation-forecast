"""Experiment 215: scan the real series for steps in solar capacity and match them to the register.

For each Primary, fits experiment 207's step path (total-variation weight 3, with 0.3 and 30 as
sensitivity) to the real April to August rows from the Primary's first usable instant to
2026-06-30, with no injection, and lists the steps above 0.5 MW. Increases dated before the
register's last commissioning date (2021-03-24) are compared with the share-weighted feed-in tariff
capacity commissioned within 60 days of the step's month boundary. The gate refits at the cut
of experiment 207 and compares the base level path at D10, farm A, weight 3, with experiment 207's
stored path.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00215-*/e00215_scan.py
"""

import argparse
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from typing import Final

import e00215_common as common
import e00215_fit as fit
import numpy as np
import polars as pl
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam import GamSpec, prepare_fit
from studies.dp_pinned_capacity import pinned_capacity, read_vintages
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SCAN_CUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""The scan uses every row in the cache, up to 2026-06-30."""
SCAN_SEASON_YEARS: Final[range] = range(2019, 2027)
WEIGHTS: Final[tuple[float, ...]] = (3.0, 0.3, 30.0)
"""Total-variation weights: the planned one first, then the sensitivity checks."""
STEP_MW: Final[float] = 0.5
CLUSTER_MW: Final[float] = 0.5
MATCH_DAYS: Final[int] = 60
REGISTER_DIR: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00045-pvdis-4-weekly-capacity-asymmetric-l1" / "outside"
)
REGISTER_END: Final[date] = date(2021, 3, 24)
"""The latest commissioning date in the feed-in tariff register."""
REGISTER_START: Final[date] = date(2019, 9, 1)
"""The first month of the cache, so the first boundary with data on both sides."""
GATE_SITE: Final[str] = "D10"
GATE_FARM: Final[str] = "A"
GATE_TOLERANCE: Final[float] = 1e-6
SITES: Final[tuple[str, ...]] = common.ALL_SITES


def month_date(*, month_index: int) -> date:
    """Return the first day of a month given as months since 1970-01."""
    return date(1970 + month_index // 12, month_index % 12 + 1, 1)


def fit_path(
    *,
    cache: FeatureCache,
    site: str,
    cut: datetime,
    shares: tuple[float, ...],
    masked_by: np.ndarray | None = None,
) -> tuple[fit.MonthDesign, dict[float, np.ndarray], int]:
    """Fit the step path of a Primary on real summer rows; return the months and levels (points).

    Args:
        cache: The cache.
        site: The Primary.
        cut: The cut; no training row ends after it.
        shares: The total-variation weights to fit.
        masked_by: An optional farm-power series (points) removed from `y` to fix the row set, as
            experiment 207 did, used only by the gate.

    Returns:
        The month design, the level path in points by weight, and the number of training rows.
    """
    features = cache.sites[site]
    masked = features if masked_by is None else replace(features, y=features.y - masked_by)
    spec = GamSpec(ridge=common.RIDGE, capacity="fitted")
    seasons = SCAN_SEASON_YEARS if cut >= SCAN_CUT else common.SEASON_YEARS
    layout, index = prepare_fit(
        cache=replace(cache, sites={**cache.sites, site: masked}),
        site=site,
        spec=spec,
        cut=cut,
        window=common.TRAINING_WINDOW,
        season_years=seasons,
    )
    index = fit.summer_rows(cache=cache, index=index)
    targets = features.y[index][None]
    design = fit.month_design(cache=cache, index=index)
    gram, rhs = fit.accumulate_month_gram(
        cache=cache, site=site, layout=layout, index=index, design=design, targets=targets
    )
    step_t = fit.step_transform(layout=layout, design=design)
    out = {}
    for share in shares:
        matrix, cost, constrained = fit.step_penalty(
            layout=layout,
            design=design,
            n_rows=len(index),
            gram=gram,
            share=share,
            transform=step_t,
        )
        theta = fit.solve(
            gram=step_t.T @ gram @ step_t,
            rhs=rhs @ step_t,
            penalty=matrix,
            cost=cost,
            constrained=constrained,
        )
        out[share] = fit.levels_from(arm="step", theta=theta, layout=layout, design=design)[0]
    return design, out, len(index)


def steps_of(
    *, design: fit.MonthDesign, levels_mw: np.ndarray, site: str, weight: float
) -> list[dict]:
    """Return the steps above `STEP_MW`: summer month-to-month changes and August-to-April nets."""
    months = design.first_month + np.arange(design.n_months)
    calendar_month = months % 12 + 1
    rows = []
    for k in range(1, len(months)):
        change = float(levels_mw[k] - levels_mw[k - 1])
        both_summer = calendar_month[k] in (5, 6, 7, 8) and calendar_month[k - 1] in (4, 5, 6, 7)
        if both_summer and abs(change) > STEP_MW:
            rows.append(
                {
                    "site": site,
                    "weight": weight,
                    "kind": "month",
                    "month_index": int(months[k]),
                    "date": month_date(month_index=int(months[k])),
                    "change_mw": change,
                }
            )
    for k in range(len(months)):
        if calendar_month[k] != 8:
            continue
        april = np.flatnonzero(months == months[k] + 8)
        if len(april) == 0:
            continue
        change = float(levels_mw[april[0]] - levels_mw[k])
        if abs(change) > STEP_MW:
            rows.append(
                {
                    "site": site,
                    "weight": weight,
                    "kind": "winter_gap",
                    "month_index": int(months[k]),
                    "date": month_date(month_index=int(months[k])),
                    "change_mw": change,
                }
            )
    return rows


def register_weights() -> pl.DataFrame:
    """Return feed-in tariff installations with their Primary and share-weighted capacity in MW."""
    installs = pl.read_parquet(REGISTER_DIR / "fit_solar_pv_installations.parquet").select(
        "postcode", "commissioning_date", "installed_capacity_kw"
    )
    shares = pl.read_parquet(REGISTER_DIR / "fit_district_polygon_shares.parquet")
    return installs.join(
        shares.select("district", "time_series_id", "share"),
        left_on="postcode",
        right_on="district",
        how="inner",
    ).with_columns(weighted_mw=pl.col("installed_capacity_kw") * pl.col("share") / 1000.0)


def window_mw(*, register: pl.DataFrame, series_id: int, centre: date) -> float:
    """Return the share-weighted register capacity commissioned within `MATCH_DAYS` of a date."""
    low, high = centre - timedelta(days=MATCH_DAYS), centre + timedelta(days=MATCH_DAYS)
    part = register.filter(
        (pl.col("time_series_id") == series_id)
        & (pl.col("commissioning_date") >= low)
        & (pl.col("commissioning_date") <= high)
    )
    return float(part["weighted_mw"].to_numpy().sum())


def base_rate(*, register: pl.DataFrame, series_id: int) -> tuple[float, int]:
    """Return the share of month boundaries from 2019-09 to 2021-03 whose window holds a cluster."""
    boundaries = [date(2019 + (8 + i) // 12, (8 + i) % 12 + 1, 1) for i in range(19)]
    hits = [
        window_mw(register=register, series_id=series_id, centre=b) >= CLUSTER_MW
        for b in boundaries
    ]
    return float(np.mean(hits)), len(boundaries)


def run_gate(*, cache: FeatureCache) -> dict:
    """Refit experiment 207's base path at D10, farm A, weight 3; compare it with the stored one."""
    features = cache.sites[GATE_SITE]
    pinned_points = pinned_capacity(
        vintages=read_vintages(), table=features.table, cut=common.ORIGIN
    ).held_points
    norm = common.farm_norm(cache=cache, farm=GATE_FARM)
    masked_by = common.INJECTED_SHARES[0] * pinned_points * norm
    design, levels, _ = fit_path(
        cache=cache, site=GATE_SITE, cut=common.ORIGIN, shares=(3.0,), masked_by=masked_by
    )
    stored = (
        pl.read_parquet(
            ML_EXPERIMENTS_DATA_DIR
            / "00207-dp-meter-regime-steps-e2-step-capacity-known-answer"
            / "work"
            / f"paths_{GATE_SITE}.parquet"
        )
        .filter(
            (pl.col("farm") == GATE_FARM) & (pl.col("arm") == "step_3") & (pl.col("target") == 0)
        )
        .sort("month")
    )
    months = design.first_month + np.arange(design.n_months)
    ours = pl.DataFrame({"month": months, "level": levels[3.0]})
    joined = stored.join(ours, on="month", how="inner", suffix="_new")
    difference = float((joined["level"] - joined["level_new"]).abs().to_numpy().max())
    return {
        "months": joined.height,
        "max_abs_difference_points": difference,
        "passed": bool(difference < GATE_TOLERANCE and joined.height == len(months)),
    }


def main() -> None:
    """Write the gate, the paths, and the steps to `work/`."""
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", default=list(SITES))
    args = parser.parse_args()
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    cache = common.load_cache()
    gate = run_gate(cache=cache)
    print("gate:", gate, flush=True)
    common.write_json_atomic(path=common.WORK_DIR / "gate.json", content=gate)
    register = register_weights()
    step_rows, path_rows, summary = [], [], {}
    for site in args.sites:
        table = cache.sites[site].table
        design, levels, n_rows = fit_path(cache=cache, site=site, cut=SCAN_CUT, shares=WEIGHTS)
        scale = table.effective_capacity_mw / 100.0
        months = design.first_month + np.arange(design.n_months)
        for weight, path in levels.items():
            path_rows += [
                {
                    "site": site,
                    "weight": weight,
                    "month_index": int(m),
                    "level_mw": float(v * scale),
                }
                for m, v in zip(months, path, strict=True)
            ]
            step_rows += steps_of(design=design, levels_mw=path * scale, site=site, weight=weight)
        rate, n_boundaries = base_rate(register=register, series_id=int(table.time_series_id))
        summary[site] = {
            "rows": n_rows,
            "months": design.n_months,
            "base_rate": rate,
            "n_boundaries": n_boundaries,
            "series_id": int(table.time_series_id),
        }
        print(site, summary[site], flush=True)
    steps = pl.DataFrame(step_rows, schema_overrides={"date": pl.Date})
    matched = []
    for row in steps.iter_rows(named=True):
        series_id = summary[row["site"]]["series_id"]
        in_register = row["kind"] == "month" and row["date"] <= REGISTER_END
        cluster = (
            window_mw(register=register, series_id=series_id, centre=row["date"])
            if in_register
            else None
        )
        matched.append(cluster)
    steps = steps.with_columns(register_mw=pl.Series(matched, dtype=pl.Float64))
    steps.write_parquet(common.WORK_DIR / "steps.parquet")
    pl.DataFrame(path_rows).write_parquet(common.WORK_DIR / "paths.parquet")
    common.write_json_atomic(path=common.WORK_DIR / "summary.json", content=summary)
    print(steps.filter(pl.col("weight") == 3.0).sort("site", "date"))


if __name__ == "__main__":
    main()
