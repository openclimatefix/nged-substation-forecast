"""Experiment 113's real-data intervals for the five decision substations.

For each cut (2025-07-01, with 2024-07-01 as the replicate), the convex model with fitted
capacity is fitted to the real target with the D1 drift and without it. For each fit the script
reports, for each hour band's total heating slope and each season-year's capacity, the point
estimate and the I1 (stratified, inflated), I1u, and I2 80% intervals, beside the weekly
real-time capacity of experiment 45's vintage averaged over the season-year
(`studies.dp_pinned_capacity`). It also reports the absorption ratio of dropping the harmonics
and drift, the bootstrap-to-sandwich variance ratio tau, and the effective sample size.

    uv run python ml-experiments/00113-*/e00113_real.py
"""

from dataclasses import replace
from datetime import datetime

import e00113_analytic as analytic_module
import e00113_common as common
import e00113_design as design_module
import numpy as np
import polars as pl
from studies.dp_calendar import SMOOTHING_HOURS
from studies.dp_features import FeatureCache
from studies.dp_gam import N_BANDS
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

REAL_ARMS: tuple[str, ...] = ("g_full", "g_h")
"""With the D1 drift (and harmonics), and without the drift (harmonics kept)."""


def real_functionals(*, design: design_module.Design) -> tuple[np.ndarray, list[str]]:
    """Return the functionals: each band's total slope, then each season-year's capacity."""
    layout = design.layout
    names = list(design_module.FUNCTIONAL_NAMES[:N_BANDS])
    columns = []
    heating = layout.slices["heating"].start
    for band in range(N_BANDS):
        column = np.zeros(design.p)
        for smoothing in range(len(SMOOTHING_HOURS)):
            column[heating + smoothing * N_BANDS + band] = 1.0
        columns.append(column)
    capacity = layout.slices["capacity"]
    for position, year in enumerate(layout.season_years):
        column = np.zeros(design.p)
        column[capacity.start : capacity.start + position + 1] = 1.0
        columns.append(column)
        names.append(f"capacity_{year}")
    return np.stack(columns, axis=1), names


def vintage_season_means(
    *, cache: FeatureCache, site: str, cut: datetime, years: tuple[int, ...]
) -> dict[int, float]:
    """Return the vintage's weekly capacity averaged over each season-year's weeks up to the cut."""
    table = cache.sites[site].table
    pinned = pinned_capacity(vintages=read_vintages(), table=table, cut=cut)
    out = {}
    for year in years:
        start = np.datetime64(f"{year}-07-01", "D")
        stop = np.datetime64(f"{year + 1}-07-01", "D")
        weeks = (
            (pinned.weeks >= start)
            & (pinned.weeks < stop)
            & (pinned.weeks <= np.datetime64(cut.date().isoformat(), "D"))
        )
        out[year] = float(pinned.points[weeks].mean()) if weeks.any() else float("nan")
    return out


def main() -> None:
    """Compute and store the real-data table."""
    common.configure_threads(threads=4)
    cache = common.load_cache()
    rows = []
    for cut_label, cut in (("scored", common.SCORED_CUT), ("replicate", common.REPLICATE_CUT)):
        for site in common.DECISION_SITES:
            base = design_module.build_design(cache=cache, site=site, cut=cut, window="all")
            matrix, names = real_functionals(design=base)
            design = replace(base, functionals=matrix)
            target = cache.sites[site].y[design.index]
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [common.EXPERIMENT_ID, 9, common.DECISION_SITES.index(site), cut.year]
                )
            )
            counts = design_module.weighting_set(design=design, rng=rng)
            gram = design_module.weighted_gram(design=design, counts=counts)
            rhs = design_module.weighted_rhs(
                counts=counts,
                month_rhs=design_module.month_cross_products(design=design, target=target),
            )
            vintage = vintage_season_means(
                cache=cache, site=site, cut=cut, years=design.layout.season_years
            )
            n_boot = common.N_BOOTSTRAP
            months_per_quarter = len(design.months) / 4.0
            for arm_name in REAL_ARMS:
                arm = design_module.ARMS[arm_name]
                theta = design_module.solve_arm(
                    design=design, arm=arm, gram=gram, rhs=rhs, pinned_increments=None
                )
                values = design_module.functional_values(design=design, theta=theta)
                analytics_arm = analytic_module._inverse_quadratic(design=design, arm=arm)
                from e00113_cell import arm_analytics

                moments = arm_analytics(design=design, arm=arm)
                residual = target - design.x @ theta[0]
                sigma2 = float(residual @ residual) / (design.n - moments.edf)
                small = analytic_module._inverse_quadratic(
                    design=design, arm=replace(design_module.ARMS["g_0"], drift_step=arm.drift_step)
                )
                boot, boot_u = values[1 : 1 + n_boot], values[1 + n_boot :]
                median = np.median(boot, axis=0)
                q_low, q_high = np.quantile(boot, [0.1, 0.9], axis=0)
                qu_low, qu_high = np.quantile(boot_u, [0.1, 0.9], axis=0)
                for position, name in enumerate(names):
                    is_capacity = name.startswith("capacity")
                    m = float(common.CAPACITY_MONTHS if is_capacity else months_per_quarter)
                    inflate = float(np.sqrt(m / (m - 1.0)))
                    sd_post = float(np.sqrt(moments.posterior[position] * sigma2))
                    sd_sand = float(np.sqrt(moments.sandwich[position] * sigma2))
                    tau = float(boot[:, position].var(ddof=1) / sd_sand**2)
                    rows.append(
                        {
                            "cut": cut_label,
                            "site": site,
                            "arm": arm_name,
                            "functional": name,
                            "point": float(values[0, position]),
                            "i1_lo": float(
                                median[position] - inflate * (median[position] - q_low[position])
                            ),
                            "i1_hi": float(
                                median[position] + inflate * (q_high[position] - median[position])
                            ),
                            "i1_raw_lo": float(q_low[position]),
                            "i1_raw_hi": float(q_high[position]),
                            "i1u_lo": float(qu_low[position]),
                            "i1u_hi": float(qu_high[position]),
                            "i2_lo": float(values[0, position] - common.INTERVAL_Z * sd_post),
                            "i2_hi": float(values[0, position] + common.INTERVAL_Z * sd_post),
                            "vintage_season_mean": vintage.get(
                                int(name.split("_")[1]), float("nan")
                            )
                            if is_capacity
                            else float("nan"),
                            "tau": tau,
                            "r_without_harmonics_and_drift": float(
                                small[position] / analytics_arm[position]
                            ),
                            "n_rows": design.n,
                            "n_eff_rows": float(design.n / tau),
                        }
                    )
            print("done", cut_label, site, flush=True)
    pl.DataFrame(rows).write_parquet(common.WORK_DIR / "real_intervals.parquet")


if __name__ == "__main__":
    main()
