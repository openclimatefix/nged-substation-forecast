# ruff: noqa: E501, PERF401, PLR0912, PLR0915
"""Gains of PERFECT and PERFECT_LAG over K0 by row group, metric and target, with intervals.

Reads the score tables of `e00283_score.py`, builds the per-row node-only curtailment cost, checks
that cost against `studies.node_curtailment.node_curtailment`, and prints every number the page
quotes. Intervals resample whole shifts (units: world and event) for rows near a shift and whole
calendar months for ordinary rows, 95%, 20,000 draws.

    uv run --no-sync python e00283_report.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
from e00283_common import WORK_DIR
from studies.node_curtailment import (
    NodeCurtailmentAssumptions,
    NodeTable,
    node_curtailment,
    table_with_scaled_limits,
)

SCORES: Final[Path] = WORK_DIR / "scores"
RESULTS: Final[Path] = WORK_DIR / "results"
ARMS: Final[tuple[str, ...]] = ("K0", "PERFECT", "PERFECT_LAG", "PERFECT_ADD", "PERFECT_LAG_ADD")
CONTRASTS: Final[tuple[str, ...]] = ("PERFECT", "PERFECT_LAG", "PERFECT_ADD", "PERFECT_LAG_ADD")
"""The `_ADD` arms (post hoc) forecast observed power, so they are scored on that target only."""
TARGETS: Final[tuple[str, ...]] = ("nra", "obs")
FRACTIONS: Final[tuple[float, ...]] = (0.5, 0.75, 0.9)
PRICE_GBP_PER_MWH: Final[float] = 2000.0
BREACH_RATIO: Final[int] = 20
ORDINARY_WEIGHT: Final[float] = 4.0
"""Only every fourth daily run's ordinary rows are scored, so each stands for four."""
N_DRAWS: Final[int] = 20_000
METRICS: Final[tuple[str, ...]] = ("qs", "ae_median", "ae_mean", "tw95", "tw99")
LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0-12 h", 0.0, 12.0),
    ("12-36 h", 12.0, 36.0),
    ("36 h-7 d", 36.0, 1.0e9),
)
SIZE_BINS: Final[tuple[tuple[str, float, float], ...]] = (
    ("under 10%", 0.0, 0.10),
    ("10% to 25%", 0.10, 0.25),
    ("over 25%", 0.25, 1.0e9),
)
DURATION_BINS: Final[tuple[tuple[str, float, float], ...]] = (
    ("under 6 h", 0.0, 6.0),
    ("6 to 24 h", 6.0, 24.0),
    ("over 1 day", 24.0, 1.0e9),
)
WINTER: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})
LEAD_MIN_HOURS: Final[float] = 12.0
LEAD_MAX_HOURS: Final[float] = 36.0


def load_wide() -> tuple[pl.DataFrame, dict[str, Any]]:
    """Read the score tables and put the three arms side by side, one row per scored row."""
    constants = json.loads((SCORES / "site_constants.json").read_text())
    frames = [pl.read_parquet(p) for p in sorted(SCORES.glob("[D]*_*.parquet"))]
    long = pl.concat(frames)
    meta = [
        "site",
        "world",
        "period",
        "row",
        "position",
        "run",
        "power_fcst_init_time",
        "lead_hours",
        "event",
        "phase",
        "in_shift",
        "event_hours",
        "fraction",
        "onset_month",
        "actual_nra",
        "actual_obs",
    ]
    values = [c for c in long.columns if c not in {*meta, "arm"}]
    wide = long.pivot(on="arm", index=meta, values=values, separator="__")
    return wide, constants


def add_cost(*, wide: pl.DataFrame, constants: dict[str, Any]) -> pl.DataFrame:
    """Add the node-only curtailment cost (GBP per half-hour) of each arm, target and fraction.

    The cost is null outside the 12 to 36 hour leads. A site's scaled limit is `f` times the
    largest power it carried before the first replicate row, in MW.
    """
    cap = np.array([constants[s]["capacity_mw"] for s in wide["site"]]) / 100.0
    peak = np.array([constants[s]["max_before_points"] for s in wide["site"]]) * cap
    lead = wide["lead_hours"].to_numpy()
    inside = (lead >= LEAD_MIN_HOURS) & (lead < LEAD_MAX_HOURS)
    new = []
    for fraction in FRACTIONS:
        limit = fraction * peak
        for target in TARGETS:
            actual = wide[f"actual_{target}"].to_numpy().astype(np.float64) * cap
            for arm in ARMS:
                q95 = wide[f"q95__{arm}"].to_numpy().astype(np.float64) * cap
                planned = np.maximum(q95 - limit, 0.0) * 0.5
                breach = np.maximum(actual - planned / 0.5 - limit, 0.0) * 0.5
                cost = PRICE_GBP_PER_MWH * (planned + BREACH_RATIO * breach)
                new.append(
                    pl.Series(f"cost{fraction}_{target}__{arm}", np.where(inside, cost, np.nan))
                )
    return wide.with_columns(new)


def check_cost_against_library(
    *, wide: pl.DataFrame, constants: dict[str, Any]
) -> dict[str, float]:
    """Compare the per-row cost with `node_curtailment` for K0 at one site in one world."""
    site = "D11"
    world = int(wide.filter(pl.col("site") == site)["world"][0])
    sub = wide.filter(pl.col("site") == site, pl.col("world") == world)
    cap = constants[site]["capacity_mw"] / 100.0
    limit = 0.5 * constants[site]["max_before_points"] * cap
    valid = pl.col("power_fcst_init_time") + pl.duration(
        microseconds=(pl.col("lead_hours").cast(pl.Float64) * 3.6e9).round().cast(pl.Int64)
    )
    frame = sub.select(
        time_series_id=pl.lit(1, dtype=pl.Int64),
        power_fcst_init_time=pl.col("power_fcst_init_time"),
        valid_time=valid,
        seed=pl.col("row").cast(pl.Int64) * 0,
        actual=pl.col("actual_obs").cast(pl.Float64) * cap,
        q05=pl.col("q05__K0").cast(pl.Float64) * cap,
        q95=pl.col("q95__K0").cast(pl.Float64) * cap,
    ).unique(subset=["power_fcst_init_time", "valid_time"], keep="first")
    nodes = pl.DataFrame(
        {
            "node_id": [site],
            "time_series_id": [1],
            "node_type": ["BSP"],
            "parent_node_id": [None],
            "export_sign": [1],
            "sign_source": ["assumed"],
        },
        schema_overrides={"parent_node_id": pl.String, "export_sign": pl.Int64},
    )
    generators = pl.DataFrame(
        schema={
            "generator_id": pl.Int64,
            "node_id": pl.String,
            "capacity_mw": pl.Float64,
            "connection_date": pl.Date,
            "is_forecast": pl.Boolean,
        }
    )
    limits = pl.DataFrame(
        schema={
            "node_id": pl.String,
            "season": pl.String,
            "export_limit_mw": pl.Float64,
            "import_limit_mw": pl.Float64,
            "limit_source": pl.String,
        }
    )
    table = NodeTable(nodes=nodes, limits=limits, generators=generators)
    scaled = pl.DataFrame(
        {"node_id": [site], "export_limit_mw": [limit], "import_limit_mw": [None]},
        schema_overrides={"import_limit_mw": pl.Float64},
    )
    result = node_curtailment(
        table=table_with_scaled_limits(table=table, scaled=scaled, fraction=0.5),
        node_forecasts={"K0": frame},
        generator_forecasts={},
        assumptions=NodeCurtailmentAssumptions(
            use_generators=False, headline_price_gbp_per_mwh=PRICE_GBP_PER_MWH
        ),
    )
    library = float(result.per_arm.filter(pl.col("is_headline_price"))["energy_proxy_gbp"][0])
    mine_rows = sub.filter(
        pl.col("lead_hours") >= LEAD_MIN_HOURS, pl.col("lead_hours") < LEAD_MAX_HOURS
    )
    mine_rows = mine_rows.unique(subset=["power_fcst_init_time", "lead_hours"], keep="first")
    mine = float(mine_rows["cost0.5_obs__K0"].sum())
    return {
        "library_gbp": library,
        "per_row_gbp": mine,
        "relative_gap": abs(library - mine) / max(library, 1e-9),
    }


def units_of(*, wide: pl.DataFrame) -> np.ndarray:
    """A unit label per row: world and event near a shift, calendar month for ordinary rows."""
    valid_month = (
        wide.select(
            (
                pl.col("power_fcst_init_time")
                + pl.duration(hours=pl.col("lead_hours").cast(pl.Int64))
            )
            .dt.strftime("%Y%m")
            .cast(pl.Int64)
        )
        .to_series()
        .to_numpy()
    )
    world, event = (
        wide["world"].to_numpy().astype(np.int64),
        wide["event"].to_numpy().astype(np.int64),
    )
    ordinary = wide["phase"].to_numpy() == "ORDINARY"
    return np.where(ordinary, -valid_month, world * 100_000 + event)


def interval(
    *,
    gain: np.ndarray,
    base: np.ndarray,
    weight: np.ndarray,
    units: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Mean gain per row and gain as a percentage of the base, with whole-unit resampling.

    Units with a negative label are ordinary-row months and units with a positive label are shifts;
    each kind is resampled within itself, so the two kinds keep their own counts.
    """
    keep = np.isfinite(gain) & np.isfinite(base)
    gain, base, weight, units = gain[keep], base[keep], weight[keep], units[keep]
    if not len(gain):
        return dict.fromkeys(
            ("n", "mean", "mean_lo", "mean_hi", "pct", "pct_lo", "pct_hi"), float("nan")
        )
    labels, codes = np.unique(units, return_inverse=True)
    g = np.bincount(codes, weights=gain * weight, minlength=len(labels))
    b = np.bincount(codes, weights=base * weight, minlength=len(labels))
    c = np.bincount(codes, weights=weight, minlength=len(labels))
    mean_draws = np.zeros(N_DRAWS)
    pct_num = np.zeros(N_DRAWS)
    pct_den = np.zeros(N_DRAWS)
    cnt = np.zeros(N_DRAWS)
    for stratum in (labels < 0, labels > 0):
        idx_units = np.flatnonzero(stratum)
        if not len(idx_units):
            continue
        draws = rng.integers(0, len(idx_units), size=(N_DRAWS, len(idx_units)))
        chosen = idx_units[draws]
        pct_num += g[chosen].sum(axis=1)
        pct_den += b[chosen].sum(axis=1)
        cnt += c[chosen].sum(axis=1)
    mean_draws = pct_num / cnt
    pct_draws = 100.0 * pct_num / pct_den
    est_mean = g.sum() / c.sum()
    est_pct = 100.0 * g.sum() / b.sum()
    m_lo, m_hi = np.percentile(mean_draws, [2.5, 97.5])
    p_lo, p_hi = np.percentile(pct_draws, [2.5, 97.5])
    return {
        "n": float(c.sum()),
        "n_units": float(len(labels)),
        "mean": float(est_mean),
        "mean_lo": float(m_lo),
        "mean_hi": float(m_hi),
        "pct": float(est_pct),
        "pct_lo": float(p_lo),
        "pct_hi": float(p_hi),
    }


def group_masks(*, wide: pl.DataFrame) -> dict[str, np.ndarray]:
    """The row groups G0 to G5."""
    phase = wide["phase"].to_numpy()
    return {
        "G0 all rows": np.ones(len(phase), dtype=bool),
        "G1 ordinary": phase == "ORDINARY",
        "G2 before alarm": phase == "BEFORE_ALARM",
        "G3 during": phase == "DURING",
        "G4 after": phase == "AFTER",
        "G5 near": phase != "ORDINARY",
    }


def strata_masks(*, wide: pl.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """The G6 strata, applied inside a group: size, duration, season, lead, substation."""
    out: dict[str, dict[str, np.ndarray]] = {}
    fraction, hours = wide["fraction"].to_numpy(), wide["event_hours"].to_numpy()
    lead, month = wide["lead_hours"].to_numpy(), wide["onset_month"].to_numpy()
    out["size"] = {name: (fraction >= lo) & (fraction < hi) for name, lo, hi in SIZE_BINS}
    out["duration"] = {name: (hours >= lo) & (hours < hi) for name, lo, hi in DURATION_BINS}
    out["season"] = {
        "November to March": np.isin(month, list(WINTER)),
        "April to October": ~np.isin(month, list(WINTER)) & (month > 0),
    }
    out["lead"] = {name: (lead >= lo) & (lead < hi) for name, lo, hi in LEAD_BANDS}
    sites = wide["site"].to_numpy()
    out["substation"] = {s: sites == s for s in sorted(set(sites))}
    return out


def main() -> None:
    """Compute and print every number the page quotes."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    wide, constants = load_wide()
    wide = add_cost(wide=wide, constants=constants)
    rng = np.random.default_rng(283)
    out: dict[str, Any] = {"n_rows": wide.height}
    out["cost_check"] = check_cost_against_library(wide=wide, constants=constants)
    print("cost check:", out["cost_check"], flush=True)
    phase = wide["phase"].to_numpy()
    weight = np.where(phase == "ORDINARY", ORDINARY_WEIGHT, 1.0)
    units = units_of(wide=wide)
    groups = group_masks(wide=wide)
    metrics = [*METRICS, *(f"cost{f}" for f in FRACTIONS)]
    table: list[dict[str, Any]] = []
    for contrast in CONTRASTS:
        for target in TARGETS:
            if contrast.endswith("_ADD") and target == "nra":
                continue
            for metric in metrics:
                k0 = (
                    wide[f"{metric}_{target}__K0"].to_numpy().astype(np.float64)
                    if not metric.startswith("cost")
                    else wide[f"{metric}_{target}__K0"].to_numpy()
                )
                arm = wide[f"{metric}_{target}__{contrast}"].to_numpy().astype(np.float64)
                gain = k0 - arm
                for group, mask in groups.items():
                    res = interval(
                        gain=gain[mask],
                        base=k0[mask],
                        weight=weight[mask],
                        units=units[mask],
                        rng=rng,
                    )
                    table.append(
                        {
                            "contrast": contrast,
                            "target": target,
                            "metric": metric,
                            "group": group,
                            **res,
                        }
                    )
        print(f"{contrast} done", flush=True)
    out["groups"] = table
    strata_rows: list[dict[str, Any]] = []
    strata = strata_masks(wide=wide)
    for contrast in CONTRASTS:
        for target, metric in (("nra", "tw95"), ("nra", "qs"), ("obs", "cost0.5"), ("obs", "tw95")):
            if contrast.endswith("_ADD") and target == "nra":
                continue
            k0 = wide[f"{metric}_{target}__K0"].to_numpy().astype(np.float64)
            arm = wide[f"{metric}_{target}__{contrast}"].to_numpy().astype(np.float64)
            gain = k0 - arm
            for group in ("G4 after", "G5 near"):
                base_mask = groups[group]
                for kind, parts in strata.items():
                    for name, mask in parts.items():
                        both = base_mask & mask
                        res = interval(
                            gain=gain[both],
                            base=k0[both],
                            weight=weight[both],
                            units=units[both],
                            rng=rng,
                        )
                        strata_rows.append(
                            {
                                "contrast": contrast,
                                "target": target,
                                "metric": metric,
                                "group": group,
                                "kind": kind,
                                "stratum": name,
                                **res,
                            }
                        )
    out["strata"] = strata_rows
    # Leave-one-substation-out of the headline contrasts.
    loo: list[dict[str, Any]] = []
    sites = wide["site"].to_numpy()
    for contrast in CONTRASTS:
        for target, metric, group in (("nra", "tw95", "G4 after"), ("obs", "cost0.5", "G5 near")):
            if contrast.endswith("_ADD") and target == "nra":
                continue
            k0 = wide[f"{metric}_{target}__K0"].to_numpy().astype(np.float64)
            arm = wide[f"{metric}_{target}__{contrast}"].to_numpy().astype(np.float64)
            for site in sorted(set(sites)):
                mask = groups[group] & (sites != site)
                res = interval(
                    gain=(k0 - arm)[mask],
                    base=k0[mask],
                    weight=weight[mask],
                    units=units[mask],
                    rng=rng,
                )
                loo.append(
                    {
                        "contrast": contrast,
                        "target": target,
                        "metric": metric,
                        "group": group,
                        "left_out": site,
                        **res,
                    }
                )
    out["leave_one_out"] = loo
    # Median minus mean forecast per site, in points (the mean forecast is the GAM-K member mean plus the correction).
    checks = []
    for site in sorted(set(sites)):
        mask = sites == site
        for arm in ARMS:
            checks.append(
                {
                    "site": site,
                    "arm": arm,
                    "median_minus_mean_error_ae": float(
                        np.nanmean(
                            wide[f"ae_median_nra__{arm}"].to_numpy()[mask]
                            - wide[f"ae_mean_nra__{arm}"].to_numpy()[mask]
                        )
                    ),
                }
            )
    out["checks"] = checks
    out["counts"] = {
        "worlds": int(wide["world"].n_unique()),
        "shifts": len(np.unique(units[units > 0])),
        "rows_by_phase": {p: int((phase == p).sum()) for p in sorted(set(phase))},
    }
    (RESULTS / "report.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out["counts"], indent=2))
    for row in table:
        if True:
            print(
                f"{row['contrast']:12s} {row['target']} {row['metric']:8s} {row['group']:16s} "
                f"n={row['n']:.0f} mean {row['mean']:+.4f} ({row['mean_lo']:+.4f}, {row['mean_hi']:+.4f}) "
                f"pct {row['pct']:+.2f} ({row['pct_lo']:+.2f}, {row['pct_hi']:+.2f})"
            )

    print("--- strata (G4 and G5, selected metrics)")
    for row in strata_rows:
        print(
            f"{row['contrast']:12s} {row['target']} {row['metric']:8s} {row['group']:10s} {row['kind']:10s} {row['stratum']:20s} "
            f"n={row['n']:.0f} pct {row['pct']:+.2f} ({row['pct_lo']:+.2f}, {row['pct_hi']:+.2f})"
        )
    print("--- leave one substation out")
    for row in loo:
        print(
            f"{row['contrast']:12s} {row['target']} {row['metric']:8s} {row['group']:10s} without {row['left_out']:4s} "
            f"pct {row['pct']:+.2f} ({row['pct_lo']:+.2f}, {row['pct_hi']:+.2f})"
        )
    print("--- median-forecast error minus mean-forecast error, per site (points)")
    for row in checks:
        print(f"{row['site']:4s} {row['arm']:16s} {row['median_minus_mean_error_ae']:+.4f}")


if __name__ == "__main__":
    main()
