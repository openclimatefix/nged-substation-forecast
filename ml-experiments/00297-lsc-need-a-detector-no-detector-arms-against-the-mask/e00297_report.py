# ruff: noqa: E501, PLR0912, PLR0915
"""Gains of PERFECT and PERFECT_LAG over K0 by row group, metric and target, with intervals.

Reads the score tables of `e00297_score.py`, builds the per-row node-only curtailment cost, checks
that cost against `studies.node_curtailment.node_curtailment`, and prints every number the page
quotes. Intervals resample whole shifts (units: world and event) for rows near a shift and whole
calendar months for ordinary rows, 95%, 20,000 draws.

    uv run --no-sync python e00297_report.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
from e00297_common import WORK_DIR
from studies.node_curtailment import (
    NodeCurtailmentAssumptions,
    NodeTable,
    node_curtailment,
    table_with_scaled_limits,
)

SCORES: Final[Path] = WORK_DIR / "scores"
RESULTS: Final[Path] = WORK_DIR / "results"
SCORES2: Final[Path] = WORK_DIR / "scores2"
SCORES3: Final[Path] = WORK_DIR / "scores3"
NO_DETECTOR: Final[tuple[str, ...]] = (
    "K0",
    "GAM_ONLY",
    "GAM_ONLY_K0SPREAD",
    "BAND_TABLE_PLAIN",
    "BAND_TABLE_DISAGREE",
    "BAND_HEAD",
    "ASOF_WIDEN",
    "LEAD_DECAY",
)
HELDOUT: Final[str] = "BAND_HEAD_HELDOUT"
DETECTOR: Final[tuple[str, ...]] = ("PERFECT", "PERFECT_ADD", "DET", "DET_ADD")
ARMS: Final[tuple[str, ...]] = (*NO_DETECTOR, HELDOUT, *DETECTOR)
"""The `_ADD` arms (post hoc) forecast observed power, so they are scored on that target only."""
KEEP: Final[tuple[str, ...]] = (
    "qs_nra",
    "qs_obs",
    "ae_median_nra",
    "ae_median_obs",
    "ae_mean_nra",
    "ae_mean_obs",
    "tw95_nra",
    "tw95_obs",
    "q01",
    "q05",
    "q95",
    "q99",
)
TARGETS: Final[tuple[str, ...]] = ("nra", "obs")
FRACTIONS: Final[tuple[float, ...]] = (0.5, 0.75, 0.9)
PRICE_GBP_PER_MWH: Final[float] = 2000.0
BREACH_RATIO: Final[int] = 20
ORDINARY_WEIGHT: Final[float] = 4.0
"""Only every fourth daily run's ordinary rows are scored, so each stands for four."""
N_DRAWS: Final[int] = 20_000
METRICS: Final[tuple[str, ...]] = ("qs", "ae_median", "ae_mean", "tw95")
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
    """Read pass 1 and pass 2 scores and put the arms side by side, one row per scored row."""
    constants = json.loads((SCORES / "site_constants.json").read_text())
    meta_cols = [
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
    long1 = pl.concat([pl.read_parquet(p) for p in sorted(SCORES.glob("D*_*.parquet"))])
    long2 = pl.concat(
        [
            pl.read_parquet(p)
            for p in [*sorted(SCORES2.glob("D*_*.parquet")), *sorted(SCORES3.glob("D*_*.parquet"))]
        ]
    )
    wide = long1.filter(pl.col("arm") == "K0").select(meta_cols)
    for long in (long1, long2):
        for arm in sorted(long["arm"].unique().to_list()):
            part = long.filter(pl.col("arm") == arm).select(
                "site", "world", "row", *[pl.col(c).alias(f"{c}__{arm}") for c in KEEP]
            )
            wide = wide.join(part, on=["site", "world", "row"], how="left", validate="1:1")
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
                    pl.Series(
                        f"cost{fraction}_{target}__{arm}", np.where(inside, cost, np.nan)
                    ).cast(pl.Float32)
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
        "Gs shift rows": np.isin(phase, ["BEFORE_ALARM", "DURING", "AFTER"]),
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


def col(wide: pl.DataFrame, metric: str, target: str, arm: str) -> np.ndarray:
    """One arm's metric column as float64."""
    return wide[f"{metric}_{target}__{arm}"].to_numpy().astype(np.float64)


def ratio_interval(
    *,
    num: np.ndarray,
    den: np.ndarray,
    weight: np.ndarray,
    units: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Ratio of summed `num` to summed `den`, resampling whole units within each kind per draw."""
    keep = np.isfinite(num) & np.isfinite(den)
    num, den, weight, units = num[keep], den[keep], weight[keep], units[keep]
    labels, codes = np.unique(units, return_inverse=True)
    n = np.bincount(codes, weights=num * weight, minlength=len(labels))
    d = np.bincount(codes, weights=den * weight, minlength=len(labels))
    sn, sd = np.zeros(N_DRAWS), np.zeros(N_DRAWS)
    for stratum in (labels < 0, labels > 0):
        idx = np.flatnonzero(stratum)
        if len(idx):
            chosen = idx[rng.integers(0, len(idx), size=(N_DRAWS, len(idx)))]
            sn += n[chosen].sum(axis=1)
            sd += d[chosen].sum(axis=1)
    draws = sn / sd
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"ratio": float(n.sum() / d.sum()), "lo": float(lo), "hi": float(hi)}


def show(tag: str, res: dict[str, float]) -> None:
    """Print one interval result."""
    print(
        f"{tag}: n={res['n']:.0f} mean {res['mean']:+.4f} ({res['mean_lo']:+.4f}, {res['mean_hi']:+.4f})"
        f" pct {res['pct']:+.2f} ({res['pct_lo']:+.2f}, {res['pct_hi']:+.2f})"
    )


def main() -> None:
    """Compute and print every number the page quotes."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    wide, constants = load_wide()
    wide = add_cost(wide=wide, constants=constants)
    rng = np.random.default_rng(297)
    out: dict[str, Any] = {"n_rows": wide.height}
    out["cost_check"] = check_cost_against_library(wide=wide, constants=constants)
    print("cost check:", out["cost_check"], flush=True)
    phase = wide["phase"].to_numpy()
    weight = np.where(phase == "ORDINARY", ORDINARY_WEIGHT, 1.0)
    units = units_of(wide=wide)
    groups = group_masks(wide=wide)
    lead = wide["lead_hours"].to_numpy()
    world = wide["world"].to_numpy()
    half_a = world % 2 == 0
    sites = wide["site"].to_numpy()

    def gain_of(
        *, base_arm: str, arm: str, metric: str, target: str, mask: np.ndarray
    ) -> dict[str, float]:
        b, a = col(wide, metric, target, base_arm), col(wide, metric, target, arm)
        return interval(
            gain=(b - a)[mask], base=b[mask], weight=weight[mask], units=units[mask], rng=rng
        )

    # 1. Every arm against K0, all rows of all worlds, by group.
    print("=== gain over K0 by group (positive is better)")
    table = []
    for arm in ARMS[1:]:
        for target, metric in (("nra", "tw95"), ("nra", "qs"), ("obs", "tw95"), ("obs", "cost0.5")):
            if arm.endswith("_ADD") and target == "nra":
                continue
            for group, mask in groups.items():
                res = gain_of(base_arm="K0", arm=arm, metric=metric, target=target, mask=mask)
                table.append(
                    {"arm": arm, "target": target, "metric": metric, "group": group, **res}
                )
                show(f"{arm:20s} {target} {metric:8s} {group:16s}", res)
    out["gain_over_k0"] = table
    # 2. Absolute scores.
    print("=== absolute mean scores (points; cost in GBP per scored half-hour)")
    absolute = []
    for arm in ARMS:
        for group, mask in groups.items():
            row = {"arm": arm, "group": group}
            for target in TARGETS:
                for metric in (*METRICS, "cost0.5"):
                    v = col(wide, metric, target, arm)[mask]
                    w = weight[mask]
                    ok = np.isfinite(v)
                    row[f"{metric}_{target}"] = float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))
            absolute.append(row)
            print(arm, group, {k: round(v, 4) for k, v in row.items() if isinstance(v, float)})
    out["absolute"] = absolute
    # 3. Decision contrasts: best no-detector arm chosen on half A, detectors read on half B.
    print("=== decision contrasts (half B), best no-detector arm chosen on half A")
    g5 = groups["G5 near"]
    decisions = []
    for target, metric in (("nra", "tw95"), ("obs", "cost0.5")):
        scores = {}
        for arm in NO_DETECTOR:
            v = col(wide, metric, target, arm)
            m = g5 & half_a & np.isfinite(v)
            scores[arm] = float(np.sum(v[m] * weight[m]) / np.sum(weight[m]))
        best = min(scores, key=lambda name: scores[name])
        print(
            f"best no-detector arm on half A for {metric} {target}: {best}",
            {k: round(v, 4) for k, v in scores.items()},
        )
        for det in DETECTOR:
            if det.endswith("_ADD") != (target == "obs"):
                continue
            extras = [("all substations", np.ones(len(sites), bool))]
            extras += [("without D11", sites != "D11"), ("without D16", sites != "D16")]
            extras += [(f"only {s}", sites == s) for s in sorted(set(sites))]
            for gname in ("G5 near", "Gs shift rows"):
                for label, extra in extras:
                    mask = groups[gname] & ~half_a & extra
                    if not mask.any():
                        continue
                    res = gain_of(base_arm=best, arm=det, metric=metric, target=target, mask=mask)
                    g1 = gain_of(
                        base_arm=best,
                        arm=det,
                        metric="qs",
                        target=target,
                        mask=groups["G1 ordinary"] & ~half_a & extra,
                    )
                    g1c = gain_of(
                        base_arm=best,
                        arm=det,
                        metric="cost0.5",
                        target="obs",
                        mask=groups["G1 ordinary"] & ~half_a & extra,
                    )
                    decisions.append(
                        {
                            "target": target,
                            "metric": metric,
                            "best": best,
                            "detector": det,
                            "group": gname,
                            "subset": label,
                            "gain": res,
                            "g1_qs": g1,
                            "g1_cost": g1c,
                        }
                    )
                    show(f"{det:12s} vs {best:14s} {gname:13s} {metric} {label:16s}", res)
                    if label in ("all substations", "without D11", "without D16"):
                        show(f"{det:12s} vs {best:14s} G1 qs {label:16s}", g1)
                        show(f"{det:12s} vs {best:14s} G1 cost {label:16s}", g1c)
    out["decisions"] = decisions
    # 4. Each no-detector arm against each detector, all worlds (no selection).
    print("=== detector against every no-detector arm, all worlds, G5 and G1")
    versus = []
    for det in DETECTOR:
        target, metric = ("obs", "cost0.5") if det.endswith("_ADD") else ("nra", "tw95")
        for arm in NO_DETECTOR:
            for group in ("G5 near", "G4 after", "G1 ordinary"):
                res = gain_of(
                    base_arm=arm, arm=det, metric=metric, target=target, mask=groups[group]
                )
                versus.append(
                    {"detector": det, "arm": arm, "group": group, "metric": metric, **res}
                )
                show(f"{det:12s} vs {arm:20s} {metric} {group:12s}", res)
    out["versus"] = versus
    # 5. P1: GAM_ONLY cost on ordinary rows at 24 to 168 h, and tail coverage after a shift.
    print("=== P1")
    band = (lead >= 24.0) & (lead <= 168.0)
    p1 = []
    for arm in ("GAM_ONLY", "GAM_ONLY_K0SPREAD"):
        res = gain_of(
            base_arm="K0", arm=arm, metric="qs", target="nra", mask=groups["G1 ordinary"] & band
        )
        p1.append({"arm": arm, "g1_24_168h": res})
        show(f"{arm} vs K0 qs nra G1 24-168 h (gain = K0 minus arm)", res)
    print("=== tail coverage in G4 against the true series (share of readings outside the band)")
    coverage = []
    g4 = groups["G4 after"]
    for arm in ARMS:
        for target in TARGETS:
            if arm.endswith("_ADD") and target == "nra":
                continue
            actual = wide[f"actual_{target}"].to_numpy().astype(np.float64)
            for side, ind in (
                ("below q01", actual < wide[f"q01__{arm}"].to_numpy()),
                ("above q99", actual > wide[f"q99__{arm}"].to_numpy()),
            ):
                res = interval(
                    gain=ind[g4].astype(np.float64),
                    base=np.ones(g4.sum()),
                    weight=weight[g4],
                    units=units[g4],
                    rng=rng,
                )
                coverage.append({"arm": arm, "target": target, "side": side, **res})
                print(
                    f"{arm:20s} {target} {side}: share {100 * res['mean']:.2f}% ({100 * res['mean_lo']:.2f}, {100 * res['mean_hi']:.2f})"
                )
    out["p1_cost"] = p1
    out["coverage_g4"] = coverage
    # 6. P2: G1 change for the band arms against K0.
    print("=== P2 G1 change of the band arms against K0 (mean qs nra, gain = K0 minus arm)")
    for arm in ("BAND_TABLE_PLAIN", "BAND_TABLE_DISAGREE", "BAND_HEAD", "ASOF_WIDEN", "LEAD_DECAY"):
        show(
            arm,
            gain_of(base_arm="K0", arm=arm, metric="qs", target="nra", mask=groups["G1 ordinary"]),
        )
    # 7. P3: recovery of the PERFECT gain by LEAD_DECAY on G4 tw95 against the true series.
    print("=== P3 recovery ratio, G4 tw95 nra, (K0 - arm) / (K0 - PERFECT)")
    recovery = []
    for arm in NO_DETECTOR[1:]:
        k0, a, p = (
            col(wide, "tw95", "nra", "K0"),
            col(wide, "tw95", "nra", arm),
            col(wide, "tw95", "nra", "PERFECT"),
        )
        res = ratio_interval(
            num=(k0 - a)[g4], den=(k0 - p)[g4], weight=weight[g4], units=units[g4], rng=rng
        )
        recovery.append({"arm": arm, **res})
        print(f"{arm:20s} recovery {res['ratio']:.3f} ({res['lo']:.3f}, {res['hi']:.3f})")
    out["recovery"] = recovery
    # 7b. Held-out head: fitted on even-seed worlds only, scored on odd-seed worlds only.
    print("=== held-out head against the cross-fitted head, odd-seed worlds only")
    odd = ~half_a
    for arm in ("BAND_HEAD", HELDOUT):
        for metric, target, gname in (
            ("tw95", "nra", "G4 after"),
            ("tw95", "nra", "G5 near"),
            ("qs", "nra", "G1 ordinary"),
            ("cost0.5", "obs", "G5 near"),
        ):
            show(
                f"{arm:18s} vs K0 {metric} {target} {gname}",
                gain_of(
                    base_arm="K0", arm=arm, metric=metric, target=target, mask=groups[gname] & odd
                ),
            )
        actual = wide["actual_nra"].to_numpy().astype(np.float64)
        for side, ind in (
            ("below q01", actual < wide[f"q01__{arm}"].to_numpy()),
            ("above q99", actual > wide[f"q99__{arm}"].to_numpy()),
        ):
            m = g4 & odd
            res = interval(
                gain=ind[m].astype(np.float64),
                base=np.ones(m.sum()),
                weight=weight[m],
                units=units[m],
                rng=rng,
            )
            print(
                f"{arm:18s} G4 {side}: share {100 * res['mean']:.2f}% ({100 * res['mean_lo']:.2f}, {100 * res['mean_hi']:.2f})"
            )
    # 8. Checks: median minus mean forecast error per site.
    print("=== median-forecast error minus mean-forecast error, per site (points)")
    checks = []
    for site in sorted(set(sites)):
        for arm in ARMS:
            m = sites == site
            diff = float(
                np.nanmean(
                    col(wide, "ae_median", "nra", arm)[m] - col(wide, "ae_mean", "nra", arm)[m]
                )
            )
            checks.append({"site": site, "arm": arm, "diff": diff})
            print(f"{site:4s} {arm:20s} {diff:+.4f}")
    out["checks"] = checks
    out["counts"] = {
        "worlds": len(set(world.tolist())),
        "shifts": len(np.unique(units[units > 0])),
        "rows_by_phase": {p: int((phase == p).sum()) for p in sorted(set(phase))},
    }
    print(json.dumps(out["counts"], indent=2))
    (RESULTS / "report.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
