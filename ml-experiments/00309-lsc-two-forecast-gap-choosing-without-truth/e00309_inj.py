# ruff: noqa: E501, PLR0912, PLR0915
"""RULE, ORACLE, and the fixed forecasts on experiment 297's injected-world rows.

Reads experiment 297's score rows (K0, GAM_ONLY with its own bands, BAND_HEAD; ensemble weather, 44
worlds, six Primary substations), builds RULE (K0 while the 24-hour mean residual of measured power
against GAM_ONLY says the arrangement is abnormal, GAM_ONLY otherwise), RULE_GAP (the same rule on
the plain gap of K0 minus GAM_ONLY, as the comparator), ORACLE (the arm with the lower quantile
score on that row and target) and ORACLE_STATE (K0 while a shift is on, GAM_ONLY otherwise), scores
everything by row group and target, and prints the numbers the page quotes. Intervals resample whole
shifts (world and event) and whole calendar months for ordinary rows, 95%, 20,000 draws.

    uv run --no-sync python e00309_inj.py r24   # the primary RULE; "d" runs the gap comparator
"""

from __future__ import annotations

import json
import sys
from typing import Any, Final

import e00309_data as dat
import e00309_harness as h
import e00309_state as st
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SRC: Final = ML_EXPERIMENTS_DATA_DIR / "00297-lsc-need-a-detector-no-detector-arms-against-the-mask"
RESULTS: Final = dat.WORK_DIR / "results"
KEEP: Final[tuple[str, ...]] = (
    "qs_nra", "qs_obs", "ae_median_nra", "ae_median_obs", "ae_mean_nra", "ae_mean_obs",
    "tw95_nra", "tw95_obs", "q01", "q05", "q50", "q95", "q99",
)  # fmt: skip
BASE_ARMS: Final[tuple[str, ...]] = ("K0", "GAM_ONLY", "BAND_HEAD")
RULES: Final[tuple[str, ...]] = tuple(
    f"RULE_{score}_{k}" for score, ks in st.THRESHOLDS.items() for k in ks
)
TARGETS: Final[tuple[str, ...]] = ("nra", "obs")
PRICE_GBP_PER_MWH: Final[float] = 2000.0
BREACH_RATIO: Final[int] = 20
ORDINARY_WEIGHT: Final[float] = 4.0
N_DRAWS: Final[int] = 20_000
LEAD_MIN_HOURS: Final[float] = 12.0
LEAD_MAX_HOURS: Final[float] = 36.0
FRACTION: Final[float] = 0.5
METRICS: Final[tuple[str, ...]] = ("qs", "tw95", "ae_median", "ae_mean")
WINTER: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})
SIZE_BINS: Final = (("under 10%", 0.0, 0.10), ("10% to 25%", 0.10, 0.25), ("over 25%", 0.25, 1e9))
DURATION_BINS: Final = (
    ("under 6 h", 0.0, 6.0),
    ("6 to 24 h", 6.0, 24.0),
    ("over 1 day", 24.0, 1e9),
)
LEAD_BANDS: Final = (("0-12 h", 0.0, 12.0), ("12-36 h", 12.0, 36.0), ("36 h-7 d", 36.0, 1e9))


def load_wide() -> tuple[pl.DataFrame, dict[str, Any]]:
    """Experiment 297's rows with K0, GAM_ONLY and BAND_HEAD side by side."""
    constants = json.loads((SRC / "scores" / "site_constants.json").read_text())
    meta = [
        "site", "world", "period", "row", "position", "run", "power_fcst_init_time", "lead_hours",
        "event", "phase", "in_shift", "event_hours", "fraction", "onset_month", "actual_nra",
        "actual_obs",
    ]  # fmt: skip
    long1 = pl.concat([pl.read_parquet(p) for p in sorted((SRC / "scores").glob("D*_*.parquet"))])
    long2 = pl.concat([pl.read_parquet(p) for p in sorted((SRC / "scores2").glob("D*_*.parquet"))])
    wide = long1.filter(pl.col("arm") == "K0").select(meta)
    for long in (long1, long2):
        for arm in BASE_ARMS:
            part = long.filter(pl.col("arm") == arm)
            if part.height == 0:
                continue
            part = part.select(
                "site",
                "world",
                "row",
                *[pl.col(c).alias(f"{c}__{arm}") for c in KEEP if c in part.columns],
            )
            wide = wide.join(part, on=["site", "world", "row"], how="left", validate="1:1")
    return wide, constants


def add_states(*, wide: pl.DataFrame) -> pl.DataFrame:
    """Add one boolean column per RULE arm: is the row's forecast origin in the abnormal state."""
    stamp = (
        wide.select(
            (
                (
                    pl.col("power_fcst_init_time").dt.replace_time_zone("UTC")
                    - pl.lit(dat.GRID_START)
                ).dt.total_seconds()
                // 1800
            ).cast(pl.Int64)
        )
        .to_series()
        .to_numpy()
    )
    site, world = wide["site"].to_numpy(), wide["world"].to_numpy()
    out = {
        f"state_{score}_{k}": np.zeros(wide.height, dtype=bool)
        for score, ks in st.THRESHOLDS.items()
        for k in ks
    }
    for s_name, series in st.SITES.items():
        for seed in np.unique(world[site == s_name]):
            path = st.GAP_DIR / f"world_{seed}_{s_name}.npz"
            mask = (site == s_name) & (world == seed)
            if not path.exists():
                continue
            z = np.load(path)
            for score, ks in st.THRESHOLDS.items():
                sigma = st.sigma_of(series=series, score=score)
                for k in ks:
                    state = st.abnormal_state(gap=z[score], sigma=sigma, k=k)
                    out[f"state_{score}_{k}"][mask] = st.state_for_origin(
                        state=state, stamp=stamp[mask]
                    )
    return wide.with_columns([pl.Series(name, v) for name, v in out.items()])


def choose(*, wide: pl.DataFrame, name: str, use_k0: np.ndarray) -> list[pl.Series]:
    """Columns of arm `name` that copy K0's columns where `use_k0` and GAM_ONLY's elsewhere."""
    cols = []
    for c in KEEP:
        a, b = f"{c}__K0", f"{c}__GAM_ONLY"
        cols.append(
            pl.Series(
                f"{c}__{name}", np.where(use_k0, wide[a].to_numpy(), wide[b].to_numpy())
            ).cast(pl.Float32)
        )
    return cols


def add_arms(*, wide: pl.DataFrame) -> pl.DataFrame:
    """RULE at each threshold, ORACLE for each target, and ORACLE_STATE."""
    new = []
    for rule in RULES:
        new += choose(
            wide=wide, name=rule, use_k0=wide[f"state_{rule.removeprefix('RULE_')}"].to_numpy()
        )
    for target in TARGETS:
        better_k0 = wide[f"qs_{target}__K0"].to_numpy() < wide[f"qs_{target}__GAM_ONLY"].to_numpy()
        new += choose(wide=wide, name=f"ORACLE_{target.upper()}", use_k0=better_k0)
    phase = wide["phase"].to_numpy()
    new += choose(wide=wide, name="ORACLE_STATE", use_k0=np.isin(phase, ["BEFORE_ALARM", "DURING"]))
    return wide.with_columns(new)


def all_arms() -> tuple[str, ...]:
    """Every arm, in the order tables print them."""
    return (
        "GAM_ONLY", "K0", "BAND_HEAD", *RULES, "ORACLE_NRA", "ORACLE_OBS", "ORACLE_STATE",
    )  # fmt: skip


def add_cost(*, wide: pl.DataFrame, constants: dict[str, Any]) -> pl.DataFrame:
    """Node-only curtailment cost (GBP per half-hour) at the scaled limit 0.5, 12 to 36 hour leads."""
    cap = np.array([constants[s]["capacity_mw"] for s in wide["site"]]) / 100.0
    peak = np.array([constants[s]["max_before_points"] for s in wide["site"]]) * cap
    lead = wide["lead_hours"].to_numpy()
    inside = (lead >= LEAD_MIN_HOURS) & (lead < LEAD_MAX_HOURS)
    limit = FRACTION * peak
    new = []
    for target in TARGETS:
        actual = wide[f"actual_{target}"].to_numpy().astype(np.float64) * cap
        for arm in all_arms():
            q95 = wide[f"q95__{arm}"].to_numpy().astype(np.float64) * cap
            planned = np.maximum(q95 - limit, 0.0) * 0.5
            breach = np.maximum(actual - planned / 0.5 - limit, 0.0) * 0.5
            cost = PRICE_GBP_PER_MWH * (planned + BREACH_RATIO * breach)
            new.append(
                pl.Series(f"cost_{target}__{arm}", np.where(inside, cost, np.nan)).cast(pl.Float32)
            )
    return wide.with_columns(new)


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


def draws_of(
    *, units: np.ndarray, rng: np.random.Generator, arrays: tuple[np.ndarray, ...]
) -> tuple[np.ndarray, ...]:
    """Bootstrap sums of each array over whole units, resampling ordinary-month and shift units apart."""
    labels, codes = np.unique(units, return_inverse=True)
    per_unit = [np.bincount(codes, weights=a, minlength=len(labels)) for a in arrays]
    sums = [np.zeros(N_DRAWS) for _ in arrays]
    for stratum in (labels < 0, labels > 0):
        idx = np.flatnonzero(stratum)
        if not len(idx):
            continue
        chosen = idx[rng.integers(0, len(idx), size=(N_DRAWS, len(idx)))]
        for total, per in zip(sums, per_unit, strict=True):
            total[:] += per[chosen].sum(axis=1)
    return tuple(sums)


def col(wide: pl.DataFrame, metric: str, target: str, arm: str) -> np.ndarray:
    """One arm's metric column as float64."""
    name = f"{metric}_{target}__{arm}"
    return wide[name].to_numpy().astype(np.float64)


def gain_interval(
    *,
    base: np.ndarray,
    arm: np.ndarray,
    weight: np.ndarray,
    units: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Mean gain (base minus arm) per row and as a percentage of the base, with whole-unit resampling."""
    keep = np.isfinite(base) & np.isfinite(arm)
    base, arm, weight, units = base[keep], arm[keep], weight[keep], units[keep]
    if not len(base):
        return dict.fromkeys(("n", "mean", "lo", "hi", "pct", "pct_lo", "pct_hi"), float("nan"))
    gain = base - arm
    g, b, c = draws_of(units=units, rng=rng, arrays=(gain * weight, base * weight, weight))
    est_g, est_b, est_c = (
        float((gain * weight).sum()),
        float((base * weight).sum()),
        float(weight.sum()),
    )
    mean_lo, mean_hi = np.percentile(g / c, [2.5, 97.5])
    pct_lo, pct_hi = np.percentile(100.0 * g / b, [2.5, 97.5])
    return {
        "n": est_c, "mean": est_g / est_c, "lo": float(mean_lo), "hi": float(mean_hi),
        "pct": 100.0 * est_g / est_b, "pct_lo": float(pct_lo), "pct_hi": float(pct_hi),
    }  # fmt: skip


def recovery_interval(
    *,
    fixed: np.ndarray,
    rule: np.ndarray,
    oracle: np.ndarray,
    weight: np.ndarray,
    units: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    """(fixed minus RULE) / (fixed minus ORACLE) summed over rows, resampling whole units."""
    keep = np.isfinite(fixed) & np.isfinite(rule) & np.isfinite(oracle)
    fixed, rule, oracle, weight, units = (
        fixed[keep],
        rule[keep],
        oracle[keep],
        weight[keep],
        units[keep],
    )
    num, den = draws_of(
        units=units, rng=rng, arrays=((fixed - rule) * weight, (fixed - oracle) * weight)
    )
    est = float(((fixed - rule) * weight).sum() / ((fixed - oracle) * weight).sum())
    lo, hi = np.percentile(num / den, [2.5, 97.5])
    return {"ratio": est, "lo": float(lo), "hi": float(hi), "n": float(weight.sum())}


def group_masks(*, wide: pl.DataFrame) -> dict[str, np.ndarray]:
    """The row groups G0 to G5, plus the three shift phases pooled."""
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
    """The G6 strata: size, duration, season, lead, substation."""
    fraction, hours = wide["fraction"].to_numpy(), wide["event_hours"].to_numpy()
    lead, month = wide["lead_hours"].to_numpy(), wide["onset_month"].to_numpy()
    sites = wide["site"].to_numpy()
    return {
        "size": {n: (fraction >= lo) & (fraction < hi) for n, lo, hi in SIZE_BINS},
        "duration": {n: (hours >= lo) & (hours < hi) for n, lo, hi in DURATION_BINS},
        "season": {
            "November to March": np.isin(month, list(WINTER)),
            "April to October": ~np.isin(month, list(WINTER)) & (month > 0),
        },
        "lead": {n: (lead >= lo) & (lead < hi) for n, lo, hi in LEAD_BANDS},
        "substation": {s: sites == s for s in sorted(set(sites))},
    }


def show(tag: str, res: dict[str, float]) -> None:
    """Print one interval result."""
    print(
        f"{tag}: n={res['n']:.0f} mean {res['mean']:+.4f} ({res['lo']:+.4f}, {res['hi']:+.4f}) pct {res['pct']:+.2f} ({res['pct_lo']:+.2f}, {res['pct_hi']:+.2f})"
    )


def main(*, score: str) -> None:
    """Compute and print every number the page quotes, with RULE reading `score` (r24 or d)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    wide, constants = load_wide()
    wide = add_states(wide=wide)
    wide = add_arms(wide=wide)
    wide = add_cost(wide=wide, constants=constants)
    rng = np.random.default_rng(309)
    phase = wide["phase"].to_numpy()
    weight = np.where(phase == "ORDINARY", ORDINARY_WEIGHT, 1.0)
    units = units_of(wide=wide)
    groups = group_masks(wide=wide)
    init_month = wide["power_fcst_init_time"].dt.month().to_numpy()
    odd = np.isin(init_month, list(h.TUNING_MONTHS))
    even = ~odd
    out: dict[str, Any] = {"n_rows": wide.height}
    print("rows", wide.height, "shift units", len(np.unique(units[units > 0])))
    print("share of rows in the abnormal state, by phase:")
    for name in RULES:
        s = wide[f"state_{name.removeprefix('RULE_')}"].to_numpy()
        print(name, {p: round(float(s[phase == p].mean()), 3) for p in sorted(set(phase))})
    # Absolute scores, even months.
    print("=== absolute mean scores, even months (points; cost GBP per scored half-hour)")
    absolute = []
    for arm in all_arms():
        for gname, mask in groups.items():
            row: dict[str, Any] = {"arm": arm, "group": gname}
            m = mask & even
            w = weight[m]
            for target in TARGETS:
                for metric in (*METRICS, "cost"):
                    v = col(wide, metric, target, arm)[m]
                    ok = np.isfinite(v)
                    row[f"{metric}_{target}"] = (
                        float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else float("nan")
                    )
            absolute.append(row)
            print(arm, gname, {k_: round(v_, 4) for k_, v_ in row.items() if isinstance(v_, float)})
    out["absolute_even"] = absolute
    # Select k on odd months: lowest pooled tw95 obs in G5.
    sel = {}
    for k in st.THRESHOLDS[score]:
        v = col(wide, "tw95", "obs", f"RULE_{score}_{k}")
        m = groups["G5 near"] & odd & np.isfinite(v)
        sel[k] = float(np.sum(v[m] * weight[m]) / np.sum(weight[m]))
    best_k = min(sel, key=lambda x: sel[x])
    print("odd-month G5 tw95 obs by k:", {k: round(v, 4) for k, v in sel.items()}, "chosen", best_k)
    out["selected_k"] = best_k
    out["selection"] = sel
    rule = f"RULE_{score}_{best_k}"
    # Recovery and G1 harm, even months, both targets.
    print("=== recovery of ORACLE's gain over the better fixed choice (even months)")
    recovery = []
    for target, oracle_arm in (("obs", "ORACLE_OBS"), ("nra", "ORACLE_NRA")):
        for gname in (
            "G5 near",
            "Gs shift rows",
            "G2 before alarm",
            "G3 during",
            "G4 after",
            "G0 all rows",
        ):
            mask = groups[gname] & even
            for metric in ("tw95", "qs", "cost") if target == "obs" else ("tw95", "qs"):
                scores = {}
                for arm in ("GAM_ONLY", "K0"):
                    v = col(wide, metric, target, arm)
                    mm = mask & np.isfinite(v)
                    scores[arm] = (
                        float(np.sum(v[mm] * weight[mm]) / np.sum(weight[mm]))
                        if mm.any()
                        else float("inf")
                    )
                fixed = min(scores, key=lambda x: scores[x])
                rec = recovery_interval(
                    fixed=col(wide, metric, target, fixed)[mask],
                    rule=col(wide, metric, target, rule)[mask],
                    oracle=col(wide, metric, target, oracle_arm)[mask],
                    weight=weight[mask],
                    units=units[mask],
                    rng=rng,
                )
                gain = gain_interval(
                    base=col(wide, metric, target, fixed)[mask],
                    arm=col(wide, metric, target, rule)[mask],
                    weight=weight[mask],
                    units=units[mask],
                    rng=rng,
                )
                recovery.append(
                    {
                        "target": target,
                        "group": gname,
                        "metric": metric,
                        "fixed": fixed,
                        "rule": rule,
                        **rec,
                        "gain": gain,
                    }
                )
                print(
                    f"{target} {gname:16s} {metric:5s} fixed={fixed:9s} recovery {rec['ratio']:+.3f} ({rec['lo']:+.3f}, {rec['hi']:+.3f}); gain over fixed {gain['mean']:+.4f} ({gain['lo']:+.4f}, {gain['hi']:+.4f}) {gain['pct']:+.1f}% n={rec['n']:.0f}"
                )
    out["recovery"] = recovery
    print(
        "=== G1 harm of RULE against each fixed arm (mean qs, gain = fixed minus RULE), even months"
    )
    harm = []
    for target in TARGETS:
        for fixed in ("GAM_ONLY", "K0", "BAND_HEAD"):
            for gname in ("G1 ordinary",):
                mask = groups[gname] & even
                res = gain_interval(
                    base=col(wide, "qs", target, fixed)[mask],
                    arm=col(wide, "qs", target, rule)[mask],
                    weight=weight[mask],
                    units=units[mask],
                    rng=rng,
                )
                harm.append({"target": target, "fixed": fixed, "group": gname, **res})
                show(f"{target} qs {gname} {fixed} minus {rule}", res)
    out["harm"] = harm
    # Every arm against GAM_ONLY and against K0, even months, G5 and G1, headline metrics.
    print("=== gain of each arm over each fixed arm, even months")
    versus = []
    for target in TARGETS:
        for base in ("GAM_ONLY", "K0"):
            for arm in all_arms():
                if arm == base:
                    continue
                for gname in ("G5 near", "Gs shift rows", "G4 after", "G1 ordinary"):
                    for metric in ("tw95", "qs", "cost") if target == "obs" else ("tw95", "qs"):
                        mask = groups[gname] & even
                        res = gain_interval(
                            base=col(wide, metric, target, base)[mask],
                            arm=col(wide, metric, target, arm)[mask],
                            weight=weight[mask],
                            units=units[mask],
                            rng=rng,
                        )
                        versus.append(
                            {
                                "target": target,
                                "base": base,
                                "arm": arm,
                                "group": gname,
                                "metric": metric,
                                **res,
                            }
                        )
    out["versus"] = versus
    # G6 strata for RULE and fixed arms (tw95 obs, G5, even months).
    print(
        "=== strata inside G5, even months: tw95 obs gain of RULE over the better fixed arm of that stratum"
    )
    strata_out = []
    for dim, parts in strata_masks(wide=wide).items():
        for name, sm in parts.items():
            mask = groups["G5 near"] & even & sm
            if mask.sum() < 50:
                continue
            scores = {}
            for arm in ("GAM_ONLY", "K0"):
                v = col(wide, "tw95", "obs", arm)
                mm = mask & np.isfinite(v)
                scores[arm] = float(np.sum(v[mm] * weight[mm]) / np.sum(weight[mm]))
            fixed = min(scores, key=lambda x: scores[x])
            rec = recovery_interval(
                fixed=col(wide, "tw95", "obs", fixed)[mask],
                rule=col(wide, "tw95", "obs", rule)[mask],
                oracle=col(wide, "tw95", "obs", "ORACLE_OBS")[mask],
                weight=weight[mask],
                units=units[mask],
                rng=rng,
            )
            strata_out.append(
                {"dim": dim, "name": name, "fixed": fixed, **rec, "n_rows": int(mask.sum())}
            )
            print(
                f"{dim:10s} {name:18s} fixed={fixed:9s} recovery {rec['ratio']:+.3f} ({rec['lo']:+.3f}, {rec['hi']:+.3f}) rows={mask.sum()}"
            )
    out["strata"] = strata_out
    # Tail coverage in G4 and G5 (share above q99), even months.
    print("=== share of readings above q99 (G4 and G5), even months")
    coverage = []
    for target in TARGETS:
        actual = wide[f"actual_{target}"].to_numpy().astype(np.float64)
        for arm in all_arms():
            for gname in ("G4 after", "G5 near", "G1 ordinary"):
                m = groups[gname] & even
                ind = (actual > wide[f"q99__{arm}"].to_numpy())[m].astype(np.float64)
                res = gain_interval(
                    base=ind, arm=np.zeros(len(ind)), weight=weight[m], units=units[m], rng=rng
                )
                coverage.append({"target": target, "arm": arm, "group": gname, **res})
                print(
                    f"{target} {arm:13s} {gname:12s} above q99 {100 * res['mean']:.2f}% ({100 * res['lo']:.2f}, {100 * res['hi']:.2f})"
                )
    out["coverage"] = coverage
    # Checks: median minus mean error per site.
    print("=== median-forecast error minus mean-forecast error, per site (points), target nra")
    checks = []
    sites = wide["site"].to_numpy()
    for site in sorted(set(sites)):
        for arm in ("GAM_ONLY", "K0", "BAND_HEAD", rule):
            m = sites == site
            diff = float(
                np.nanmean(
                    col(wide, "ae_median", "nra", arm)[m] - col(wide, "ae_mean", "nra", arm)[m]
                )
            )
            checks.append({"site": site, "arm": arm, "diff": diff})
            print(f"{site:4s} {arm:13s} {diff:+.4f}")
    out["checks"] = checks
    # Leave-one-site-out for the headline recovery.
    print("=== leave one site out: recovery, tw95 obs, G5, even months")
    loo = []
    for site in sorted(set(sites)):
        mask = groups["G5 near"] & even & (sites != site)
        scores = {}
        for arm in ("GAM_ONLY", "K0"):
            v = col(wide, "tw95", "obs", arm)
            mm = mask & np.isfinite(v)
            scores[arm] = float(np.sum(v[mm] * weight[mm]) / np.sum(weight[mm]))
        fixed = min(scores, key=lambda x: scores[x])
        rec = recovery_interval(
            fixed=col(wide, "tw95", "obs", fixed)[mask],
            rule=col(wide, "tw95", "obs", rule)[mask],
            oracle=col(wide, "tw95", "obs", "ORACLE_OBS")[mask],
            weight=weight[mask],
            units=units[mask],
            rng=rng,
        )
        loo.append({"left_out": site, "fixed": fixed, **rec})
        print(
            f"without {site}: fixed={fixed} recovery {rec['ratio']:+.3f} ({rec['lo']:+.3f}, {rec['hi']:+.3f})"
        )
    out["leave_one_out"] = loo
    (RESULTS / f"injected_{score}.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main(score=sys.argv[1])
