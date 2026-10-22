# ruff: noqa: E501, PLR0912, PLR0915, T201
"""Print every number the page quotes and write the tables the charts read.

Reads the injected-world scores of `e00208_injected.py`, the real-record scores of `e00208_real.py`,
and experiment 203's stored ordinary-day scores.

    OMP_NUM_THREADS=1 uv run --no-sync python e00208_report.py
"""

from __future__ import annotations

import glob
import json
from typing import Final

import e00208_alarms as al
import e00208_fed as fed
import numpy as np
import polars as pl
from e00208_common import WORK_DIR
from studies.sources import ML_EXPERIMENTS_DATA_DIR
from studies.tail_calibration import pinball

SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16", "D9")
PHASES: Final[tuple[str, ...]] = ("BEFORE_ALARM", "DURING", "AFTER")
N_DRAWS: Final[int] = 20_000
CHUNK: Final[int] = 1_000
E203_DIR: Final = ML_EXPERIMENTS_DATA_DIR / "00203-switching-replay-learned-score-in-masked-forecast"
REPORT_DIR: Final = WORK_DIR / "report"
WINTER: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})
SIZE_BANDS: Final[dict[str, tuple[float, float]]] = {
    "under 10%": (0.0, 0.10),
    "10% to 25%": (0.10, 0.25),
    "25% or more": (0.25, 9.0),
}
DURATION_BANDS: Final[dict[str, tuple[float, float]]] = {
    "2 to 6 h": (0.0, 6.0),
    "6 to 24 h": (6.0, 24.0),
    "over 1 d": (24.0, 1e9),
}
LEAD_BANDS: Final[dict[str, tuple[float, float]]] = {
    "0 to 12 h": (0.0, 12.0),
    "12 to 36 h": (12.0, 36.0),
    "36 h to 7 d": (36.0, 1e9),
}
SWEEPS: Final[dict[str, list[str]]] = {
    "recall": ["RECALL_0.25|MASK", "RECALL_0.5|MASK", "RECALL_0.75|MASK", "PERFECT|MASK"],
    "onset delay (half-hours)": ["PERFECT|MASK", "DELAY_6|MASK", "DELAY_12|MASK", "DELAY_24|MASK", "DELAY_48|MASK", "DELAYX_96|MASK", "DELAYX_144|MASK", "DELAYX_192|MASK"],
    "MW error (subtraction)": ["PERFECT|SUB", "BLUR_0.25|SUB", "BLUR_0.5|SUB"],
    "repair of RO-NC misses": ["DET|MASK", "REPAIR_0.25|MASK", "REPAIR_0.5|MASK", "REPAIR_0.75|MASK", "REPAIR_1.0|MASK"],
}
QCOLS: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")


def fmt(t: tuple[float, float, float], digits: int = 3) -> str:
    """Format an estimate with its interval."""
    return f"{t[0]:+.{digits}f} ({t[1]:+.{digits}f} to {t[2]:+.{digits}f})"


def boot_matrix(*, sums: np.ndarray, counts: np.ndarray, seed: int) -> np.ndarray:
    """Resample units whole: returns draws of each column's sum over the sampled units' count.

    Args:
        sums: Per-unit totals, shape (units, columns).
        counts: Per-unit row counts, shape (units,).
        seed: The seed.

    Returns:
        Array of shape (N_DRAWS, columns): the ratio-of-sums mean of each column.
    """
    rng = np.random.default_rng(seed)
    n = len(counts)
    out = np.empty((N_DRAWS, sums.shape[1]))
    for lo in range(0, N_DRAWS, CHUNK):
        picks = rng.integers(0, n, size=(CHUNK, n))
        d = counts[picks].sum(axis=1)
        out[lo : lo + CHUNK] = sums[picks].sum(axis=1) / d[:, None]
    return out


def interval(draws: np.ndarray) -> tuple[float, float]:
    """The 95% percentile interval of draws."""
    return float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def load_injected() -> tuple[pl.DataFrame, pl.DataFrame]:
    """The wide per-row table of quantile scores (one column per arm) and the tail table."""
    wides, tails = [], []
    keep_tail = {"K0|NONE", "CLEAN|NONE", "PERFECT|MASK", "PERFECT_LAG|MASK", "DET|MASK"}
    for path in sorted(glob.glob(str(WORK_DIR / "injected" / "*.parquet"))):
        f = pl.read_parquet(path)
        key = ["site", "world", "row", "period", "event", "phase", "lead_hours", "event_hours", "fraction", "onset_month"]
        wides.append(
            f.pivot(on="arm", index=key, values="qs").with_columns(pl.col(pl.Float64).cast(pl.Float32))
        )
        tails.append(
            f.filter(pl.col("arm").is_in(keep_tail)).select(
                *key, "arm", "actual", "q01", "q05", "q95", "q99", "err_median", "err_mean"
            )
        )
    wide = pl.concat(wides, how="diagonal")
    extras = []
    for path in sorted(glob.glob(str(WORK_DIR / "injected_extra" / "*.parquet"))):
        f = pl.read_parquet(path).filter(pl.col("arm") != "K0|NONE")
        extras.append(f.pivot(on="arm", index=["site", "world", "row"], values="qs").with_columns(pl.col(pl.Float64).cast(pl.Float32)))
    if extras:
        wide = wide.join(pl.concat(extras, how="diagonal"), on=["site", "world", "row"], how="left")
    return wide, pl.concat(tails)


def unit_of(frame: pl.DataFrame) -> np.ndarray:
    """A unit label per row: the injected transfer (world and event)."""
    return (frame["world"].cast(pl.Int64) * 1_000_000 + frame["event"]).to_numpy()


def gains_by_unit(*, frame: pl.DataFrame, arms: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit sums of (K0 score minus arm score) and row counts."""
    units = unit_of(frame)
    labels, inverse = np.unique(units, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    sums = np.empty((len(labels), len(arms)))
    k0 = frame["K0|NONE"].to_numpy().astype(float)
    for i, arm in enumerate(arms):
        gain = k0 - frame[arm].to_numpy().astype(float)
        sums[:, i] = np.bincount(inverse, weights=np.nan_to_num(gain), minlength=len(labels))
    return sums, counts


def stratum_table(*, frame: pl.DataFrame, arms: list[str], reference: dict[str, str], seed: int) -> pl.DataFrame:
    """Gain and share of the perfect gain for each arm on a set of rows, with intervals."""
    sums, counts = gains_by_unit(frame=frame, arms=arms)
    if len(counts) < 2:
        return pl.DataFrame()
    draws = boot_matrix(sums=sums, counts=counts, seed=seed)
    est = sums.sum(axis=0) / counts.sum()
    rows = []
    for i, arm in enumerate(arms):
        lo, hi = interval(draws[:, i])
        ref = arms.index(reference.get(arm, "PERFECT|MASK")) if reference.get(arm, "PERFECT|MASK") in arms else None
        share = share_lo = share_hi = dropped = None
        if ref is not None and est[ref] > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                s = draws[:, i] / draws[:, ref]
            share = float(est[i] / est[ref])
            share_lo, share_hi = interval(np.where(draws[:, ref] > 0, s, np.nan))
            dropped = float((draws[:, ref] <= 0).mean())
        rows.append(
            {"arm": arm, "gain": float(est[i]), "gain_lo": lo, "gain_hi": hi, "share": share, "share_lo": share_lo, "share_hi": share_hi, "share_draws_dropped": dropped, "units": len(counts), "rows": int(counts.sum())}
        )
    return pl.DataFrame(rows)


def main() -> None:
    """Compute and print the injected-world and real-record results."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(220)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    wide, tails = load_injected()
    arms = [c for c in wide.columns if "|" in c]
    gain_arms = [a for a in arms if a not in {"K0|NONE"}]
    reference = {a: ("PERFECT|SUB" if a.endswith("|SUB") else "PERFECT|MASK") for a in gain_arms}
    reference["PERFECT_LAG|SUB"] = "PERFECT|SUB"
    print("injected rows:", wide.height, "units:", len(np.unique(unit_of(wide))), "worlds:", wide["world"].n_unique())
    print(wide.group_by("phase").agg(rows=pl.len(), units=pl.struct("world", "event").n_unique()))

    # ---- control: CLEAN equals K0 before the first injected stamp is asserted in the job ----
    strata: dict[str, pl.Expr] = {"all transfers": pl.lit(True)}
    strata["headline (25% or more, over 1 d)"] = (pl.col("fraction") >= 0.25) & (pl.col("event_hours") > 24)
    for name, (lo, hi) in SIZE_BANDS.items():
        strata[f"size {name}"] = (pl.col("fraction") >= lo) & (pl.col("fraction") < hi)
    for name, (lo, hi) in DURATION_BANDS.items():
        strata[f"duration {name}"] = (pl.col("event_hours") > lo) & (pl.col("event_hours") <= hi)
    strata["season winter (Nov to Mar)"] = pl.col("onset_month").is_in(list(WINTER))
    strata["season summer (Apr to Oct)"] = ~pl.col("onset_month").is_in(list(WINTER))
    for name, (lo, hi) in LEAD_BANDS.items():
        strata[f"lead {name}"] = (pl.col("lead_hours") >= lo) & (pl.col("lead_hours") < hi)
    for site in SITES:
        strata[f"substation {site}"] = pl.col("site") == site

    tables = []
    for phase in PHASES:
        for name, expr in strata.items():
            sub = wide.filter((pl.col("phase") == phase) & expr)
            t = stratum_table(frame=sub, arms=gain_arms + ["K0|NONE"], reference=reference, seed=208 + len(tables))
            if t.height:
                tables.append(t.filter(pl.col("arm") != "K0|NONE").with_columns(phase=pl.lit(phase), stratum=pl.lit(name)))
    result = pl.concat(tables)
    result.write_parquet(REPORT_DIR / "injected_strata.parquet")

    show = ["PERFECT|MASK", "PERFECT_LAG|MASK", "DET|MASK", "DET_NOFALSE|MASK", "CLEAN|NONE", "GAM_ONLY|NONE", "PERFECT|SUB", "PERFECT_LAG|SUB", "DET|SUB"]
    print("\n== Headline: gain over K0 (points; positive is better) and share of the PERFECT|MASK gain, by phase ==")
    for stratum in ("all transfers", "headline (25% or more, over 1 d)"):
        print(f"\n-- {stratum} --")
        print(
            result.filter((pl.col("stratum") == stratum) & pl.col("arm").is_in(show))
            .select("phase", "arm", "gain", "gain_lo", "gain_hi", "share", "share_lo", "share_hi", "units", "rows")
            .sort("phase", "arm")
        )

    print("\n== Quality sweeps: share of PERFECT gain recovered, headline stratum and all transfers ==")
    for stratum in ("headline (25% or more, over 1 d)", "all transfers"):
        for axis, members in SWEEPS.items():
            print(f"\n-- {stratum} / {axis} --")
            print(
                result.filter((pl.col("stratum") == stratum) & pl.col("arm").is_in(members))
                .select("phase", "arm", "gain", "gain_lo", "gain_hi", "share", "share_lo", "share_hi")
                .sort("phase", "arm")
            )

    print("\n== Strata: PERFECT|MASK and DET|MASK gain by phase ==")
    print(
        result.filter(pl.col("arm").is_in(["PERFECT|MASK", "DET|MASK"]))
        .select("stratum", "phase", "arm", "gain", "gain_lo", "gain_hi", "share", "units")
        .sort("stratum", "phase", "arm")
    )

    # ---- leave one substation out ----
    print("\n== Leave one substation out: AFTER phase, headline stratum, share recovered ==")
    head = (pl.col("fraction") >= 0.25) & (pl.col("event_hours") > 24)
    for site in (None, *SITES):
        sub = wide.filter((pl.col("phase") == "AFTER") & head & ((pl.col("site") != site) if site else pl.lit(True)))
        t = stratum_table(frame=sub, arms=["K0|NONE", "PERFECT|MASK", "DET|MASK", "PERFECT_LAG|MASK"], reference=reference, seed=999)
        print(site or "none left out", t.filter(pl.col("arm") != "K0|NONE").select("arm", "gain", "share", "units").to_dicts())

    # ---- whole-year pooled gain of perfect knowledge ----
    n_rows = 0
    for site in SITES:
        d = fed.get_site(site)
        n_rows += int((np.isfinite(d.actual) & np.isin(d.run_period, ["scored", "replicate"])[d.run_of_row]).sum())
    worlds = wide["world"].n_unique()
    total = {a: float((wide["K0|NONE"].cast(pl.Float64) - wide[a].cast(pl.Float64)).sum()) for a in ("PERFECT|MASK", "DET|MASK", "PERFECT_LAG|MASK")}
    print("\nWhole-year rows (six substations, scored+replicate periods, per world):", n_rows)
    print("Whole-year pooled gain (points, over all rows of all worlds):", {a: v / (worlds * n_rows) for a, v in total.items()})

    # ---- detector recall in the injected worlds ----
    print("\n== RO-NC recall in the injected worlds (share of injected transfers with an interval starting within 2 hours of the onset) ==")
    hits = tot = 0
    by_size: dict[str, list[int]] = {}
    for path in sorted(glob.glob(str(ML_EXPERIMENTS_DATA_DIR / "00141-switching-detector-fed-into-gam-k" / "worlds" / "world_*.npz"))):
        seed = int(path.split("_")[-1].split(".")[0])
        iv = pl.read_parquet(path.replace(".npz", ".parquet")).filter(pl.col("setting") == "R0_M48")
        with np.load(path) as z:
            ev = z["events"]
            for site, series in fed.SERIES_OF_SITE.items():
                delta = z[f"delta_{series}"]
                mine = iv.filter(pl.col("series") == series)["onset"].to_numpy()
                for row in ev:
                    o, e = int(row[2]), int(row[3])
                    if not np.any(np.nan_to_num(delta[o:e])):
                        continue
                    found = bool(np.any(np.abs(mine - o) <= 4))
                    hits += found
                    tot += 1
                    band = "25% or more" if row[4] >= 0.25 else ("10% to 25%" if row[4] >= 0.10 else "under 10%")
                    by_size.setdefault(band, []).append(int(found))
    print(f"recall {hits}/{tot} = {hits / tot:.3f}", {k: (sum(v), len(v)) for k, v in by_size.items()})

    # ---- explanation test: why the injected gain is smaller than the real-record gain ----
    print("\n== Explanation test: size of the injected MW against the logged transfers' MW ==")
    pts = []
    for path in sorted(glob.glob(str(ML_EXPERIMENTS_DATA_DIR / "00141-switching-detector-fed-into-gam-k" / "worlds" / "world_*.npz"))):
        with np.load(path) as z:
            for row in z["events"]:
                if row[4] < 0.25 or (row[3] - row[2]) / 2 <= 24:
                    continue
                for site, series in fed.SERIES_OF_SITE.items():
                    d = np.nan_to_num(z[f"delta_{series}"][int(row[2]) : int(row[3])])
                    if np.any(d):
                        pts.append(100.0 * float(np.mean(np.abs(d))) / fed.CAPACITY_MW[site])
    print("injected headline transfers: mean |MW| in points of capacity per substation", float(np.mean(pts)), "median", float(np.median(pts)), "n", len(pts))
    lab = pl.read_parquet(ML_EXPERIMENTS_DATA_DIR / "00041-sw2-0a-labeller-gate" / "real" / "labelled_set.parquet").filter(~pl.col("is_sentinel") & (pl.col("kind") == "real_operation") & pl.col("moved_mw_p50").is_not_null())
    real_pts = []
    for r in lab.iter_rows(named=True):
        for site, series in fed.SERIES_OF_SITE.items():
            if str(series) in {str(r["donor"]), str(r["side_a"]), str(r["side_b"])}:
                real_pts.append(100.0 * abs(float(r["moved_mw_p50"])) / fed.CAPACITY_MW[site])
    print("logged real transfers touching the six substations: median |MW| in points", float(np.median(real_pts)), "mean", float(np.mean(real_pts)), "n", len(real_pts))
    k0_after = wide.filter((pl.col("phase") == "AFTER") & head)
    print("injected headline AFTER rows: K0 mean score", float(k0_after["K0|NONE"].cast(pl.Float64).mean()), "CLEAN mean score", float(k0_after["CLEAN|NONE"].cast(pl.Float64).mean()))

    # ---- tails (headline stratum, AFTER phase) ----
    print("\n== Tail coverage and pinball loss, AFTER phase, headline stratum, injected worlds ==")
    sub = tails.filter((pl.col("phase") == "AFTER") & (pl.col("fraction") >= 0.25) & (pl.col("event_hours") > 24))
    tail_rows = []
    for arm in sub["arm"].unique().sort():
        a = sub.filter(pl.col("arm") == arm)
        actual = a["actual"].to_numpy()
        for col, level in (("q01", 0.01), ("q05", 0.05), ("q95", 0.95), ("q99", 0.99)):
            q = a[col].to_numpy()
            tail_rows.append({"arm": arm, "level": level, "coverage": float((actual <= q).mean()), "pinball": float(pinball(actual=actual, quantile=q, level=level).mean()), "rows": len(actual)})
    tail_table = pl.DataFrame(tail_rows)
    print(tail_table.sort("arm", "level"))
    tail_table.write_parquet(REPORT_DIR / "injected_tails.parquet")
    print("\n== Median minus mean forecast, per substation (points), AFTER phase ==")
    mm = tails.filter((pl.col("phase") == "AFTER") & pl.col("arm").is_in(["K0|NONE", "PERFECT|MASK"])).group_by("site", "arm").agg(median_minus_mean=(pl.col("err_median") - pl.col("err_mean")).mean())
    print(mm.sort("site", "arm"))
    print("\n== Error of the mean forecast (mean absolute, points) beside the score gain, headline stratum ==")
    for phase in PHASES:
        s = tails.filter((pl.col("phase") == phase) & (pl.col("fraction") >= 0.25) & (pl.col("event_hours") > 24)).group_by("arm").agg(mae_mean=pl.col("err_mean").abs().mean(), bias=pl.col("err_mean").mean())
        print(phase, s.sort("arm").to_dicts())

    # ================= real record =================
    print("\n\n================ REAL RECORD ================")
    real_dir = WORK_DIR / "real"
    scores = pl.concat([pl.read_parquet(real_dir / f"event_scores_{s}.parquet").with_columns(site=pl.lit(s)) for s in SITES if (real_dir / f"event_scores_{s}.parquet").exists()])
    rows = pl.concat([pl.read_parquet(real_dir / f"event_rows_{s}.parquet") for s in SITES if (real_dir / f"event_rows_{s}.parquet").exists()])
    fold_of = al.card_folds()
    rows = rows.with_columns(fold=pl.col("card").replace_strict(fold_of, default=-1, return_dtype=pl.Int64))
    # an arm without folds has fold -1; a learned arm uses the fold that holds out the card's Primary pair
    missing = rows.filter(pl.col("fold") == -1)["card"].unique().to_list()
    if missing:
        raise AssertionError(f"cards without a fold: {missing}")
    j = rows.join(scores, on=["site", "row"], how="inner") if "site" in scores.columns else rows.join(scores, on="row")
    j = j.filter((pl.col("fold_right") == -1) | (pl.col("fold_right") == pl.col("fold"))) if "fold_right" in j.columns else j
    wide_r = j.pivot(on="arm", index=["site", "card", "row", "phase", "in_163_after", "cutoff_ok", "hours", "onset_month", "lead_hours"], values="qs")
    arms_r = ["LOGGED", "RONC", "PAIR_NEW", "PAIR_GATED", "LEARNED_R", "LEARNED_F", "RONC_195"]
    print("windows (card x substation):", wide_r.select(pl.struct("site", "card").n_unique()).item(), "cards:", wide_r["card"].n_unique())
    ctrl = wide_r.filter(pl.col("in_163_after"))
    wide_r = wide_r.filter(pl.col("cutoff_ok"))
    print("positive control, rows in experiment 163's AFTER set (expect K0 10.129, PAIR_NEW 6.083 for the 11 transfers):", {a: float(ctrl[a].mean()) for a in ("K0", "PAIR_NEW", "LOGGED")}, "rows", ctrl.height)
    real_rows = []
    for phase in (*PHASES, "ALL"):
        for scope in ("all", "over 1 d"):
            sub = wide_r.filter(((pl.col("phase") == phase) if phase != "ALL" else pl.lit(True)) & ((pl.col("hours") > 24) if scope == "over 1 d" else pl.lit(True)))
            labels, inv = np.unique(sub["card"].to_numpy() if sub.height else np.array([]), return_inverse=True)
            if len(labels) < 2:
                continue
            counts = np.bincount(inv).astype(float)
            sums = np.stack([np.bincount(inv, weights=np.nan_to_num((sub["K0"] - sub[a]).to_numpy().astype(float)), minlength=len(labels)) for a in arms_r], axis=1)
            draws = boot_matrix(sums=sums, counts=counts, seed=77)
            est = sums.sum(axis=0) / counts.sum()
            for i, a in enumerate(arms_r):
                lo, hi = interval(draws[:, i])
                ref = arms_r.index("LOGGED")
                share = float(est[i] / est[ref]) if est[ref] > 0 else None
                real_rows.append({"phase": phase, "scope": scope, "arm": a, "gain": float(est[i]), "gain_lo": lo, "gain_hi": hi, "share_of_logged": share, "cards": len(labels), "rows": int(counts.sum())})
    real = pl.DataFrame(real_rows)
    real.write_parquet(REPORT_DIR / "real_gain.parquet")
    print(real.sort("phase", "scope", "arm"))
    print("K0 mean score by phase (points):", wide_r.group_by("phase").agg(pl.col("K0").mean()).to_dicts())

    # ---- real record: gain pooled over ordinary rows plus experiment 163's AFTER rows ----
    after163 = j.filter(pl.col("phase").is_in(["AFTER", "TAIL"]) | pl.col("in_163_after"))
    ctrl_rows = wide_r.filter(pl.col("in_163_after") | True).filter(pl.col("in_163_after"))
    pooled = {a: float((ctrl_rows["K0"] - ctrl_rows[a]).sum() / (1_579_711 + ctrl_rows.height)) for a in arms_r}
    print("\nReal record, gain pooled over ordinary rows plus experiment 163's AFTER rows (points; the AFTER rows are in the scored and replicate months only):", pooled, "AFTER rows", ctrl_rows.height)
    del after163

    # ---- real-record tails, AFTER ----
    t_real = j.filter(pl.col("phase") == "AFTER")
    for arm in ("K0", "LOGGED", "PAIR_NEW", "LEARNED_R", "RONC"):
        a = t_real.filter(pl.col("arm") == arm)
        actual = a["actual"].to_numpy()
        print(arm, {c: (round(float((actual <= a[c].to_numpy()).mean()), 3), round(float(pinball(actual=actual, quantile=a[c].to_numpy(), level=lv).mean()), 3)) for c, lv in (("q01", 0.01), ("q05", 0.05), ("q95", 0.95), ("q99", 0.99))})

    # ---- ordinary days: stored scores of experiment 203 ----
    print("\n== Ordinary-day harm (arm minus K0, points over all ordinary rows) from experiment 203's stored scores ==")
    ordinary = pl.concat([pl.read_parquet(E203_DIR / f"meta_{s}.parquet").filter(pl.col("ordinary")).select("site", "row", "valid_time") for s in SITES])
    sc = pl.concat([pl.read_parquet(E203_DIR / f"scores_{s}.parquet").with_columns(site=pl.lit(s)) for s in SITES])
    sc = sc.join(ordinary, on=["site", "row"], how="inner").with_columns(month=pl.col("valid_time").dt.strftime("%Y-%m"))
    wide_o = sc.group_by("site", "row", "month", "arm").agg(qs=pl.col("qs").mean()).pivot(on="arm", index=["site", "row", "month"], values="qs")
    months, inv = np.unique(wide_o["month"].to_numpy(), return_inverse=True)
    counts = np.bincount(inv).astype(float)
    harm_arms = ["PAIR_GATED", "PAIR_NEW", "LEARNED_R", "LEARNED_F", "RONC_195"]
    sums = np.stack([np.bincount(inv, weights=(wide_o[a] - wide_o["K0"]).to_numpy().astype(float), minlength=len(months)) for a in harm_arms], axis=1)
    draws = boot_matrix(sums=sums, counts=counts, seed=203)
    est = sums.sum(axis=0) / counts.sum()
    harm_rows = []
    for i, a in enumerate(harm_arms):
        lo, hi = interval(draws[:, i])
        harm_rows.append({"source": a, "harm": float(est[i]), "lo": lo, "hi": hi})
        print(f"{a:12s} {fmt((est[i], lo, hi))}  inside +/-0.05: {lo >= -0.05 and hi <= 0.05}")
    print("ordinary rows:", wide_o.height, "K0 mean score:", float(wide_o["K0"].mean()))

    # ---- false-alarm precision sweep ----
    print("\n== Precision sweep: LOGGED plus false alarms minus LOGGED alone, ordinary rows ==")
    fa = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(str(real_dir / "false_alarm_*.parquet")))])
    oc = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(str(real_dir / "ordinary_count_*.parquet")))])
    # months are counted since 1970-01 plus 1 in the job; the denominator uses the same labels
    month_labels = np.array(sorted(set(oc["month"].to_list())))
    denominators = np.array([oc.filter(pl.col("month") == m)["len"].sum() for m in month_labels], dtype=float)
    prec_rows = []
    for precision in sorted(fa["precision"].unique().to_list(), reverse=True):
        f = fa.filter(pl.col("precision") == precision)
        n_draws = f["draw"].n_unique()
        s = np.array([f.filter(pl.col("month") == m)["diff"].sum() / n_draws for m in month_labels])
        d = boot_matrix(sums=s[:, None], counts=denominators, seed=int(precision * 100))
        lo, hi = interval(d[:, 0])
        est_p = float(s.sum() / denominators.sum())
        prec_rows.append({"precision": precision, "harm": est_p, "lo": lo, "hi": hi, "n_false_per_draw": float(f.group_by("site", "draw").agg(pl.col("n_false").first()).group_by("draw").agg(pl.col("n_false").sum())["n_false"].mean())})
        print(f"precision {precision}: {fmt((est_p, lo, hi))}  false alarms per draw (all substations) {prec_rows[-1]['n_false_per_draw']:.1f}")
    pl.DataFrame(prec_rows).write_parquet(REPORT_DIR / "precision_harm.parquet")
    pl.DataFrame(harm_rows).write_parquet(REPORT_DIR / "ordinary_harm.parquet")

    # ---- alarm counts and precision of the stored detectors on the logged transfers ----
    print("\n== Alarms and logged transfers found, per source (real record, six substations) ==")
    import e00208_events as ev_

    tables_ = ev_.alarm_tables()
    for name in ("RONC", "NEW", "GATED"):
        t = tables_[name].filter(pl.col("series").is_in(list(fed.SERIES_OF_SITE.values())))
        found = 0
        n_logged = 0
        for site in SITES:
            e = ev_.site_events(site=site, tables=tables_, offset=0)
            col = {"RONC": "known_ronc", "NEW": "known_new", "GATED": "known_gated"}[name]
            found += int((e[col] >= 0).sum())
            n_logged += e.height
        print(f"{name}: {t.height} alarms; logged transfers matched {found} of {n_logged}")

    summary = {
        "injected_units": int(len(np.unique(unit_of(wide)))),
        "worlds": int(worlds),
        "whole_year_pooled": {a: v / (worlds * n_rows) for a, v in total.items()},
    }
    (REPORT_DIR / "report.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
