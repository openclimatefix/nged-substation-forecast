"""Join the stack's per-row scores to BG2 and the references, and print every number the page quotes.

Reads the private `rows_*.parquet`, `events_*.parquet`, `rowsets_*.parquet` and `quiet_*.parquet`
files of `e00163_run.py`, experiment 87's stored predictions (BG2, the manual heuristic, and
climatology), and experiment 155's arm C. Writes `report.json` and `joined.parquet` beside them.

    OMP_NUM_THREADS=1 uv run --no-sync python e00163_report.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Final

import numpy as np
import polars as pl
from e00163_common import WORK_DIR
from e00163_g_arms import LEVELS, quantile_score
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16", "D9")
STACK: Final[tuple[str, ...]] = ("K0", "GAM_ONLY", "MASK_NEW", "MASK_RONC", "MASK_UNION")
EXTERNAL: Final[tuple[str, ...]] = ("BG2", "MANUAL", "CLIM", "ARM_C")
ARMS: Final[tuple[str, ...]] = (*STACK, *EXTERNAL)
E87: Final = ML_EXPERIMENTS_DATA_DIR / "00087-bg2-best-guess-assembled-with-drop-one-ablations"
E155: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00155-dp-bg-stage2-assembly-night-offset-and-error-correction"
    / "work"
    / "dp_preds"
)
STORED: Final[dict[str, str]] = {
    "BG2": "p_s6_bg2",
    "MANUAL": "ref_manual_heuristic",
    "CLIM": "ref_climatology",
}
QCOLS: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
TAILS: Final[tuple[tuple[str, float], ...]] = (
    ("q01", 0.01),
    ("q05", 0.05),
    ("q95", 0.95),
    ("q99", 0.99),
)
N_DRAWS: Final[int] = 20_000
WINTER: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})
KEYS: Final[list[str]] = ["site", "power_fcst_init_time", "valid_time"]


def external_scores() -> pl.DataFrame:
    """Per-row score and tail quantiles for BG2, the references, and arm C, in points."""
    frames = []
    for arm, name in STORED.items():
        f = pl.read_parquet(E87 / "predictions" / name / "predictions.parquet").filter(
            pl.col("site").is_in(SITES)
        )
        frames.append(
            f.select(
                *KEYS,
                pl.col("lead_hours").cast(pl.Float64),
                (pl.col("actual").cast(pl.Float64) * 100).alias("actual"),
                *[(pl.col(c).cast(pl.Float64) * 100).alias(c) for c in QCOLS],
            ).with_columns(arm=pl.lit(arm))
        )
    arm_c = []
    for site in SITES:
        c = pl.read_parquet(E155 / f"c_scored_{site}.parquet")
        arm_c.append(
            c.with_columns(site=pl.lit(site)).select(
                *KEYS,
                pl.col("lead_hours").cast(pl.Float64),
                pl.col("actual").cast(pl.Float64),
                *[pl.col(c_).cast(pl.Float64) for c_ in QCOLS],
            )
        )
    frames.append(pl.concat(arm_c).with_columns(arm=pl.lit("ARM_C")))
    out = pl.concat(frames)
    q = out.select(QCOLS).to_numpy()
    score = quantile_score(actual=out["actual"].to_numpy(), quantiles=q)
    out = out.with_columns(qs=pl.Series(score), mean_forecast=pl.col("q50"))
    return out.filter(pl.col("qs").is_finite())


def load_joined() -> pl.DataFrame:
    """The long table of every arm's score on the rows the stack and BG2 share."""
    rows = pl.concat([pl.read_parquet(WORK_DIR / f"rows_{s}.parquet") for s in SITES])
    rows = rows.with_columns(
        pl.col("power_fcst_init_time").dt.replace_time_zone("UTC"),
        pl.col("valid_time").dt.replace_time_zone("UTC"),
    )
    ext = external_scores()
    long = pl.concat(
        [
            rows.select(
                "arm", *KEYS, "row", "lead_hours", "actual", "qs", *[c for c in QCOLS if c in rows.columns], "mean_forecast"
            ),
            ext.select("arm", *KEYS, "lead_hours", "actual", "qs", *QCOLS, "mean_forecast").with_columns(
                row=pl.lit(None, dtype=pl.Int64)
            ),
        ],
        how="diagonal_relaxed",
    )
    return long


def boot(
    *, num: np.ndarray, den: np.ndarray | None, groups: np.ndarray, seed: int = 163
) -> tuple[float, float, float]:
    """Ratio of sums (a mean when `den` is None) with a 95% interval from resampling groups."""
    den = np.ones(len(num)) if den is None else den
    labels, inverse = np.unique(groups, return_inverse=True)
    n = np.bincount(inverse, weights=num, minlength=len(labels))
    d = np.bincount(inverse, weights=den, minlength=len(labels))
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(labels), size=(N_DRAWS, len(labels)))
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = n[picks].sum(axis=1) / d[picks].sum(axis=1)
    return (
        float(n.sum() / d.sum()),
        float(np.nanpercentile(draws, 2.5)),
        float(np.nanpercentile(draws, 97.5)),
    )


def fmt(t: tuple[float, float, float]) -> str:
    """Format a point estimate with its interval."""
    return f"{t[0]:+.3f} ({t[1]:+.3f} to {t[2]:+.3f})"


def build() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return the wide per-row scores, the event rows, and the long table with tail quantiles."""
    rows = pl.concat([pl.read_parquet(WORK_DIR / f"rows_{s}.parquet") for s in SITES]).with_columns(
        pl.col("power_fcst_init_time").dt.replace_time_zone("UTC"),
        pl.col("valid_time").dt.replace_time_zone("UTC"),
    )
    ext = external_scores()
    stack_wide = rows.pivot(on="arm", index=[*KEYS, "row", "actual", "lead_hours"], values="qs")
    ext_wide = ext.pivot(on="arm", index=KEYS, values="qs", aggregate_function="first")
    wide = stack_wide.join(ext_wide, on=KEYS, how="left").with_columns(
        mo=pl.col("valid_time").dt.month(),
        month=pl.col("valid_time").dt.strftime("%Y-%m"),
        shared=pl.col("BG2").is_not_null(),
    )
    sets = pl.concat([pl.read_parquet(WORK_DIR / f"rowsets_{s}.parquet") for s in SITES])
    events = pl.concat([pl.read_parquet(WORK_DIR / f"events_{s}.parquet") for s in SITES])
    ev = sets.join(wide, on=["site", "row"], how="inner").join(
        events.select("card", "site", "known_new", "known_ronc"), on=["card", "site"]
    )
    long = pl.concat(
        [
            rows.select("arm", *KEYS, "actual", "qs", *[c for c in QCOLS if c in rows.columns], "mean_forecast"),
            ext.select("arm", *KEYS, "actual", "qs", *QCOLS, "mean_forecast"),
        ],
        how="diagonal_relaxed",
    )
    return wide, ev, long


def main() -> None:  # noqa: PLR0915
    """Print every number the write-up quotes and write report.json."""
    wide, ev, long = build()
    report: dict = {}
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_fmt_str_lengths(8)
    check = wide.select(*KEYS, "actual").join(
        long.filter(pl.col("arm") == "BG2").select(*KEYS, pl.col("actual").alias("actual_bg2")), on=KEYS
    )
    print("rows in both:", check.height, "max |actual stack - actual BG2| (points):", float((check["actual"] - check["actual_bg2"]).abs().max()))
    ev = ev.with_columns(
        winter=pl.col("mo").is_in(list(WINTER)),
        period=pl.when(pl.col("power_fcst_init_time") >= datetime(2025, 7, 1, tzinfo=UTC)).then(pl.lit("scored")).otherwise(pl.lit("replicate")),
    )
    cards = ev.group_by("set").agg(pl.len().alias("rows"), pl.col("card").n_unique().alias("cards"), pl.struct("card", "site").n_unique().alias("events"))
    print(cards)
    per_event = (
        ev.group_by("set", "period", "card", "site")
        .agg(pl.len().alias("rows"), *[pl.col(a).mean().round(2) for a in [*STACK, "BG2", "ORACLE"]], pl.col("known_new").first(), pl.col("known_ronc").first())
        .with_columns(found_new=pl.col("known_new") >= 0, found_ronc=pl.col("known_ronc") >= 0)
        .drop("known_new", "known_ronc")
        .sort("period", "card", "site", "set")
    )
    print(per_event)
    report["rows_by_set"] = cards.to_dicts()

    def pooled(frame: pl.DataFrame, a: str, b: str | None = None, *, groups: str = "card") -> tuple[float, float, float]:
        sub = frame.filter(pl.col(a).is_not_null() & (pl.col(b).is_not_null() if b else pl.lit(True)))
        diff = sub[a] - (sub[b] if b else 0.0)
        return boot(num=diff.to_numpy(), den=None, groups=sub[groups].to_numpy())

    def gap_closed(frame: pl.DataFrame, arm: str) -> tuple[float, float, float]:
        sub = frame.filter(pl.col("ORACLE").is_not_null())
        sub = sub.join(
            sub.group_by("card", "site").agg((pl.col("K0") - pl.col("ORACLE")).mean().alias("g")).filter(pl.col("g") > 0).select("card", "site"),
            on=["card", "site"],
        )
        return boot(
            num=(sub["K0"] - sub[arm]).to_numpy(), den=(sub["K0"] - sub["ORACLE"]).to_numpy(), groups=sub["card"].to_numpy()
        )

    print("\n== means by set, scope, and season (points of effective capacity) ==")
    for scope in ("shared", "stack_only"):
        for season in ("all", "winter", "other"):
            for name in ("BEFORE", "INSIDE", "AFTER"):
                f = ev.filter(pl.col("set") == name)
                if scope == "shared":
                    f = f.filter(pl.col("shared"))
                if season != "all":
                    f = f.filter(pl.col("winter") == (season == "winter"))
                if f.is_empty():
                    continue
                means = {a: round(float(f[a].mean()), 3) for a in [*ARMS, "ORACLE"] if f[a].null_count() < f.height}
                print(f"{scope:10s} {season:6s} {name:6s} rows {f.height:6d} cards {f['card'].n_unique():2d}", means)
                report[f"means|{scope}|{season}|{name}"] = {"rows": f.height, "cards": f["card"].n_unique()} | means
    print("\n== planned contrasts on AFTER ==")
    after = ev.filter(pl.col("set") == "AFTER")
    sh = after.filter(pl.col("shared"))
    for label, a, b in (
        ("C1 MASK_NEW minus BG2", "MASK_NEW", "BG2"),
        ("C2 MASK_UNION minus BG2", "MASK_UNION", "BG2"),
        ("K0 minus BG2", "K0", "BG2"),
        ("GAM_ONLY minus BG2", "GAM_ONLY", "BG2"),
        ("ARM_C minus BG2", "ARM_C", "BG2"),
    ):
        report[label] = pooled(sh, a, b)
        print(label, "shared rows:", fmt(report[label]), "cards", sh["card"].n_unique())
    for label, a, b in (("MASK_NEW minus K0", "MASK_NEW", "K0"), ("MASK_UNION minus K0", "MASK_UNION", "K0"), ("MASK_RONC minus K0", "MASK_RONC", "K0"), ("GAM_ONLY minus K0", "GAM_ONLY", "K0")):
        report[label + " (AFTER, stack only)"] = pooled(after, a, b)
        print(label, "AFTER stack-only (incl. replicate months):", fmt(report[label + " (AFTER, stack only)"]), "cards", after["card"].n_unique())
    print("\n== C3: share of the K0-to-ORACLE gap closed on AFTER ==")
    for scope, f in (("shared", sh), ("stack_only", after)):
        for arm in ("MASK_NEW", "MASK_UNION", "MASK_RONC", "GAM_ONLY", "BG2", "ARM_C"):
            if f[arm].null_count() == f.height:
                continue
            g = gap_closed(f.filter(pl.col(arm).is_not_null()), arm)
            report[f"C3|{scope}|{arm}"] = g
            n_ev = f.filter(pl.col("ORACLE").is_not_null()).select(pl.struct("card", "site")).n_unique()
            print(f"{scope:10s} {arm:10s} {g[0]:.3f} ({g[1]:.3f} to {g[2]:.3f})  events with oracle: {n_ev}")
    print("\n== per-event share of the gap closed (AFTER; events with K0 minus ORACLE above 0.25 points) ==")
    pe = (
        after.filter(pl.col("ORACLE").is_not_null())
        .group_by("card", "site", "period")
        .agg(pl.len().alias("rows"), *[pl.col(a).mean() for a in ("K0", "ORACLE", "MASK_NEW", "MASK_UNION", "MASK_RONC", "GAM_ONLY", "BG2")])
        .with_columns(gap=pl.col("K0") - pl.col("ORACLE"))
        .filter(pl.col("gap") > 0.25)
    )
    for arm in ("MASK_NEW", "MASK_UNION", "MASK_RONC", "GAM_ONLY", "BG2"):
        sh_ = ((pe["K0"] - pe[arm]) / pe["gap"]).drop_nulls()
        print(f"{arm:10s} events {len(sh_):2d} median {sh_.median():+.2f} mean {sh_.mean():+.2f} min {sh_.min():+.2f} max {sh_.max():+.2f}")
        report[f"per_event_share|{arm}"] = sh_.to_list()
    print(pe.with_columns(pl.col(pl.Float64).round(2)).sort("gap", descending=True))
    print("\n== BEFORE and INSIDE contrasts (descriptive; observed power contains the transfer) ==")
    for name in ("BEFORE", "INSIDE"):
        f = ev.filter((pl.col("set") == name) & pl.col("shared"))
        if f.is_empty():
            continue
        for label, a, b in (("MASK_NEW minus K0", "MASK_NEW", "K0"), ("MASK_UNION minus K0", "MASK_UNION", "K0"), ("BG2 minus K0", "BG2", "K0"), ("GAM_ONLY minus K0", "GAM_ONLY", "K0")):
            report[f"{name}|{label}"] = pooled(f, a, b)
            print(name, label, fmt(report[f'{name}|{label}']), "cards", f["card"].n_unique())
    print("\n== leave one substation out, AFTER shared ==")
    for site in SITES:
        f = sh.filter(pl.col("site") != site)
        if f.is_empty():
            continue
        print(site, "left out:", "C1", fmt(pooled(f, "MASK_NEW", "BG2")), "C2", fmt(pooled(f, "MASK_UNION", "BG2")))
    print("\n== no harm: quiet rows (outside logged transfers and 72 h after, outside hand-filtered switching) ==")
    quiet = pl.concat([pl.read_parquet(WORK_DIR / f"quiet_{s}.parquet") for s in SITES])
    q = wide.join(quiet, on=["site", "row"], how="inner").filter(pl.col("shared"))
    q = q.with_columns(winter=pl.col("mo").is_in(list(WINTER)))
    report["quiet_rows"] = q.height
    for season, f in (("all", q), ("winter", q.filter(pl.col("winter"))), ("other", q.filter(~pl.col("winter")))):
        means = {a: round(float(f[a].mean()), 3) for a in ARMS}
        print(season, f.height, means)
        report[f"quiet_means|{season}"] = means
        for label, a, b in (
            ("MASK_NEW minus K0", "MASK_NEW", "K0"), ("MASK_UNION minus K0", "MASK_UNION", "K0"), ("MASK_RONC minus K0", "MASK_RONC", "K0"),
            ("GAM_ONLY minus K0", "GAM_ONLY", "K0"), ("K0 minus BG2", "K0", "BG2"), ("MASK_NEW minus BG2", "MASK_NEW", "BG2"),
            ("MASK_UNION minus BG2", "MASK_UNION", "BG2"), ("ARM_C minus BG2", "ARM_C", "BG2"),
        ):
            ci = pooled(f, a, b, groups="month")
            report[f"quiet|{season}|{label}"] = ci
            print("  ", label, fmt(ci))
    print("\n== tails (coverage; pinball loss in points) on quiet rows and on AFTER shared rows ==")
    key_sets = {"quiet": q.select(KEYS), "AFTER": sh.select(KEYS).unique()}
    for name, keys in key_sets.items():
        for arm in ARMS:
            sub = long.filter(pl.col("arm") == arm).join(keys, on=KEYS, how="inner")
            act = sub["actual"].to_numpy()
            out = {}
            for col, level in TAILS:
                qv = sub[col].to_numpy()
                err = act - qv
                out[col] = (round(float(np.mean(act <= qv)), 4), round(float(np.mean(np.maximum(level * err, (level - 1) * err))), 4))
            print(name, f"{arm:10s}", out)
            report[f"tails|{name}|{arm}"] = out
    print("\n== checks: median forecast minus mean forecast (stack arms), and error of the median forecast on AFTER ==")
    for arm in ("K0", "MASK_NEW", "MASK_UNION"):
        sub = long.filter(pl.col("arm") == arm).join(q.select(KEYS), on=KEYS, how="inner")
        d = (sub["q50"] - sub["mean_forecast"]).group_by(sub["site"]).mean() if False else sub.group_by("site").agg((pl.col("q50") - pl.col("mean_forecast")).mean().alias("d")).sort("site")
        print(arm, {r["site"]: round(r["d"], 3) for r in d.iter_rows(named=True)})
        report[f"median_minus_mean|{arm}"] = {r["site"]: r["d"] for r in d.iter_rows(named=True)}
    mae = {}
    for arm in ("K0", "GAM_ONLY", "MASK_NEW", "MASK_UNION", "BG2"):
        sub = long.filter(pl.col("arm") == arm).join(sh.select(KEYS).unique(), on=KEYS, how="inner")
        mae[arm] = round(float((sub["q50"] - sub["actual"]).abs().mean()), 3)
    print("absolute error of the median forecast on AFTER shared rows:", mae)
    report["mae_median_after"] = mae
    (WORK_DIR / "report.json").write_text(json.dumps(report, indent=2, default=float))
    ev.write_parquet(WORK_DIR / "event_rows.parquet")


if __name__ == "__main__":
    main()
