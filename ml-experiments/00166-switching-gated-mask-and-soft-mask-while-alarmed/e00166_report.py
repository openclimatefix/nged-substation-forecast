# ruff: noqa: E501, PLR0912, RET504
"""Join the stack's per-row scores to BG2 and the references, and print every number the page quotes.

Reads the private `rows_*.parquet`, `events_*.parquet`, `rowsets_*.parquet`, `flags_*.parquet` and
`quiet_*.parquet` files of `e00166_run.py`, experiment 87's stored predictions (BG2, the manual
heuristic, and climatology), and experiment 155's arm C. Writes `report.json`, `wide.parquet`, and
`event_rows.parquet` beside them.

    OMP_NUM_THREADS=1 uv run --no-sync python e00166_report.py
"""

from __future__ import annotations

import json
from typing import Final

import e00166_events as ev_module
import numpy as np
import polars as pl
from e00166_common import WORK_DIR
from e00166_g_arms import quantile_score
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16", "D9")
MASKS: Final[tuple[str, ...]] = ("MASK_NEW", "MASK_RONC", "MASK_UNION", "MASK_GATED")
SOFT_PRE: Final[tuple[str, ...]] = ("SOFT_F10", "SOFT_F100", "SOFT_F1000")
SOFT: Final[tuple[str, ...]] = (*SOFT_PRE, "SOFT_F10000", "SOFT_F100000", "SOFT_F1000000")
STACK: Final[tuple[str, ...]] = ("K0", "GAM_ONLY", *MASKS, *SOFT)
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
D159: Final = ML_EXPERIMENTS_DATA_DIR / "00159-switching-d8-multi-day-drops-recurrence-and-report"
D8_SERIES: Final[int] = 26
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
                "arm",
                *KEYS,
                "row",
                "lead_hours",
                "actual",
                "qs",
                *[c for c in QCOLS if c in rows.columns],
                "mean_forecast",
            ),
            ext.select(
                "arm", *KEYS, "lead_hours", "actual", "qs", *QCOLS, "mean_forecast"
            ).with_columns(row=pl.lit(None, dtype=pl.Int64)),
        ],
        how="diagonal_relaxed",
    )
    return long


def boot(
    *, num: np.ndarray, den: np.ndarray | None, groups: np.ndarray, seed: int = 166
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
        events.select("card", "site", "known_new", "known_ronc", "known_gated", "end"),
        on=["card", "site"],
    )
    long = pl.concat(
        [
            rows.select(
                "arm",
                *KEYS,
                "actual",
                "qs",
                *[c for c in QCOLS if c in rows.columns],
                "mean_forecast",
            ),
            ext.select("arm", *KEYS, "actual", "qs", *QCOLS, "mean_forecast"),
        ],
        how="diagonal_relaxed",
    )
    return wide, ev, long


def decide(*, report: dict, ev: pl.DataFrame, outside: pl.DataFrame, pairs: pl.DataFrame) -> None:
    """Apply the decision rule of the plan to the numbers in `report` and print each test."""
    print("\n== DECISION RULE (pre-registered in result.md) ==")
    gated_pairs = (
        ev.filter(pl.col("set") == "AFTER").select("card", "site", "known_gated", "end").unique()
    )
    with_alarm = gated_pairs.filter(
        (pl.col("known_gated") >= 0) & (pl.col("known_gated") < pl.col("end"))
    ).height
    print(
        "transfer-substation pairs with AFTER rows:",
        gated_pairs.height,
        "of which with a gated alarm known:",
        with_alarm,
    )
    enough = with_alarm >= 5
    g1 = report["outside|all|MASK_GATED minus K0"]
    after = report["means|shared|all|AFTER"]
    hard_gain = after["K0"] - after["MASK_UNION"]
    gated_gain = after["K0"] - after["MASK_GATED"]
    g1_pass = g1[1] >= -0.03 and g1[2] <= 0.03
    g2_pass = gated_gain >= 0.7 * hard_gain
    print(f"G1 MASK_GATED minus K0 outside {fmt(g1)}: inside +/-0.03 -> {g1_pass}")
    print(
        f"G2 AFTER gain K0 - GATED {gated_gain:.3f} vs 0.7 x hard gain {0.7 * hard_gain:.3f} "
        f"(hard {hard_gain:.3f}) -> {g2_pass}"
    )
    part1 = (
        "inconclusive" if not enough else ("supported" if g1_pass and g2_pass else "not supported")
    )
    print("PART 1:", part1)
    inside = report["means|shared|all|INSIDE"]
    hard_harm = inside["MASK_UNION"] - inside["K0"]
    best: tuple[float, str] | None = None
    for arm in SOFT_PRE:
        s1 = report[f"outside|all|{arm} minus MASK_UNION"][0]
        s1_pass = s1 <= -0.051
        gain = after["K0"] - after[arm]
        s2_pass = gain >= 0.8 * hard_gain
        harm = inside[arm] - inside["K0"]
        s3_pass = harm <= 1.2 * hard_harm
        outside_change = report[f"outside|all|{arm} minus K0"][0]
        print(
            f"{arm}: S1 outside minus MASK_UNION {s1:+.3f} (<= -0.051) {s1_pass}; "
            f"S2 AFTER gain {gain:.3f} (>= {0.8 * hard_gain:.3f}) {s2_pass}; "
            f"S3 INSIDE harm {harm:+.3f} (<= {1.2 * hard_harm:+.3f}) {s3_pass}; "
            f"outside minus K0 {outside_change:+.3f}"
        )
        report[f"decision|{arm}"] = [s1_pass, s2_pass, s3_pass]
        if s2_pass and s3_pass and (best is None or outside_change < best[0]):
            best = (outside_change, arm)
    passing = [a for a in SOFT_PRE if all(report[f"decision|{a}"])]
    print("best factor among those passing S2 and S3:", best)
    part2 = "supported" if passing else "not supported"
    print("PART 2:", part2, "factors passing S1, S2, and S3:", passing)
    n_supported = (part1 == "supported") + (part2 == "supported")
    if n_supported == 2:
        verdict = "supported"
    elif n_supported == 1:
        verdict = "partly supported"
    else:
        verdict = "inconclusive" if part1 == "inconclusive" else "not supported"
    print("VERDICT by the pre-registered rule:", verdict)
    report["decision"] = {"part1": part1, "part2": part2, "verdict": verdict, "best": best}


def main() -> None:  # noqa: PLR0915
    """Print every number the write-up quotes and write report.json."""
    wide, ev, long = build()
    flags = pl.concat([pl.read_parquet(WORK_DIR / f"flags_{s}.parquet") for s in SITES])
    wide = wide.join(flags, on=["site", "row"], how="left").with_columns(
        winter=pl.col("mo").is_in(list(WINTER))
    )
    ev = ev.with_columns(winter=pl.col("mo").is_in(list(WINTER)))
    report: dict = {}
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_fmt_str_lengths(8)
    arms = [*ARMS, "ORACLE"]

    def pooled(
        frame: pl.DataFrame, a: str, b: str, *, groups: str = "card"
    ) -> tuple[float, float, float]:
        sub = frame.filter(pl.col(a).is_not_null() & pl.col(b).is_not_null())
        return boot(num=(sub[a] - sub[b]).to_numpy(), den=None, groups=sub[groups].to_numpy())

    def means(frame: pl.DataFrame, names: list[str]) -> dict[str, float]:
        return {
            a: round(float(frame[a].mean()), 3)  # ty: ignore[invalid-argument-type]
            for a in names
            if frame[a].null_count() < frame.height
        }

    print("rows by site:", wide.group_by("site").len().sort("site").to_dicts())
    print("stack rows (every scored row with a reading):", wide.height)
    print("shared with BG2:", wide.filter(pl.col("shared")).height)
    for key in ("in_logged", "in_switching_filter"):
        share = float(wide[key].mean())  # ty: ignore[invalid-argument-type]
        print(key, "rows:", int(wide[key].sum()), "share", round(share, 4))
    for arm in ("MASK_NEW", "MASK_RONC", "MASK_UNION", "MASK_GATED"):
        a, p = wide[f"active_{arm}"], wide[f"post_{arm}"]
        a_share, p_share = float(a.mean()), float(p.mean())  # ty: ignore[invalid-argument-type]
        print(
            f"{arm}: alarm active at the forecast origin on {a.sum()} rows "
            f"({a_share:.3f}); 72 h post-end on {p.sum()} rows ({p_share:.3f})"
        )
        by_site = wide.group_by("site").agg(pl.col(f"active_{arm}").mean().round(3)).sort("site")
        print(
            "   active share by site:",
            {r["site"]: r[f"active_{arm}"] for r in by_site.iter_rows(named=True)},
        )
        report[f"active_share|{arm}"] = a_share

    outside = wide.filter(~pl.col("in_logged"))
    outside_163 = outside.filter(~pl.col("in_switching_filter"))
    report["rows_outside"] = outside.height
    report["rows_outside_163"] = outside_163.height
    print("\n== OUTSIDE events: not within a logged transfer from onset to 72 h after its end ==")
    print(
        "rows",
        outside.height,
        "(163's narrower rows, also outside hand-filtered switching:",
        outside_163.height,
        ")",
    )
    contrasts = (
        ("MASK_UNION minus K0", "MASK_UNION", "K0"),
        ("MASK_NEW minus K0", "MASK_NEW", "K0"),
        ("MASK_RONC minus K0", "MASK_RONC", "K0"),
        ("GAM_ONLY minus K0", "GAM_ONLY", "K0"),
        ("MASK_GATED minus K0", "MASK_GATED", "K0"),
        ("MASK_GATED minus MASK_UNION", "MASK_GATED", "MASK_UNION"),
        ("MASK_GATED minus BG2", "MASK_GATED", "BG2"),
        *[(f"{a} minus K0", a, "K0") for a in SOFT],
        *[(f"{a} minus MASK_UNION", a, "MASK_UNION") for a in SOFT],
        *[(f"{a} minus MASK_GATED", a, "MASK_GATED") for a in SOFT],
        *[(f"{a} minus BG2", a, "BG2") for a in SOFT],
        ("K0 minus BG2", "K0", "BG2"),
        ("MASK_UNION minus BG2", "MASK_UNION", "BG2"),
        ("MASK_NEW minus BG2", "MASK_NEW", "BG2"),
        ("ARM_C minus BG2", "ARM_C", "BG2"),
    )
    for scope, frame in (("outside", outside), ("outside_163rows", outside_163)):
        for season, f in (
            ("all", frame),
            ("winter", frame.filter(pl.col("winter"))),
            ("other", frame.filter(~pl.col("winter"))),
        ):
            m = means(f.filter(pl.col("shared")), arms)
            print(scope, season, f.height, "shared-row means:", m)
            report[f"means|{scope}|{season}"] = m
            for label, a, b in contrasts:
                src = f.filter(pl.col("shared")) if "BG2" in (a, b) or "ARM_C" in (a, b) else f
                ci = pooled(src, a, b, groups="month")
                report[f"{scope}|{season}|{label}"] = ci
                print("  ", label, fmt(ci))

    print("\n== leave one substation out, OUTSIDE, MASK_UNION minus K0 and MASK_NEW minus K0 ==")
    for site in SITES:
        f = outside.filter(pl.col("site") != site)
        print(
            site,
            "left out:",
            fmt(pooled(f, "MASK_UNION", "K0", groups="month")),
            fmt(pooled(f, "MASK_NEW", "K0", groups="month")),
            "gated",
            fmt(pooled(f, "MASK_GATED", "K0", groups="month")),
            "soft",
            [fmt(pooled(f, a, "K0", groups="month")) for a in SOFT],
        )
    print("\n== per substation, OUTSIDE, mean score ==")
    for site in SITES:
        f = outside.filter(pl.col("site") == site)
        print(site, means(f, ["K0", "MASK_NEW", "MASK_UNION", "MASK_GATED", *SOFT, "GAM_ONLY"]))

    print("\n== rows by the alarm active at the forecast origin (descriptive) ==")
    for arm in ("MASK_NEW", "MASK_UNION", "MASK_GATED"):
        for scope, base in (("all rows", wide), ("outside logged transfers", outside)):
            for kind in ("active", "post"):
                f = base.filter(pl.col(f"{kind}_{arm}"))
                if f.is_empty():
                    continue
                m = means(f.filter(pl.col("shared")), [arm, "K0", "GAM_ONLY", "BG2"])
                ci = pooled(f, arm, "K0", groups="month")
                ci_bg = pooled(f.filter(pl.col("shared")), arm, "BG2", groups="month")
                print(
                    f"{arm} {kind:6s} {scope:25s} rows {f.height:6d}",
                    m,
                    "minus K0",
                    fmt(ci),
                    "minus BG2",
                    fmt(ci_bg),
                )
                report[f"alarm|{arm}|{kind}|{scope}|minus K0"] = ci
                report[f"alarm|{arm}|{kind}|{scope}|minus BG2"] = ci_bg
    for arm in ("MASK_NEW", "MASK_UNION", "MASK_GATED"):
        quiet_alarm = outside.filter(~pl.col(f"active_{arm}") & ~pl.col(f"post_{arm}"))
        ci = pooled(quiet_alarm, arm, "K0", groups="month")
        print(
            f"OUTSIDE and no {arm} alarm active or post: {arm} minus K0",
            fmt(ci),
            "rows",
            quiet_alarm.height,
        )
        report[f"outside_no_alarm|{arm} minus K0"] = ci
    print("\n== logged transfers: row sets and cards ==")
    cards = ev.group_by("set").agg(
        pl.len().alias("rows"),
        pl.col("card").n_unique().alias("cards"),
        pl.struct("card", "site").n_unique().alias("events"),
    )
    print(cards)
    report["rows_by_set"] = cards.to_dicts()
    for scope in ("shared", "stack_only"):
        for name in ("BEFORE", "INSIDE", "AFTER"):
            f = ev.filter(pl.col("set") == name)
            if scope == "shared":
                f = f.filter(pl.col("shared"))
            for season, g in (
                ("all", f),
                ("winter", f.filter(pl.col("winter"))),
                ("other", f.filter(~pl.col("winter"))),
            ):
                if g.is_empty():
                    continue
                m = means(g, arms)
                print(scope, season, name, "rows", g.height, "cards", g["card"].n_unique(), m)
                report[f"means|{scope}|{season}|{name}"] = {
                    "rows": g.height,
                    "cards": g["card"].n_unique(),
                } | m
    print("\n== C2 to C5 contrasts by logged-transfer row set (resampling transfers) ==")
    for name in ("BEFORE", "INSIDE", "AFTER"):
        for scope in ("stack_only", "shared"):
            f = ev.filter(pl.col("set") == name)
            if scope == "shared":
                f = f.filter(pl.col("shared"))
            if f.is_empty():
                continue
            for label, a, b in (
                ("MASK_UNION minus K0", "MASK_UNION", "K0"),
                ("MASK_NEW minus K0", "MASK_NEW", "K0"),
                ("GAM_ONLY minus K0", "GAM_ONLY", "K0"),
                ("MASK_GATED minus K0", "MASK_GATED", "K0"),
                ("MASK_GATED minus MASK_UNION", "MASK_GATED", "MASK_UNION"),
                ("MASK_GATED minus BG2", "MASK_GATED", "BG2"),
                *[(f"{a} minus K0", a, "K0") for a in SOFT],
                *[(f"{a} minus MASK_UNION", a, "MASK_UNION") for a in SOFT],
                *[(f"{a} minus MASK_GATED", a, "MASK_GATED") for a in SOFT],
                ("MASK_UNION minus BG2", "MASK_UNION", "BG2"),
                ("MASK_NEW minus BG2", "MASK_NEW", "BG2"),
                ("BG2 minus K0", "BG2", "K0"),
                ("MASK_UNION minus ORACLE", "MASK_UNION", "ORACLE"),
            ):
                if "BG2" in (a, b) and scope == "stack_only":
                    continue
                ci = pooled(f, a, b)
                report[f"{name}|{scope}|{label}"] = ci
                print(
                    name,
                    scope,
                    label,
                    fmt(ci),
                    "cards",
                    f.filter(pl.col(a).is_not_null() & pl.col(b).is_not_null())["card"].n_unique(),
                )
    after = ev.filter(pl.col("set") == "AFTER")
    pairs = (
        after.group_by("card", "site")
        .agg(
            pl.len().alias("rows"),
            *[
                pl.col(a).mean()
                for a in (
                    "K0",
                    "MASK_NEW",
                    "MASK_UNION",
                    "MASK_GATED",
                    *SOFT,
                    "GAM_ONLY",
                    "BG2",
                    "ORACLE",
                )
            ],
        )
        .sort("card", "site")
    )
    print(pairs.with_columns(pl.col(pl.Float64).round(2)))
    pairs_n = pairs.filter(pl.col("K0").is_not_null()).height
    report["after_pairs"] = pairs_n
    print("transfer-substation pairs with AFTER rows:", pairs_n)
    print("leave one substation out, AFTER, MASK_UNION minus K0:")
    for site in SITES:
        f = after.filter(pl.col("site") != site)
        if f.is_empty():
            continue
        print(
            " ",
            site,
            "left out:",
            fmt(pooled(f, "MASK_UNION", "K0")),
            "pair alarm",
            fmt(pooled(f, "MASK_NEW", "K0")),
            "gated",
            fmt(pooled(f, "MASK_GATED", "K0")),
            "soft",
            [fmt(pooled(f, a, "K0")) for a in SOFT],
        )

    print(
        "\n== tails (coverage; pinball loss in points) on OUTSIDE rows and on AFTER shared rows =="
    )
    key_sets = {
        "outside": outside.filter(pl.col("shared")).select(KEYS),
        "AFTER": after.filter(pl.col("shared")).select(KEYS).unique(),
    }
    for name, keys in key_sets.items():
        for arm in ARMS:
            sub = long.filter(pl.col("arm") == arm).join(keys, on=KEYS, how="inner")
            act = sub["actual"].to_numpy()
            out = {}
            for col, level in TAILS:
                qv = sub[col].to_numpy()
                err = act - qv
                out[col] = (
                    round(float(np.mean(act <= qv)), 4),
                    round(float(np.mean(np.maximum(level * err, (level - 1) * err))), 4),
                )
            print(name, f"{arm:10s}", out)
            report[f"tails|{name}|{arm}"] = out
    print(
        "\n== checks: median forecast minus mean forecast, and absolute error of the median forecast =="
    )
    for arm in ("K0", "MASK_NEW", "MASK_UNION", "MASK_GATED", *SOFT):
        sub = long.filter(pl.col("arm") == arm).join(outside.select(KEYS), on=KEYS, how="inner")
        d = (
            sub.group_by("site")
            .agg((pl.col("q50") - pl.col("mean_forecast")).mean().alias("d"))
            .sort("site")
        )
        print(arm, {r["site"]: round(r["d"], 3) for r in d.iter_rows(named=True)})
        report[f"median_minus_mean|{arm}"] = {r["site"]: r["d"] for r in d.iter_rows(named=True)}
    for name, keys in (("outside", key_sets["outside"]), ("AFTER", key_sets["AFTER"])):
        mae = {}
        for arm in ("K0", "GAM_ONLY", "MASK_NEW", "MASK_UNION", "MASK_GATED", *SOFT, "BG2"):
            sub = long.filter(pl.col("arm") == arm).join(keys, on=KEYS, how="inner")
            mae[arm] = round(float((sub["q50"] - sub["actual"]).abs().mean()), 3)  # ty: ignore[invalid-argument-type]
        print("absolute error of the median forecast,", name, "shared rows:", mae)
        report[f"mae_median|{name}"] = mae

    decide(report=report, ev=ev, outside=outside, pairs=pairs)
    print("\n== D8 multi-day drops of experiment 159: alarms within 24 hours of a drop start ==")
    tables = ev_module.alarm_tables()
    drops = pl.read_parquet(D159 / "drops.parquet")
    starts = drops["first_stamp"].to_numpy()
    for name, table in tables.items():
        on = table.filter(pl.col("series") == D8_SERIES)["onset"].to_numpy()
        hit = [bool(np.any(np.abs(on - st) <= 48)) for st in starts]
        print(
            f"{name}: {sum(hit)} of {len(hit)} drops with an alarm onset within 24 h of the start; {len(on)} alarms on D8 in all"
        )
        report[f"d8|{name}"] = [sum(hit), len(hit), len(on)]
    (WORK_DIR / "report.json").write_text(json.dumps(report, indent=2, default=float))
    wide.write_parquet(WORK_DIR / "wide.parquet")
    ev.write_parquet(WORK_DIR / "event_rows.parquet")


if __name__ == "__main__":
    main()
