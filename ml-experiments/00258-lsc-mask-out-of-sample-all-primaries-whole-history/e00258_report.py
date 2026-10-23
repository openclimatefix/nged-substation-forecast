# ruff: noqa: E501, PLR0912, PLR0915
"""Gain after load shifts, ordinary-day harm, and the net annual figure from the scored tables.

Reads the tables of `e00258_score.py` and the gates of `e00258_gates.py`, applies the plan's verdict
rule, and writes the printed report and the tables the charts read.

    uv run --no-sync python e00258_report.py
"""

from __future__ import annotations

import json
from typing import Final

import e00258_data as dat
import e00258_harness as h
import e00258_score as sc
import numpy as np
import polars as pl

N_DRAWS: Final[int] = 20_000
N_DRAWS_LOO: Final[int] = 2_000
SEED: Final[int] = 258
BASES: Final[tuple[str, ...]] = ("LR_b20", "LR_b80", "RO_b80", "HALF", "GAM_ONLY", "ORACLE")
FOLD_BASES: Final[frozenset[str]] = frozenset({"LR_b20", "LR_b80", "HALF"})
HARM_LIMIT: Final[float] = 0.05
SHARE_BAR: Final[float] = 0.5
LONG_HOURS: Final[float] = 24.0
METRIC_COLUMNS: Final[tuple[str, ...]] = sc.METRICS
OUT: Final = h.HARNESS_DIR


def base_of(arm: str) -> str:
    """The arm name without its fold suffix."""
    return arm.rsplit("_f", 1)[0] if "_f" in arm else arm


def load_after() -> pl.DataFrame:
    """Sums per (card, series, arm), over lead bands, with the arm's base name and fold."""
    frame = pl.concat([pl.read_parquet(h.HARNESS_DIR / f"after_{s}.parquet") for s in h.SERIES])
    return frame.with_columns(
        base=pl.col("arm").map_elements(base_of, return_dtype=pl.String),
    )


def load_ordinary() -> pl.DataFrame:
    """Sums per (series, variant, month, base): fold arms averaged over their folds."""
    frame = pl.concat([pl.read_parquet(h.HARNESS_DIR / f"ordinary_{s}.parquet") for s in h.SERIES])
    frame = frame.with_columns(base=pl.col("arm").map_elements(base_of, return_dtype=pl.String))
    return frame.group_by("series", "variant", "year_month", "odd", "base").agg(
        n=pl.col("n").first(), **{m: pl.col(m).mean() for m in METRIC_COLUMNS}
    )


def after_units() -> pl.DataFrame:
    """One row per (load shift, series) with the sums of every base arm, using the pair's fold."""
    events = h.labelled_events()
    folds = dat.pair_folds()
    after = load_after()
    rows = []
    for e in events.iter_rows(named=True):
        fold = folds[e["pair"]]
        for s in e["series"]:
            if s not in h.SERIES:
                continue
            unit = after.filter((pl.col("card") == e["card_id"]) & (pl.col("series") == s))
            if unit.is_empty():
                continue
            row = {
                "card": e["card_id"],
                "series": s,
                "pair": e["pair"],
                "hours": float(e["hours"]),
                "n": int(unit.filter(pl.col("arm") == "K0")["n"].sum()),
            }
            for base in ("K0", *BASES):
                arm = f"{base}_f{fold}" if base in FOLD_BASES else base
                row[f"qs_{base}"] = float(unit.filter(pl.col("arm") == arm)["qs"].sum())
            rows.append(row)
    return pl.DataFrame(rows)


def interval(draws: np.ndarray) -> tuple[float, float]:
    """The 2.5% and 97.5% points of bootstrap draws."""
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return float(low), float(high)


def pair_bootstrap(*, units: pl.DataFrame, n_draws: int, rng: np.random.Generator) -> dict:
    """Pooled gain and share of ORACLE's gain per base arm, resampling whole Primary pairs."""
    pairs = sorted(units["pair"].unique())
    g = {
        base: np.array(
            [
                units.filter(pl.col("pair") == p)
                .select(pl.col("qs_K0") - pl.col(f"qs_{base}"))
                .sum()
                .item()
                for p in pairs
            ]
        )
        for base in BASES
    }
    n = np.array([units.filter(pl.col("pair") == p)["n"].sum() for p in pairs], dtype=float)
    draw = rng.integers(0, len(pairs), size=(n_draws, len(pairs)))
    total_n = n[draw].sum(axis=1)
    denominator = g["ORACLE"][draw].sum(axis=1)
    out = {}
    for base in BASES:
        numerator = g[base][draw].sum(axis=1)
        point_gain = g[base].sum() / n.sum()
        point_share = g[base].sum() / g["ORACLE"].sum()
        out[base] = {
            "gain": float(point_gain),
            "gain_interval": interval(numerator / total_n),
            "share": float(point_share),
            "share_interval": interval(numerator / denominator),
        }
    out["rows"] = int(n.sum())
    out["n_pairs"] = len(pairs)
    return out


def month_bootstrap(*, cells: pl.DataFrame, n_draws: int, rng: np.random.Generator) -> dict:
    """Mean score difference per base arm over the cells, resampling whole calendar months."""
    months = sorted(cells["year_month"].unique())
    out = {}
    n_month = np.array(
        [
            cells.filter((pl.col("year_month") == m) & (pl.col("base") == "K0"))["n"].sum()
            for m in months
        ],
        dtype=float,
    )
    draw = rng.integers(0, len(months), size=(n_draws, len(months)))
    for base in BASES:
        diff = np.array(
            [
                cells.filter((pl.col("year_month") == m) & (pl.col("base") == base))["qs"].sum()
                - cells.filter((pl.col("year_month") == m) & (pl.col("base") == "K0"))["qs"].sum()
                for m in months
            ]
        )
        out[base] = {
            "harm": float(diff.sum() / n_month.sum()),
            "interval": interval(diff[draw].sum(axis=1) / n_month[draw].sum(axis=1)),
        }
    out["rows"] = int(n_month.sum())
    out["n_months"] = len(months)
    return out


def main() -> None:
    """Write the printed report and the tables."""
    rng = np.random.default_rng(SEED)
    gates = json.loads((OUT / "gates.json").read_text())
    units = after_units()
    units.write_parquet(OUT / "after_units.parquet")
    long_units = units.filter(pl.col("hours") > LONG_HOURS)
    short_units = units.filter(pl.col("hours") <= LONG_HOURS)
    ordinary = load_ordinary()
    report: dict = {"gates": gates}

    print("== AFTER: gain over K0 in the 72 hours after a labelled load shift ==")
    headline = pair_bootstrap(units=long_units, n_draws=N_DRAWS, rng=rng)
    short = pair_bootstrap(units=short_units, n_draws=N_DRAWS_LOO, rng=rng)
    report["after_long"], report["after_short"] = headline, short
    print(
        f"35 load shifts over 1 day: {long_units['card'].n_unique()} shifts, "
        f"{long_units.height} shift-series units, {headline['rows']} rows, {headline['n_pairs']} pairs"
    )
    for base in BASES:
        r = headline[base]
        print(
            f"  {base:9s} gain {r['gain']:+.3f} ({r['gain_interval'][0]:+.3f} to {r['gain_interval'][1]:+.3f})"
            f"  share {100 * r['share']:6.1f}% ({100 * r['share_interval'][0]:.1f}% to {100 * r['share_interval'][1]:.1f}%)"
        )
    print(f"shorter load shifts: {short_units['card'].n_unique()} shifts, {short['rows']} rows")
    for base in BASES:
        r = short[base]
        print(f"  {base:9s} gain {r['gain']:+.3f}  share {100 * r['share']:6.1f}%")
    loo_after = {}
    for s in sorted(long_units["series"].unique()):
        rest = long_units.filter(pl.col("series") != s)
        loo_after[int(s)] = pair_bootstrap(units=rest, n_draws=N_DRAWS_LOO, rng=rng)
    report["after_loo"] = loo_after
    print("leave-one-Primary-out share of ORACLE's gain, LR_b20:")
    for s, r in loo_after.items():
        print(
            f"  without {dat.label_of(series=s):9s} {100 * r['LR_b20']['share']:6.1f}% "
            f"({100 * r['LR_b20']['share_interval'][0]:.1f}% to {100 * r['LR_b20']['share_interval'][1]:.1f}%)"
        )

    per_shift = (
        long_units.group_by("card")
        .agg(
            hours=pl.col("hours").first(),
            pair=pl.col("pair").first(),
            n=pl.col("n").sum(),
            **{f"gain_{b}": (pl.col("qs_K0") - pl.col(f"qs_{b}")).sum() for b in BASES},
        )
        .with_columns(
            **{
                f"share_{b}": pl.col(f"gain_{b}") / pl.col("gain_ORACLE")
                for b in BASES
                if b != "ORACLE"
            },
            pooled_gain_ORACLE=pl.col("gain_ORACLE") / pl.col("n"),
        )
        .sort("gain_ORACLE", descending=True)
    )
    per_shift.write_parquet(OUT / "after_per_shift.parquet")
    positive = per_shift.filter(pl.col("gain_ORACLE") > 0)
    report["median_share"] = {
        b: float(positive[f"share_{b}"].median()) for b in BASES if b != "ORACLE"
    }
    print(
        f"median per-shift share (shifts where ORACLE gains, n={positive.height}):",
        {b: round(100 * v, 1) for b, v in report["median_share"].items()},
    )
    top = per_shift.sort("gain_ORACLE", descending=True)
    for base in ("LR_b20", "ORACLE"):
        biggest = per_shift.sort(f"gain_{base}", descending=True)[f"gain_{base}"]
        report[f"top_shift_share_{base}"] = float(biggest[0] / biggest.sum())
        print(
            f"largest single shift's share of {base}'s total gain: {biggest[0] / biggest.sum():.3f}"
        )
    print(
        "top 3 shifts' share of ORACLE's total gain:",
        [
            round(float(x), 3)
            for x in (top["gain_ORACLE"].head(3) / top["gain_ORACLE"].sum()).to_list()
        ],
    )

    print("\n== ORDINARY rows: score difference from K0 (points; negative is an improvement) ==")
    report["harm"] = {}
    for variant in (
        "ordinary",
        "ordinary_clean",
        "ordinary_no_d8_d9",
        "ordinary_wide",
        "ordinary_no_balanced",
    ):
        for odd in (True, False):
            cells = ordinary.filter(
                (pl.col("variant") == variant) & (pl.col("odd") == odd)
            ).with_columns(qs=pl.col("qs"))
            result = month_bootstrap(cells=cells, n_draws=N_DRAWS, rng=rng)
            report["harm"][f"{variant}|{'odd' if odd else 'even'}"] = result
            tag = "odd (out of sample)" if odd else "even (in sample)"
            print(f"{variant:21s} {tag:20s} rows {result['rows']}")
            for base in BASES:
                r = result[base]
                print(
                    f"   {base:9s} {r['harm']:+.4f} ({r['interval'][0]:+.4f} to {r['interval'][1]:+.4f})"
                )
    cells = ordinary.filter((pl.col("variant") == "ordinary") & pl.col("odd"))
    by_primary: dict[str, dict] = {}
    loo_harm = {}
    for s in h.SERIES:
        for odd in (True, False):
            one = ordinary.filter(
                (pl.col("variant") == "ordinary") & (pl.col("odd") == odd) & (pl.col("series") == s)
            )
            by_primary[f"{s}|{'odd' if odd else 'even'}"] = month_bootstrap(
                cells=one, n_draws=N_DRAWS_LOO, rng=rng
            )
        loo_harm[s] = month_bootstrap(
            cells=cells.filter(pl.col("series") != s), n_draws=N_DRAWS_LOO, rng=rng
        )
    report["harm_by_primary"], report["harm_loo"] = by_primary, loo_harm
    print("\nodd-month harm of LR_b20 by Primary:")
    for s in h.SERIES:
        r = by_primary[f"{s}|odd"]["LR_b20"]
        print(
            f"  {dat.label_of(series=s):9s} {r['harm']:+.4f} ({r['interval'][0]:+.4f} to {r['interval'][1]:+.4f})"
            f"  without it, all others: {loo_harm[s]['LR_b20']['harm']:+.4f}"
        )

    print("\n== Net annual figure: every row of each series-year, difference from K0 (points) ==")
    all_rows = ordinary.filter(pl.col("variant") == "all_rows").with_columns(
        year=pl.col("year_month") // 100
    )
    net = (
        all_rows.group_by("series", "year", "base")
        .agg(n=pl.col("n").sum(), qs=pl.col("qs").sum())
        .pivot(on="base", index=["series", "year", "n"], values="qs")
        .with_columns(**{f"d_{b}": (pl.col(b) - pl.col("K0")) / pl.col("n") for b in BASES})
        .sort("series", "year")
    )
    net.write_parquet(OUT / "net_annual.parquet")
    pooled = (
        net.group_by("year")
        .agg(
            n=pl.col("n").sum(),
            **{
                f"d_{b}": ((pl.col(b) - pl.col("K0")).sum() / pl.col("n").sum())
                for b in ("LR_b20", "GAM_ONLY")
            },
        )
        .sort("year")
    )
    print(pooled)
    report["net_annual"] = pooled.to_dicts()
    overall = {
        b: float((net[b] - net["K0"]).sum() / net["n"].sum())
        for b in ("LR_b20", "LR_b80", "RO_b80", "HALF", "GAM_ONLY", "ORACLE")
    }
    report["net_overall"] = overall
    print(
        "net over all years and series (arm minus K0, points):",
        {k: round(v, 4) for k, v in overall.items()},
    )
    by_series = (
        net.group_by("series")
        .agg(n=pl.col("n").sum(), d=(pl.col("LR_b20") - pl.col("K0")).sum())
        .with_columns(share=pl.col("d") / pl.col("d").sum(), mean=pl.col("d") / pl.col("n"))
        .sort("d", descending=True)
    )
    print("LR_b20 net loss by series (share of the sum over series):")
    print(by_series.head(5))
    report["net_by_series"] = by_series.to_dicts()

    print("\n== Tails and bias (long load shifts AFTER rows, and odd-month ordinary rows) ==")
    after = load_after()
    tails = []
    for base in ("K0", *BASES):
        arm_rows = after.filter(
            pl.col("card").is_in(long_units["card"].unique()) & (pl.col("base") == base)
        )
        if base in FOLD_BASES:
            fold_of_card = {
                e["card_id"]: dat.pair_folds()[e["pair"]]
                for e in h.labelled_events().iter_rows(named=True)
            }
            arm_rows = arm_rows.filter(
                pl.col("arm")
                == pl.col("card").replace_strict(
                    {c: f"{base}_f{f}" for c, f in fold_of_card.items()}, return_dtype=pl.String
                )
            )
        n = arm_rows["n"].sum()
        tails.append(
            {
                "set": "after_long",
                "arm": base,
                "n": int(n),
                **{
                    f"below_q{h.LEVELS[i]:.2f}": float(arm_rows[f"below_{i}"].sum() / n)
                    for i in sc.TAIL_LEVELS
                },
                **{
                    f"pin_q{h.LEVELS[i]:.2f}": float(arm_rows[f"pin_{i}"].sum() / n)
                    for i in sc.TAIL_LEVELS
                },
                "err_mean": float(arm_rows["err_mean"].sum() / n),
                "err_median": float(arm_rows["err_median"].sum() / n),
                "abs_mean": float(arm_rows["abs_mean"].sum() / n),
                "abs_median": float(arm_rows["abs_median"].sum() / n),
                "qs": float(arm_rows["qs"].sum() / n),
            }
        )
    odd = ordinary.filter((pl.col("variant") == "ordinary") & pl.col("odd"))
    for base in ("K0", *BASES):
        arm_rows = odd.filter(pl.col("base") == base)
        n = arm_rows["n"].sum()
        tails.append(
            {
                "set": "ordinary_odd",
                "arm": base,
                "n": int(n),
                **{
                    f"below_q{h.LEVELS[i]:.2f}": float(arm_rows[f"below_{i}"].sum() / n)
                    for i in sc.TAIL_LEVELS
                },
                **{
                    f"pin_q{h.LEVELS[i]:.2f}": float(arm_rows[f"pin_{i}"].sum() / n)
                    for i in sc.TAIL_LEVELS
                },
                "err_mean": float(arm_rows["err_mean"].sum() / n),
                "err_median": float(arm_rows["err_median"].sum() / n),
                "abs_mean": float(arm_rows["abs_mean"].sum() / n),
                "abs_median": float(arm_rows["abs_median"].sum() / n),
                "qs": float(arm_rows["qs"].sum() / n),
            }
        )
    tail_frame = pl.DataFrame(tails)
    tail_frame.write_parquet(OUT / "tails.parquet")
    print(
        tail_frame.select(
            "set", "arm", "n", "qs", "abs_mean", "abs_median", "err_mean", "err_median"
        )
    )
    print("tails: share of readings below each tail quantile, and the pinball loss at it")
    print(tail_frame.select("set", "arm", pl.col("^below_.*$"), pl.col("^pin_.*$")))
    for name in ("after_long", "ordinary_odd"):
        one = tail_frame.filter(pl.col("set") == name)
        k0 = one.filter(pl.col("arm") == "K0").row(0, named=True)
        print(f"direction check, {name}: K0 minus arm (positive is an improvement)")
        for r in one.iter_rows(named=True):
            if r["arm"] != "K0":
                print(
                    f"   {r['arm']:9s} qs {k0['qs'] - r['qs']:+.4f}  mean abs error "
                    f"{k0['abs_mean'] - r['abs_mean']:+.4f}  median abs error {k0['abs_median'] - r['abs_median']:+.4f}"
                )
    k0_odd = (
        odd.filter(pl.col("base") == "K0")
        .group_by("series")
        .agg(
            n=pl.col("n").sum(),
            err_mean=pl.col("err_mean").sum() / pl.col("n").sum(),
            err_median=pl.col("err_median").sum() / pl.col("n").sum(),
        )
        .sort("series")
    )
    print("K0 odd-month ordinary rows, mean and median forecast minus reading, by series:")
    print(k0_odd)
    report["median_minus_mean"] = k0_odd.to_dicts()

    spans = pl.concat(
        [
            pl.read_parquet(OUT / f"spans_{s}.parquet").with_columns(series=pl.lit(s))
            for s in h.SERIES
        ]
    )
    folds_of = dat.series_folds()
    span_counts = {}
    for arm, group in spans.group_by("arm"):
        name = arm[0]
        weights = np.array(
            [
                1.0 / len(folds_of[s]) if base_of(name) in FOLD_BASES else 1.0
                for s in group["series"]
            ]
        )
        span_counts[name] = float(weights.sum())
    summed: dict[str, float] = {}
    for name, value in span_counts.items():
        summed[base_of(name)] = summed.get(base_of(name), 0.0) + value
    report["alarms"] = summed
    print(
        "\nalarms (spans) per source, fold arms weighted by 1 over the folds of the series:", summed
    )

    ordinary_alarms = {"all": 0.0, "on_ordinary_stamp": 0.0, "balanced_on_ordinary_stamp": 0.0}
    for s in h.SERIES:
        stamp_ok = ~sc.closure_blocks(series=s, neighbours=False)
        for fold in folds_of[s]:
            table = dat.alarm_tables(source="LR_b20")[fold].filter(pl.col("series") == s)
            for r in table.iter_rows(named=True):
                w = 1.0 / len(folds_of[s])
                ordinary_alarms["all"] += w
                if stamp_ok[min(int(r["onset"]), h.N_GRID - 1)]:
                    ordinary_alarms["on_ordinary_stamp"] += w
                    ratio, n_opp = r.get("opp_ratio"), r.get("n_opposite")
                    if (
                        ratio is not None
                        and n_opp is not None
                        and n_opp >= 1
                        and sc.BALANCED_RANGE[0] <= ratio <= sc.BALANCED_RANGE[1]
                    ):
                        ordinary_alarms["balanced_on_ordinary_stamp"] += w
    report["ordinary_alarms"] = ordinary_alarms
    print("LR_b20 alarms (fold-weighted):", ordinary_alarms)

    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))
    verdict_inputs = {
        "gates_pass": gates["G1"]["passes"] and gates["G2"]["passes"],
        "share": headline["LR_b20"]["share"],
        "share_low": headline["LR_b20"]["share_interval"][0],
        "harm": report["harm"]["ordinary|odd"]["LR_b20"],
    }
    print("\nverdict inputs:", verdict_inputs)


if __name__ == "__main__":
    main()
