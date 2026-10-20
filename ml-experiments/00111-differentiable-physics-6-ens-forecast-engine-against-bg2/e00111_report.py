"""Score experiment 111's arms and print every number the write-up quotes.

Reads the arms that exist (a missing arm's contrasts are skipped and listed), puts the arms of each
contrast on the rows they share, and writes `work/report.md` and `work/results.json`. Intervals
resample whole calendar months of run initialisations, paired across arms: the four screening
contrasts C1 to C4 are read at 98.75% (a Bonferroni correction across four), the rest at 95%.

    uv run python ml-experiments/00111-*/e00111_report.py [--periods scored replicate]
"""

import argparse
import contextlib
from typing import Final

import e00111_common as common
import e00111_score as score
import numpy as np
import polars as pl

SCREENING: Final[dict[str, tuple[str, str]]] = {
    "C1 GAM-0 minus X-bar": ("gam0", "xnl"),
    "C2 GAM-K minus X-bar+K": ("gamk", "xnl_k"),
    "C3 GAM-margin minus Xlag-bar": ("gam_margin", "xlag_d"),
    "C4 GAM-K minus GAM-0": ("gamk", "gam0"),
}
EXPLORATORY: Final[dict[str, tuple[str, str]]] = {
    "GAM-K minus BG2 (not like for like)": ("gamk", "bg2"),
    "X-nl minus BG2 (SIMP1's lag gap)": ("xnl", "bg2"),
    "GAM-K minus Xlag-bar (not like for like)": ("gamk", "xlag_d"),
    "GAM-margin minus BG2's recipe is C5; not run": ("gam_margin", "bg2"),
    "GAM-K-era5 minus GAM-K": ("gamk_era5", "gamk"),
    "GAM-E minus GAM-K": ("game", "gamk"),
    "GAM-ens minus GAM-0": ("gamens", "gam0"),
    "X-nl-o minus X-nl-z": ("xnl_o", "xnl_z"),
    "X-nl-o minus X-nl": ("xnl_o", "xnl"),
    "Xlag-o minus Xlag-z": ("xlag_o", "xlag_z"),
    "Xlag-z minus Xlag-d": ("xlag_z", "xlag_d"),
    "X-nl+K minus X-nl": ("xnl_k", "xnl"),
}
DP_ARMS: Final[tuple[str, ...]] = ("gam0", "gamk", "game", "gamk_era5", "gamens")
XGB_ARMS: Final[tuple[str, ...]] = (
    "xnl_z",
    "xnl_o",
    "xlag_d",
    "xlag_z",
    "xlag_o",
    "gam_margin",
)
REFERENCES: Final[dict[str, str]] = {
    "bg2": "p_s6_bg2",
    "climatology": "ref_climatology",
    "manual_heuristic": "ref_manual_heuristic",
}
WINTER_MONTHS: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})


def load_all(*, periods: set[str], sites: tuple[str, ...]) -> dict[str, pl.DataFrame]:
    """Load every arm that exists for the periods and sites."""
    arms_found: dict[str, pl.DataFrame] = {}
    for name in DP_ARMS:
        try:
            arms_found[name] = score.load_dp(arm=name, periods=periods, sites=sites)
        except ValueError, FileNotFoundError:
            continue
    for name in ("xnl", *XGB_ARMS):
        try:
            frame = score.load_xgb(arm=name, periods=periods, sites=sites)
        except ValueError, FileNotFoundError:
            frame = None
        if frame is not None and frame.height:
            arms_found[name] = frame
    with contextlib.suppress(ValueError, FileNotFoundError):
        arms_found["xnl_k"] = score.load_xnl_k(periods=periods, sites=sites)
    if "scored" in periods:
        arms_found["xnl"] = score.load_simp1_xnl(sites=sites)
        for key, folder in REFERENCES.items():
            arms_found[key] = score.load_e87(name=folder, sites=sites)
    return arms_found


def summarise(
    *,
    name: str,
    pair: tuple[str, str],
    loaded: dict[str, pl.DataFrame],
    level: float,
    months: tuple[str, ...] = (),
) -> dict | None:
    """Return a contrast's estimate and interval overall, by lead band, and by robustness cut."""
    if pair[0] not in loaded or pair[1] not in loaded:
        return None
    shared = score.common_rows(arms_by_name={pair[0]: loaded[pair[0]], pair[1]: loaded[pair[1]]})
    if months:
        shared = {
            arm: frame.filter(
                pl.col("power_fcst_init_time").dt.strftime("%Y-%m").is_in(list(months))
            )
            for arm, frame in shared.items()
        }
    losses = score.stack_losses(arms_by_name=shared)
    out: dict = {
        "name": name,
        "treatment": pair[0],
        "reference": pair[1],
        "rows": shared[pair[0]].height,
    }
    out["quantile_score"] = score.contrast(
        losses=losses, treatment=pair[0], reference=pair[1], metric="qs", level=level
    )
    out["median_error"] = score.contrast(
        losses=losses, treatment=pair[0], reference=pair[1], metric="abs_median", level=95.0
    )
    out["absolute"] = {
        arm: float(score.quantile_score(frame=shared[arm]).to_numpy().mean()) for arm in pair
    }
    out["by_lead_band"] = {}
    for band in score.LEAD_BANDS:
        subset = {key: score.lead_filter(frame=frame, band=band) for key, frame in shared.items()}
        if subset[pair[0]].height == 0:
            continue
        out["by_lead_band"][band[0]] = score.contrast(
            losses=score.stack_losses(arms_by_name=subset),
            treatment=pair[0],
            reference=pair[1],
            level=95.0,
        )
    out["leave_one_site_out"] = {}
    cache = common.load_cache()
    for site in common.DECISION_SITES:
        series = cache.sites[site].table.time_series_id
        part = losses.filter(pl.col("site") != series)
        if part.filter(pl.col("arm") == pair[0]).height and part.height < losses.height:
            out["leave_one_site_out"][site] = score.contrast(
                losses=part, treatment=pair[0], reference=pair[1], level=95.0
            )
    month = losses["month"].str.slice(5, 2).cast(pl.Int32)
    for label, keep in (
        ("November to March", month.is_in(list(WINTER_MONTHS))),
        ("April to October", ~month.is_in(list(WINTER_MONTHS))),
    ):
        part = losses.filter(keep)
        if part.filter(pl.col("arm") == pair[0]).height:
            out.setdefault("seasons", {})[label] = score.contrast(
                losses=part, treatment=pair[0], reference=pair[1], level=95.0
            )
    return out


TAIL_LEVELS: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
TAIL_PROBABILITY: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}


def tail_table(*, loaded: dict[str, pl.DataFrame]) -> list[dict]:
    """Return coverage and pinball loss at the four tail levels, per arm and lead band.

    Coverage is the share of rows whose observed power lies at or below the level's quantile; the
    nominal value is the level itself. The pinball loss is in points of effective capacity.
    """
    rows = []
    for arm, frame in loaded.items():
        for band in score.LEAD_BANDS[:4]:
            part = score.lead_filter(frame=frame, band=band)
            if part.height == 0:
                continue
            actual = part["actual"].to_numpy().astype(np.float64)
            for level in TAIL_LEVELS:
                quantile = part[level].to_numpy().astype(np.float64)
                tau = TAIL_PROBABILITY[level]
                error = actual - quantile
                rows.append(
                    {
                        "arm": arm,
                        "lead_band": band[0],
                        "level": level,
                        "nominal": tau,
                        "coverage": float(np.mean(actual <= quantile)),
                        "pinball": float(np.mean(np.maximum(tau * error, (tau - 1.0) * error))),
                    }
                )
    return rows


def median_minus_mean(*, loaded: dict[str, pl.DataFrame], sites: tuple[str, ...]) -> list[dict]:
    """Return, per DP arm and site, the mean of (median forecast minus mean forecast), in points."""
    rows = []
    for arm in DP_ARMS:
        if arm not in loaded:
            continue
        pieces = []
        for period in ("scored",):
            for site in sites:
                path = score.dp_run.out_path(arm=arm, period=period, site=site)
                if path.exists():
                    pieces.append(
                        pl.read_parquet(path)
                        .filter(~pl.col("switching_mask"))
                        .select(site=pl.lit(site), gap=pl.col("q50") - pl.col("mean"))
                    )
        if pieces:
            table = pl.concat(pieces).group_by("site").agg(pl.col("gap").mean()).sort("site")
            rows += [
                {"arm": arm, "site": r["site"], "median_minus_mean": r["gap"]}
                for r in table.iter_rows(named=True)
            ]
    return rows


HEADLINE_ARMS: Final[tuple[str, ...]] = (
    "gam0",
    "gamk",
    "gam_margin",
    "xnl",
    "xnl_k",
    "xlag_d",
    "bg2",
    "climatology",
    "manual_heuristic",
)


def headline_scores(*, loaded: dict[str, pl.DataFrame]) -> list[dict]:
    """Return each headline arm's mean quantile score by site and lead day, on shared rows."""
    present = {arm: loaded[arm] for arm in HEADLINE_ARMS if arm in loaded}
    if len(present) < 2:
        return []
    shared = score.common_rows(arms_by_name=present)
    rows = []
    for arm, frame in shared.items():
        table = (
            frame.with_columns(
                score.quantile_score(frame=frame),
                day=((pl.col("lead_hours") - 1e-9) // 24).cast(pl.Int32) + 1,
            )
            .group_by("site", "day")
            .agg(pl.col("qs").mean(), n=pl.len())
            .sort("site", "day")
        )
        rows += [{"arm": arm, **r} for r in table.iter_rows(named=True)]
    return rows


def render_report(*, results: dict) -> list[str]:
    """Return the report's lines: every contrast, then the checks tables."""
    lines = ["# Experiment 111 report", ""]
    for item in results["contrasts"]:
        q = item["quantile_score"]
        kind = "planned" if item["planned"] else "exploratory"
        lines.append(
            f"- {item['name']} [{item['sites']}, {kind}]: "
            f"{q['estimate']:+.3f} ({q['lower']:+.3f} to {q['upper']:+.3f}), "
            f"{q['sites_better']} of {q['n_sites']} sites better, {item['rows']:,} rows; "
            f"median-forecast error {item['median_error']['estimate']:+.3f}"
        )
        absolute = ", ".join(f"{arm} {value:.3f}" for arm, value in item["absolute"].items())
        lines.append(f"    - absolute quantile scores: {absolute}")
        bands = "; ".join(
            f"{band} {c['estimate']:+.3f} ({c['lower']:+.3f} to {c['upper']:+.3f})"
            for band, c in item["by_lead_band"].items()
        )
        lines.append(f"    - by lead band: {bands}")
        seasons = "; ".join(
            f"{name} {c['estimate']:+.3f}" for name, c in item.get("seasons", {}).items()
        )
        lines.append(f"    - by season: {seasons}")
        loso = "; ".join(
            f"without {site} {c['estimate']:+.3f} ({c['lower']:+.3f} to {c['upper']:+.3f})"
            for site, c in item["leave_one_site_out"].items()
        )
        lines.append(f"    - leave one site out: {loso}")
    lines += ["", "Skipped (arm missing): " + "; ".join(results["skipped"])]
    for sites_label in ("decision",):
        lines += ["", f"## Median forecast minus mean forecast ({sites_label} sites, points)"]
        table = pl.DataFrame(results[f"median_minus_mean_{sites_label}"])
        if table.height:
            lines.append(table.pivot(on="site", index="arm", values="median_minus_mean").__str__())
        lines += [
            "",
            f"## Tails ({sites_label} sites): coverage at q01, q05, q95, q99 by lead band",
        ]
        tails = pl.DataFrame(results[f"tails_{sites_label}"])
        if tails.height:
            wide = tails.pivot(on="level", index=["arm", "lead_band"], values="coverage")
            lines.append(wide.__str__())
            pinball = tails.pivot(on="level", index=["arm", "lead_band"], values="pinball")
            lines += ["", "Pinball loss at the same levels (points):", pinball.__str__()]
    return lines


def main() -> None:
    """Score and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", nargs="+", default=["scored"])
    parser.add_argument(
        "--months", nargs="+", default=[], help="keep only runs of these months, as YYYY-MM"
    )
    args = parser.parse_args()
    periods = set(args.periods)
    results: dict = {"periods": sorted(periods), "contrasts": [], "skipped": []}
    for sites_label, sites in (
        ("decision", common.DECISION_SITES),
        ("scored", common.SCORED_SITES),
    ):
        loaded = load_all(periods=periods, sites=sites)
        results[f"arms_{sites_label}"] = sorted(loaded)
        results[f"tails_{sites_label}"] = tail_table(loaded=loaded)
        results[f"headline_scores_{sites_label}"] = headline_scores(loaded=loaded)
        results[f"median_minus_mean_{sites_label}"] = median_minus_mean(loaded=loaded, sites=sites)
        for name, pair in SCREENING.items():
            summary = summarise(
                name=name,
                pair=pair,
                loaded=loaded,
                level=score.BONFERRONI_LEVEL,
                months=tuple(args.months),
            )
            if summary is None:
                results["skipped"].append(f"{name} at {sites_label}")
                continue
            summary["sites"] = sites_label
            summary["planned"] = True
            results["contrasts"].append(summary)
        for name, pair in EXPLORATORY.items():
            summary = summarise(
                name=name, pair=pair, loaded=loaded, level=95.0, months=tuple(args.months)
            )
            if summary is None:
                results["skipped"].append(f"{name} at {sites_label}")
                continue
            summary["sites"] = sites_label
            summary["planned"] = False
            results["contrasts"].append(summary)
    tag = "" if periods == {"scored"} else "_" + "_".join(sorted(periods))
    if args.months:
        tag += "_" + "_".join(args.months)
    common.write_json_atomic(path=common.WORK_DIR / f"results{tag}.json", content=results)
    lines = render_report(results=results)
    (common.WORK_DIR / f"report{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
