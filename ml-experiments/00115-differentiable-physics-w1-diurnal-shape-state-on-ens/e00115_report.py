"""Score experiment 115 and print every number its write-up quotes.

Writes `work/results.json` and prints the same numbers as text; `docs/ml-experiments/00115-*/` pages
quote this output only. Intervals resample whole calendar months, paired across arms. The four
planned contrasts use 98.75% intervals (Bonferroni across four); every other contrast uses 95%.

    uv run python ml-experiments/00115-*/e00115_report.py [--part core|exploratory|tails|checks]
"""

import argparse
import json
from typing import Final

import e00115_common as common
import e00115_score as score
import numpy as np
import polars as pl
from studies.tail_calibration import tail_calibration_table

PERIODS: Final[tuple[str, ...]] = ("replicate", "scored")
PLANNED: Final[dict[str, tuple[str, str]]] = {
    "C1 shape state (KW - K0)": ("KW", "K0"),
    "C2 lead-band tuning (KW-lb - KW)": ("KW-lb", "KW"),
    "C3 newest intraday readings (KW - KW-day)": ("KW", "KW-day"),
    "C4 state against recipe (KW - R)": ("KW", "R"),
}
EXPLORATORY: Final[dict[str, tuple[str, str]]] = {
    "K0-lb - K0": ("K0-lb", "K0"),
    "KS - KW": ("KS", "KW"),
    "KW-nomid - KW": ("KW-nomid", "KW"),
    "KS1 - KW": ("KS1", "KW"),
    "R75 - K0": ("R75", "K0"),
    "KW-d0 - K0-d0": ("KW-d0", "K0-d0"),
    "KW-d24 - K0-d24": ("KW-d24", "K0-d24"),
    "KW-d33 - K0-d33": ("KW-d33", "K0-d33"),
    "KW - K0 at 3.5 h": ("KW", "K0"),
    "KW-pl52 - K0": ("KW-pl52", "K0"),
    "KW-pl52 - KW": ("KW-pl52", "KW"),
    "R - K0": ("R", "K0"),
    "KW-fast - K0 (post hoc)": ("KW-fast", "K0"),
    "KW-fast - KW (post hoc)": ("KW-fast", "KW"),
    "KW-fast - R (post hoc)": ("KW-fast", "R"),
}
WINTER: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})
HEADLINE: Final[str] = "D+1"
BAND_LABELS: Final[tuple[str, ...]] = ("night", "morning", "midday", "evening")
RESULTS_PATH = common.WORK_DIR / "results.json"


def load(*, arms: list[str], period: str, sites: tuple[str, ...]) -> dict[str, pl.DataFrame]:
    """Load the W1 arms for a period."""
    return {arm: score.load_arm(arm=arm, period=period, sites=sites) for arm in arms}


def day_type_of(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the day type (weekday, Saturday, Sunday, bank holiday) of each row's valid time."""
    cache = common.load_cache()
    position = np.searchsorted(
        cache.end_times, frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    )
    return cache.calendar.day_type[np.minimum(position, len(cache.end_times) - 1)]


def contrast_block(
    *,
    pair: tuple[str, str],
    frames: dict[str, pl.DataFrame],
    level: float,
    robust: bool,
) -> dict | None:
    """Return a contrast overall, by lead band, and (if asked) across the robustness rows."""
    treatment, reference = pair
    if treatment not in frames or reference not in frames:
        return None
    shared = score.common_rows(
        arms_by_name={treatment: frames[treatment], reference: frames[reference]}
    )
    out: dict = {"treatment": treatment, "reference": reference, "bands": {}}
    for band in score.LEAD_BANDS:
        subset = {k: score.lead_filter(frame=f, band=band) for k, f in shared.items()}
        if subset[treatment].height == 0:
            continue
        losses = score.stack_losses(arms_by_name=subset)
        entry = {
            "qs": score.contrast(
                losses=losses, treatment=treatment, reference=reference, level=level
            ),
            "mae_median": score.contrast(
                losses=losses, treatment=treatment, reference=reference, metric="abs_median"
            ),
            "absolute_qs": {
                arm: score.mean_of(score.quantile_score(frame=subset[arm])) for arm in pair
            },
            "absolute_mae_median": {
                arm: score.mean_of((subset[arm]["actual"] - subset[arm]["q50"]).abs())
                for arm in pair
            },
        }
        if "abs_mean" in losses.columns:
            entry["mae_mean"] = score.contrast(
                losses=losses, treatment=treatment, reference=reference, metric="abs_mean"
            )
        out["bands"][band[0]] = entry
    if robust:
        out["robust"] = robustness(pair=pair, shared=shared, level=level)
    return out


def robustness(*, pair: tuple[str, str], shared: dict[str, pl.DataFrame], level: float) -> dict:
    """Return a contrast at D+1 on the robustness rows."""
    treatment, reference = pair
    d1 = {k: score.lead_filter(frame=f, band=score.LEAD_BANDS[0]) for k, f in shared.items()}
    out: dict = {"leave_one_site_out": {}, "season": {}, "hour_band": {}, "day_type": {}}

    def run(subset: dict[str, pl.DataFrame]) -> dict | None:
        if subset[treatment].height < 200:
            return None
        return score.contrast(
            losses=score.stack_losses(arms_by_name=subset),
            treatment=treatment,
            reference=reference,
            level=level,
        )

    for site in common.SCORED_SITES:
        subset = {k: f.filter(pl.col("site") != site) for k, f in d1.items()}
        out["leave_one_site_out"][site] = run(subset)
    month = {k: f["valid_time"].dt.month().to_numpy() for k, f in d1.items()}
    for label, wanted in (("November to March", True), ("April to October", False)):
        subset = {
            k: f.filter(pl.Series(np.isin(month[k], list(WINTER)) == wanted)) for k, f in d1.items()
        }
        out["season"][label] = run(subset)
    for band_index, label in enumerate(BAND_LABELS):
        subset = {k: f.filter(pl.col("hour_band") == band_index) for k, f in d1.items()}
        out["hour_band"][label] = run(subset)
    day = {k: day_type_of(frame=f) for k, f in d1.items()}
    for label, wanted in (("working days", True), ("weekends and bank holidays", False)):
        subset = {k: f.filter(pl.Series((day[k] == 0) == wanted)) for k, f in d1.items()}
        out["day_type"][label] = run(subset)
    return out


def planned_contrasts() -> dict:
    """Return C1 to C4 for the replicate months and the scored year."""
    names = sorted({a for pair in PLANNED.values() for a in pair})
    out: dict = {}
    for period in PERIODS:
        frames = load(arms=names, period=period, sites=common.DECISION_SITES)
        out[period] = {
            label: contrast_block(
                pair=pair, frames=frames, level=score.BONFERRONI_LEVEL, robust=True
            )
            for label, pair in PLANNED.items()
        }
        print(period, "planned contrasts done", flush=True)
    return out


REPORTED_ARMS: Final[tuple[str, ...]] = (
    "K0",
    "KW",
    "KW-lb",
    "K0-lb",
    "KW-day",
    "R",
    "R75",
    "KS",
    "KW-nomid",
    "KS1",
    "KW-pl52",
    "KW-d0",
    "KW-d24",
    "KW-d33",
    "K0-d0",
    "K0-d24",
    "K0-d33",
    "KW-fast",
)
TAIL_ARMS: Final[tuple[str, ...]] = ("K0", "KW", "KW-lb", "R", "KW-day", "KW-shift")
REFERENCES: Final[dict[str, str]] = {
    "GAM-K (stored)": "gamk",
    "GAM-0": "gam0",
    "X-nl+K": "xnl_k",
    "Xlag-d": "xlag_d",
    "X-nl": "xnl",
}
E87_REFERENCES: Final[dict[str, str]] = {
    "BG2": "p_s6_bg2",
    "climatology": "ref_climatology",
    "manual_heuristic": "ref_manual_heuristic",
}
TAIL_LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.95, 0.99)


def available(*, arms: tuple[str, ...]) -> list[str]:
    """Return the arms whose frames exist at every scored site."""
    import e00115_tune as tune

    return [
        a for a in arms if all(tune.frame_path(arm=a, site=s).exists() for s in common.SCORED_SITES)
    ]


def with_shift_only(*, frames: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Add `KW-shift`: K0's quantiles moved by KW's correction.

    The correction is the same for every ensemble member, so the shift is the difference of the two
    arms' corrected mean forecasts. The spread, and so the tails, stay K0's.
    """
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    joined = frames["K0"].join(
        frames["KW"].select([*keys, pl.col("mean").alias("kw_mean")]), on=keys, how="inner"
    )
    shift = joined["kw_mean"] - joined["mean"]
    shifted = (
        joined.with_columns([(pl.col(q) + shift).alias(q) for q in score.LEVEL_COLUMNS])
        .with_columns(mean=pl.col("kw_mean"))
        .drop("kw_mean")
    )
    return {**frames, "KW-shift": shifted}


def absolute_table(*, frames: dict[str, pl.DataFrame]) -> dict:
    """Return each arm's mean quantile score and MAE of the median by lead band, on common rows."""
    shared = score.common_rows(arms_by_name=frames)
    out: dict = {}
    for arm, frame in shared.items():
        out[arm] = {}
        for band in score.LEAD_BANDS:
            subset = score.lead_filter(frame=frame, band=band)
            out[arm][band[0]] = {
                "qs": score.mean_of(score.quantile_score(frame=subset)),
                "mae_median": score.mean_of((subset["actual"] - subset["q50"]).abs()),
                "rows": subset.height,
            }
    return out


def tails_table(*, frames: dict[str, pl.DataFrame]) -> dict:
    """Return coverage and pinball at q01, q05, q95, and q99 per arm and lead band."""
    shared = score.common_rows(arms_by_name=frames)
    out: dict = {}
    for arm, frame in shared.items():
        out[arm] = {}
        for band in score.LEAD_BANDS[:5]:
            subset = score.lead_filter(frame=frame, band=band)
            table = tail_calibration_table(frame=subset, levels=TAIL_LEVELS)
            out[arm][band[0]] = table.to_dicts()
    return out


def checks_table(*, frames: dict[str, pl.DataFrame]) -> dict:
    """Return the median forecast minus the mean forecast per site and arm, at D+1."""
    out: dict = {}
    for arm, frame in frames.items():
        subset = score.lead_filter(frame=frame, band=score.LEAD_BANDS[0])
        if "mean" not in subset.columns:
            continue
        grouped = subset.group_by("site").agg(gap=(pl.col("q50") - pl.col("mean")).mean())
        out[arm] = dict(zip(grouped["site"].to_list(), grouped["gap"].to_list(), strict=True))
    return out


def exploratory(*, period: str) -> dict:
    """Return the exploratory contrasts of a period at 95%."""
    names = sorted({a for pair in EXPLORATORY.values() for a in pair})
    names = [a for a in names if a in available(arms=tuple(names))]
    frames = load(arms=names, period=period, sites=common.DECISION_SITES)
    out = {}
    for label, pair in EXPLORATORY.items():
        block = contrast_block(pair=pair, frames=frames, level=95.0, robust=False)
        if block is not None:
            out[label] = block
    return out


def reference_scores(*, period: str, arms: list[str]) -> dict:
    """Return the stored references and the W1 arms on their common rows, and best-arm contrasts."""
    sites = common.DECISION_SITES
    frames = load(arms=arms, period=period, sites=sites)
    for label, name in REFERENCES.items():
        try:
            frames[label] = score.stored_reference(name=name, period=period, sites=sites)
        except ValueError, FileNotFoundError:
            continue
    if period == "scored":
        for label, folder in E87_REFERENCES.items():
            frames[label] = score.e87_reference(folder=folder, sites=sites)
    shared = score.common_rows(arms_by_name=frames)
    out: dict = {"absolute": {}, "rows": next(iter(shared.values())).height}
    for arm, frame in shared.items():
        out["absolute"][arm] = {
            band[0]: score.mean_of(
                score.quantile_score(frame=score.lead_filter(frame=frame, band=band))
            )
            for band in score.LEAD_BANDS
        }
    return out


def reference_contrasts(*, period: str) -> dict:
    """Return KW, R, and K0 minus each stored reference at 95% on the rows both hold."""
    sites = common.DECISION_SITES
    frames = load(arms=["K0", "KW", "R", "KW-fast"], period=period, sites=sites)
    for label, name in REFERENCES.items():
        frames[label] = score.stored_reference(name=name, period=period, sites=sites)
    if period == "scored":
        for label, folder in E87_REFERENCES.items():
            frames[label] = score.e87_reference(folder=folder, sites=sites)
    out = {}
    for treatment in ("K0", "KW", "R", "KW-fast"):
        for reference in frames:
            if reference in {"K0", "KW", "R", "KW-fast"}:
                continue
            block = contrast_block(
                pair=(treatment, reference),
                frames={treatment: frames[treatment], reference: frames[reference]},
                level=95.0,
                robust=False,
            )
            if block is not None:
                out[f"{treatment} - {reference}"] = block
    return out


COLD_SNAP_C: Final[float] = 3.0
"""A cold-snap day is a local day from November to March whose mean ERA5 temperature at the site is
below this many degrees Celsius. The rule was fixed before any cold-snap score was read."""
MIN_COLD_DAYS: Final[int] = 10
"""A site-year with fewer cold-snap days than this is reported as too few days."""
WINTER_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("KW", "K0"),
    ("R", "K0"),
    ("KW", "R"),
    ("KW-fast", "K0"),
    ("KW-fast", "R"),
    ("KW", "X-nl+K"),
    ("KW", "Xlag-d"),
    ("KW", "BG2"),
    ("K0", "BG2"),
)


def cold_snap_mask(*, frame: pl.DataFrame, site: str) -> np.ndarray:
    """Return True for rows whose local day is a cold-snap day at the site."""
    cache = common.load_cache()
    features = cache.sites[site]
    dates = cache.calendar.local_date.astype("datetime64[D]").astype(np.int64)
    unique, inverse = np.unique(dates, return_inverse=True)
    mean = np.bincount(inverse, weights=features.temperature_c) / np.bincount(inverse)
    months = unique.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64) % 12 + 1
    cold_days = unique[(mean < COLD_SNAP_C) & np.isin(months, list(WINTER))]
    position = np.searchsorted(
        cache.end_times, frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    )
    row_dates = dates[np.minimum(position, len(dates) - 1)]
    return np.isin(row_dates, cold_days)


def winter_breakdown(*, period: str) -> dict:
    """Return winter (November to March) contrasts and absolute scores, a labelled post hoc table.

    Rows are those of the period at day D+1 and over all leads, in winter months, the evening hour
    band within winter, and cold-snap days (see `COLD_SNAP_C`). Pairs against stored references
    use the rows both hold.
    """
    sites = common.DECISION_SITES
    frames = load(arms=["K0", "KW", "R", "KW-fast"], period=period, sites=sites)
    for label, name in REFERENCES.items():
        if label in {"X-nl+K", "Xlag-d"}:
            frames[label] = score.stored_reference(name=name, period=period, sites=sites)
    if period == "scored":
        frames["BG2"] = score.e87_reference(folder=E87_REFERENCES["BG2"], sites=sites)
    out: dict = {"cold_days": {}, "pairs": {}}
    for site in sites:
        part = score.lead_filter(frame=frames["K0"], band=score.LEAD_BANDS[0]).filter(
            pl.col("site") == site
        )
        mask = cold_snap_mask(frame=part, site=site)
        out["cold_days"][site] = int(
            part.filter(pl.Series(mask))["valid_time"].dt.date().n_unique()
        )
    for treatment, reference in WINTER_PAIRS:
        if treatment not in frames or reference not in frames:
            continue
        shared = score.common_rows(
            arms_by_name={treatment: frames[treatment], reference: frames[reference]}
        )
        month = {k: f["valid_time"].dt.month().to_numpy() for k, f in shared.items()}
        winter = {
            k: f.filter(pl.Series(np.isin(month[k], list(WINTER)))) for k, f in shared.items()
        }
        rows: dict[str, dict[str, pl.DataFrame]] = {
            "winter, D+1": {
                k: score.lead_filter(frame=f, band=score.LEAD_BANDS[0]) for k, f in winter.items()
            },
            "winter, all leads": {
                k: score.lead_filter(frame=f, band=score.LEAD_BANDS[6]) for k, f in winter.items()
            },
        }
        if "hour_band" in winter[treatment].columns and "hour_band" in winter[reference].columns:
            rows["winter evening, D+1"] = {
                k: score.lead_filter(frame=f, band=score.LEAD_BANDS[0]).filter(
                    pl.col("hour_band") == 3
                )
                for k, f in winter.items()
            }
        cold = {}
        for k, f in winter.items():
            flags = np.zeros(f.height, dtype=bool)
            for site in sites:
                where = (f["site"] == site).to_numpy()
                if where.any():
                    flags[where] = cold_snap_mask(frame=f.filter(pl.col("site") == site), site=site)
            cold[k] = f.filter(pl.Series(flags))
        rows["cold snap days, D+1"] = {
            k: score.lead_filter(frame=f, band=score.LEAD_BANDS[0]) for k, f in cold.items()
        }
        entry = {}
        for label, subset in rows.items():
            if subset[treatment].height < 500:
                entry[label] = "too few rows"
                continue
            losses = score.stack_losses(arms_by_name=subset)
            entry[label] = {
                "qs": score.contrast(losses=losses, treatment=treatment, reference=reference),
                "absolute": {
                    arm: float(score.quantile_score(frame=subset[arm]).mean())  # ty: ignore[invalid-argument-type]
                    for arm in (treatment, reference)
                },
                "rows": subset[treatment].height,
            }
        out["pairs"][f"{treatment} - {reference}"] = entry
    return out


def sister_intervals() -> dict:
    """Return the intervals the audit's sixth trigger compares, all at 98.75% at day D+1.

    The sisters of C1 (KW - K0) are R75 - K0 on the same rows (scored year and replicate months)
    and experiment 111's GAM-K - GAM-0 from its stored predictions on the scored rows.
    """
    sites = common.DECISION_SITES
    out: dict = {}
    for period in PERIODS:
        frames = load(arms=["K0", "KW", "R75"], period=period, sites=sites)
        stored = {
            "GAM-K (stored)": score.stored_reference(name="gamk", period=period, sites=sites),
            "GAM-0": score.stored_reference(name="gam0", period=period, sites=sites),
        }
        out[period] = {}
        for label, pair, source in (
            ("C1 KW - K0", ("KW", "K0"), frames),
            ("R75 - K0", ("R75", "K0"), frames),
            ("GAM-K - GAM-0 (experiment 111, stored)", ("GAM-K (stored)", "GAM-0"), stored),
        ):
            shared = score.common_rows(arms_by_name={a: source[a] for a in pair})
            subset = {
                k: score.lead_filter(frame=f, band=score.LEAD_BANDS[0]) for k, f in shared.items()
            }
            result = score.contrast(
                losses=score.stack_losses(arms_by_name=subset),
                treatment=pair[0],
                reference=pair[1],
                level=score.BONFERRONI_LEVEL,
            )
            result["width"] = result["upper"] - result["lower"]
            out[period][label] = result
    return out


def main() -> None:  # noqa: PLR0912
    """Print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", default="core")
    args = parser.parse_args()
    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    if args.part == "core":
        results["planned"] = planned_contrasts()
    if args.part == "exploratory":
        results["exploratory"] = {period: exploratory(period=period) for period in PERIODS}
    if args.part == "absolute":
        arms = available(arms=REPORTED_ARMS)
        results["absolute"] = {
            period: absolute_table(
                frames=load(arms=arms, period=period, sites=common.DECISION_SITES)
            )
            for period in PERIODS
        }
        results["absolute_d9"] = {
            period: absolute_table(frames=load(arms=["K0", "KW"], period=period, sites=("D9",)))
            for period in PERIODS
        }
    if args.part == "references":
        arms = available(arms=("K0", "KW", "KW-lb", "R", "KW-fast"))
        results["references"] = {p: reference_scores(period=p, arms=arms) for p in PERIODS}
    if args.part == "refcontrasts":
        results["reference_contrasts"] = {p: reference_contrasts(period=p) for p in PERIODS}
    if args.part == "sisters":
        results["sisters"] = sister_intervals()
    if args.part == "winter":
        results["winter"] = {p: winter_breakdown(period=p) for p in PERIODS}
    if args.part == "tails":
        arms = ["K0", "KW", "KW-lb", "R", "KW-day"]
        results["tails"] = {}
        for period in PERIODS:
            frames = with_shift_only(
                frames=load(arms=arms, period=period, sites=common.DECISION_SITES)
            )
            results["tails"][period] = tails_table(frames={a: frames[a] for a in TAIL_ARMS})
            results.setdefault("checks", {})[period] = checks_table(frames=frames)
    common.write_json_atomic(path=RESULTS_PATH, content=results)
    for period, block in results.get("planned", {}).items():
        for label, entry in block.items():
            for band_name in ("D+1", "0.5-168 h"):
                if band_name not in entry["bands"]:
                    continue
                q = entry["bands"][band_name]["qs"]
                m = entry["bands"][band_name]["mae_median"]
                print(
                    f"{period:9s} {label:45s} {band_name:9s} qs {q['estimate']:+.4f} "
                    f"[{q['lower']:+.4f}, {q['upper']:+.4f}] sites "
                    f"{q['sites_better']}/{q['n_sites']} "
                    f"| mae median {m['estimate']:+.4f}"
                )


if __name__ == "__main__":
    main()
