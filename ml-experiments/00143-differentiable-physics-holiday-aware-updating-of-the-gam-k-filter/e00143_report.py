"""Score experiment 143 and print every number its write-up quotes.

Writes `work/results.json` and `work/daily.parquet`, and prints the numbers as text, which is
committed beside the write-up as `report.txt`. Contrasts on holiday categories use a cluster
bootstrap (a block is one holiday cluster with its holiday and following days); contrasts on the
other days and on all rows use calendar-month blocks. All are paired across arms, 99% intervals.

    uv run --no-sync python ml-experiments/00143-*/e00143_report.py \
        > docs/ml-experiments/00143-*/report.txt
"""

import json
from datetime import date
from typing import Final

import e00143_arms as arms
import e00143_common as common
import e00143_holidays as hol
import e00143_score as score
import numpy as np
import polars as pl
from studies.tail_calibration import tail_calibration_table

PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "development", "scored")
"""`development` pools the tuning and replicate rows: each alone holds 1 to 3 holiday clusters."""
TAGS: Final[dict[str, str]] = {
    "K0": "k0",
    "K-skip": "ks",
    "K-skip-core": "kc",
    "K-skip+1": "k1",
    "K-zero": "kz",
}
LEVEL: Final[float] = 99.0
N_RESAMPLES: Final[int] = 5000
SEED: Final[int] = 143
WINTER: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})
EVENING: Final[range] = range(16, 20)
NIGHT_HOURS: Final[tuple[int, ...]] = (22, 23, 0, 1, 2, 3, 4, 5, 6)
COLD_SNAP_C: Final[float] = 3.0
TAIL_LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.95, 0.99)
D1: Final = score.LEAD_BANDS[0]


def mean_of(series: pl.Series) -> float:
    """Return the mean of a series as a float."""
    return float(np.mean(series.to_numpy()))


def cluster_ids(*, local_day: np.ndarray, holiday: np.ndarray) -> np.ndarray:
    """Return the block of each local day: its holiday cluster plus 5 following days, else -1."""
    out = np.full(local_day.shape, -1, dtype=np.int64)
    for index, (first, last) in enumerate(hol.clusters(holiday=holiday)):
        out[(local_day >= first) & (local_day <= last + hol.AFTER_DAYS)] = index
    return out


def wide_frame(*, period: str, sites: tuple[str, ...]) -> pl.DataFrame:
    """Return the arms side by side on the rows every arm holds, with day labels."""
    if period == "development":
        return pl.concat([wide_frame(period=p, sites=sites) for p in ("tuning", "replicate")])
    shared = score.common_rows(
        arms_by_name={arm: score.load_arm(arm=arm, period=period, sites=sites) for arm in arms.ARMS}
    )
    base = shared["K0"].select(
        "site", "time_series_id", "power_fcst_init_time", "valid_time", "lead_hours", "actual"
    )
    columns = {}
    for arm, tag in TAGS.items():
        losses = score.with_losses(frame=shared[arm])
        columns[f"qs_{tag}"] = losses["qs"]
        columns[f"med_{tag}"] = losses["abs_median"]
        columns[f"mean_{tag}"] = (shared[arm]["actual"] - shared[arm]["mean"]).abs()
        columns[f"q95_{tag}"] = shared[arm]["q95"]
        columns[f"q50_{tag}"] = shared[arm]["q50"]
        columns[f"avg_{tag}"] = shared[arm]["mean"]
    base = base.with_columns(**{k: pl.Series(k, v) for k, v in columns.items()})
    local = pl.col("valid_time").dt.convert_time_zone("Europe/London")
    base = base.with_columns(
        local_date=local.dt.date(), local_hour=local.dt.hour(), month=local.dt.month()
    )
    days = base["local_date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    holiday = arms.holiday_numbers()
    return base.with_columns(
        category=pl.Series(hol.categorise(local_day=days, holiday=holiday)),
        cluster=pl.Series(cluster_ids(local_day=days, holiday=holiday)),
        month_block=pl.Series(
            base["valid_time"].dt.year().to_numpy() * 100 + base["month"].to_numpy()
        ),
    )


def cold_flags(*, frame: pl.DataFrame) -> np.ndarray:
    """Return True for rows on a winter day whose daily mean temperature is under 3 C."""
    cache = common.load_cache()
    dates = cache.calendar.local_date.astype("datetime64[D]").astype(np.int64)
    unique, inverse = np.unique(dates, return_inverse=True)
    flags = np.zeros(frame.height, dtype=bool)
    row_days = frame["local_date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    for site in common.SCORED_SITES:
        where = (frame["site"] == site).to_numpy()
        if not where.any():
            continue
        temperature = cache.sites[site].temperature_c
        mean = np.bincount(inverse, weights=temperature) / np.bincount(inverse)
        months = unique.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64) % 12 + 1
        cold_days = unique[(mean < COLD_SNAP_C) & np.isin(months, list(WINTER))]
        flags[where] = np.isin(row_days[where], cold_days)
    return flags


def bootstrap(*, difference: np.ndarray, block: np.ndarray, rng: np.random.Generator) -> tuple:
    """Return (estimate, lower, upper, n_blocks) of the mean of `difference`, blocks resampled."""
    if len(difference) == 0:
        return (np.nan, np.nan, np.nan, 0)
    unique, inverse = np.unique(block, return_inverse=True)
    sums = np.bincount(inverse, weights=difference)
    counts = np.bincount(inverse).astype(float)
    draws = rng.integers(0, len(unique), size=(N_RESAMPLES, len(unique)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (100.0 - LEVEL) / 2.0
    lower, upper = np.percentile(means, [tail, 100.0 - tail])
    return (float(sums.sum() / counts.sum()), float(lower), float(upper), len(unique))


def contrast(
    *,
    frame: pl.DataFrame,
    mask: np.ndarray,
    treatment: str,
    reference: str,
    blocks: str,
    metric: str = "qs",
) -> dict:
    """Return a paired contrast on the masked rows: estimate, interval, blocks, rows, sites."""
    part = frame.filter(pl.Series(mask))
    if part.height == 0:
        return {"estimate": np.nan, "lower": np.nan, "upper": np.nan, "blocks": 0, "rows": 0}
    t, r = TAGS[treatment], TAGS[reference]
    difference = (part[f"{metric}_{t}"] - part[f"{metric}_{r}"]).to_numpy().astype(np.float64)
    rng = np.random.default_rng(SEED)
    estimate, lower, upper, n_blocks = bootstrap(
        difference=difference, block=part[blocks].to_numpy(), rng=rng
    )
    by_site = part.group_by("site").agg(
        d=(pl.col(f"{metric}_{t}") - pl.col(f"{metric}_{r}")).mean()
    )
    return {
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "blocks": n_blocks,
        "rows": part.height,
        "dates": int(part["local_date"].n_unique()),
        "sites_better": int((by_site["d"] < 0).sum()),
        "n_sites": by_site.height,
    }


def text(entry: dict) -> str:
    """Return a contrast as `estimate [lower, upper]`, with its block and row counts."""
    if entry["rows"] == 0:
        return "no rows"
    return (
        f"{entry['estimate']:+.4f} [{entry['lower']:+.4f}, {entry['upper']:+.4f}] "
        f"blocks {entry['blocks']}, dates {entry['dates']}, rows {entry['rows']:,}, "
        f"sites better {entry['sites_better']}/{entry['n_sites']}"
    )


def date_mask(*, frame: pl.DataFrame, month: int, days: tuple[int, ...]) -> np.ndarray:
    """Return True for rows whose local date is one of `days` of `month`."""
    dates = frame["local_date"]
    return ((dates.dt.month() == month) & dates.dt.day().is_in(list(days))).to_numpy()


def mean_difference(*, frame: pl.DataFrame) -> float:
    """Return the mean of K-skip's score minus K0's over a frame, or NaN for no rows."""
    if frame.height == 0:
        return float("nan")
    return mean_of(frame["qs_ks"] - frame["qs_k0"])


def residual_diagnostic() -> dict:
    """Return each holiday cluster's mean filter residual on its holiday days and the days after.

    The residual is observed minus the fitted forecast driven by the analysis proxy, in points,
    averaged over the five decision substations and over days with readings. If the holiday
    residual and the following days' residual have the same sign, the filter's carried level helps
    the following days; with opposite signs, skipping the holiday readings helps.
    """
    import e00143_frames as frames

    holiday = arms.holiday_numbers()
    out: dict = {}
    for first, last in hol.clusters(holiday=holiday):
        held, after = [], []
        for site in common.DECISION_SITES:
            data = frames.get_site(site)
            days = arms.grid_days(data=data)
            on = np.isin(days, holiday) & (days >= first) & (days <= last)
            nxt = (days > last) & (days <= last + hol.AFTER_DAYS) & ~np.isin(days, holiday)
            if np.isfinite(data.stream[on]).any() and np.isfinite(data.stream[nxt]).any():
                held.append(float(np.nanmean(data.stream[on])))
                after.append(float(np.nanmean(data.stream[nxt])))
        if held:
            first_date = date.fromordinal(date(1970, 1, 1).toordinal() + first)
            out[str(first_date)] = {"holiday": float(np.mean(held)), "after": float(np.mean(after))}
            print(
                f"residual | cluster from {first_date}: mean residual on holiday days "
                f"{np.mean(held):+.2f}, on the {hol.AFTER_DAYS} days after "
                f"{np.mean(after):+.2f} points"
            )
    return out


def main() -> None:  # noqa: PLR0912, PLR0915
    """Score every period and print the numbers."""
    gate = json.loads((common.WORK_DIR / "gate.json").read_text())
    print(f"== Gate conditions: {json.dumps(gate['conditions'])}")
    results: dict = {}
    daily = []
    for period in PERIODS:
        wide = wide_frame(period=period, sites=common.DECISION_SITES)
        d1 = wide.filter((pl.col("lead_hours") > D1[1]) & (pl.col("lead_hours") <= D1[2]))
        category = d1["category"].to_numpy()
        is_holiday = np.isin(category, ["christmas_window", "bank_holiday"])
        is_after = np.isin(category, ["after_christmas", "after_bank_holiday"])
        is_other = category == "other"
        everything = np.ones(d1.height, dtype=bool)
        out: dict = {}
        print(f"\n== {period}: D+1, {d1.height:,} rows, 5 decision substations, 99% intervals ==")
        counts = d1.group_by("category").agg(
            dates=pl.col("local_date").n_unique(),
            rows=pl.len(),
            clusters=pl.col("cluster").n_unique(),
        )
        print(f"{period} | category sizes: {counts.sort('category').to_dicts()}")
        planned = {
            "C1 K-skip - K0, leakage days": (is_after, "K-skip", "K0", "cluster"),
            "C2 K-skip - K0, holiday days": (is_holiday, "K-skip", "K0", "cluster"),
            "C3 K-skip - K0, other days": (is_other, "K-skip", "K0", "month_block"),
            "C4 K-skip+1 - K-skip, leakage days": (is_after, "K-skip+1", "K-skip", "cluster"),
            "C5 K-zero - K-skip, holiday days": (is_holiday, "K-zero", "K-skip", "cluster"),
            "C6 K-skip - K0, all rows": (everything, "K-skip", "K0", "month_block"),
        }
        for label, (mask, t, r, blocks) in planned.items():
            out[label] = {
                metric: contrast(
                    frame=d1, mask=mask, treatment=t, reference=r, blocks=blocks, metric=metric
                )
                for metric in ("qs", "med", "mean")
            }
            print(
                f"{period} | {label} | {text(out[label]['qs'])}; MAE median "
                f"{out[label]['med']['estimate']:+.4f}, MAE mean "
                f"{out[label]['mean']['estimate']:+.4f}"
            )
        # Christmas: 27 to 30 December, and the whole window, by arm pair.
        boxing = date_mask(frame=d1, month=12, days=(27, 28, 29, 30))
        for t, r in (("K-skip-core", "K0"), ("K-skip-core", "K-skip"), ("K-skip", "K0")):
            label = f"C7 {t} - {r}, 27-30 December"
            out[label] = contrast(frame=d1, mask=boxing, treatment=t, reference=r, blocks="cluster")
            print(f"{period} | {label} | {text(out[label])}")
        for t, r in (("K-skip", "K0"), ("K-skip+1", "K0"), ("K-zero", "K0"), ("K-skip-core", "K0")):
            for name, mask in (
                ("Christmas window", category == "christmas_window"),
                ("bank holidays outside it", category == "bank_holiday"),
                ("after Christmas", category == "after_christmas"),
                ("after other bank holidays", category == "after_bank_holiday"),
            ):
                label = f"{t} - {r}, {name}"
                out[label] = contrast(
                    frame=d1, mask=mask, treatment=t, reference=r, blocks="cluster"
                )
                print(f"{period} | {label} | {text(out[label])}")
        # Breakdowns of K-skip - K0 and K-skip+1 - K0.
        winter = d1["month"].is_in(list(WINTER)).to_numpy()
        evening = d1["local_hour"].is_in(list(EVENING)).to_numpy()
        cold = cold_flags(frame=d1)
        night = d1["local_hour"].is_in(list(NIGHT_HOURS)).to_numpy()
        breakdowns = {
            "winter": winter,
            "winter evening 16-20": winter & evening,
            "cold-snap days": cold,
            "cold-snap nights": cold & night,
            "non-winter": ~winter,
        }
        for t in ("K-skip", "K-skip+1"):
            for name, mask in breakdowns.items():
                label = f"{t} - K0, {name}"
                out[label] = contrast(
                    frame=d1, mask=mask, treatment=t, reference="K0", blocks="month_block"
                )
                print(f"{period} | {label} | {text(out[label])}")
        cold_rows = d1.filter(pl.Series(cold & night))
        for arm, tag in TAGS.items():
            if cold_rows.height:
                coverage = mean_of(cold_rows["actual"] <= cold_rows[f"q95_{tag}"])
                print(
                    f"{period} | cold-snap nights q95 coverage {arm}: {coverage:.4f} "
                    f"({cold_rows.height:,} rows)"
                )
        # Every lead band beside D+1, for K-skip - K0.
        for name, low, high in score.LEAD_BANDS:
            part = wide.filter((pl.col("lead_hours") > low) & (pl.col("lead_hours") <= high))
            cat = part["category"].to_numpy()
            for label, mask in (
                ("leakage days", np.isin(cat, ["after_christmas", "after_bank_holiday"])),
                ("holiday days", np.isin(cat, ["christmas_window", "bank_holiday"])),
                ("other days", cat == "other"),
            ):
                entry = contrast(
                    frame=part,
                    mask=mask,
                    treatment="K-skip",
                    reference="K0",
                    blocks="cluster" if label != "other days" else "month_block",
                )
                out[f"lead {name} {label}"] = entry
                print(f"{period} | K-skip - K0 | lead {name} | {label} | {text(entry)}")
        # Per-cluster numbers.
        holiday = arms.holiday_numbers()
        for index, (first, last) in enumerate(hol.clusters(holiday=holiday)):
            part = d1.filter(pl.col("cluster") == index)
            if part.height == 0:
                continue
            first_date = date.fromordinal(date(1970, 1, 1).toordinal() + first)
            row = {arm: mean_of(part[f"qs_{tag}"]) for arm, tag in TAGS.items()}
            h = part.filter(pl.col("category").is_in(["christmas_window", "bank_holiday"]))
            a = part.filter(pl.col("category").is_in(["after_christmas", "after_bank_holiday"]))
            out[f"cluster {first_date}"] = {
                "holiday_days": last - first + 1,
                "scored_dates": int(part["local_date"].n_unique()),
                "holiday_diff": mean_of(h["qs_ks"] - h["qs_k0"]) if h.height else None,
                "after_diff": mean_of(a["qs_ks"] - a["qs_k0"]) if a.height else None,
                "core_after_diff": mean_of(a["qs_kc"] - a["qs_k0"]) if a.height else None,
            }
            print(
                f"{period} | cluster from {first_date} ({last - first + 1} holiday days, "
                f"{part['local_date'].n_unique()} scored dates): mean D+1 score "
                + ", ".join(f"{k} {v:.3f}" for k, v in row.items())
                + f"; K-skip-K0 holiday {mean_difference(frame=h):+.3f}"
                + f", after {mean_difference(frame=a):+.3f}"
            )
        # Absolute scores, tails, median minus mean.
        print(
            f"{period} | absolute D+1 score (all rows): "
            + ", ".join(f"{arm} {mean_of(d1[f'qs_{tag}']):.4f}" for arm, tag in TAGS.items())
        )
        print(f"{period} | absolute D+1 score by category:")
        for cat_name in hol.CATEGORIES:
            part = d1.filter(pl.col("category") == cat_name)
            if part.height:
                print(
                    f"    {cat_name} ({part.height:,} rows): "
                    + ", ".join(
                        f"{arm} {mean_of(part[f'qs_{tag}']):.3f}" for arm, tag in TAGS.items()
                    )
                )
        tails = {}
        for arm in arms.ARMS:
            frame = score.lead_filter(
                frame=score.common_rows(
                    arms_by_name={
                        arm: score.load_arm(arm=arm, period=period, sites=common.DECISION_SITES)
                    }
                )[arm],
                band=D1,
            )
            tails[arm] = tail_calibration_table(frame=frame, levels=TAIL_LEVELS).to_dicts()
            print(
                f"{period} | tails {arm}: "
                + ", ".join(
                    f"q{round(r['level'] * 100):02d} {100 * r['gap']:+.2f} {r['pinball']:.4f}"
                    for r in tails[arm]
                )
            )
        for arm, tag in TAGS.items():
            gap = (
                d1.group_by("site")
                .agg(g=(pl.col(f"q50_{tag}") - pl.col(f"avg_{tag}")).mean())
                .sort("site")
            )
            print(
                f"{period} | median minus mean {arm}: "
                + ", ".join(f"{s} {g:+.3f}" for s, g in gap.iter_rows())
            )
        # Leave one site out for C1.
        for site in common.DECISION_SITES:
            entry = contrast(
                frame=d1,
                mask=is_after & (d1["site"] != site).to_numpy(),
                treatment="K-skip",
                reference="K0",
                blocks="cluster",
            )
            out[f"C1 without {site}"] = entry
            print(f"{period} | C1 without {site} | {text(entry)}")
        # Power check: mean absolute difference between K0 and K-skip corrections on leakage days.
        diff = (d1["avg_ks"] - d1["avg_k0"]).abs()
        print(
            f"{period} | mean |K-skip mean forecast - K0 mean forecast|: leakage days "
            f"{mean_of(diff.filter(pl.Series(is_after))):.3f}, holiday days "
            f"{mean_of(diff.filter(pl.Series(is_holiday))):.3f}, other days "
            f"{mean_of(diff.filter(pl.Series(is_other))):.3f} points"
        )
        results[period] = out
        daily.append(
            d1.group_by(["local_date", "category"])
            .agg([pl.col(f"qs_{tag}").mean() for tag in TAGS.values()] + [pl.len().alias("rows")])
            .with_columns(period=pl.lit(period))
        )
    print("\n== Residual diagnostic (does the holiday residual predict the days after?) ==")
    results["residual_diagnostic"] = residual_diagnostic()
    pl.concat(daily).sort("local_date").write_parquet(common.WORK_DIR / "daily.parquet")
    common.write_json_atomic(path=common.WORK_DIR / "results.json", content=results)


if __name__ == "__main__":
    main()
