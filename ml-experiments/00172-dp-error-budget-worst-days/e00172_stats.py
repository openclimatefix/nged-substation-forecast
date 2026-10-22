"""Shares, bootstraps, label tables, and controls for experiment 172."""

import math
from typing import Final

import e00172_common as common
import numpy as np
import polars as pl

CURVE_FRACTIONS: Final[np.ndarray] = np.linspace(0.0, 1.0, 201)
INJECTION_REPEATS: Final[int] = 200


def share_of_worst(*, excess_by_site: list[np.ndarray], fraction: float) -> float:
    """Return the pooled share of the positive excess carried by each site's worst `fraction`."""
    top, total = 0.0, 0.0
    for excess in excess_by_site:
        ordered = np.sort(excess)[::-1]
        top += ordered[: math.ceil(len(ordered) * fraction)].sum()
        total += ordered[ordered > 0].sum()
    return float(top / total)


def pooled_share(*, days: pl.DataFrame, sites: tuple[str, ...], column: str = "excess") -> float:
    """Return the pooled worst-5% share over `sites` for the chosen excess column."""
    arrays = [days.filter(pl.col("site") == s)[column].to_numpy() for s in sites]
    return share_of_worst(excess_by_site=arrays, fraction=common.WORST_FRACTION)


def month_blocks(
    *, days: pl.DataFrame, rows: pl.DataFrame, sites: tuple[str, ...], cold_c: float
) -> dict:
    """Return, per month of the scored year and site, the sums the bootstrap recombines.

    The pre-registered row-level sums (`cold_excess`, `all_excess`, `cold_rows`, `all_rows`) use a
    row's score minus the site's median day score. The day-level sums (`cold_pos`, `all_pos`,
    `cold_days`, `all_days`) use the positive day excess. The night sums are the score of the
    night rows (local 18:00 to 06:00) on cold days and on the other winter days.
    """
    median = days.select("site", "site_median").unique()
    night = (pl.col("local_hour") >= 18) | (pl.col("local_hour") < 6)
    cold = pl.col("day_temperature_c") < cold_c
    winter = pl.col("month").is_in(sorted(common.WINTER_MONTHS))
    marked = rows.join(median, on="site").with_columns(
        row_excess=pl.col("score") - pl.col("site_median"),
        cold_night=night & cold,
        other_winter_night=night & ~cold & winter,
        year_month=pl.col("local_date").dt.year() * 100 + pl.col("month"),
    )
    blocks: dict = {}
    for month in sorted(days["year_month"].unique().to_list()):
        per_site = {}
        for site in sites:
            day_part = days.filter((pl.col("site") == site) & (pl.col("year_month") == month))
            row_part = marked.filter((pl.col("site") == site) & (pl.col("year_month") == month))
            cold_rows = row_part.filter(pl.col("cold_night"))
            other_rows = row_part.filter(pl.col("other_winter_night"))
            positive = day_part["excess"].clip(lower_bound=0.0)
            cold_day = day_part["day_temperature_c"] < cold_c
            per_site[site] = {
                "excess": day_part["excess"].to_numpy(),
                "cold_excess": float(cold_rows["row_excess"].sum()),
                "all_excess": float(row_part["row_excess"].sum()),
                "cold_rows": cold_rows.height,
                "all_rows": row_part.height,
                "cold_pos": float(positive.filter(cold_day).sum()),
                "all_pos": float(positive.sum()),
                "cold_days": int(cold_day.sum()),
                "all_days": day_part.height,
                "cold_night_score": float(cold_rows["score"].sum()),
                "cold_night_n": cold_rows.height,
                "other_night_score": float(other_rows["score"].sum()),
                "other_night_n": other_rows.height,
            }
        blocks[month] = per_site
    return blocks


def row_ratio(*, parts: list[dict]) -> float:
    """Return the pre-registered ratio: cold nights' share of row excess over their row share."""
    excess_all = sum(p["all_excess"] for p in parts)
    rows_cold = sum(p["cold_rows"] for p in parts)
    if rows_cold == 0 or excess_all <= 0:
        return float("nan")
    share = sum(p["cold_excess"] for p in parts) / excess_all
    return share / (rows_cold / sum(p["all_rows"] for p in parts))


def day_ratio(*, parts: list[dict]) -> float:
    """Return the cold days' share of the positive day excess over their share of the days."""
    days_cold = sum(p["cold_days"] for p in parts)
    if days_cold == 0:
        return float("nan")
    share = sum(p["cold_pos"] for p in parts) / sum(p["all_pos"] for p in parts)
    return share / (days_cold / sum(p["all_days"] for p in parts))


def night_ratio(*, parts: list[dict]) -> float:
    """Return the mean night score on cold days over that on the other winter days."""
    cold_n = sum(p["cold_night_n"] for p in parts)
    other_n = sum(p["other_night_n"] for p in parts)
    if cold_n == 0 or other_n == 0:
        return float("nan")
    cold = sum(p["cold_night_score"] for p in parts) / cold_n
    return cold / (sum(p["other_night_score"] for p in parts) / other_n)


def bootstrap(*, blocks: dict, sites: tuple[str, ...], rng: np.random.Generator) -> dict:
    """Resample months with replacement; return draws of the share and the three ratios."""
    months = list(blocks)
    names = ("share", "row_ratio", "day_ratio", "night_ratio")
    draws = {name: np.empty(common.N_RESAMPLES) for name in names}
    for i in range(common.N_RESAMPLES):
        chosen = [months[j] for j in rng.integers(0, len(months), size=len(months))]
        arrays = [np.concatenate([blocks[m][s]["excess"] for m in chosen]) for s in sites]
        parts = [blocks[m][s] for m in chosen for s in sites]
        draws["share"][i] = share_of_worst(excess_by_site=arrays, fraction=common.WORST_FRACTION)
        draws["row_ratio"][i] = row_ratio(parts=parts)
        draws["day_ratio"][i] = day_ratio(parts=parts)
        draws["night_ratio"][i] = night_ratio(parts=parts)
    return draws


def interval(*, draws: np.ndarray) -> tuple[float, float]:
    """Return the 95% percentile interval of the finite bootstrap draws."""
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return float(low), float(high)


def label_table(*, days: pl.DataFrame, worst: pl.DataFrame, sites: tuple[str, ...]) -> pl.DataFrame:
    """Return, per label, its rate among all days, among the worst days, and its loss share."""
    scope = days.filter(pl.col("site").is_in(sites))
    top = worst.filter(pl.col("site").is_in(sites))
    positive = scope.with_columns(positive=pl.col("excess").clip(lower_bound=0.0))
    total = float(positive.select(pl.col("positive").sum()).item())
    return pl.DataFrame(
        [
            {
                "label": name,
                "share_of_all_days": float(scope.select(pl.col(name).mean()).item()),
                "share_of_worst_days": float(top.select(pl.col(name).mean()).item()),
                "n_worst_days": int(top[name].sum()),
                "share_of_total_excess": float(
                    positive.filter(pl.col(name)).select(pl.col("positive").sum()).item() / total
                ),
            }
            for name in (*common.LABELS, "labelled")
        ]
    )


def curve(*, days: pl.DataFrame, sites: tuple[str, ...]) -> pl.DataFrame:
    """Return the cumulative share of the positive excess against the fraction of worst days."""
    arrays = [
        np.sort(days.filter(pl.col("site") == s)["excess"].clip(lower_bound=0.0).to_numpy())[::-1]
        for s in sites
    ]
    total = sum(a[a > 0].sum() for a in arrays)
    shares = [
        sum(a[: math.ceil(len(a) * f)].sum() for a in arrays) / total for f in CURVE_FRACTIONS
    ]
    return pl.DataFrame({"fraction_of_days": CURVE_FRACTIONS, "share_of_excess": shares})


def random_control(*, days: pl.DataFrame, sites: tuple[str, ...], rng: np.random.Generator) -> dict:
    """Return the real share's place among shares of randomly chosen days of the same number."""
    arrays = [days.filter(pl.col("site") == s)["excess"].to_numpy() for s in sites]
    total = sum(a[a > 0].sum() for a in arrays)
    counts = [math.ceil(len(a) * common.WORST_FRACTION) for a in arrays]
    draws = np.array(
        [
            sum(
                rng.choice(a, size=n, replace=False).sum()
                for a, n in zip(arrays, counts, strict=True)
            )
            / total
            for _ in range(common.N_RANDOM_DRAWS)
        ]
    )
    real = share_of_worst(excess_by_site=arrays, fraction=common.WORST_FRACTION)
    return {
        "draw_mean": float(draws.mean()),
        "draw_max": float(draws.max()),
        "real": real,
        "draws_at_or_above_real": int((draws >= real).sum()),
    }


def injected_control(
    *,
    days: pl.DataFrame,
    sites: tuple[str, ...],
    rng: np.random.Generator,
    points: float = common.INJECTED_POINTS,
) -> dict[str, float]:
    """Return, per site and pooled, the share of injected days that land in the worst 5%."""
    hits_by_site = dict.fromkeys(sites, 0)
    per_site = common.INJECTED_DAYS * INJECTION_REPEATS
    for _ in range(INJECTION_REPEATS):
        for site in sites:
            score = days.filter(pl.col("site") == site)["score"].to_numpy().copy()
            chosen = rng.choice(len(score), size=common.INJECTED_DAYS, replace=False)
            score[chosen] += points
            excess = score - np.median(score)
            worst = np.argsort(-excess)[: math.ceil(len(score) * common.WORST_FRACTION)]
            hits_by_site[site] += len(set(worst) & set(chosen))
    out = {site: hits / per_site for site, hits in hits_by_site.items()}
    out["pooled"] = sum(hits_by_site.values()) / (per_site * len(sites))
    return out
