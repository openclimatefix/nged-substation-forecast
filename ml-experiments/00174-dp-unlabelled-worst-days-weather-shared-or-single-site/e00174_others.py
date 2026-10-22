"""Which other Primaries are in their own worst 5% of days on the same day (experiment 174).

The other Primaries' days come from experiment 87's BG2 forecasts, which cover 16 Primaries over
the scored year, so the ranking of their days does not depend on arm D. BG2 has no switching mask,
so the hand-filtered switching rows stay in its scores.
"""

from pathlib import Path
from typing import Final

import e00174_common as common
import numpy as np
import polars as pl
from studies.sources import REPO_DATA_DIR

HIERARCHY_DIR: Final[Path] = REPO_DATA_DIR / "NGED" / "substation_hierarchy_2026-10-15"


def bg2_days() -> pl.DataFrame:
    """Return BG2's day table (D+1 band, local days) for the 16 Primaries."""
    rows = (
        pl.read_parquet(common.BG2_PREDICTIONS)
        .filter(
            pl.col("actual").is_finite(),
            pl.col("lead_hours") >= common.D1_LEAD_HOURS[0],
            pl.col("lead_hours") < common.D1_LEAD_HOURS[1],
            pl.col("valid_time") >= common.SCORED_FIRST,
            pl.col("valid_time") <= common.SCORED_LAST,
        )
        .with_columns(score=common.pinball_score_expr(), local_date=common.local_date_expr())
    )
    days = rows.group_by(["site", "local_date"]).agg(n_rows=pl.len(), score=pl.col("score").mean())
    days = days.filter(pl.col("n_rows") >= common.MIN_ROWS_PER_DAY)
    return days.with_columns(excess=pl.col("score") - pl.col("score").median().over("site")).sort(
        ["site", "local_date"]
    )


def worst_flags(*, days: pl.DataFrame, fraction: float = common.WORST_FRACTION) -> pl.DataFrame:
    """Return each day with `in_worst`: in its site's worst `fraction` of days (rounded up)."""
    ranked = days.with_columns(
        rank=pl.col("excess").rank(method="ordinal", descending=True).over("site"),
        n_days=pl.len().over("site"),
    )
    return ranked.with_columns(in_worst=pl.col("rank") <= (pl.col("n_days") * fraction).ceil())


def primary_numbers() -> dict[str, int]:
    """Return each BG2 series' D-label mapped to the substation number of its Primary.

    Two series can share one Primary (a transformer each), so two labels can map to one number.
    """
    series = pl.read_parquet(common.BG2_PREDICTIONS, columns=["site", "time_series_id"]).unique()
    monitor = pl.read_csv(HIERARCHY_DIR / "Monitor Direction.csv").select(
        "TimeSeriesID", "SubstationNumber"
    )
    joined = series.join(monitor, left_on="time_series_id", right_on="TimeSeriesID", how="left")
    if joined["SubstationNumber"].null_count() or joined["site"].n_unique() != joined.height:
        msg = "a BG2 series has no unique substation number"
        raise ValueError(msg)
    return {row["site"]: int(row["SubstationNumber"]) for row in joined.iter_rows(named=True)}


def neighbours() -> dict[str, set[str]]:
    """Return, for each BG2 D-label, the D-labels of the Primaries interconnected with its Primary.

    A label that shares its Primary with the key (the other transformer) is not a neighbour.
    """
    number_of = primary_numbers()
    labels_of: dict[int, set[str]] = {}
    for label, number in number_of.items():
        labels_of.setdefault(number, set()).add(label)
    pairs = pl.read_csv(HIERARCHY_DIR / "Primary Substation Interconnections.csv")
    out: dict[str, set[str]] = {label: set() for label in number_of}
    for first, second in pairs.iter_rows():
        for here, there in ((first, second), (second, first)):
            for label in labels_of.get(here, set()):
                out[label] |= labels_of.get(there, set())
    return out


def same_primary() -> dict[str, set[str]]:
    """Return, for each D-label, the D-labels of series on the same Primary, itself included."""
    number_of = primary_numbers()
    return {
        label: {other for other, number in number_of.items() if number == number_of[label]}
        for label in number_of
    }


def coincidence(
    *, flags: pl.DataFrame, targets: pl.DataFrame, links: dict[str, set[str]]
) -> pl.DataFrame:
    """Add `n_others` and `n_neighbours` to the target (site, local_date) days.

    Args:
        flags: BG2's days with `in_worst`.
        targets: Days with `site` and `local_date`.
        links: Each site's interconnected sites.

    Returns:
        The targets with the number of other Primaries in their own worst 5% on the same day,
        and the number of those that are interconnected with the target's substation.
    """
    worst = flags.filter(pl.col("in_worst")).group_by("local_date").agg(sites=pl.col("site"))
    by_date = {row["local_date"]: set(row["sites"]) for row in worst.iter_rows(named=True)}
    own = same_primary()
    others, near = [], []
    for site, day in targets.select("site", "local_date").iter_rows():
        present = by_date.get(day, set()) - own[site]
        others.append(len(present))
        near.append(len(present & links.get(site, set())))
    return targets.with_columns(
        n_others=pl.Series(others, dtype=pl.Int32), n_neighbours=pl.Series(near, dtype=pl.Int32)
    )


def shifted_rate(
    *,
    flags: pl.DataFrame,
    targets: pl.DataFrame,
    rng: np.random.Generator,
    n_draws: int = common.N_SHIFT_DRAWS,
    common_shift: bool = False,
) -> dict[str, np.ndarray]:
    """Return the share of target days with a coincidence when the other Primaries' days move.

    Each other Primary's worst-day indicator series is rolled by its own random whole number of
    days (1 to the length minus 1), which keeps the number of worst days and their runs together
    and removes any link to the target's day.

    Args:
        flags: BG2's days with `in_worst`.
        targets: The target (site, local_date) days.
        rng: The random generator.
        n_draws: The number of draws.
        common_shift: Roll all the other Primaries by one shared amount per draw, which keeps
            their worst days together on the same days, instead of rolling each by its own amount.

    Returns:
        `any_other` and `any_neighbour`: one share per draw.
    """
    sites = sorted(flags["site"].unique().to_list())
    first = flags["local_date"].min()
    last = flags["local_date"].max()
    n_days = (last - first).days + 1  # ty: ignore[unsupported-operator]
    indicator = {site: np.zeros(n_days, dtype=bool) for site in sites}
    for site, day in flags.filter(pl.col("in_worst")).select("site", "local_date").iter_rows():
        indicator[site][(day - first).days] = True
    links = neighbours()
    own = same_primary()
    target_rows = [
        (site, (day - first).days) for site, day in targets.select("site", "local_date").iter_rows()
    ]
    any_other, any_neighbour = [], []
    for _ in range(n_draws):
        shared_amount = int(rng.integers(1, n_days))
        rolled = {
            site: np.roll(values, shared_amount if common_shift else int(rng.integers(1, n_days)))
            for site, values in indicator.items()
        }
        hit_other = hit_near = 0
        for site, index in target_rows:
            others = [s for s in sites if s not in own[site] and rolled[s][index]]
            hit_other += bool(others)
            hit_near += any(s in links.get(site, set()) for s in others)
        any_other.append(hit_other / len(target_rows))
        any_neighbour.append(hit_near / len(target_rows))
    return {"any_other": np.array(any_other), "any_neighbour": np.array(any_neighbour)}
