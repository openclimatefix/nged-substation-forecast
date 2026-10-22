r"""Print every number of the experiment 253 write-up, and the verdict the decision rule gives.

Reads the tables that `e00253_run.py` wrote. The unit is a side (a donor or a metered receiver) of
an injected transfer that has at least one metered receiver, weighted by the MW moved divided by
the number of sides of the transfer.

    uv run --no-sync python ml-experiments/00253-*/e00253_contrasts.py
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl
from e00253_common import WORK_DIR, stamp_months, stamp_year_months

RUN_DIR: Final = WORK_DIR / "run"
N_RESAMPLES: Final[int] = 20_000
LEVEL: Final[float] = 1 - 0.05 / 4
"""The Bonferroni level for four contrasts: 98.75%."""
MARGIN: Final[float] = -0.02
MAX_FALSE_END: Final[float] = 0.10
SETTINGS: Final[tuple[str, ...]] = ("OFF", "R0", "P1", "P2", "P3", "P1G", "P2G")
CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("P2", "R0"),
    ("P1", "R0"),
    ("P3", "P2"),
    ("P3", "R0"),
    ("P1G", "R0"),
    ("P2G", "R0"),
)
POST_HOC: Final[frozenset[str]] = frozenset({"P1G", "P2G"})
"""Settings added after the first results; their contrasts are not part of the 4 planned."""
SEED: Final[int] = 253


def sides() -> pl.DataFrame:
    """One row per side, setting, shape, and variant, for pair transfers starting in even months."""
    events = pl.read_parquet(RUN_DIR / "events.parquet")
    onset = events["onset"].to_numpy()
    events = events.with_columns(
        year_month=pl.Series(stamp_year_months()[onset]),
        even=pl.Series(stamp_months()[onset] % 2 == 0),
    )
    rows = pl.read_parquet(RUN_DIR / "results.parquet").join(
        events.select("event_id", "year_month", "even", "metered_receivers", "hours"),
        on="event_id",
        validate="m:1",
    )
    return rows.filter(pl.col("even"))


def month_sums(*, frame: pl.DataFrame, value: str) -> tuple[np.ndarray, np.ndarray]:
    """Per calendar month, the weighted sum of `value` and of the weights."""
    grouped = (
        frame.with_columns(num=pl.col("w") * pl.col(value).cast(pl.Float64))
        .group_by("year_month")
        .agg(pl.col("num").sum(), pl.col("w").sum())
        .sort("year_month")
    )
    return grouped["num"].to_numpy(), grouped["w"].to_numpy()


def interval(*, picks: np.ndarray, num: np.ndarray, den: np.ndarray, level: float) -> tuple:
    """The weighted share and its month-resampled interval at `level`."""
    share = num.sum() / den.sum()
    draws = num[picks].sum(axis=1) / den[picks].sum(axis=1)
    return share, *np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])


def analyse(*, variant: str, data: pl.DataFrame, exactly_one: bool = False) -> dict:
    """Print the table of one variant and return the share found of every setting."""
    wanted = pl.col("metered_receivers") == 1 if exactly_one else pl.col("metered_receivers") >= 1
    side = data.filter(pl.col("kind") == "side", pl.col("variant") == variant, wanted).with_columns(
        w=pl.col("moved_mw") / (1 + pl.col("metered_receivers"))
    )
    months = side["year_month"].unique().sort()
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(months), size=(N_RESAMPLES, len(months)))
    note = ", exactly one metered receiver (post hoc)" if exactly_one else ""
    n_rows = side.filter(pl.col("setting") == "R0").height
    print(f"\n== variant {variant}{note}: {n_rows} side rows ==")
    sums: dict[str, tuple] = {}
    rows: list[dict] = []
    for setting in SETTINGS:
        one = side.filter(pl.col("setting") == setting)
        found = month_sums(frame=one, value="found")
        onset = one.filter(pl.col("onset_found"))
        false = month_sums(frame=onset, value="premature")
        sums[setting] = (found, false, month_sums(frame=one, value="onset_found"))
        share = interval(picks=picks, num=found[0], den=found[1], level=0.95)
        false_share = onset.select((pl.col("w") * pl.col("premature")).sum() / pl.col("w").sum())
        error = onset["end_error_hours"].abs()
        rows.append(
            {
                "kind": "share",
                "setting": setting,
                "point": share[0],
                "low": share[1],
                "high": share[2],
            }
        )
        print(
            f"{setting}: share found {share[0]:.3f} ({share[1]:.3f} to {share[2]:.3f}), "
            f"false-end {false_share.item():.3f}, onset found "
            f"{sums[setting][2][0].sum() / sums[setting][2][1].sum():.3f}, "
            f"median |end error| {error.median():.2f} h, lag {onset['lag_hours'].median():.1f} h"
        )
    for first, second in CONTRASTS:
        a, b = sums[first][0], sums[second][0]
        draws = a[0][picks].sum(1) / a[1][picks].sum(1) - b[0][picks].sum(1) / b[1][picks].sum(1)
        point = a[0].sum() / a[1].sum() - b[0].sum() / b[1].sum()
        low, high = np.quantile(draws, [(1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2])
        rows.append(
            {
                "kind": "contrast",
                "setting": f"{first} minus {second}",
                "point": point,
                "low": low,
                "high": high,
            }
        )
        tag = " (post hoc)" if {first, second} & POST_HOC else ""
        print(f"  {first} minus {second}: {point:+.3f} ({low:+.3f} to {high:+.3f}){tag}")
    print("  disagreement between sides (hours), transfers where 2+ sides alarmed:")
    events = data.filter(pl.col("kind") == "event", pl.col("variant") == variant, wanted).filter(
        pl.col("disagreement_hours").is_not_null() & pl.col("disagreement_hours").is_finite()
    )
    for setting in SETTINGS:
        d = events.filter(pl.col("setting") == setting)["disagreement_hours"]
        rows.append(
            {
                "kind": "disagreement",
                "setting": setting,
                "point": float(np.mean(d.to_numpy() > 6)),
            }
        )
        print(
            f"    {setting}: n {d.len()}, median {d.median():.2f}, mean {d.mean():.2f}, "
            f"share over 6 h {(d > 6).mean():.3f}"
        )
    return {s: (sums[s][0][0].sum() / sums[s][0][1].sum(), sums[s]) for s in SETTINGS} | {
        "picks": picks,
        "rows": rows,
    }


def check_r0(*, data: pl.DataFrame) -> None:
    """R0 on the full draw, scored as experiment 148 scored a transfer, reproduces 0.794."""
    events = data.filter(
        pl.col("kind") == "event", pl.col("variant") == "clean", pl.col("setting") == "R0"
    ).with_columns(w=pl.col("moved_mw"))
    print(
        "\nCheck: R0 best-of-sides share found, full draw, clean:",
        f"{events.select((pl.col('w') * pl.col('found')).sum() / pl.col('w').sum()).item():.3f}",
        "(experiment 148: 0.794)",
    )


def verdict(*, results: dict[str, dict]) -> str:
    """Apply the decision rule to P2 minus R0 on the clean, C, and U10 variants."""

    def contrast(variant: str) -> tuple[float, float, float, float]:
        res = results[variant]
        picks = res["picks"]
        a, b = res["P2"][1][0], res["R0"][1][0]
        draws = a[0][picks].sum(1) / a[1][picks].sum(1) - b[0][picks].sum(1) / b[1][picks].sum(1)
        low, high = np.quantile(draws, [(1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2])
        false = res["P2"][1][1]
        return (
            a[0].sum() / a[1].sum() - b[0].sum() / b[1].sum(),
            low,
            high,
            false[0].sum() / false[1].sum(),
        )

    clean, unrelated, unequal = contrast("clean"), contrast("C"), contrast("U10")
    holds_clean = clean[1] > MARGIN and clean[3] <= MAX_FALSE_END
    holds_unequal = unequal[1] > MARGIN and unequal[3] <= MAX_FALSE_END
    helps = unrelated[1] > 0
    if holds_clean and holds_unequal and helps:
        return "supported"
    if holds_clean and holds_unequal:
        return "partly supported"
    if clean[2] < MARGIN or unrelated[2] < 0:
        return "refuted"
    return "inconclusive"


def real_pairs() -> None:
    """Describe the labelled real transfers, side by side."""
    real = pl.read_parquet(RUN_DIR / "real_results.parquet")
    print("\n== real labelled transfers with 2+ metered sides ==")
    for setting in SETTINGS:
        one = real.filter(pl.col("setting") == setting)
        cards = one["card_id"].n_unique()
        alarmed = one.filter(pl.col("onset_found"))
        d = (
            alarmed.group_by("card_id")
            .agg(pl.col("disagreement_hours").first())
            .filter(pl.col("disagreement_hours").is_finite())["disagreement_hours"]
        )
        print(
            f"{setting}: {cards} transfers, {one.height} sides, onset found {alarmed.height}, "
            f"end within tolerance {one['found'].sum()}, median |end error| "
            f"{alarmed['end_error_hours'].abs().median():.1f} h, disagreement n {d.len()}, "
            f"median {d.median() if d.len() else float('nan'):.1f} h, under 48 h "
            f"{int((d < 48).sum())}"
        )


def print_counts(*, data: pl.DataFrame) -> None:
    """Print the transfers behind each population, and the side rows of blanked series 12."""
    base = data.filter(
        pl.col("kind") == "event", pl.col("variant") == "clean", pl.col("setting") == "R0"
    )
    print(
        f"Even-month transfer rows (3 shapes): {base.height}; with a metered receiver "
        f"{base.filter(pl.col('metered_receivers') >= 1).height}"
    )
    blank = data.filter(
        pl.col("kind") == "side",
        pl.col("variant") == "clean",
        pl.col("setting") == "R0",
        pl.col("side") == 12,
        pl.col("metered_receivers") >= 1,
    )
    print(
        f"Side rows on blanked series 12: {blank.height}, start found {blank['onset_found'].sum()}"
    )


def main() -> None:
    """Print every table."""
    data = sides()
    print_counts(data=data)
    results = {v: analyse(variant=v, data=data) for v in ("clean", "C", "U10")}
    for variant in ("clean", "C", "U10"):
        analyse(variant=variant, data=data, exactly_one=True)
    check_r0(data=data)
    print("\nVerdict by the decision rule:", verdict(results=results))
    real_pairs()


if __name__ == "__main__":
    main()
