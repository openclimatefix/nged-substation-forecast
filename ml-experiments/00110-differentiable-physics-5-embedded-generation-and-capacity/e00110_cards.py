"""Print the substation cards of experiment 110 as markdown tables, from the report.

One card per decision substation and one per MVA substation (D-labels only), from the arms fitted
at the 2025-07-01 cut. The real generator case G-c has no card: it has no forecast arm.

**Provisional: fitted to past weather on a screening run.** The capacity intervals are weakly
identified at real noise levels (experiment 112); compare across substations with care.

    uv run python ml-experiments/00110-*/e00110_cards.py > docs/ml-experiments/00110-*/cards.txt
"""

import e00110_charts as charts
import polars as pl

DECISION_SITES: tuple[str, ...] = ("D5", "D10", "D11", "D14", "D16")


def _markdown(*, frame: pl.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = [
        "| " + " | ".join("" if value is None else str(value) for value in row) + " |"
        for row in frame.iter_rows()
    ]
    return "\n".join([header, divider, *rows])


def decision_cards(*, sections: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """One row per decision substation: capacity, vintage, node, and clip."""
    capacity = charts._section(sections=sections, start="Capacity by site").filter(
        pl.col("cut") == "scored", pl.col("arm") == "k_base"
    )
    node = charts._section(sections=sections, start="Node: deepest outage").filter(
        pl.col("cut") == "scored", pl.col("arm") == "b_gen"
    )
    clip = charts._section(sections=sections, start="Clip: profiled").filter(
        pl.col("cut") == "scored", pl.col("arm") == "b_clip"
    )
    rows = []
    for site in DECISION_SITES:
        cap = capacity.filter(pl.col("site") == site).row(0, named=True)
        node_row = node.filter(pl.col("site") == site).row(0, named=True)
        clip_row = clip.filter(pl.col("site") == site).row(0, named=True)
        rows.append(
            {
                "substation": site,
                "K-base capacity (points)": f"{cap['capacity']:.0f}",
                "90% range": f"{cap['range_low']:.0f} to {cap['range_high']:.0f}",
                "vintage mean": f"{cap['vintage_mean']:.0f}",
                "ratio to vintage": f"{cap['ratio_to_vintage']:.2f}",
                "node depth (points)": f"{node_row['evidence points']:.0f}",
                "clip m": f"{clip_row['multiplier']:.2f}",
                "clip ceiling share": f"{clip_row['ceiling share']:.2f}",
            }
        )
    return pl.DataFrame(rows)


def meter_cards(*, sections: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """One row per MVA substation: reverse flow and the reactive pair at the scored cut."""
    meter = charts._section(sections=sections, start="Meter: reactive pair").filter(
        pl.col("cut") == "scored", pl.col("arm") == "lq"
    )
    return meter.select(
        pl.col("site").alias("substation"),
        pl.col("reactive").alias("reactive pair"),
        pl.col("reverse-flow share").round(3),
        pl.col("power factor").round(3),
        pl.col("q0 points").round(2),
    ).sort("substation")


def main() -> None:
    """Print both card tables."""
    sections = charts.read_sections()
    print("Provisional: fitted to past weather on a screening run.\n")
    print(_markdown(frame=decision_cards(sections=sections)))
    print()
    print(_markdown(frame=meter_cards(sections=sections)))


if __name__ == "__main__":
    main()
