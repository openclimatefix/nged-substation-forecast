"""Print and save every number the experiment 00088 write-up quotes.

Reads the per-case cells of `e00088_cases.py` and the checks of `e00088_controls.py`, then writes
tables to `work/results/*.parquet` and prints a report to standard output.

    uv run python ml-experiments/00088-*/e00088_report.py
"""

from typing import Final

import e00088_common as common
import e00088_stats as stats
import polars as pl

HEADLINE: Final[tuple[str, ...]] = stats.HEADLINE_SHAPES
RESULTS_DIR = common.RESULTS_DIR


def load() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the case table and the controls' cells."""
    cells = pl.concat(
        [
            frame
            for path in stats.cases_dir_files(pattern="cells_*.parquet")
            if (frame := pl.read_parquet(path)).height
        ]
    )
    events = pl.concat(
        [pl.read_parquet(path) for path in stats.cases_dir_files(pattern="events_*.parquet")]
    )
    return stats.case_table(cells=cells, events=events), events


def save(*, name: str, frame: pl.DataFrame) -> None:
    """Write a result table to the private results folder."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(RESULTS_DIR / f"{name}.parquet")


def family_a_bins(*, cases: pl.DataFrame) -> list[tuple[str, pl.DataFrame]]:
    """Return family A cases pooled and split by time since onset at the origin."""
    family_a = cases.filter(pl.col("family") == "A")
    out = [("all", family_a)]
    out.extend(
        (
            name,
            family_a.filter(
                (pl.col("hours_at_origin") >= low) & (pl.col("hours_at_origin") < high)
            ),
        )
        for name, low, high in stats.ONSET_BINS
    )
    return out


def headline(*, cases: pl.DataFrame) -> pl.DataFrame:
    """Print and return the four planned contrasts and the shares of the gap."""
    scored = cases.filter(pl.col("scored") & (pl.col("family") == "A"))
    rows = []
    for name, arm_a, arm_b in (
        ("C1 arm4_caus - arm1", "arm4_caus", "arm1"),
        ("C2 arm3_inj - arm1", "arm3_inj", "arm1"),
        ("C3 arm4_caus - arm2_caus", "arm4_caus", "arm2_caus"),
        ("C4 arm2_caus - arm1", "arm2_caus", "arm1"),
    ):
        interval = stats.contrast(frame=scored, shapes=HEADLINE, arm_a=arm_a, arm_b=arm_b)
        signs = stats.month_signs(frame=scored, shapes=HEADLINE, arm_a=arm_a, arm_b=arm_b)
        rows.append(
            {
                "contrast": name,
                "point": interval.point,
                "low": interval.low,
                "high": interval.high,
                "months_negative": sum(v < 0 for v in signs.values()),
                "months": len(signs),
            }
        )
    table = pl.DataFrame(rows)
    print(f"\nPlanned contrasts, family A, shapes {HEADLINE}, level {stats.LEVEL:.2f}%")
    print(table)
    return table


def gap_shares(*, cases: pl.DataFrame, label: str, shapes: tuple[str, ...]) -> pl.DataFrame:
    """Return the share of the arm-1-to-ceiling gap that arms 2, 3, and 4 close, by time bin."""
    rows = []
    for bin_name, frame in family_a_bins(cases=cases.filter(pl.col("scored"))):
        for arm in ("arm2_inj", "arm2_caus", "arm3_inj", "arm3_caus", "arm4_inj", "arm4_caus"):
            share, gap = stats.share_of_gap(frame=frame, shapes=shapes, arm=arm)
            rows.append(
                {
                    "set": label,
                    "bin": bin_name,
                    "arm": arm,
                    "share": share.point,
                    "share_low": share.low,
                    "share_high": share.high,
                    "gap": gap.point,
                    "gap_low": gap.low,
                    "gap_high": gap.high,
                    "cases": frame.filter(pl.col("shape") == shapes[0]).height,
                }
            )
    return pl.DataFrame(rows)


def main() -> None:
    """Compute every table and print it."""
    cases, events = load()
    print(f"cases: {cases.height:,}; events: {events.select('shape', 'event_id').n_unique():,}")
    print(cases.group_by("shape", "family", "scored").len().sort("shape", "family", "scored"))
    save(name="planned_contrasts", frame=headline(cases=cases))
    for label, shapes in (("headline", HEADLINE), ("S1", ("S1",))):
        table = gap_shares(cases=cases, label=label, shapes=shapes)
        save(name=f"gap_shares_{label}", frame=table)
        print(f"\nShare of the gap closed, {label}")
        print(table)
    scored_a = cases.filter(pl.col("scored") & (pl.col("family") == "A"))
    absolute = stats.absolute_scores(frame=scored_a, shapes=HEADLINE)
    save(name="absolute_family_a", frame=absolute)
    print("\nAbsolute scores, family A, headline shapes")
    print(absolute.filter(pl.col("metric").is_in(["mae", "qs"])))


if __name__ == "__main__":
    main()
