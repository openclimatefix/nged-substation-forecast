"""Print and save every number the experiment 00096 write-up quotes.

Reads the per-case cells of `e00096_cases.py` and, for the plain booster, the ceiling, and the
nine-level reproduction check, experiment 00088's cells. Only transfers that start in even months
are scored: the synthetic training transfers start in odd months. Writes tables to
`work/results/*.parquet` and prints a report to standard output.

    uv run python ml-experiments/00096-*/e00096_report.py
"""

from pathlib import Path
from typing import Final

import e00096_common as common
import e00096_stats as stats
import numpy as np
import polars as pl

HEADLINE: Final[tuple[str, ...]] = stats.HEADLINE_SHAPES
RESULTS_DIR = common.RESULTS_DIR
OLD_CELLS_DIR = common.EXPERIMENT_00088_WORK_DIR / "cases"
OLD_CAUS_DIR = common.EXPERIMENT_00088_WORK_DIR / "cases_caus"
OLD_VARIANTS: Final[tuple[str, ...]] = ("plain", "ceiling", "nra_oracle_inj")
OLD_CAUS_VARIANTS: Final[tuple[str, ...]] = ("nra_oracle_caus",)


def _read(*, directory: Path, pattern: str, variants: tuple[str, ...]) -> pl.DataFrame:
    frames = [
        frame for path in sorted(directory.glob(pattern)) if (frame := pl.read_parquet(path)).height
    ]
    return pl.concat(frames, how="diagonal").filter(pl.col("variant").is_in(variants))


def load() -> tuple[pl.DataFrame, pl.DataFrame, list[pl.DataFrame]]:
    """Return the case table, the events, and the cell frames."""
    mine = _read(
        directory=common.CASES_DIR,
        pattern="cells_*.parquet",
        variants=tuple(v for v in stats.ARMS.values() if v.startswith(("nra_real", "nra_synth"))),
    )
    old = _read(directory=OLD_CELLS_DIR, pattern="cells_*.parquet", variants=OLD_VARIANTS)
    old_caus = _read(directory=OLD_CAUS_DIR, pattern="cells_*.parquet", variants=OLD_CAUS_VARIANTS)
    events = pl.concat(
        [pl.read_parquet(path) for path in sorted(common.CASES_DIR.glob("events_*.parquet"))]
    )
    cells = [mine, old, old_caus]
    return stats.case_table(cells=cells, events=events), events, cells


def save(*, name: str, frame: pl.DataFrame) -> None:
    """Write a result table to the private results folder."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(RESULTS_DIR / f"{name}.parquet")


def headline(*, cases: pl.DataFrame) -> pl.DataFrame:
    """Print and return the four planned contrasts and, as post hoc, the shares of the gap."""
    scored = cases.filter(pl.col("scored") & (pl.col("family") == "A"))
    rows = []
    for name, arm_a, arm_b in (
        ("C1 synth_inj - real_inj", "synth_inj", "real_inj"),
        ("C2 synth_caus - real_caus", "synth_caus", "real_caus"),
        ("C3 synth_cmax - real_cmax", "synth_cmax", "real_cmax"),
        ("C4 synth_cmax - plain", "synth_cmax", "plain"),
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


def absolute(*, cases: pl.DataFrame, family: str) -> pl.DataFrame:
    """Print and return every arm's weighted mean of each metric with a 95% interval."""
    frame = cases.filter(pl.col("scored") & (pl.col("family") == family))
    table = stats.absolute_scores(frame=frame, shapes=HEADLINE)
    print(f"\nAbsolute scores, family {family} (95% interval)")
    print(table.pivot(on="metric", index="arm", values="value").sort("arm"))
    return table


def gap_shares(*, cases: pl.DataFrame) -> pl.DataFrame:
    """Return the share of the plain-to-ceiling gap each arm closes, family A."""
    frame = cases.filter(pl.col("scored") & (pl.col("family") == "A"))
    rows = []
    for arm in (
        "old_inj",
        "real_inj",
        "real_caus",
        "real_cmax",
        "synth_inj",
        "synth_caus",
        "synth_cmax",
    ):
        share, gap = stats.share_of_gap(frame=frame, shapes=HEADLINE, arm=arm)
        rows.append(
            {
                "arm": arm,
                "share": share.point,
                "share_low": share.low,
                "share_high": share.high,
                "gap": gap.point,
            }
        )
    table = pl.DataFrame(rows)
    print("\nShare of the plain-to-ceiling gap closed (95%)")
    print(table)
    return table


def by_onset_bin(*, cases: pl.DataFrame) -> pl.DataFrame:
    """Return the three twin contrasts by time since onset, family A, 95% intervals."""
    frame = cases.filter(pl.col("scored") & (pl.col("family") == "A"))
    rows = []
    for name, low, high in stats.ONSET_BINS:
        part = frame.filter((pl.col("hours_at_origin") >= low) & (pl.col("hours_at_origin") < high))
        for arm, twin in stats.BASELINE.items():
            interval = stats.contrast(
                frame=part, shapes=HEADLINE, arm_a=arm, arm_b=twin, level=95.0
            )
            rows.append(
                {
                    "bin": name,
                    "contrast": f"{arm} - {twin}",
                    "point": interval.point,
                    "low": interval.low,
                    "high": interval.high,
                    "cases": part.filter(pl.col("shape") == "S2").height,
                }
            )
    table = pl.DataFrame(rows)
    print("\nTwin contrasts by hours since onset, family A (95%)")
    print(table)
    return table


def family_b(*, cases: pl.DataFrame) -> pl.DataFrame:
    """Return the twin contrasts after the transfer ended, by days since the end."""
    frame = cases.filter(pl.col("scored") & (pl.col("family") == "B"))
    bins = (("0-24 h", 0.0, 24.0), ("1-3 d", 24.0, 72.0), ("3-7 d", 72.0, 1e9))
    rows = []
    for name, low, high in bins:
        part = frame.filter((pl.col("hours") >= low) & (pl.col("hours") < high))
        for arm, twin in stats.BASELINE.items():
            interval = stats.contrast(
                frame=part, shapes=HEADLINE, arm_a=arm, arm_b=twin, level=95.0
            )
            rows.append(
                {
                    "bin": name,
                    "contrast": f"{arm} - {twin}",
                    "point": interval.point,
                    "low": interval.low,
                    "high": interval.high,
                }
            )
    table = pl.DataFrame(rows)
    print("\nFamily B twin contrasts by hours since the end (95%)")
    print(table)
    return table


def lead_bands(*, cells: list[pl.DataFrame], events: pl.DataFrame) -> pl.DataFrame:
    """Return each variant's MW-weighted mean error by lead band, family A, headline shapes."""
    weights = events.select("shape", "event_id", "moved_mw", "onset_month").with_columns(
        w=pl.col("moved_mw").clip(lower_bound=0.0), month=pl.col("onset_month") % 100
    )
    frames = []
    for part in cells:
        work = (
            part.filter((pl.col("family") == "A") & pl.col("shape").is_in(HEADLINE))
            .join(weights, on=["shape", "event_id"], how="left")
            .filter(pl.col("month").is_in(list(stats.SCORED_MONTH_NUMBERS)))
        )
        frames.append(
            work.group_by("band", "variant").agg(
                mae=(pl.col("mae") * pl.col("w") * pl.col("n")).sum()
                / (pl.col("w") * pl.col("n")).sum(),
            )
        )
    table = pl.concat(frames).sort("band", "variant")
    print("\nMAE by lead band, family A")
    print(table.pivot(on="variant", index="band", values="mae"))
    return table


def reproduction(*, cases: pl.DataFrame) -> None:
    """Print how far this experiment's real-trained scores are from experiment 00088's."""
    frame = cases.filter(pl.col("scored") & (pl.col("family") == "A"))
    for new, old in (("real_inj", "old_inj"), ("real_caus", "old_caus")):
        interval = stats.contrast(frame=frame, shapes=HEADLINE, arm_a=new, arm_b=old, level=95.0)
        print(f"\nReproduction check {new} - {old}: {interval.text()}")


def tails(*, cases: pl.DataFrame) -> pl.DataFrame:
    """Print the tail-calibration table of the real and synthetic arms (family A)."""
    frame = cases.filter(pl.col("scored") & (pl.col("family") == "A"))
    rows = []
    for arm in ("real_inj", "real_caus", "real_cmax", "synth_inj", "synth_caus", "synth_cmax"):
        row = {"arm": arm}
        for metric in (
            "below01",
            "below05",
            "ex95",
            "ex99",
            "pin01",
            "pin05",
            "pin95",
            "pin99",
            "cov90",
            "cov98",
            "qs",
        ):
            work = frame.with_columns(pl.col(f"{metric}_{stats.ARMS[arm]}").alias("_v")).drop_nulls(
                "_v"
            )
            row[metric] = stats.mean_value(frame=work, shapes=HEADLINE, column="_v").point
        rows.append(row)
    table = pl.DataFrame(rows)
    print(
        "\nTail calibration, family A (nominal: below01 0.01, below05 0.05, ex95 0.05, ex99 0.01)"
    )
    print(table)
    return table


def main() -> None:
    """Print the report and save its tables."""
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_rows(80)
    cases, events, cells = load()
    scored = cases.filter(pl.col("scored") & (pl.col("family") == "A"))
    print(
        f"cases: {cases.height:,}; "
        f"family A scored per shape: {scored.filter(pl.col('shape') == 'S2').height}"
    )
    save(name="headline", frame=headline(cases=cases))
    save(name="absolute_a", frame=absolute(cases=cases, family="A"))
    save(name="absolute_b", frame=absolute(cases=cases, family="B"))
    save(name="gap_shares", frame=gap_shares(cases=cases))
    save(name="by_onset", frame=by_onset_bin(cases=cases))
    save(name="family_b", frame=family_b(cases=cases))
    save(name="lead_bands", frame=lead_bands(cells=cells, events=events))
    save(name="tails", frame=tails(cases=cases))
    reproduction(cases=cases)
    np.set_printoptions(precision=3)


if __name__ == "__main__":
    main()
