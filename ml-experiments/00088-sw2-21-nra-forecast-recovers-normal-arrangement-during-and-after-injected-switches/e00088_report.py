"""Print and save every number the experiment 00088 write-up quotes.

Reads the per-case cells of `e00088_cases.py` and the checks of `e00088_controls.py`, then writes
tables to `work/results/*.parquet` and prints a report to standard output.

    uv run python ml-experiments/00088-*/e00088_report.py
"""

from typing import Final

import e00088_common as common
import e00088_stats as stats
import numpy as np
import polars as pl

HEADLINE: Final[tuple[str, ...]] = stats.HEADLINE_SHAPES
RESULTS_DIR = common.RESULTS_DIR


def load() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return the case table, the events, and the per-band cells."""
    first_pass = pl.concat(
        [
            frame
            for path in stats.cases_dir_files(pattern="cells_*.parquet")
            if (frame := pl.read_parquet(path)).height
        ]
    )
    second_pass = pl.concat(
        [
            frame
            for path in stats.cases_dir_files(
                pattern="cells_*.parquet", directory=common.WORK_DIR / "cases_caus"
            )
            if (frame := pl.read_parquet(path)).height
        ]
    )
    # The first pass forecast every variant; its causal-MW variants summed duplicate intervals, so
    # the second pass (which de-duplicates them and adds the max-merge variants) replaces them.
    redone = second_pass["variant"].unique().to_list()
    cells = pl.concat(
        [first_pass.filter(~pl.col("variant").is_in(redone)), second_pass], how="diagonal"
    )
    events = pl.concat(
        [pl.read_parquet(path) for path in stats.cases_dir_files(pattern="events_*.parquet")]
    )
    return stats.case_table(cells=cells, events=events), events, cells


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
        ("post hoc arm4_cmax - arm1", "arm4_cmax", "arm1"),
        ("post hoc arm2_cmax - arm1", "arm2_cmax", "arm1"),
        ("post hoc arm4_cmax - arm2_cmax", "arm4_cmax", "arm2_cmax"),
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
        for arm in (
            "arm2_inj",
            "arm2_caus",
            "arm2_cmax",
            "arm3_inj",
            "arm3_caus",
            "arm4_inj",
            "arm4_caus",
            "arm4_cmax",
        ):
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


def arm_means(*, frame: pl.DataFrame, shapes: tuple[str, ...]) -> dict[str, float]:
    """Return each arm's weighted mean absolute error, shapes averaged with equal weight."""
    out = {}
    for arm, variant in stats.ARMS.items():
        work = frame.with_columns(pl.col(f"mae_{variant}").alias("_v")).drop_nulls("_v")
        out[arm] = stats.mean_value(frame=work, shapes=shapes, column="_v").point
    return out


def detection(*, frame: pl.DataFrame) -> dict[str, float]:
    """Return how often the causal detector had marked the transfer at the newest lag.

    Cases are weighted by the event's MW. `found` is the share with a non-zero causal MW at the
    newest lag, `right_sign` the share with the injected sign, and `ratio` the weighted median of
    the causal over the injected MW among cases with the right sign.
    """
    w = frame["w"].to_numpy()
    caus, inj = frame["caus_mw"].to_numpy(), frame["inj_mw"].to_numpy()
    found = caus != 0
    right = found & (np.sign(caus) == np.sign(inj)) & (inj != 0)
    wrong = found & ~right
    ratio = caus[right] / inj[right]
    order = np.argsort(ratio)
    cumulative = np.cumsum(w[right][order]) / max(w[right].sum(), 1e-12)
    median = float(ratio[order][np.searchsorted(cumulative, 0.5)]) if right.any() else float("nan")
    total = max(w.sum(), 1e-12)
    return {
        "found": float((w * found).sum() / total),
        "right_sign": float((w * right).sum() / total),
        "wrong_or_stray": float((w * wrong).sum() / total),
        "ratio_median": median,
    }


def by_bins(
    *, cases: pl.DataFrame, family: str, column: str, bins: tuple[tuple[str, float, float], ...]
) -> pl.DataFrame:
    """Return arm means, C1, and detection for bins of a column, headline shapes pooled."""
    rows = []
    frame = cases.filter(pl.col("scored") & (pl.col("family") == family))
    for name, low, high in bins:
        part = frame.filter((pl.col(column) >= low) & (pl.col(column) < high))
        if part.filter(pl.col("shape").is_in(HEADLINE)).is_empty():
            continue
        means = arm_means(frame=part, shapes=HEADLINE)
        c1 = stats.contrast(frame=part, shapes=HEADLINE, arm_a="arm4_caus", arm_b="arm1", level=95)
        c2 = stats.contrast(frame=part, shapes=HEADLINE, arm_a="arm3_inj", arm_b="arm1", level=95)
        rows.append(
            {
                "family": family,
                "column": column,
                "bin": name,
                "cases": part.filter(pl.col("shape") == "S2").height,
                **{f"mae_{arm}": value for arm, value in means.items()},
                "c1": c1.point,
                "c1_low": c1.low,
                "c1_high": c1.high,
                "c2": c2.point,
                "c2_low": c2.low,
                "c2_high": c2.high,
                **detection(frame=part.filter(pl.col("shape").is_in(HEADLINE))),
            }
        )
    return pl.DataFrame(rows)


def lead_band_table(*, cells: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """Return each variant's MW-weighted mean error by lead band, family A, headline shapes."""
    weights = events.select("shape", "event_id", "moved_mw", "onset_month").with_columns(
        w=pl.col("moved_mw").clip(lower_bound=0.0), month=pl.col("onset_month") % 100
    )
    work = (
        cells.filter((pl.col("family") == "A") & pl.col("shape").is_in(HEADLINE))
        .join(weights, on=["shape", "event_id"], how="left")
        .filter(pl.col("month").is_in(list(stats.SCORED_MONTH_NUMBERS)))
    )
    return (
        work.group_by("band", "variant")
        .agg(
            mae=(pl.col("mae") * pl.col("w") * pl.col("n")).sum()
            / (pl.col("w") * pl.col("n")).sum(),
            qs=(pl.col("qs") * pl.col("w") * pl.col("n")).sum() / (pl.col("w") * pl.col("n")).sum(),
        )
        .sort("band", "variant")
    )


def odd_month_check(*, cases: pl.DataFrame) -> pl.DataFrame:
    """Return the headline contrasts on events starting in the odd months (XL tuned there)."""
    odd = cases.filter(~pl.col("scored") & (pl.col("family") == "A"))
    rows = []
    for name, arm_a, arm_b in (
        ("arm4_caus - arm1", "arm4_caus", "arm1"),
        ("arm4_cmax - arm1", "arm4_cmax", "arm1"),
        ("arm3_inj - arm1", "arm3_inj", "arm1"),
    ):
        interval = stats.contrast(frame=odd, shapes=HEADLINE, arm_a=arm_a, arm_b=arm_b, level=95.0)
        rows.append(
            {"contrast": name, "point": interval.point, "low": interval.low, "high": interval.high}
        )
    return pl.DataFrame(rows)


def controls() -> None:
    """Print the false-alarm and real-event checks when their files exist."""
    false_path = common.CASES_DIR / "controls_false.parquet"
    if false_path.exists():
        table = pl.read_parquet(false_path)
        summary = (
            table.group_by("variant")
            .agg(
                mae=(pl.col("mae") * pl.col("n")).sum() / pl.col("n").sum(),
                qs=(pl.col("qs") * pl.col("n")).sum() / pl.col("n").sum(),
                cases=pl.struct("site", "origin").n_unique(),
                marked=(pl.col("caus_mw_newest") != 0).mean(),
            )
            .sort("variant")
        )
        print("\nFalse-alarm check (500 pairs with no transfer)")
        print(summary)
        save(name="controls_false", frame=summary)
    real_path = common.CASES_DIR / "controls_real.parquet"
    if real_path.exists():
        table = pl.read_parquet(real_path)
        summary = (
            table.group_by("variant")
            .agg(
                mae=(pl.col("mae") * pl.col("n")).sum() / pl.col("n").sum(),
                qs=(pl.col("qs") * pl.col("n")).sum() / pl.col("n").sum(),
                events=pl.col("card_id").n_unique(),
                cases=pl.struct("site", "origin", "card_id").n_unique(),
            )
            .sort("variant")
        )
        print("\nReal-event check (NRA estimates as the truth)")
        print(summary)
        save(name="controls_real", frame=summary)


def main() -> None:
    """Compute every table and print it."""
    cases, events, cells = load()
    print(f"cases: {cases.height:,}; events: {events.select('shape', 'event_id').n_unique():,}")
    print(cases.group_by("shape", "family", "scored").len().sort("shape", "family", "scored"))
    save(name="planned_contrasts", frame=headline(cases=cases))
    for label, shapes in (("headline", HEADLINE), ("S1", ("S1",))):
        table = gap_shares(cases=cases, label=label, shapes=shapes)
        save(name=f"gap_shares_{label}", frame=table)
        print(f"\nShare of the gap closed, {label}")
        print(table)
    hours_b = (("0-24 h", 0.0, 24.0), ("1-3 d", 24.0, 72.0), ("3-7 d", 72.0, 1e9))
    for name, family, column, bins in (
        ("by_onset", "A", "hours_at_origin", stats.ONSET_BINS),
        ("by_step_a", "A", "donor_step_z", stats.STEP_BINS),
        ("by_since_end", "B", "hours_at_origin", hours_b),
        ("by_step_b", "B", "donor_step_z", stats.STEP_BINS),
    ):
        table = by_bins(cases=cases, family=family, column=column, bins=bins)
        save(name=name, frame=table)
        print(f"\n{name}: family {family} by {column}")
        with pl.Config(tbl_rows=30, tbl_cols=40, fmt_str_lengths=30):
            print(table)
    bands = lead_band_table(cells=cells, events=events)
    save(name="lead_bands", frame=bands)
    print("\nBy lead band (family A, headline shapes, MW-weighted mean absolute error)")
    print(
        bands.pivot(on="variant", index="band", values="mae").with_columns(
            pl.col(pl.Float64).round(3)
        )
    )
    odd = odd_month_check(cases=cases)
    save(name="odd_months", frame=odd)
    print("\nOdd-month check (detector tuned on these months), 95% intervals")
    print(odd)
    controls()
    scored_a = cases.filter(pl.col("scored") & (pl.col("family") == "A"))
    absolute = stats.absolute_scores(frame=scored_a, shapes=HEADLINE)
    save(name="absolute_family_a", frame=absolute)
    print("\nAbsolute scores, family A, headline shapes")
    print(absolute.filter(pl.col("metric").is_in(["mae", "qs"])))


if __name__ == "__main__":
    main()
