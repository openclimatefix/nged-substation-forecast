"""Print every number the write-up of experiment 00037 quotes, and save the small tables.

uv run python ml-experiments/00037-switching-first-look/e00037_report.py | tee report.log
"""

import numpy as np
import polars as pl
from e00037_common import WORK_DIR

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(240)


def q1_tables(*, real: pl.DataFrame, placebo: pl.DataFrame) -> None:
    """Confirmed transfers by duration band, kind, pair units, and size band, against placebo."""
    seeds = placebo["placebo_seed"].n_unique()
    print("\n== Q1: eligible closures (2022 onwards, 2 h to 60 d), one row per sheet and start ==")
    print(
        f"eligible {real.height}; confirmed {real['confirmed'].sum()} "
        f"({real['confirmed'].mean():.3f}); placebo confirmed per seed "
        f"{placebo['confirmed'].sum() / seeds:.1f} of {placebo.height / seeds:.0f} "
        f"({placebo['confirmed'].mean():.3f})"
    )
    for name, frame, divisor in (("real", real, 1), ("placebo (mean of seeds)", placebo, seeds)):
        print(f"-- {name}")
        print(
            frame.group_by("duration_band", "kind")
            .agg(n=pl.len() / divisor, confirmed=pl.col("confirmed").sum() / divisor)
            .sort("duration_band", "kind")
        )
    two = real.filter(pl.col("kind") == "two_sided", pl.col("series_a").is_not_null())
    print("-- two-sided pairs: confirmation rate by pair units (MVA = either side non-directional)")
    print(
        two.group_by("has_mva").agg(
            n=pl.len(), confirmed=pl.col("confirmed").sum(), rate=pl.col("confirmed").mean()
        )
    )
    print("-- confirmed transfers by split, kind, and size band (share of the receiver's load)")
    print(
        real.filter("confirmed")
        .group_by("split", "kind", "size_band")
        .len()
        .sort("split", "kind", "size_band")
    )
    print("-- eligible closures by duration band and split")
    print(real.group_by("split", "duration_band").len().sort("split", "duration_band"))
    ratios = real.filter("confirmed", pl.col("kind") == "two_sided").with_columns(
        r=(pl.col("step_b") / pl.col("step_a")).abs()
    )
    print("-- |step_b / step_a| of two-sided confirmed transfers (1 = exact conservation)")
    print(
        ratios.select(
            n=pl.len(),
            q25=pl.col("r").quantile(0.25),
            median=pl.col("r").median(),
            q75=pl.col("r").quantile(0.75),
        )
    )


def step_variation(*, real: pl.DataFrame) -> None:
    """Whether the relative step differs between the first and second half of an event."""
    print("\n== Does the step vary within an event? (confirmed, >= 6 h, larger-step side) ==")
    long = real.filter("confirmed", pl.col("hours") >= 6)
    rows = []
    for row in long.iter_rows(named=True):
        side = "a" if abs(row["step_a"] or 0) >= abs(row["step_b"] or 0) else "b"
        first, second = row[f"rel_first_{side}"], row[f"rel_second_{side}"]
        if first is None or second is None:
            continue
        rows.append(
            {
                "sheet": row["sheet"],
                "start_utc": row["start_utc"],
                "hours": row["hours"],
                "rel_first": first,
                "rel_second": second,
            }
        )
    frame = pl.DataFrame(rows).with_columns(
        ratio=pl.col("rel_second") / pl.col("rel_first"),
        same_sign=pl.col("rel_second") * pl.col("rel_first") > 0,
    )
    frame.write_parquet(WORK_DIR / "step_variation.parquet")
    print(
        frame.select(
            n=pl.len(),
            same_sign=pl.col("same_sign").sum(),
            median_ratio=pl.col("ratio").median(),
            within_factor_2=((pl.col("ratio") > 0.5) & (pl.col("ratio") < 2)).sum(),
        )
    )
    print(frame.sort("hours", descending=True))


def multi_receiver(*, intervals: pl.DataFrame, real: pl.DataFrame) -> None:  # noqa: PLR0912
    """Operations in which one series sits on several sheets, so load may split across receivers."""
    print("\n== Logged operations with more than one receiver (one series on >= 2 sheets) ==")
    rows = []
    for operation, group in intervals.filter(pl.col("start_utc").dt.year() >= 2022).group_by(
        "operation"
    ):
        sheets = group.unique(subset=["sheet"])
        counts: dict[str, set[str]] = {}
        for row in sheets.iter_rows(named=True):
            for label, other in ((row["side_a"], row["side_b"]), (row["side_b"], row["side_a"])):
                if label not in (None, "unknown") and other not in (None, "unknown"):
                    counts.setdefault(label, set()).add(other)
        for hub, partners in counts.items():
            if len(partners) >= 2:
                rows.append(
                    {
                        "operation": operation[0],
                        "hub": hub,
                        "receivers": ",".join(sorted(partners)),
                        "n_receivers": len(partners),
                    }
                )
    frame = pl.DataFrame(rows) if rows else pl.DataFrame()
    n_ops = intervals.filter(pl.col("start_utc").dt.year() >= 2022)["operation"].n_unique()
    print(
        f"operations since 2022: {n_ops}; "
        f"with a multi-receiver hub: {frame['operation'].n_unique()}"
    )
    step = real.select(
        "operation",
        "sheet",
        "series_a",
        "series_b",
        "step_a",
        "step_b",
        "confirmed",
        "hours",
        "start_utc",
        "split",
    )
    summary = []
    for row in frame.iter_rows(named=True):
        group = step.filter(pl.col("operation") == row["operation"])
        hub = int(row["hub"])
        hub_steps, receiver_steps = [], {}
        for g in group.iter_rows(named=True):
            for tag, other in (("a", "b"), ("b", "a")):
                if g[f"series_{tag}"] == hub and g[f"step_{tag}"] is not None:
                    hub_steps.append(g[f"step_{tag}"])
                    if g[f"series_{other}"] is not None and g[f"step_{other}"] is not None:
                        receiver_steps[g[f"series_{other}"]] = g[f"step_{other}"]
        if hub_steps and receiver_steps:
            total = sum(receiver_steps.values())
            summary.append(
                {
                    **row,
                    "hub_step_mw": float(np.median(hub_steps)),
                    "receiver_sum_mw": total,
                    "ratio": -total / np.median(hub_steps),
                    "fractions": ";".join(
                        f"{k}:{v / total:.2f}" for k, v in receiver_steps.items()
                    ),
                    "start_utc": group["start_utc"][0],
                    "any_confirmed": bool(group["confirmed"].any()),
                }
            )
    if summary:
        out = pl.DataFrame(summary)
        out.write_parquet(WORK_DIR / "multi_receiver.parquet")
        print(out.sort("start_utc"))
    else:
        print("no multi-receiver operation has telemetry on the hub and a receiver")


def alarm_day_share() -> None:
    """Share of alarms on the 1st of the month, for the monthly-refit check."""
    print("\n== Share of alarms on the 1st of a month (chance level 0.033) ==")
    chosen = pl.read_parquet(WORK_DIR / "chosen.parquet").filter(pl.col("grid") == "pre_registered")
    from datetime import timedelta

    from e00037_grid import GRID_START

    rows = []
    for row in chosen.iter_rows(named=True):
        table = pl.read_parquet(
            WORK_DIR / f"alarms_{row['detector']}_{row['window']}_{row['h']}.parquet"
        )
        days = [(GRID_START + timedelta(minutes=30 * s)).day for s in table["stamp"].to_list()]
        rows.append(
            {
                "detector": row["detector"],
                "window": row["window"],
                "alarms": len(days),
                "share_first_day": float(np.mean([d == 1 for d in days])) if days else None,
            }
        )
    print(pl.DataFrame(rows))


def partner_counts() -> None:
    """How many neighbours the balance detector found at each alarm, and the split fractions."""
    print("\n== D1 partners per alarm (pre-registered operating point, w=8) ==")
    table = pl.read_parquet(WORK_DIR / "alarms_D1_8_4.0.parquet").with_columns(
        n_partners=pl.col("partners").str.count_matches(":")
    )
    print(table.group_by("n_partners").len().sort("n_partners"))
    multi = table.filter(pl.col("n_partners") >= 2)
    print(f"alarms with 2 or more partners: {multi.height} of {table.height}")
    print(table["balance_ratio"].describe())


def main() -> None:
    """Print all tables."""
    real = pl.read_parquet(WORK_DIR / "oracle_real.parquet").filter("primary")
    placebo = pl.read_parquet(WORK_DIR / "oracle_placebo.parquet").filter("primary")
    intervals = pl.read_parquet(WORK_DIR / "intervals.parquet")
    q1_tables(real=real, placebo=placebo)
    step_variation(real=real)
    multi_receiver(intervals=intervals, real=real)
    print("\n== Detector curve (validation) and chosen operating points ==")
    print(pl.read_parquet(WORK_DIR / "curve.parquet"))
    print(
        pl.read_parquet(WORK_DIR / "chosen.parquet").transpose(
            include_header=True,
            column_names=[
                f"{r['grid'][:3]}_{r['detector']}_w{r['window']}"
                for r in pl.read_parquet(WORK_DIR / "chosen.parquet").iter_rows(named=True)
            ],
        )
    )
    alarm_day_share()
    partner_counts()
    print("\n== Stack tail (Q2) ==")
    print(pl.read_parquet(WORK_DIR / "tail_shares.parquet"))
    print(pl.read_parquet(WORK_DIR / "tail_conserved_by_series.parquet"))


if __name__ == "__main__":
    main()
