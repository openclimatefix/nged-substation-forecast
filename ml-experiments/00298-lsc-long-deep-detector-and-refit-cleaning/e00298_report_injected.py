"""Print Part 1's injected-shift tables: recall by duration, the stratum split, errors, placebo.

Scores the even-month onsets only (the odd-month onsets chose the variants). The stratum is
experiment 302's rule applied to injected shifts: 2 days or more is "long and deep" (every injected
size is at least 10% of load); a 1-day shift is "the rest". MW weights are size times the Primary's
median daily power.

    uv run --no-sync python e00298_report_injected.py > printed_injected.txt
"""

from __future__ import annotations

import e00298_common as common
import polars as pl

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(200)


def main() -> None:
    """Print the tables."""
    panel = pl.read_parquet(common.PANEL_PATH).group_by("site").agg(level=pl.col("power_mw").median())
    d = pl.read_parquet(common.EXPERIMENT_DIR / "injected_detection.parquet").join(panel, on="site")
    d = d.filter(pl.col("onset").dt.month() % 2 == 0).with_columns(
        weight=pl.col("size") * pl.col("level"),
        stratum=pl.when(pl.col("duration") == 1).then(pl.lit("rest (1 day)")).otherwise(pl.lit("long (2+ days)")),
    )
    true = d.filter(pl.col("run") == "true", pl.col("duration") != -1)
    print("== Recall by duration (unmetered receiver, share shape, 25% of load) ==")
    sel = true.filter(~pl.col("metered"), pl.col("shape") == "share", pl.col("size") == 0.25)
    print(sel.group_by("variant", "duration").agg(r=pl.col("found").mean()).pivot(on="duration", index="variant", values="r").sort("variant"))
    print("== Recall by size (unmetered, share, 7 to 90 days) ==")
    sel = true.filter(~pl.col("metered"), pl.col("shape") == "share", pl.col("duration") >= 7)
    print(sel.group_by("variant", "size").agg(r=pl.col("found").mean()).pivot(on="size", index="variant", values="r").sort("variant"))
    print("== Recall by shape (unmetered, 25%, 7 to 90 days) ==")
    sel = true.filter(~pl.col("metered"), pl.col("size") == 0.25, pl.col("duration") >= 7)
    print(sel.group_by("variant", "shape").agg(r=pl.col("found").mean()).pivot(on="shape", index="variant", values="r").sort("variant"))
    print("== Stratum split: matches, alarms per match, MW-weighted matches (unmetered and metered shown apart) ==")
    for met in (False, True):
        sel = true.filter(pl.col("metered") == met)
        out = sel.group_by("variant", "stratum").agg(
            n=pl.len(), matches=pl.col("found").sum(),
            alarms=pl.col("found").sum() + pl.col("n_spans_outside").sum(),
            mw=(pl.col("found").cast(pl.Float64) * pl.col("weight")).sum() / pl.col("weight").sum(),
        ).with_columns(apm=pl.col("alarms") / pl.col("matches"), recall=pl.col("matches") / pl.col("n"))
        print(f"-- metered={met}")
        print(out.sort("stratum", "variant"))
    print("== Timing among found worlds (long, unmetered): medians in hours ==")
    sel = true.filter(~pl.col("metered"), pl.col("found"), pl.col("duration") >= 6)
    print(sel.group_by("variant").agg(lag=pl.col("alarm_lag_h").median(), onset_err=pl.col("onset_err_h").median(), end_err=pl.col("end_err_h").median(), n=pl.len()).sort("variant"))
    print("== Neighbour checks: true partner against shifted-week placebo (metered, 25%, share, 2+ days) ==")
    sel = d.filter(pl.col("metered"), pl.col("size") == 0.25, pl.col("duration") >= 2, pl.col("variant").str.contains("^[DEF]"))
    print(sel.group_by("variant", "run").agg(r=pl.col("found").mean(), explained=pl.col("explained").mean()).pivot(on="run", index="variant", values="r").sort("variant"))
    print("== Unmetered receivers (no partner): D to F recall is the chance level ==")
    sel = d.filter(~pl.col("metered"), pl.col("size") == 0.25, pl.col("shape") == "share", pl.col("duration") >= 2, pl.col("variant").str.contains("^[DEF]"), pl.col("run") == "true")
    print(sel.group_by("variant").agg(r=pl.col("found").mean()).sort("variant"))
    print("== Permanent steps: share alarmed ==")
    steps = d.filter(pl.col("run") == "true", pl.col("duration") == -1)
    print(steps.group_by("variant", "metered").agg(r=pl.col("found").mean(), n=pl.len()).pivot(on="metered", index="variant", values="r").sort("variant"))


if __name__ == "__main__":
    main()
