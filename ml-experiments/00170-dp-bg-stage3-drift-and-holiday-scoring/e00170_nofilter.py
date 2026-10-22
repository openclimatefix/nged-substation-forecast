"""Score the stage 1 arms with and without the Kalman filter (post hoc diagnostic).

Experiment 167 found that a drift term lowered the score by 0.40 points on a forecast without the
filter. This script tests whether the filter explains why the drift term does not help here: it
scores each stage 1 arm and its `_nf` twin (the same fits, the filter's correction left out) on the
scored year's common rows at the five decision substations, over leads 0.5 to 168 hours.

    uv run --no-sync python ml-experiments/00170-*/e00170_nofilter.py
"""

import e00170_report as report
import e00170_run as run
import e00170_stage2_common as common
import polars as pl


def main() -> None:
    """Score the six stage 1 arms and write `work/report/nofilter.json`."""
    cache = common.load_cache()
    arms = [*run.ARM_NAMES, *run.NO_FILTER_ARMS]
    frames = {}
    for arm in arms:
        pieces = [
            pl.read_parquet(common.STAGE1_DIR / f"{arm}_scored_{site}.parquet").with_columns(
                site=pl.lit(site)
            )
            for site in common.SCORED_SITES
        ]
        frame = pl.concat(pieces).select(report.BASE_COLUMNS)
        frames[arm] = report.annotate(
            frame=report.in_period(frame=frame, period="scored"), cache=cache
        )
    losses = report.stack(frames=report.common_rows(frames=frames))
    decision = report.lead_filter(
        frame=losses.filter(pl.col("site").is_in(list(common.DECISION_SITES))), low=0.0, high=168.0
    )
    table = (
        decision.group_by("arm")
        .agg(qs=pl.col("qs").mean(), abs_mean=pl.col("abs_mean").mean(), rows=pl.len())
        .sort("arm")
        .to_dicts()
    )
    contrasts = {}
    for label, (treatment, reference) in {
        "drift m24 minus no drift, with filter": ("m24_d1", "a2_repro"),
        "drift m24 minus no drift, no filter": ("m24_d1_nf", "a2_repro_nf"),
        "drift all years minus no drift, with filter": ("all_d1", "a2_repro"),
        "drift all years minus no drift, no filter": ("all_d1_nf", "a2_repro_nf"),
        "filter minus no filter, no drift": ("a2_repro", "a2_repro_nf"),
        "filter minus no filter, drift m24": ("m24_d1", "m24_d1_nf"),
    }.items():
        contrasts[label] = {
            "qs": report.contrast(losses=decision, treatment=treatment, reference=reference),
            "abs_mean": report.contrast(
                losses=decision, treatment=treatment, reference=reference, metric="abs_mean"
            ),
        }
    common.write_json_atomic(
        path=common.WORK_DIR / "report" / "nofilter.json",
        content={"table": table, "contrasts": contrasts},
    )
    print("nofilter written")


if __name__ == "__main__":
    main()
