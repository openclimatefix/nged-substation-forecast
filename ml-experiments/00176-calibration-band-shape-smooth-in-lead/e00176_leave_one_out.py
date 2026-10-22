"""Leave one decision substation out of the pooled quantile-score contrasts of experiment 176.

uv run --no-sync python ml-experiments/00176-*/e00176_leave_one_out.py
"""

import e00176_common as common
import e00176_report as report
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.experiment_store import losses_for_bootstrap

ARMS: tuple[str, ...] = ("p", "s", "ps")


def main() -> None:
    """Print each arm's pooled contrast with every decision substation left out in turn."""
    cache = common.load_cache()
    out: dict = {}
    for period in ("scored", "replicate"):
        frames = {
            arm: report.with_losses(frame=report.load_arm(arm=arm, period=period, cache=cache))
            for arm in ("a", *ARMS)
        }
        scored = {
            arm: frame.filter(
                ~pl.col("switching_mask")
                & pl.col("actual").is_finite()
                & (pl.col("lead_hours") <= common.HORIZON_HOURS)
            )
            for arm, frame in frames.items()
        }
        ids = {
            row["site"]: row["time_series_id"]
            for row in scored["a"].select("site", "time_series_id").unique().to_dicts()
        }
        keep = [*report.KEYS, "lead_hours", "seed", "qs"]
        losses = losses_for_bootstrap(arm_losses={arm: f.select(keep) for arm, f in scored.items()})
        for left_out in (None, *common.DECISION_SITES):
            sites = [ids[s] for s in common.DECISION_SITES if s != left_out]
            part = losses.filter(pl.col("site").is_in(sites))
            for arm in ARMS:
                t = part.filter(pl.col("arm") == arm)["qs"].to_numpy()
                r = part.filter(pl.col("arm") == "a")["qs"].to_numpy()
                lower, upper = bootstrap_difference_at_level(
                    losses=part,
                    treatment=arm,
                    reference="a",
                    metric="qs",
                    level=report.LEVEL_PERCENT,
                    n_resamples=report.N_RESAMPLES,
                )
                key = f"{period}|{arm}|without {left_out}"
                out[key] = [float(t.mean() - r.mean()), float(lower), float(upper)]
                print(key, [round(v, 4) for v in out[key]], flush=True)
    common.write_json_atomic(path=common.WORK_DIR / "report" / "leave_one_out.json", content=out)


if __name__ == "__main__":
    main()
