"""Placebo control of experiment 155: the winter offset with the winter indicator moved six months.

Scores the placebo against arm A and against arm B (all leads, scored year, five decision
substations, outside the switching events) and writes `work/placebo.json`.

    uv run --no-sync python ml-experiments/00155-*/e00155_placebo.py
"""

import e00155_common as common
import e00155_corrections as corr
import e00155_report as report
import numpy as np
import polars as pl


def main() -> None:
    """Build the placebo arm at the decision substations and print its score change."""
    cache = common.load_cache()
    rows = {}
    for site in common.DECISION_SITES:
        base = corr.add_calendar(frame=common.read_base(site=site))
        shift, _ = corr.winter_offset_shift(frame=base, variant="placebo")
        arm = corr.apply_shift_and_refit_spread(frame=base, shift=shift)
        part = arm.filter(pl.col("period") == "scored").with_columns(site=pl.lit(site))
        rows[site] = part.select(report.BASE_COLUMNS)
    placebo = report.annotate(frame=pl.concat(rows.values()), cache=cache)
    out = {}
    for name in ("a", "b"):
        frames = [
            pl.read_parquet(common.out_path(arm=name, period="scored", site=s)).with_columns(
                site=pl.lit(s)
            )
            for s in common.DECISION_SITES
        ]
        other = report.annotate(frame=pl.concat(frames).select(report.BASE_COLUMNS), cache=cache)
        shared = report.common_rows(frames={"placebo": placebo, name: other})
        scores = {k: float(report.with_losses(frame=v)["qs"].mean()) for k, v in shared.items()}
        out[name] = {**scores, "placebo_minus": scores["placebo"] - scores[name]}
    common.write_json_atomic(path=common.WORK_DIR / "placebo.json", content=out)
    print(out)
    assert np.isfinite(out["a"]["placebo_minus"])


if __name__ == "__main__":
    main()
