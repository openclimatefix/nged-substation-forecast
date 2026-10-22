# ruff: noqa: E501
"""Score the post hoc variants of experiment 221 and print the in-sample diagnostic.

uv run --no-sync python ml-experiments/00221-wind-correction-head-on-stored-predictions/e00221_posthoc_score.py
"""

import json

import e00221_common as common
import numpy as np
import polars as pl
from e00221_score import BANDS, PRIMARY_BAND, SECOND_BAND, contrast, losses, read_arm

NAMES = ("A0", "A1", "A2", "A1_short", "A2_short", "A1_linear", "A2_linear")


def main() -> None:
    """Print the variants' scores and contrasts against A0, and the in-sample gap."""
    arms = {name: read_arm(name=name) for name in NAMES}
    out = []
    for name in NAMES:
        for band in (PRIMARY_BAND, SECOND_BAND, None):
            frame = arms[name] if band is None else arms[name].filter(pl.col("band") == band)
            row = {
                "arm": name,
                "band": "all" if band is None else BANDS[band],
                "qs": float(frame["qs"].to_numpy().mean()),
                "covered": float(frame["covered"].to_numpy().mean()),
            }
            if name != "A0":
                c = contrast(arms=arms, treatment=name, reference="A0", band=band)
                row |= {"diff": c["difference"], "lower": c["lower"], "upper": c["upper"]}
            out.append(row)
            print(json.dumps(row))
    print("\nIn-sample (training rows), mean quantile score by arm, days 1-2:")
    frames = [
        pl.read_parquet(p) for p in sorted((common.WORK_DIR / "posthoc").glob("insample_*.parquet"))
    ]
    table = pl.concat(frames)
    for arm in ("A0", "A1", "A2"):
        part = table.filter(pl.col("arm") == arm).drop("arm", "month")
        scored = losses(
            frame=part.sort("time_series_id", "power_fcst_init_time", "valid_time", "seed")
        )
        for band in (PRIMARY_BAND, SECOND_BAND):
            sub = scored.filter(pl.col("band") == band)
            print(
                arm,
                BANDS[band],
                round(float(sub["qs"].to_numpy().mean()), 3),
                "coverage",
                round(float(sub["covered"].to_numpy().mean()), 3),
            )
    print("\nOracle rescaling of the best guess about its median, fitted on the scored rows:")
    frames = [
        pl.read_parquet(p) for p in sorted((common.WORK_DIR / "arms").glob("A0_seed*.parquet"))
    ]
    a0 = pl.concat(frames)
    for index in (PRIMARY_BAND, SECOND_BAND):
        name, low, high = common.LEAD_DAY_BANDS[index]
        part = a0.filter((pl.col("lead_day") >= low) & (pl.col("lead_day") <= high))
        quantiles = part.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
        actual = part["actual_norm"].to_numpy().astype(np.float64)
        best = min(
            common.SPREAD_GRID,
            key=lambda k: common.quantile_score(
                quantiles=common.scale_about_median(quantiles=quantiles, factor=float(k)),
                actual=actual,
            ).mean(),
        )
        scaled = common.scale_about_median(quantiles=quantiles, factor=float(best))
        print(
            name,
            "factor",
            float(best),
            "score",
            round(100 * float(common.quantile_score(quantiles=scaled, actual=actual).mean()), 3),
            "against",
            round(100 * float(common.quantile_score(quantiles=quantiles, actual=actual).mean()), 3),
            "coverage",
            round(float(common.covered(quantiles=scaled, actual=actual).mean()), 3),
        )
    (common.WORK_DIR / "results" / "posthoc.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
