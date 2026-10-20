"""Write experiment 111's predictions as named sets in the experiment store.

One set per arm and period, through `studies.experiment_store.write_predictions`, never
overwriting. Every set holds, in points of effective capacity: the keys (`time_series_id`,
`power_fcst_init_time`, `valid_time`), `seed`, `lead_hours`, the 11 quantiles `q01` to `q99`,
`actual`, `arm`, and `switching_mask` (true inside the hand-filtered switching events, so a scorer
can leave those rows out). The Kalman arms add `level` and `anomaly` (the filter's two parts), and
`gam_margin` adds `margin`, the ensemble mean of the DP's forecast that its XGBoost model took as
the base margin.

The scored sets are named `<arm>_scored` (row set `fold_a_daily_gap_v1`, the 292 runs) and the
replicate sets `<arm>_replicate` (row set `dp_replicate_v1`, the even months of 2024-07 to
2025-06). `xnl_scored` is SIMP1's stored monthly no-lag refits, copied for the scored runs. D18 is
not in these sets except in the DP arms.

    uv run python ml-experiments/00111-*/e00111_predictions.py
"""

from typing import Final

import e00111_arms as arms
import e00111_common as common
import e00111_dp_run as dp_run
import e00111_ens as ens
import e00111_score as score
import numpy as np
import polars as pl
from studies.dp_ens import member_means
from studies.experiment_store import HOLDOUT_START, write_predictions

REPLICATE_ROW_SET: Final[str] = "dp_replicate_v1"
DP_ARMS: Final[tuple[str, ...]] = ("gam0", "gamk", "game", "gamk_era5")
XGB_ARMS: Final[tuple[str, ...]] = ("xnl_z", "xnl_o", "xlag_d", "xlag_o", "gam_margin")


def with_switching(*, frame: pl.DataFrame, sites: tuple[str, ...]) -> pl.DataFrame:
    """Add `switching_mask` to a frame that lacks it, from the feature cache's masks."""
    if "switching_mask" in frame.columns:
        return frame
    cache = common.load_cache()
    pieces = []
    for site in sites:
        part = frame.filter(pl.col("site") == site)
        position = np.searchsorted(
            cache.end_times, part["valid_time"].dt.replace_time_zone(None).to_numpy()
        )
        last = len(cache.end_times) - 1
        masked = cache.sites[site].switching_mask[np.minimum(position, last)]
        pieces.append(part.with_columns(switching_mask=pl.Series(masked)))
    return pl.concat(pieces)


def dp_frame(*, arm: str, period: str, sites: tuple[str, ...]) -> pl.DataFrame:
    """Return a DP arm's stored forecasts for a period with the `level` and `anomaly` columns."""
    pieces = []
    for site in sites:
        path = dp_run.out_path(arm=arm, period=period, site=site)
        if path.exists():
            pieces.append(pl.read_parquet(path).with_columns(site=pl.lit(site)))
    return score.in_period(frame=pl.concat(pieces), periods={period})


def margin_column(*, frame: pl.DataFrame, sites: tuple[str, ...]) -> pl.DataFrame:
    """Add the ensemble-mean DP margin (points) to the rows of `gam_margin`."""
    cache = common.load_cache()
    runs = common.run_dates()
    pieces = []
    for site in sites:
        setup = ens.load_setup(cache=cache, site=site).with_calibration(
            calibration=arms.load_calibration(site=site)
        )
        fits = arms.load_arm_fits(arm=arms.ARMS["gam0"], site=site)
        part = frame.filter(pl.col("site") == site)
        tables = []
        for origin in part["power_fcst_init_time"].unique().to_list():
            index = runs.index(origin - common.PUBLICATION_DELAY)
            if (
                len(cache.end_times) - setup.runs[index].origin_position - 1
                < ens.MIN_WINDOW_HALF_HOURS
            ):
                continue
            result = member_means(
                cache=cache,
                context=setup.context,
                run=setup.runs[index],
                origin=origin,
                fits=fits,
            )
            times = pl.Series(cache.end_times[result.window]).dt.replace_time_zone("UTC")
            tables.append(
                pl.DataFrame(
                    {
                        "power_fcst_init_time": [origin] * len(times),
                        "valid_time": times,
                        "margin": result.means.mean(axis=0),
                    }
                ).with_columns(pl.col("power_fcst_init_time").dt.replace_time_zone("UTC"))
            )
        pieces.append(
            part.join(pl.concat(tables), on=["power_fcst_init_time", "valid_time"], how="left")
        )
    return pl.concat(pieces)


def store(*, name: str, frame: pl.DataFrame, row_set_id: str) -> None:
    """Write one named set with the stored columns.

    Rows with no reading, and rows on or after the held-out start, are left out.
    """
    frame = frame.filter(pl.col("actual").is_finite() & (pl.col("valid_time") < HOLDOUT_START))
    columns = [
        "time_series_id",
        "power_fcst_init_time",
        "valid_time",
        "seed",
        "lead_hours",
        *score.LEVEL_COLUMNS,
        "actual",
        "arm",
        "switching_mask",
        *[c for c in ("level", "anomaly", "margin") if c in frame.columns],
    ]
    write_predictions(
        experiment_id=common.EXPERIMENT_ID,
        name=name,
        frame=frame.select(columns),
        kind="quantiles",
        row_set_id=row_set_id,
    )
    print(f"wrote {name}: {frame.height:,} rows", flush=True)


def main() -> None:
    """Write every arm's scored and replicate sets."""
    sites = common.SCORED_SITES
    for period, row_set in (("scored", common.ROW_SET_ID), ("replicate", REPLICATE_ROW_SET)):
        for arm in DP_ARMS:
            store(
                name=f"{arm}_{period}",
                frame=dp_frame(arm=arm, period=period, sites=sites).with_columns(arm=pl.lit(arm)),
                row_set_id=row_set,
            )
        for arm in XGB_ARMS:
            frame = score.load_xgb_raw(arm=arm, sites=sites)
            frame = score.in_period(frame=frame, periods={period})
            if frame.is_empty():
                continue
            frame = with_switching(frame=frame, sites=sites).with_columns(arm=pl.lit(arm))
            if arm == "gam_margin":
                frame = margin_column(frame=frame, sites=sites)
            store(name=f"{arm}_{period}", frame=frame, row_set_id=row_set)
    own = score.load_xgb_raw(arm="xnl", sites=sites)
    replicate = score.in_period(frame=own, periods={"replicate"})
    store(
        name="xnl_replicate",
        frame=with_switching(frame=replicate, sites=sites).with_columns(arm=pl.lit("xnl")),
        row_set_id=REPLICATE_ROW_SET,
    )
    scored = score.in_period(frame=score.load_simp1_raw(sites=sites), periods={"scored"})
    store(
        name="xnl_scored",
        frame=with_switching(frame=scored, sites=sites).with_columns(arm=pl.lit("xnl")),
        row_set_id=common.ROW_SET_ID,
    )


if __name__ == "__main__":
    main()
